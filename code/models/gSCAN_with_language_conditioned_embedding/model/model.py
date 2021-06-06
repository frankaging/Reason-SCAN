import logging
import os
import shutil

import dgl
import torch as th
import torch.nn as nn

from model.cnn_model import ConvolutionalNet
from .config import cfg
from .decoder import Decoder
from .encoder import Encoder
from .gnn import LGCNLayer

logger = logging.getLogger(__name__)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

class GSCAN_model(nn.Module):
    def __init__(self, pad_idx, target_eos_idx, input_vocab_size, target_vocab_size, output_directory=None,
                 is_baseline=False, multigpu=False,
                 ):
        super().__init__()

        self.num_vocab = input_vocab_size

        self.encoder = Encoder(pad_idx, input_vocab_size)
        self.decoder = None
        # self.situation_encoder = None
        self.lgcn = None
        self.size_embedding = nn.Linear(4, 16, bias=False)  # 64
        self.shape_embedding = nn.Linear(4, 16, bias=False) # ReaSCAN dimension bump +1
        self.yrgb_embedding = nn.Linear(4, 16, bias=False)
        self.agent_embedding = nn.Linear(5, 16, bias=False)  # skip the first bit
        self.is_baseline = is_baseline
        # if CNN first then LGCN && embedding, num_channels = 256, num_conv_channels=50, SITU_D_FEAT = 150;
        # if LGCN first then CNN && embedding, num_channels = cfg.SITU_D_CTX, num_conv_channels = 50, SITU_D_FEAT = 256
        self.situation_encoder = None
        if is_baseline:
            # self.situation_encoder = ConvolutionalNet(num_channels=64,
            #                                           cnn_kernel_size=7,
            #                                           num_conv_channels=50,
            #                                           dropout_probability=0.1,
            #                                           flatten_output=True)  # ablation
            self.situation_encoder = ConvolutionalNet(num_channels=16,
                                                      cnn_kernel_size=7,
                                                      num_conv_channels=50,
                                                      dropout_probability=0.1,
                                                      flatten_output=True)  # baseline
            self.decoder = Decoder(target_vocab_size, pad_idx, visual_key_size=50 * 3, is_baseline=is_baseline)
        else:
            self.situation_encoder = ConvolutionalNet(num_channels=cfg.SITU_D_CTX,
                                                      cnn_kernel_size=7,
                                                      num_conv_channels=cfg.SITU_D_CNN_OUTPUT,
                                                      dropout_probability=0.1,
                                                      flatten_output=True)
            self.lgcn = LGCNLayer()
            self.decoder = Decoder(target_vocab_size, pad_idx, is_baseline=is_baseline)

        if multigpu:
            raise Exception("Buggy implmentation!")
            self.encoder = th.nn.DataParallel(self.encoder)
            self.decoder = th.nn.DataParallel(self.decoder)
            if self.lgcn is not None:
                self.lgcn = th.nn.DataParallel(self.lgcn)
            self.situation_encoder = th.nn.DataParallel(self.situation_encoder)

        self.loss_criterion = nn.NLLLoss(ignore_index=pad_idx)
        self.tanh = nn.Tanh()
        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = pad_idx
        self.auxiliary_task = cfg.AUXILIARY_TASK

        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    def nonzero_extractor(self, x, cnn_out=None):
        lx = []
        for i in range(x.size(0)):
            sum_x = th.sum(x[i], dim=-1)
            if cnn_out is None:
                lx.append(x[i, sum_x.gt(0), :])
            else:
                lx.append(cnn_out[i, sum_x.gt(0), :])
        return lx

    def nonzero_insertor(self, node_out, situation_batch):
        padding_len = node_out[0].size()[-1] - situation_batch.size()[-1]
        # assume B X H X W X K size for situation_batch
        B, H, W, _ = situation_batch.size()
        situation_paddng = th.zeros(B, H, W, padding_len).to(situation_batch.device)
        situation_batch = th.cat((situation_batch, situation_paddng), dim=-1)
        for i in range(situation_batch.size(0)):
            sum_x = th.sum(situation_batch[i], dim=-1)
            situation_batch[i, sum_x.gt(0), :] = node_out[i]

        return situation_batch

    @staticmethod
    def remove_start_of_sequence(input_tensor, target_pad_idx=0):
        """Get rid of SOS-tokens in targets batch and append a padding token to each example in the batch."""
        # return input_tensor
        batch_size, max_time = input_tensor.size()
        input_tensor = input_tensor[:, 1:]
        output_tensor = th.cat([input_tensor, th.zeros([batch_size, 1], device=device, dtype=th.long) + target_pad_idx],
                               dim=1)
        return output_tensor

    def get_metrics(self, target_scores, targets):
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with th.no_grad():
            targets = self.remove_start_of_sequence(targets, target_pad_idx=self.target_pad_idx)
            mask = (targets != self.target_pad_idx).long()
            total = mask.sum().data.item()
            predicted_targets = target_scores.max(dim=2)[1]
            equal_targets = th.eq(targets.data, predicted_targets.data).long()
            match_targets = (equal_targets * mask)
            match_sum_per_example = match_targets.sum(dim=1)
            expected_sum_per_example = mask.sum(dim=1)
            batch_size = expected_sum_per_example.size(0)
            exact_match = 100. * (match_sum_per_example == expected_sum_per_example).sum().data.item() / batch_size
            match_targets_sum = match_targets.sum().data.item()
            accuracy = 100. * match_targets_sum / total
        return accuracy, exact_match

    def encode_input(self, cmd_batch, situation_batch):
        batchSize = cmd_batch[0].size(0)
        # print("batch size is ", batchSize)
        cmdIndices, cmdLengths = cmd_batch[0], cmd_batch[1]
        # tgtIndices, tgtLengths = tgt_batch[0], tgt_batch[1]

        # LSTM
        cmd_out, cmd_h = self.encoder(cmdIndices, cmdLengths)

        # convert situation to embedding
        size = self.size_embedding(situation_batch[:, :, :, :4])
        shape = self.shape_embedding(situation_batch[:, :, :, 4:8])
        rgb = self.yrgb_embedding(situation_batch[:, :, :, 8:12])
        agent = self.agent_embedding(situation_batch[:, :, :, 12:])
        embedded_situation = th.cat([size, shape, rgb, agent], dim=-1)

        if self.is_baseline:
            # situation_out = self.situation_encoder(embedded_situation)  # ablation
            situation_out = self.situation_encoder(situation_batch)  # baseline
            batch_size, image_num_memory, _ = situation_out.size()
            situations_lengths = [image_num_memory for _ in range(batch_size)]
        else:
            # LGCN first, then CNN
            xs = self.nonzero_extractor(situation_batch, embedded_situation)
            gs = []
            graph_membership = []
            dgl_gs = [dgl.DGLGraph() for _ in range(batchSize)]
            for i, x in enumerate(xs):
                dgl_gs[i] = dgl_gs[i].to(self.device)
                dgl_gs[i].add_nodes(x.size(0))
                src_l, dst_l = [], []
                for j in range(x.size(0)):
                    for k in range(x.size(0)):
                        if j != k:
                            src_l.append(j)
                            dst_l.append(k)
                dgl_gs[i].add_edges(src_l, dst_l)
                graph_membership += [i for _ in range(x.size(0))]
            batch_g = dgl.batch(dgl_gs)
            situation_X = th.cat(xs, dim=0)
            graph_membership = th.tensor(graph_membership, dtype=th.long, device=self.device)

            # LGCN
            situation_out_node = self.lgcn(situation_X, batch_g, cmd_h, cmd_out, cmdLengths, batchSize,
                                           graph_membership)
            situation_batch = self.nonzero_insertor(situation_out_node, situation_batch)
            situation_out = self.situation_encoder(situation_batch)
            batch_size, image_num_memory, _ = situation_out.size()
            situations_lengths = [image_num_memory for _ in range(batch_size)]

        return cmd_h, cmd_out, cmdLengths, situation_out, situations_lengths

    def forward(self, cmd_batch, situation_batch, tgt_batch):
        '''
        cmd_batch[0]: batchsize x max_length
        cmd_batch[1]: batchsize
        situation_batch[0]: batchsize x grid x grid x k
        '''
        batchSize = cmd_batch[0].size(0)
        # print("batch size is ", batchSize)
        # cmdIndices, cmdLengths = cmd_batch[0], cmd_batch[1]
        tgtIndices, tgtLengths = tgt_batch[0], tgt_batch[1]

        cmd_h, cmd_out, cmdLengths, situation_out, situations_lengths = self.encode_input(cmd_batch, situation_batch)

        # Decoder
        decoder_output, context_situation = self.decoder(tgtIndices, tgtLengths, initial_hidden=cmd_h,
                                                         encoded_commands=cmd_out, commands_lengths=cmdLengths,
                                                         encoded_situations=situation_out,
                                                         situations_lengths=situations_lengths)

        if self.auxiliary_task:
            target_position_scores = self.auxilaiary_task_forward(context_situation)
        else:
            target_position_scores = th.zeros(1), th.zeros(1)

        return (decoder_output.transpose(0, 1),
                target_position_scores)  # decoder_output shape: [batch_size, max_target_seq_length, target_vocab_size]

    def get_loss(self, target_score, target):
        target = self.remove_start_of_sequence(target, target_pad_idx=self.target_pad_idx)

        # Calculate the loss
        _, _, vocabulary_size = target_score.size()
        target_score_2d = target_score.reshape(-1, vocabulary_size)
        loss = self.loss_criterion(target_score_2d, target.view(-1))
        return loss

    def update_state(self, is_best: bool, accuracy=None, exact_match=None) -> {}:
        self.trained_iterations += 1
        if is_best:
            self.best_exact_match = exact_match
            self.best_accuracy = accuracy
            self.best_iteration = self.trained_iterations

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """
        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        th.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path

    def get_current_state(self):
        return {'model_state_dict': self.state_dict()}

    def load_model(self, path):
        d = th.load(path)
        self.load_state_dict(d['model_state_dict'])
        return d['optimizer_state_dict']
