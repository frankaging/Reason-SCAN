import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List
from typing import Dict
from typing import Tuple
import os
import shutil

from seq2seq.cnn_model import ConvolutionalNet
from seq2seq.cnn_model import DownSamplingConvolutionalNet
from seq2seq.seq2seq_model import EncoderRNN
from seq2seq.seq2seq_model import Attention
from seq2seq.seq2seq_model import LuongAttentionDecoderRNN
from seq2seq.seq2seq_model import BahdanauAttentionDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


class Model(nn.Module):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, target_vocabulary_size: int, encoder_dropout_p: float,
                 encoder_bidirectional: bool, num_decoder_layers: int, decoder_dropout_p: float,
                 decoder_hidden_size: int, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, conditional_attention: bool, auxiliary_task: bool,
                 simple_situation_representation: bool, attention_type: str, **kwargs):
        super(Model, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        # Attention over the output features of the ConvolutionalNet.
        # Input: [bsz, 1, decoder_hidden_size], [bsz, image_width * image_width, cnn_hidden_num_channels * 3]
        # Output: [bsz, 1, decoder_hidden_size], [bsz, 1, image_width * image_width]
        self.visual_attention = Attention(key_size=cnn_hidden_num_channels * 3, query_size=decoder_hidden_size,
                                          hidden_size=decoder_hidden_size)

        self.auxiliary_task = auxiliary_task
        if auxiliary_task:
            self.auxiliary_loss_criterion = nn.NLLLoss()

        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.encoder = EncoderRNN(input_size=input_vocabulary_size,
                                  embedding_dim=embedding_dimension,
                                  rnn_input_size=embedding_dimension,
                                  hidden_size=encoder_hidden_size, num_layers=num_encoder_layers,
                                  dropout_probability=encoder_dropout_p, bidirectional=encoder_bidirectional,
                                  padding_idx=input_padding_idx)
        # Used to project the final encoder state to the decoder hidden state such that it can be initialized with it.
        self.enc_hidden_to_dec_hidden = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.textual_attention = Attention(key_size=encoder_hidden_size, query_size=decoder_hidden_size,
                                           hidden_size=decoder_hidden_size)

        # Input: [batch_size, max_target_length], initial hidden: ([batch_size, hidden_size], [batch_size, hidden_size])
        # Input for attention: [batch_size, max_input_length, hidden_size],
        #                      [batch_size, image_width * image_width, hidden_size]
        # Output: [max_target_length, batch_size, target_vocabulary_size]
        self.attention_type = attention_type
        if attention_type == "bahdanau":
            self.attention_decoder = BahdanauAttentionDecoderRNN(hidden_size=decoder_hidden_size,
                                                                 output_size=target_vocabulary_size,
                                                                 num_layers=num_decoder_layers,
                                                                 dropout_probability=decoder_dropout_p,
                                                                 padding_idx=target_pad_idx,
                                                                 textual_attention=self.textual_attention,
                                                                 visual_attention=self.visual_attention,
                                                                 conditional_attention=conditional_attention)
        elif attention_type == "luong":
            logger.warning("Luong attention not correctly implemented.")
            self.attention_decoder = LuongAttentionDecoderRNN(hidden_size=decoder_hidden_size,
                                                              output_size=target_vocabulary_size,
                                                              num_layers=num_decoder_layers,
                                                              dropout_probability=decoder_dropout_p,
                                                              conditional_attention=conditional_attention)
        else:
            raise ValueError("Unknown attention type {} specified.".format(attention_type))

        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.loss_criterion = nn.NLLLoss(ignore_index=target_pad_idx)
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    @staticmethod
    def remove_start_of_sequence(input_tensor: torch.Tensor) -> torch.Tensor:
        """Get rid of SOS-tokens in targets batch and append a padding token to each example in the batch."""
        batch_size, max_time = input_tensor.size()
        input_tensor = input_tensor[:, 1:]
        output_tensor = torch.cat([input_tensor, torch.zeros(batch_size, device=device, dtype=torch.long).unsqueeze(
            dim=1)], dim=1)
        return output_tensor

    def get_metrics(self, target_scores: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            targets = self.remove_start_of_sequence(targets)
            mask = (targets != self.target_pad_idx).long()
            total = mask.sum().data.item()
            predicted_targets = target_scores.max(dim=2)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long()
            match_targets = (equal_targets * mask)
            match_sum_per_example = match_targets.sum(dim=1)
            expected_sum_per_example = mask.sum(dim=1)
            batch_size = expected_sum_per_example.size(0)
            exact_match = 100. * (match_sum_per_example == expected_sum_per_example).sum().data.item() / batch_size
            match_targets_sum = match_targets.sum().data.item()
            accuracy = 100. * match_targets_sum / total
        return accuracy, exact_match

    @staticmethod
    def get_auxiliary_accuracy(target_scores: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            predicted_targets = target_scores.max(dim=1)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long().sum().data.item()
            accuracy = 100. * equal_targets / len(targets)
        return accuracy

    def get_loss(self, target_scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        targets = self.remove_start_of_sequence(targets)

        # Calculate the loss.
        _, _, vocabulary_size = target_scores.size()
        target_scores_2d = target_scores.reshape(-1, vocabulary_size)
        loss = self.loss_criterion(target_scores_2d, targets.view(-1))
        return loss

    def get_auxiliary_loss(self, auxiliary_scores_target: torch.Tensor, target_target_positions: torch.Tensor):
        target_loss = self.auxiliary_loss_criterion(auxiliary_scores_target, target_target_positions.view(-1))
        return target_loss

    def auxiliary_task_forward(self, output_scores_target_pos: torch.Tensor) -> torch.Tensor:
        assert self.auxiliary_task, "Please set auxiliary_task to True if using it."
        batch_size, _ = output_scores_target_pos.size()
        output_scores_target_pos = F.log_softmax(output_scores_target_pos, -1)
        return output_scores_target_pos

    def encode_input(self, commands_input: torch.LongTensor, commands_lengths: List[int],
                     situations_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        hidden, encoder_outputs = self.encoder(commands_input, commands_lengths)
        return {"encoded_situations": encoded_image, "encoded_commands": encoder_outputs, "hidden_states": hidden}

    def decode_input(self, target_token: torch.LongTensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoder_outputs: torch.Tensor, input_lengths: List[int],
                     encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decoding step based on the previous hidden state of the decoder and the previous target token."""
        return self.attention_decoder.forward_step(input_tokens=target_token, last_hidden=hidden,
                                                   encoded_commands=encoder_outputs, commands_lengths=input_lengths,
                                                   encoded_situations=encoded_situations)

    def decode_input_batched(self, target_batch: torch.LongTensor, target_lengths: List[int],
                             initial_hidden: torch.Tensor, encoded_commands: torch.Tensor,
                             command_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                    torch.Tensor]:
        """Decode a batch of input sequences."""
        initial_hidden = self.attention_decoder.initialize_hidden(
            self.tanh(self.enc_hidden_to_dec_hidden(initial_hidden)))
        decoder_output_batched, _, context_situation = self.attention_decoder(input_tokens=target_batch,
                                                                              input_lengths=target_lengths,
                                                                              init_hidden=initial_hidden,
                                                                              encoded_commands=encoded_commands,
                                                                              commands_lengths=command_lengths,
                                                                              encoded_situations=encoded_situations)
        decoder_output_batched = F.log_softmax(decoder_output_batched, dim=-1)
        return decoder_output_batched, context_situation

    def forward(self, commands_input: torch.LongTensor, commands_lengths: List[int], situations_input: torch.Tensor,
                target_batch: torch.LongTensor, target_lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.encode_input(commands_input=commands_input, commands_lengths=commands_lengths,
                                           situations_input=situations_input)
        decoder_output, context_situation = self.decode_input_batched(
            target_batch=target_batch, target_lengths=target_lengths, initial_hidden=encoder_output["hidden_states"],
            encoded_commands=encoder_output["encoded_commands"]["encoder_outputs"], command_lengths=commands_lengths,
            encoded_situations=encoder_output["encoded_situations"])
        if self.auxiliary_task:
            target_position_scores = self.auxiliary_task_forward(context_situation)
        else:
            target_position_scores = torch.zeros(1), torch.zeros(1)
        return (decoder_output.transpose(0, 1),  # [batch_size, max_target_seq_length, target_vocabulary_size]
                target_position_scores)

    def update_state(self, is_best: bool, accuracy=None, exact_match=None) -> {}:
        self.trained_iterations += 1
        if is_best:
            self.best_exact_match = exact_match
            self.best_accuracy = accuracy
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

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
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path
