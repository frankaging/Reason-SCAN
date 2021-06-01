from .config import cfg
from .utils import *

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, pad_idx, input_size):
        super().__init__()  # general
        self.device = device

        # configs
        self.d_x = input_size
        self.d_embed = cfg.CMD_D_EMBED

        # layers
        self.embedding = nn.Embedding(self.d_x, self.d_embed,
                                      padding_idx=pad_idx)  # \Pay attention to whether d_x include sos, eos, etc.
        self.enc_input_drop = nn.Dropout(1 - cfg.encInputDropout)
        self.rnn0 = BiLSTM()
        self.cmd_h_drop = nn.Dropout(1 - cfg.cmdDropout)

    def forward(self, cmdIdx, cmdLengths):
        cmd = self.embedding(cmdIdx)
        cmd = self.enc_input_drop(cmd)

        # RNN (LSTM)
        cmds_out, cmds_h = self.rnn0(cmd, cmdLengths)
        cmds_h = self.cmd_h_drop(cmds_h)

        return cmds_out, cmds_h

class BiLSTM(nn.Module):
    def __init__(self, forget_gate_bias=1.):
        super().__init__()
        self.bilstm = torch.nn.LSTM(
            input_size=cfg.CMD_D_EMBED, hidden_size=cfg.CMD_D_ENC // 2,
            num_layers=1, batch_first=True, bidirectional=True)

        d = cfg.CMD_D_ENC // 2  # before is ENC_DIM

        # initialize LSTM weights (to be consistent with TensorFlow)
        fan_avg = (d * 4 + (d + cfg.CMD_D_EMBED)) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.bilstm.weight_ih_l0, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_hh_l0, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_ih_l0_reverse, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_hh_l0_reverse, -bound, bound)

        # initialize LSTM forget gate bias (to be consistent with TensorFlow)
        self.bilstm.bias_ih_l0.data[...] = 0.
        self.bilstm.bias_ih_l0.data[d:2 * d] = forget_gate_bias
        self.bilstm.bias_hh_l0.data[...] = 0.
        self.bilstm.bias_hh_l0.requires_grad = False
        self.bilstm.bias_ih_l0_reverse.data[...] = 0.
        self.bilstm.bias_ih_l0_reverse.data[d:2 * d] = forget_gate_bias
        self.bilstm.bias_hh_l0_reverse.data[...] = 0.
        self.bilstm.bias_hh_l0_reverse.requires_grad = False

    def forward(self, questions, questionLengths):
        # sort samples according to question length (descending)
        sorted_lengths, indices = torch.sort(questionLengths, descending=True)
        sorted_questions = questions[indices]
        _, desorted_indices = torch.sort(indices, descending=False)

        # pack questions for LSTM forwarding

        packed_questions = nn.utils.rnn.pack_padded_sequence(
            sorted_questions, sorted_lengths.cpu(), batch_first=True)
        packed_output, (sorted_h_n, _) = self.bilstm(packed_questions)
        sorted_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=questions.size(1))
        sorted_h_n = torch.transpose(sorted_h_n, 1, 0).reshape(
            questions.size(0), -1)

        # sort back to the original sample order
        output = sorted_output[desorted_indices]
        h_n = sorted_h_n[desorted_indices]

        return output, h_n
