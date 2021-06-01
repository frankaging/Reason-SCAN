from typing import Tuple

from .config import cfg
from .utils import *


class Attention(nn.Module):

    def __init__(self, key_size: int, query_size: int, hidden_size: int):
        super(Attention, self).__init__()
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, queries: torch.Tensor, projected_keys: torch.Tensor, values: torch.Tensor,
                memory_lengths: List[int]):
        """
        Key-value memory which takes queries and retrieves weighted combinations of values
          This version masks out certain memories, so that you can differing numbers of memories per batch.
        :param queries: [batch_size, 1, query_dim]
        :param projected_keys: [batch_size, num_memory, query_dim]
        :param values: [batch_size, num_memory, value_dim]
        :param memory_lengths: [batch_size] actual number of keys in each batch
        :return:
            soft_values_retrieval : soft-retrieval of values; [batch_size, 1, value_dim]
            attention_weights : soft-retrieval of values; [batch_size, 1, n_memory]
        """
        batch_size = projected_keys.size(0)
        assert len(memory_lengths) == batch_size
        if type(memory_lengths) == list:
            memory_lengths = torch.tensor(memory_lengths, dtype=torch.long, device=self.device)

        # Project queries down to the correct dimension.
        # [bsz, 1, query_dimension] X [bsz, query_dimension, hidden_dim] = [bsz, 1, hidden_dim]
        queries = self.query_layer(queries)

        # [bsz, 1, query_dim] X [bsz, query_dim, num_memory] = [bsz, num_memory, 1]
        scores = self.energy_layer(torch.tanh(queries + projected_keys))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out keys that are on a padding location.encoded_commands
        mask = sequence_mask(memory_lengths)  # [batch_size, num_memory]
        mask = mask.unsqueeze(1)  # [batch_size, 1, num_memory]
        scores = scores.masked_fill(mask == 0, float('-inf'))  # fill with large negative numbers
        attention_weights = F.softmax(scores, dim=2)  # [batch_size, 1, num_memory]

        # [bsz, 1, num_memory] X [bsz, num_memory, value_dim] = [bsz, 1, value_dim]
        soft_values_retrieval = torch.bmm(attention_weights, values)
        return soft_values_retrieval, attention_weights


class BahdanauAttentionDecoderRNN(nn.Module):
    """One-step batch decoder with Luong et al. attention"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, textual_attention: Attention,
                 visual_attention: Attention, dropout_probability=0.1, padding_idx=0,
                 conditional_attention=False, is_baseline=True):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(BahdanauAttentionDecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.conditional_attention = conditional_attention
        if self.conditional_attention:
            self.queries_to_keys = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(hidden_size * 3, hidden_size, num_layers=num_layers, dropout=dropout_probability)
        self.textual_attention = textual_attention
        self.visual_attention = visual_attention
        self.output_to_hidden = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_baseline = is_baseline

    def forward_step(self, input_tokens, last_hidden,
                     encoded_commands, commands_lengths,
                     encoded_situations, situations_lengths):
        """
        Run batch decoder forward for a single time step.
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)
        :param input_tokens: one time step inputs tokens of length batch_size
        :param last_hidden: previous decoder state, which is pair of tensors [num_layers, batch_size, hidden_size]
        (pair for hidden and cell)
        :param encoded_commands: all encoder outputs, [max_input_length, batch_size, hidden_size]
        :param commands_lengths: length of each padded input seqencoded_commandsuence that were passed to the encoder.
        :param encoded_situations: the situation encoder outputs, [image_dimension * image_dimension, batch_size,
         hidden_size]
        :return: output : un-normalized output probabilities, [batch_size, output_size]
          hidden : current decoder state, which is a pair of tensors [num_layers, batch_size, hidden_size]
           (pair for hidden and cell)
          attention_weights : attention weights, [batch_size, 1, max_input_length]
        """
        last_hidden, last_cell = last_hidden

        # Embed each input symbol
        embedded_input = self.embedding(input_tokens)  # [batch_size, hidden_size]
        embedded_input = self.dropout(embedded_input)
        embedded_input = embedded_input.unsqueeze(0)  # [1, batch_size, hidden_size]

        # Bahdanau attention
        context_command, attention_weights_commands = self.textual_attention(
            queries=last_hidden.transpose(0, 1), projected_keys=encoded_commands,
            values=encoded_commands, memory_lengths=commands_lengths)
        batch_size, image_num_memory, _ = encoded_situations.size()
        # situation_lengths = [image_num_memory for _ in range(batch_size)]

        if self.conditional_attention:
            queries = torch.cat([last_hidden.transpose(0, 1), context_command], dim=-1)
            queries = self.tanh(self.queries_to_keys(queries))
        else:
            queries = last_hidden.transpose(0, 1)

        # visual attention
        context_situation, attention_weights_situations = self.visual_attention(
            queries=queries, projected_keys=encoded_situations,
            values=encoded_situations, memory_lengths=situations_lengths)
        # context : [batch_size, 1, hidden_size]
        # attention_weights : [batch_size, 1, max_input_length]

        # Concatenate the context vector and RNN hidden state, and map to an output
        attention_weights_commands = attention_weights_commands.squeeze(1)  # [batch_size, max_input_length]
        attention_weights_situations = attention_weights_situations.squeeze(1)  # [batch_size, im_dim * im_dim]
        concat_input = torch.cat([embedded_input,
                                  context_command.transpose(0, 1),
                                  context_situation.transpose(0, 1)], dim=2)  # [1, batch_size hidden_size*3]

        last_hidden = (last_hidden, last_cell)
        lstm_output, hidden = self.lstm(concat_input, last_hidden)
        # lstm_output: [1, batch_size, hidden_size]
        # hidden: tuple of each [num_layers, batch_size, hidden_size] (pair for hidden and cell)

        # Concatenate all outputs and project to output size.
        if self.is_baseline:
            pre_output = torch.cat([embedded_input, lstm_output,
                                    context_command.transpose(0, 1), context_situation.transpose(0, 1)], dim=2)
            pre_output = self.output_to_hidden(pre_output)  # [1, batch_size, hidden_size]
            output = self.hidden_to_output(pre_output)  # [1, batch_size, output_size]
        else:
            output = self.hidden_to_output(lstm_output)

        output = output.squeeze(dim=0)  # [batch_size, output_size]

        return (output, hidden, attention_weights_situations.squeeze(dim=1), attention_weights_commands,
                attention_weights_situations)
        # output : [un-normalized probabilities] [batch_size, output_size]
        # hidden: tuple of size [num_layers, batch_size, hidden_size] (for hidden and cell)
        # attention_weights: [batch_size, max_input_length]

    def forward(self, input_tokens, input_lengths,
                init_hidden, encoded_commands,
                commands_lengths, encoded_situations, situations_lengths):
        """
        Run batch attention decoder forward for a series of steps
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)
        :param input_tokens: [batch_size, max_length];  padded target sequences
        :param input_lengths: [batch_size] for sequence length of each padded target sequence
        :param init_hidden: tuple of tensors [num_layers, batch_size, hidden_size] (for hidden and cell)
        :param encoded_commands: [max_input_length, batch_size, embedding_dim]
        :param commands_lengths: [batch_size] sequence length of each encoder sequence (without padding)
        :param encoded_situations: [batch_size, image_width * image_width, image_features]; encoded image situations.
        :return: output : unnormalized log-score, [max_length, batch_size, output_size]
          hidden : current decoder state, tuple with each [num_layers, batch_size, hidden_size] (for hidden and cell)
        """
        batch_size, max_time = input_tokens.size()

        # Sort the sequences by length in descending order
        # input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=self.device)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_tokens_sorted = input_tokens.index_select(dim=0, index=perm_idx)
        initial_h, initial_c = init_hidden

        hidden = (initial_h.index_select(dim=1, index=perm_idx),
                  initial_c.index_select(dim=1, index=perm_idx))

        encoded_commands = encoded_commands.index_select(dim=0, index=perm_idx)  # change from 1 to 0
        commands_lengths = torch.tensor(commands_lengths, device=self.device)
        commands_lengths = commands_lengths.index_select(dim=0, index=perm_idx)
        encoded_situations = encoded_situations.index_select(dim=0, index=perm_idx)

        # For efficiency
        projected_keys_visual = self.visual_attention.key_layer(
            encoded_situations)  # [batch_size, situation_length, dec_hidden_dim]
        projected_keys_textual = self.textual_attention.key_layer(
            encoded_commands)  # [max_input_length, batch_size, dec_hidden_dim]

        all_attention_weights = []
        lstm_output = []
        for time in range(max_time):
            input_token = input_tokens_sorted[:, time]
            (output, hidden, context_situation, attention_weights_commands,
             attention_weights_situations) = self.forward_step(input_token, hidden, projected_keys_textual,
                                                               commands_lengths,
                                                               projected_keys_visual, situations_lengths)
            all_attention_weights.append(attention_weights_situations.unsqueeze(0))
            lstm_output.append(output.unsqueeze(0))
        lstm_output = torch.cat(lstm_output, dim=0)  # [max_time, batch_size, output_size]
        attention_weights = torch.cat(all_attention_weights, dim=0)  # [max_time, batch_size, situation_dim**2]

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = lstm_output.index_select(dim=1, index=unperm_idx)  # [max_time, batch_size, output_size]
        seq_len = input_lengths[unperm_idx].tolist()
        attention_weights = attention_weights.index_select(dim=1, index=unperm_idx)

        return lstm_output, seq_len, attention_weights.sum(dim=0)
        # output : [unnormalized log-score] [max_length, batch_size, output_size]
        # seq_len : length of each output sequence

    def initialize_hidden(self, encoder_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populate the hidden variables with a message from the encoder.
        All layers, and both the hidden and cell vectors, are filled with the same message.
        :param encoder_message:  [batch_size, hidden_size] tensor
        :return: tuple of Tensors representing the hidden and cell state of shape: [num_layers, batch_size, hidden_dim]
        """
        encoder_message = encoder_message.unsqueeze(0)  # [1, batch_size, hidden_size]
        encoder_message = encoder_message.expand(self.num_layers, -1,
                                                 -1).contiguous()  # [num_layers, batch_size, hidden_size]
        return encoder_message.clone(), encoder_message.clone()

    def extra_repr(self) -> str:
        return "AttentionDecoderRNN\n num_layers={}\n hidden_size={}\n dropout={}\n num_output_symbols={}\n".format(
            self.num_layers, self.hidden_size, self.dropout_probability, self.output_size
        )


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, target_pad_idx, visual_key_size=cfg.SITU_D_CNN_OUTPUT * 3, \
                 visual_query_size=cfg.DEC_D_H, visual_hidden_size=cfg.DEC_D_H,
                 is_baseline=True):
        super().__init__()
        # if CNN then LGCN: visual_key_size = cfg.SITU_D_CTX
        self.visual_attention = Attention(key_size=visual_key_size, \
                                          query_size=visual_query_size, hidden_size=visual_hidden_size)
        self.textual_attention = Attention(key_size=cfg.CMD_D_H, query_size=cfg.DEC_D_H, hidden_size=cfg.DEC_D_H)

        self.attentionDecoder = BahdanauAttentionDecoderRNN(hidden_size=cfg.DEC_D_H,
                                                            output_size=target_vocab_size,
                                                            num_layers=cfg.DEC_NUM_LAYER,
                                                            dropout_probability=cfg.decoderDropout,
                                                            padding_idx=target_pad_idx,
                                                            textual_attention=self.textual_attention,
                                                            visual_attention=self.visual_attention,
                                                            conditional_attention=cfg.DEC_CONDITIONAL_ATTENTION,
                                                            is_baseline=is_baseline)

        self.tanh = nn.Tanh()
        self.enc_hidden_to_dec_hidden = nn.Linear(cfg.CMD_D_H, cfg.DEC_D_H)

    def forward(self, target_batch, target_length, initial_hidden, encoded_commands,
                commands_lengths, encoded_situations, situations_lengths):
        # print("initial_hidden size is ", initial_hidden.size())

        initial_hidden = self.attentionDecoder.initialize_hidden(
            self.tanh(self.enc_hidden_to_dec_hidden(initial_hidden))
        )

        decoder_output, _, context_situation = self.attentionDecoder(input_tokens=target_batch,
                                                                     input_lengths=target_length,
                                                                     init_hidden=initial_hidden,
                                                                     encoded_commands=encoded_commands,
                                                                     commands_lengths=commands_lengths,
                                                                     encoded_situations=encoded_situations,
                                                                     situations_lengths=situations_lengths)
        decoder_output = F.log_softmax(decoder_output, dim=-1)

        return decoder_output, context_situation
