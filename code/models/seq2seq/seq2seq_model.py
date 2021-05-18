# Code adapted from:
#   https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import logging
from typing import List
from typing import Tuple

from seq2seq.helpers import sequence_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class EncoderRNN(nn.Module):
    """
    Embed a sequence of symbols using an LSTM.

    The RNN hidden vector (not cell vector) at each step is captured,
      for transfer to an attention-based decoder
    """
    def __init__(self, input_size: int, embedding_dim: int, rnn_input_size: int, hidden_size: int, num_layers: int,
                 dropout_probability: float, bidirectional: bool, padding_idx: int):
        """
        :param input_size: number of input symbols
        :param embedding_dim: number of hidden units in RNN encoder, and size of all embeddings
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        :param bidirectional: use a bidirectional LSTM instead and sum of the resulting embeddings
        """
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.dropout_probability = dropout_probability
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout_probability, bidirectional=bidirectional)

    def forward(self, input_batch: torch.LongTensor, input_lengths: List[int]) -> Tuple[torch.Tensor, dict]:
        """
        :param input_batch: [batch_size, max_length]; batched padded input sequences
        :param input_lengths: length of each padded input sequence.
        :return: hidden states for last layer of last time step, the output of the last layer per time step and
        the sequence lengths per example in the batch.
        NB: The hidden states in the bidirectional case represent the final hidden state of each directional encoder,
        meaning the whole sequence in both directions, whereas the output per time step represents different parts of
        the sequences (0:t for the forward LSTM, t:T for the backward LSTM).
        """
        assert input_batch.size(0) == len(input_lengths), "Wrong amount of lengths passed to .forward()"
        input_embeddings = self.embedding(input_batch)  # [batch_size, max_length, embedding_dim]
        input_embeddings = self.dropout(input_embeddings)  # [batch_size, max_length, embedding_dim]

        # Sort the sequences by length in descending order.
        batch_size = len(input_lengths)
        max_length = max(input_lengths)
        input_lengths = torch.tensor(input_lengths, device=device, dtype=torch.long)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_embeddings = input_embeddings.index_select(dim=0, index=perm_idx)

        # RNN embedding.
        packed_input = pack_padded_sequence(input_embeddings, input_lengths.cpu(), batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        # hidden, cell [num_layers * num_directions, batch_size, embedding_dim]
        # hidden and cell are unpacked, such that they store the last hidden state for each sequence in the batch.
        output_per_timestep, _ = pad_packed_sequence(
            packed_output)  # [max_length, batch_size, hidden_size * num_directions]

        # If biLSTM, sum the outputs for each direction
        if self.bidirectional:
            output_per_timestep = output_per_timestep.view(int(max_length), batch_size, 2, self.hidden_size)
            output_per_timestep = torch.sum(output_per_timestep, 2)  # [max_length, batch_size, hidden_size]
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            hidden = torch.sum(hidden, 1)  # [num_layers, batch_size, hidden_size]
        hidden = hidden[-1, :, :]  # [batch_size, hidden_size] (get the last layer)

        # Reverse the sorting.
        _, unperm_idx = perm_idx.sort(0)
        hidden = hidden.index_select(dim=0, index=unperm_idx)
        output_per_timestep = output_per_timestep.index_select(dim=1, index=unperm_idx)
        input_lengths = input_lengths[unperm_idx].tolist()
        return hidden, {"encoder_outputs": output_per_timestep, "sequence_lengths": input_lengths}

    def extra_repr(self) -> str:
        return "EncoderRNN\n bidirectional={} \n num_layers={}\n hidden_size={}\n dropout={}\n "\
               "n_input_symbols={}\n".format(self.bidirectional, self.num_lauers, self.hidden_size,
                                             self.dropout_probability, self.input_size)


class Attention(nn.Module):

    def __init__(self, key_size: int, query_size: int, hidden_size: int):
        super(Attention, self).__init__()
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

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
        memory_lengths = torch.tensor(memory_lengths, dtype=torch.long, device=device)

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

        # [bsz, 1, num_memory] X [bsz, num_memory, value_dim] = [bsz, 1, value_dim]
        soft_values_retrieval = torch.bmm(attention_weights, values)
        return soft_values_retrieval, attention_weights


class LuongAttentionDecoderRNN(nn.Module):
    """One-step batch decoder with Luong et al. attention"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, dropout_probability=0.1,
                 conditional_attention=False):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(LuongAttentionDecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.conditional_attention = conditional_attention
        if self.conditional_attention:
            self.queries_to_keys = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)  # TODO: change
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_probability)
        self.attention = Attention()
        self.hidden_context_to_hidden = nn.Linear(hidden_size * 3, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward_step(self, input_tokens: torch.LongTensor, last_hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoded_commands: torch.Tensor, commands_lengths: List[int],
                     encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run batch decoder forward for a single time step.
         Each decoder step considers all of the encoder_outputs through attention.
             Attention retrieval is based on decoder hidden state (not cell state)

        :param input_tokens: one time step inputs tokens of length batch_size
        :param last_hidden: previous decoder state, which is pair of tensors [num_layers, batch_size, hidden_size]
        (pair for hidden and cell)
        :param encoded_commands: all encoder outputs, [max_input_length, batch_size, hidden_size]  # TODO: embedding dim is hidden dim
        :param commands_lengths: length of each padded input sequence that were passed to the encoder.
        :param encoded_situations: the situation encoder outputs, [image_dimension * image_dimension, batch_size,
         hidden_size]
        :return: output : un-normalized output probabilities, [batch_size, output_size]
          hidden : current decoder state, which is a pair of tensors [num_layers, batch_size, hidden_size]
           (pair for hidden and cell)
          attention_weights : attention weights, [batch_size, 1, max_input_length]
        """

        # Embed each input symbol
        embedded_input = self.embedding(input_tokens)  # [batch_size, hidden_size]
        embedded_input = self.dropout(embedded_input)
        embedded_input = embedded_input.unsqueeze(0)  # [1, batch_size, hidden_size]

        lstm_output, hidden = self.lstm(embedded_input, last_hidden)
        # lstm_output: [1, batch_size, hidden_size]
        # hidden: tuple of each [num_layers, batch_size, hidden_size] (pair for hidden and cell)
        context_command, attention_weights_commands = self.attention.forward_masked(
            queries=lstm_output.transpose(0, 1), keys=encoded_commands.transpose(0, 1),
            values=encoded_commands.transpose(0, 1), memory_lengths=commands_lengths)
        batch_size, image_num_memory, _ = encoded_situations.size()
        situation_lengths = [image_num_memory for _ in range(batch_size)]

        if self.conditional_attention:
            queries = torch.cat([lstm_output.transpose(0, 1), context_command], dim=-1)
            queries = self.queries_to_keys(queries)
            queries = self.tanh(queries)
        else:
            queries = lstm_output.transpose(0, 1)

        context_situation, attention_weights_situations = self.attention.forward_masked(
            queries=queries, keys=encoded_situations,
            values=encoded_situations, memory_lengths=situation_lengths)
        # context : [batch_size, 1, hidden_size]
        # attention_weights : [batch_size, 1, max_input_length]

        # Concatenate the context vector and RNN hidden state, and map to an output
        lstm_output = lstm_output.squeeze(0)  # [batch_size, hidden_size]
        context_command = context_command.squeeze(1)  # [batch_size, hidden_size]
        context_situation = context_situation.squeeze(1)  # [batch_size, hidden_size]
        attention_weights_commands = attention_weights_commands.squeeze(1)  # [batch_size, max_input_length]
        attention_weights_situations = attention_weights_situations.squeeze(1)  # [batch_size, im_dim * im_dim]
        concat_input = torch.cat([lstm_output,
                                  context_command,
                                  context_situation], dim=1)  # [batch_size, hidden_size*3]
        concat_output = self.tanh(self.hidden_context_to_hidden(concat_input))  # [batch_size, hidden_size]
        # concat_output = self.dropout(concat_output)
        output = self.hidden_to_output(concat_output)  # [batch_size, output_size]
        return (output, hidden, attention_weights_situations.squeeze(dim=1), attention_weights_commands,
                attention_weights_situations)
        # output : [un-normalized probabilities] [batch_size, output_size]
        # hidden: tuple of size [num_layers, batch_size, hidden_size] (for hidden and cell)
        # attention_weights: [batch_size, max_input_length]

    def forward(self, input_tokens: torch.LongTensor, input_lengths: List[int],
                init_hidden: Tuple[torch.Tensor, torch.Tensor], encoded_commands: torch.Tensor,
                commands_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, List[int],
                                                                                        torch.Tensor]:
        """
        Run batch attention decoder forward for a series of steps
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param input_tokens: [batch_size, max_length];  padded target sequences
        :param input_lengths: [batch_size] for sequence length of each padded target sequence
        :param init_hidden: tuple of tensors [num_layers, batch_size, hidden_size] (for hidden and cell)
        :param encoded_commands: [max_input_length, batch_size, embedding_dim]
        :param commands_lengths: [batch_size] sequence length of each encoder sequence (without padding)
        :param encoded_situations: [] TODO
        :return: output : unnormalized log-score, [max_length, batch_size, output_size]
          hidden : current decoder state, tuple with each [num_layers, batch_size, hidden_size] (for hidden and cell)
        """
        input_embedded = self.embedding(input_tokens)  # [batch_size, max_length, embedding_dim]
        input_embedded = self.dropout(input_embedded)  # [batch_size, max_length, embedding_dim]

        # Sort the sequences by length in descending order
        input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_embedded = input_embedded.index_select(dim=0, index=perm_idx)
        initial_h, initial_c = init_hidden
        init_hidden = (initial_h.index_select(dim=1, index=perm_idx),
                       initial_c.index_select(dim=1, index=perm_idx))

        # RNN decoder
        packed_input = pack_padded_sequence(input_embedded, input_lengths.cpu(), batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input, init_hidden)
        # hidden is [num_layers, batch_size, hidden_size] (pair for hidden and cell)
        lstm_output, _ = pad_packed_sequence(packed_output)  # [max_length, batch_size, hidden_size]

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = lstm_output.index_select(dim=1, index=unperm_idx)  # [max_length, batch_size, hidden_size]
        seq_len = input_lengths[unperm_idx].tolist()

        # Compute context vector using attention
        context_commands, attention_weights = self.attention.forward_masked(queries=lstm_output.transpose(0, 1),
                                                                            keys=encoded_commands.transpose(0, 1),
                                                                            values=encoded_commands.transpose(0, 1),
                                                                            memory_lengths=commands_lengths)
        # context_commands = self.dropout(context_commands)

        # Compute context vector using attention
        if self.conditional_attention:
            queries = torch.cat([lstm_output.transpose(0, 1), context_commands], dim=-1)
            queries = self.queries_to_keys(queries)
            queries = self.tanh(queries)
        else:
            queries = lstm_output.transpose(0, 1)
        batch_size, image_num_memory, _ = encoded_situations.size()
        situation_lengths = [image_num_memory for _ in range(batch_size)]
        context_situation, attention_weights = self.attention.forward_masked(queries=queries,
                                                                             keys=encoded_situations,
                                                                             values=encoded_situations,
                                                                             memory_lengths=situation_lengths)
        # context_situation = self.dropout(context_situation)

        # context: [batch_size, max_length, hidden_size]
        # attention_weights: [batch_size, max_length, max_input_length]

        # Concatenate the context vector and RNN hidden state, and map to an output
        concat_input = torch.cat([lstm_output,
                                  context_commands.transpose(0, 1),
                                  context_situation.transpose(0, 1)], 2)  # [max_length, batch_size, hidden_size*3]
        concat_output = self.tanh(self.hidden_context_to_hidden(concat_input))  # [max_length, batch_size, hidden_size]
        # concat_output = self.dropout(concat_output)
        output = self.hidden_to_output(concat_output)  # [max_length, batch_size, output_size]
        return output, seq_len, attention_weights.sum(dim=1)
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


class BahdanauAttentionDecoderRNN(nn.Module):
    """One-step batch decoder with Luong et al. attention"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, textual_attention: Attention,
                 visual_attention: Attention, dropout_probability=0.1, padding_idx=0,
                 conditional_attention=False):
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

    def forward_step(self, input_tokens: torch.LongTensor, last_hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoded_commands: torch.Tensor, commands_lengths: torch.Tensor,
                     encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
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
            queries=last_hidden.transpose(0, 1), projected_keys=encoded_commands.transpose(0, 1),
            values=encoded_commands.transpose(0, 1), memory_lengths=commands_lengths)
        batch_size, image_num_memory, _ = encoded_situations.size()
        situation_lengths = [image_num_memory for _ in range(batch_size)]

        if self.conditional_attention:
            queries = torch.cat([last_hidden.transpose(0, 1), context_command], dim=-1)
            queries = self.tanh(self.queries_to_keys(queries))
        else:
            queries = last_hidden.transpose(0, 1)

        context_situation, attention_weights_situations = self.visual_attention(
            queries=queries, projected_keys=encoded_situations,
            values=encoded_situations, memory_lengths=situation_lengths)
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
        # output = self.hidden_to_output(lstm_output)  # [batch_size, output_size]
        # output = output.squeeze(dim=0)

        # Concatenate all outputs and project to output size.
        pre_output = torch.cat([embedded_input, lstm_output,
                                context_command.transpose(0, 1), context_situation.transpose(0, 1)], dim=2)
        pre_output = self.output_to_hidden(pre_output)  # [1, batch_size, hidden_size]
        output = self.hidden_to_output(pre_output)  # [batch_size, output_size]
        output = output.squeeze(dim=0)   # [batch_size, output_size]

        return (output, hidden, attention_weights_situations.squeeze(dim=1), attention_weights_commands,
                attention_weights_situations)
        # output : [un-normalized probabilities] [batch_size, output_size]
        # hidden: tuple of size [num_layers, batch_size, hidden_size] (for hidden and cell)
        # attention_weights: [batch_size, max_input_length]

    def forward(self, input_tokens: torch.LongTensor, input_lengths: List[int],
                init_hidden: Tuple[torch.Tensor, torch.Tensor], encoded_commands: torch.Tensor,
                commands_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, List[int],
                                                                                        torch.Tensor]:
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
        input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_tokens_sorted = input_tokens.index_select(dim=0, index=perm_idx)
        initial_h, initial_c = init_hidden
        hidden = (initial_h.index_select(dim=1, index=perm_idx),
                  initial_c.index_select(dim=1, index=perm_idx))
        encoded_commands = encoded_commands.index_select(dim=1, index=perm_idx)
        commands_lengths = torch.tensor(commands_lengths, device=device)
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
                                                               projected_keys_visual)
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


class DecoderRNN(nn.Module):
    """One-step simple batch RNN decoder"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, dropout_probability=0.1):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_probability)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, input_tokens: torch.LongTensor, last_hidden: torch.Tensor):
        """
        Run batch decoder forward for a single time step.

        :param input_tokens: [batch_size]
        :param last_hidden: previous decoder state, tuple of [num_layers, batch_size, hidden_size] (for hidden and cell)
        :return:
          output : un-normalized output probabilities, [batch_size, output_size]
          hidden : current decoder state, tuple of [num_layers, batch_size, hidden_size] (for hidden and cell)
        """

        # Embed each input symbol
        embedding = self.embedding(input_tokens)  # [batch_size, hidden_size]
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0)  # [1, batch_size, hidden_size]
        lstm_output, hidden = self.lstm(embedding, last_hidden)
        # rnn_output is [1, batch_size, hidden_size]
        # hidden is [num_layers, batch_size, hidden_size] (pair for hidden and cell)
        lstm_output = lstm_output.squeeze(0)  # [batch_size, hidden_size]
        output = self.hidden_to_output(lstm_output)  # [batch_size, output_size]
        return output, hidden
        # output : un-normalized probabilities [batch_size, output_size]
        # hidden: pair of size [num_layers, batch_size, hidden_size] (for hidden and cell)

    def init_hidden(self, encoder_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populate the hidden variables with a message from the decoder.
        All layers, and both the hidden and cell vectors, are filled with the same message.
        :param encoder_message: [batch_size, hidden_size]
        :return:
        """
        encoder_message = encoder_message.unsqueeze(0)  # 1, batch_size, hidden_size
        encoder_message = encoder_message.expand(self.num_layers, -1,
                                                 -1).contiguous()  # nlayers, batch_size, hidden_size tensor
        return encoder_message.clone(), encoder_message.clone()

    def extra_repr(self) -> str:
        return "DecoderRNN\n num_layers={}\n hidden_size={}\n dropout={}\n num_output_symbols={}\n".format(
            self.num_layers, self.hidden_size, self.dropout_probability, self.output_size
        )
