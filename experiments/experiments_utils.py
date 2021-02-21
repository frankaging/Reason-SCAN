import argparse
import logging
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '../multimodal_seq2seq_gSCAN/'))
import random
import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Iterator
import time
import json

from seq2seq.gSCAN_dataset import GroundedScanDataset
from seq2seq.model import Model
from seq2seq.train import train
from seq2seq.predict import predict_and_save
from tqdm import tqdm, trange
from GroundedScan.dataset import GroundedScan

from typing import List
from typing import Tuple
from collections import defaultdict
from collections import Counter
import json
import numpy as np

from seq2seq.gSCAN_dataset import Vocabulary
from seq2seq.helpers import sequence_accuracy

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger(__name__)
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
use_cuda = True if torch.cuda.is_available() and not isnotebook() else False
device = "cuda" if use_cuda else "cpu"

if use_cuda:
    logger.info("Using CUDA.")
    logger.info("Cuda version: {}".format(torch.version.cuda))

def get_gSCAN_parser():
    parser = argparse.ArgumentParser(description="Sequence to sequence models for Grounded SCAN")

    # General arguments
    parser.add_argument("--mode", type=str, default="run_tests", help="train, test or predict", required=True)
    parser.add_argument("--output_directory", type=str, default="output", help="In this directory the models will be "
                                                                               "saved. Will be created if doesn't exist.")
    parser.add_argument("--resume_from_file", type=str, default="", help="Full path to previously saved model to load.")

    # Data arguments
    parser.add_argument("--split", type=str, default="test", help="Which split to get from Grounded Scan.")
    parser.add_argument("--data_directory", type=str, default="data/uniform_dataset", help="Path to folder with data.")
    parser.add_argument("--input_vocab_path", type=str, default="training_input_vocab.txt",
                        help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
    parser.add_argument("--target_vocab_path", type=str, default="training_target_vocab.txt",
                        help="Path to file with target vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
    parser.add_argument("--generate_vocabularies", dest="generate_vocabularies", default=False, action="store_true",
                        help="Whether to generate vocabularies based on the data.")
    parser.add_argument("--load_vocabularies", dest="generate_vocabularies", default=True, action="store_false",
                        help="Whether to use previously saved vocabularies.")

    # Training and learning arguments
    parser.add_argument("--training_batch_size", type=int, default=50)
    parser.add_argument("--k", type=int, default=0, help="How many examples from the adverb_1 split to move to train.")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Currently only 1 supported due to decoder.")
    parser.add_argument("--max_training_examples", type=int, default=None, help="If None all are used.")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_decay_steps', type=float, default=20000)
    parser.add_argument("--adam_beta_1", type=float, default=0.9)
    parser.add_argument("--adam_beta_2", type=float, default=0.999)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--evaluate_every", type=int, default=1000, help="How often to evaluate the model by decoding the "
                                                                         "test set (without teacher forcing).")
    parser.add_argument("--max_training_iterations", type=int, default=100000)
    parser.add_argument("--weight_target_loss", type=float, default=0.3, help="Only used if --auxiliary_task set.")

    # Testing and predicting arguments
    parser.add_argument("--max_testing_examples", type=int, default=None)
    parser.add_argument("--splits", type=str, default="test", help="comma-separated list of splits to predict for.")
    parser.add_argument("--max_decoding_steps", type=int, default=30, help="After 30 decoding steps, the decoding process "
                                                                           "is stopped regardless of whether an EOS token "
                                                                           "was generated.")
    parser.add_argument("--output_file_name", type=str, default="predict.json")

    # Situation Encoder arguments
    parser.add_argument("--simple_situation_representation", dest="simple_situation_representation", default=True,
                        action="store_true", help="Represent the situation with 1 vector per grid cell. "
                                                  "For more information, read grounded SCAN documentation.")
    parser.add_argument("--image_situation_representation", dest="simple_situation_representation", default=False,
                        action="store_false", help="Represent the situation with the full gridworld RGB image. "
                                                   "For more information, read grounded SCAN documentation.")
    parser.add_argument("--cnn_hidden_num_channels", type=int, default=50)
    parser.add_argument("--cnn_kernel_size", type=int, default=7, help="Size of the largest filter in the world state "
                                                                       "model.")
    parser.add_argument("--cnn_dropout_p", type=float, default=0.1, help="Dropout applied to the output features of the "
                                                                         "world state model.")
    parser.add_argument("--auxiliary_task", dest="auxiliary_task", default=False, action="store_true",
                        help="If set to true, the model predicts the target location from the joint attention over the "
                             "input instruction and world state.")
    parser.add_argument("--no_auxiliary_task", dest="auxiliary_task", default=True, action="store_false")

    # Command Encoder arguments
    parser.add_argument("--embedding_dimension", type=int, default=25)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--encoder_hidden_size", type=int, default=100)
    parser.add_argument("--encoder_dropout_p", type=float, default=0.3, help="Dropout on instruction embeddings and LSTM.")
    parser.add_argument("--encoder_bidirectional", dest="encoder_bidirectional", default=True, action="store_true")
    parser.add_argument("--encoder_unidirectional", dest="encoder_bidirectional", default=False, action="store_false")

    # Decoder arguments
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--attention_type", type=str, default='bahdanau', choices=['bahdanau', 'luong'],
                        help="Luong not properly implemented.")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.3, help="Dropout on decoder embedding and LSTM.")
    parser.add_argument("--decoder_hidden_size", type=int, default=100)
    parser.add_argument("--conditional_attention", dest="conditional_attention", default=True, action="store_true",
                        help="If set to true joint attention over the world state conditioned on the input instruction is"
                             " used.")
    parser.add_argument("--no_conditional_attention", dest="conditional_attention", default=False, action="store_false")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corrupt_methods", type=str, default="random") 
    parser.add_argument("--save_eval_result_dict", default=False, action="store_true") 
    
    return parser

def predict_single(example: dict, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
                   eos_idx: int, device: str) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param example: single example to play with
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    """
    # Disable dropout and other regularization.
    model.eval()

    input_sequence = example["input_tensor"]
    target_sequence = example["target_tensor"]
    input_lengths = [example["input_tensor"].size(1)]
    target_lengths = [example["target_tensor"].size(1)]
    situation = example["situation_tensor"]
    situation_spec = [example["situation_representation"]]
    derivation_spec = [example["derivation_representation"]]
    agent_positions = example["agent_position"]
    target_positions = example["target_position"]

    input_sequence = input_sequence.to(device)
    target_sequence = target_sequence.to(device)
    situation = situation.to(device)
    
    # Encode the input sequence.
    encoded_input = model.encode_input(commands_input=input_sequence,
                                       commands_lengths=input_lengths,
                                       situations_input=situation)

    # For efficiency
    projected_keys_visual = model.visual_attention.key_layer(
        encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
    projected_keys_textual = model.textual_attention.key_layer(
        encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

    # Iteratively decode the output.
    output_sequence = []
    contexts_situation = []
    hidden = model.attention_decoder.initialize_hidden(
        model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
    token = torch.tensor([sos_idx], dtype=torch.long, device=device)
    decoding_iteration = 0
    attention_weights_commands = []
    attention_weights_situations = []
    while token != eos_idx and decoding_iteration <= max_decoding_steps:
        (output, hidden, context_situation, attention_weights_command,
         attention_weights_situation) = model.decode_input(
            target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
            input_lengths=input_lengths, encoded_situations=projected_keys_visual)
        output = F.log_softmax(output, dim=-1)
        token = output.max(dim=-1)[1]
        output_sequence.append(token.data[0].item())
        attention_weights_commands.append(attention_weights_command.tolist())
        attention_weights_situations.append(attention_weights_situation.tolist())
        contexts_situation.append(context_situation.unsqueeze(1))
        decoding_iteration += 1

    if output_sequence[-1] == eos_idx:
        output_sequence.pop()
        attention_weights_commands.pop()
        attention_weights_situations.pop()
    if model.auxiliary_task:
        target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
        auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
    else:
        auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
    return (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
            attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

def make_corrupt_example(raw_example, corrupt_methods="systematic"):
    if corrupt_methods == "systematic":
        ret_example = copy.deepcopy(raw_example)
        new_command = ret_example['input_command']
        if "while" in ret_example['input_command'][-1]:
            # we move while into the front
            new_command = ret_example['input_command'][-1:] + ret_example['input_command'][:-1]
        elif ret_example['input_command'][-1][-2:] == "ly":
            # this is the adv
            new_command = ret_example['input_command'][-1:] + ret_example['input_command'][:-1]
        # we can also switch words in the middle
        # circle, square, cylinder
        # use a as a maker
        start_index = new_command.index('a')
        if "circle" in new_command:
            end_index = new_command.index('circle')
        elif "square" in new_command:
            end_index = new_command.index('square')
        elif "cylinder" in new_command:
            end_index = new_command.index('cylinder')
        if end_index - start_index > 2:
            # there are two adj then
            new_command[start_index+1:end_index] = new_command[start_index+1:end_index][::-1]
        ret_example['input_command'] = new_command
    elif corrupt_methods == "random":
        ret_example = copy.deepcopy(raw_example)
        random.shuffle(ret_example['input_command'])
    return ret_example

def levenshteinDistance(s1, s2):
    """
    The Levenshtein distance allows deletion, insertion and substitution:
    https://en.wikipedia.org/wiki/Edit_distance
    Implementation reference: 
    https://stackoverflow.com/questions/2460177/edit-distance-in-python
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    norm_dist = distances[-1]/max(len(s1), len(s2))
    return norm_dist
    
class DummyGroundedScanDataset(object):
    """
    Loads a GroundedScan instance from a specified location.
    """

    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False):
        logger.info("Initializing dummy gSCAN dataset for adverserial experiments...")
        assert os.path.exists(path_to_data), "Trying to read a gSCAN dataset from a non-existing file {}.".format(
            path_to_data)
        if not generate_vocabulary:
            assert os.path.exists(os.path.join(save_directory, input_vocabulary_file)) and os.path.exists(
                os.path.join(save_directory, target_vocabulary_file)), \
                "Trying to load vocabularies from non-existing files."
        if split == "test" and generate_vocabulary:
            logger.warning("WARNING: generating a vocabulary from the test set.")
        # self.dataset = GroundedScan.load_dataset_from_file(path_to_data, save_directory=save_directory, k=k)
        # pre-load just to get the grid size
        with open(path_to_data, 'r') as infile:
            all_data = json.load(infile)
        
        self.image_dimensions = all_data["grid_size"]
        self.image_channels = 3
        self.split = split
        self.directory = save_directory

        # Keeping track of data.
        self._examples = np.array([])
        self._input_lengths = np.array([])
        self._target_lengths = np.array([])
        if generate_vocabulary:
            logger.info("Generating vocabularies...")
            self.input_vocabulary = Vocabulary()
            self.target_vocabulary = Vocabulary()
            self.read_vocabularies()
            logger.info("Done generating vocabularies.")
        else:
            logger.info("Loading vocabularies...")
            self.input_vocabulary = Vocabulary.load(os.path.join(save_directory, input_vocabulary_file))
            self.target_vocabulary = Vocabulary.load(os.path.join(save_directory, target_vocabulary_file))
            logger.info("Done loading vocabularies.")

    def read_vocabularies(self) -> {}:
        """
        Loop over all examples in the dataset and add the words in them to the vocabularies.
        """
        logger.info("Populating vocabulary...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split)):
            self.input_vocabulary.add_sentence(example["input_command"])
            self.target_vocabulary.add_sentence(example["target_command"])

    def save_vocabularies(self, input_vocabulary_file: str, target_vocabulary_file: str):
        self.input_vocabulary.save(os.path.join(self.directory, input_vocabulary_file))
        self.target_vocabulary.save(os.path.join(self.directory, target_vocabulary_file))

    def get_vocabulary(self, vocabulary: str) -> Vocabulary:
        if vocabulary == "input":
            vocab = self.input_vocabulary
        elif vocabulary == "target":
            vocab = self.target_vocabulary
        else:
            raise ValueError("Specified unknown vocabulary in sentence_to_array: {}".format(vocabulary))
        return vocab

    def shuffle_data(self) -> {}:
        """
        Reorder the data examples and reorder the lengths of the input and target commands accordingly.
        """
        random_permutation = np.random.permutation(len(self._examples))
        self._examples = self._examples[random_permutation]
        self._target_lengths = self._target_lengths[random_permutation]
        self._input_lengths = self._input_lengths[random_permutation]

    def get_data_iterator(self, batch_size=10) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[dict],
                                                        torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Iterate over batches of example tensors, pad them to the max length in the batch and yield.
        :param batch_size: how many examples to put in each batch.
        :param auxiliary_task: if true, also batches agent and target positions (flattened, so
        agent row * agent columns = agent_position)
        :return: tuple of input commands batch, corresponding input lengths, situation image batch,
        list of corresponding situation representations, target commands batch and corresponding target lengths.
        """
        for example_i in range(0, len(self._examples), batch_size):
            if example_i + batch_size > len(self._examples):
                batch_size = len(self._examples) - example_i
            examples = self._examples[example_i:example_i + batch_size]
            input_lengths = self._input_lengths[example_i:example_i + batch_size]
            target_lengths = self._target_lengths[example_i:example_i + batch_size]
            max_input_length = np.max(input_lengths)
            max_target_length = np.max(target_lengths)
            input_batch = []
            target_batch = []
            situation_batch = []
            situation_representation_batch = []
            derivation_representation_batch = []
            agent_positions_batch = []
            target_positions_batch = []
            for example in examples:
                to_pad_input = max_input_length - example["input_tensor"].size(1)
                to_pad_target = max_target_length - example["target_tensor"].size(1)
                padded_input = torch.cat([
                    example["input_tensor"],
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                # padded_input = torch.cat([
                #     torch.zeros_like(example["input_tensor"], dtype=torch.long, device=device),
                #     torch.zeros(int(to_pad_input), dtype=torch.long, device=devicedevice).unsqueeze(0)], dim=1) # TODO: change back
                padded_target = torch.cat([
                    example["target_tensor"],
                    torch.zeros(int(to_pad_target), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                input_batch.append(padded_input)
                target_batch.append(padded_target)
                situation_batch.append(example["situation_tensor"])
                situation_representation_batch.append(example["situation_representation"])
                derivation_representation_batch.append(example["derivation_representation"])
                agent_positions_batch.append(example["agent_position"])
                target_positions_batch.append(example["target_position"])

            yield (torch.cat(input_batch, dim=0), input_lengths, derivation_representation_batch,
                   torch.cat(situation_batch, dim=0), situation_representation_batch, torch.cat(target_batch, dim=0),
                   target_lengths, torch.cat(agent_positions_batch, dim=0), torch.cat(target_positions_batch, dim=0))

    def process(self, example):

        empty_example = {}
        input_commands = example["input_command"]
        target_commands = example["target_command"]
        #equivalent_target_commands = example["equivalent_target_command"]
        situation_image = example["situation_image"]
        self.image_dimensions = situation_image.shape[0]
        self.image_channels = situation_image.shape[-1]
        situation_repr = example["situation_representation"]
        input_array = self.sentence_to_array(input_commands, vocabulary="input")
        target_array = self.sentence_to_array(target_commands, vocabulary="target")
        #equivalent_target_array = self.sentence_to_array(equivalent_target_commands, vocabulary="target")
        empty_example["input_tensor"] = torch.tensor(input_array, dtype=torch.long, device=device).unsqueeze(
            dim=0)
        empty_example["target_tensor"] = torch.tensor(target_array, dtype=torch.long, device=device).unsqueeze(
            dim=0)
        #empty_example["equivalent_target_tensor"] = torch.tensor(equivalent_target_array, dtype=torch.long,
        #                                                         device=device).unsqueeze(dim=0)
        empty_example["situation_tensor"] = torch.tensor(situation_image, dtype=torch.float, device=device
                                                         ).unsqueeze(dim=0)
        empty_example["situation_representation"] = situation_repr
        empty_example["derivation_representation"] = example["derivation_representation"]
        empty_example["agent_position"] = torch.tensor(
            (int(situation_repr["agent_position"]["row"]) * int(situation_repr["grid_size"])) +
            int(situation_repr["agent_position"]["column"]), dtype=torch.long,
            device=device).unsqueeze(dim=0)
        empty_example["target_position"] = torch.tensor(
            (int(situation_repr["target_object"]["position"]["row"]) * int(situation_repr["grid_size"])) +
            int(situation_repr["target_object"]["position"]["column"]),
            dtype=torch.long, device=device).unsqueeze(dim=0)
        return empty_example

    def read_dataset(self, max_examples=None, simple_situation_representation=True) -> {}:
        """
        Loop over the data examples in GroundedScan and convert them to tensors, also save the lengths
        for input and target sequences that are needed for padding.
        :param max_examples: how many examples to read maximally, read all if None.
        :param simple_situation_representation: whether to read the full situation image in RGB or the simplified
        smaller representation.
        """
        logger.info("Converting dataset to tensors...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split, simple_situation_representation)):
            if max_examples:
                if len(self._examples) > max_examples:
                    return
            empty_example = {}
            input_commands = example["input_command"]
            target_commands = example["target_command"]
            #equivalent_target_commands = example["equivalent_target_command"]
            situation_image = example["situation_image"]
            if i == 0:
                self.image_dimensions = situation_image.shape[0]
                self.image_channels = situation_image.shape[-1]
            situation_repr = example["situation_representation"]
            input_array = self.sentence_to_array(input_commands, vocabulary="input")
            target_array = self.sentence_to_array(target_commands, vocabulary="target")
            #equivalent_target_array = self.sentence_to_array(equivalent_target_commands, vocabulary="target")
            empty_example["input_tensor"] = torch.tensor(input_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)
            empty_example["target_tensor"] = torch.tensor(target_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)
            #empty_example["equivalent_target_tensor"] = torch.tensor(equivalent_target_array, dtype=torch.long,
            #                                                         device=device).unsqueeze(dim=0)
            empty_example["situation_tensor"] = torch.tensor(situation_image, dtype=torch.float, device=device
                                                             ).unsqueeze(dim=0)
            empty_example["situation_representation"] = situation_repr
            empty_example["derivation_representation"] = example["derivation_representation"]
            empty_example["agent_position"] = torch.tensor(
                (int(situation_repr["agent_position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["agent_position"]["column"]), dtype=torch.long,
                device=device).unsqueeze(dim=0)
            empty_example["target_position"] = torch.tensor(
                (int(situation_repr["target_object"]["position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["target_object"]["position"]["column"]),
                dtype=torch.long, device=device).unsqueeze(dim=0)
            self._input_lengths = np.append(self._input_lengths, [len(input_array)])
            self._target_lengths = np.append(self._target_lengths, [len(target_array)])
            self._examples = np.append(self._examples, [empty_example])

    def sentence_to_array(self, sentence: List[str], vocabulary: str) -> List[int]:
        """
        Convert each string word in a sentence to the corresponding integer from the vocabulary and append
        a start-of-sequence and end-of-sequence token.
        :param sentence: the sentence in words (strings)
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in integers.
        """
        vocab = self.get_vocabulary(vocabulary)
        sentence_array = [vocab.sos_idx]
        for word in sentence:
            sentence_array.append(vocab.word_to_idx(word))
        sentence_array.append(vocab.eos_idx)
        return sentence_array

    def array_to_sentence(self, sentence_array: List[int], vocabulary: str) -> List[str]:
        """
        Translate each integer in a sentence array to the corresponding word.
        :param sentence_array: array with integers representing words from the vocabulary.
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in words.
        """
        vocab = self.get_vocabulary(vocabulary)
        return [vocab.idx_to_word(word_idx) for word_idx in sentence_array]

    @property
    def num_examples(self):
        return len(self._examples)

    @property
    def input_vocabulary_size(self):
        return self.input_vocabulary.size

    @property
    def target_vocabulary_size(self):
        return self.target_vocabulary.size

