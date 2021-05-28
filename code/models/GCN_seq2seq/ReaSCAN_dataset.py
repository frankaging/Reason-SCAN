import os
from typing import List
from typing import Tuple
import logging
from collections import defaultdict
from collections import Counter
import json
import torch
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', 'dataset'))
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
if isnotebook():
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger(__name__)

from world import *
from vocabulary import Vocabulary as ReaSCANVocabulary
from object_vocabulary import *

class Vocabulary(object):
    """
    Object that maps words in string form to indices to be processed by numerical models.
    """

    def __init__(self, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>"):
        """
        NB: <PAD> token is by construction idx 0.
        """
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self._idx_to_word = [pad_token, sos_token, eos_token]
        self._word_to_idx = defaultdict(lambda: self._idx_to_word.index(self.pad_token))
        self._word_to_idx[sos_token] = 1
        self._word_to_idx[eos_token] = 2
        self._word_frequencies = Counter()

    def word_to_idx(self, word: str) -> int:
        return self._word_to_idx[word]

    def idx_to_word(self, idx: int) -> str:
        return self._idx_to_word[idx]

    def add_sentence(self, sentence: List[str]):
        for word in sentence:
            if word not in self._word_to_idx:
                self._word_to_idx[word] = self.size
                self._idx_to_word.append(word)
            self._word_frequencies[word] += 1

    def most_common(self, n=10):
        return self._word_frequencies.most_common(n=n)

    @property
    def pad_idx(self):
        return self.word_to_idx(self.pad_token)

    @property
    def sos_idx(self):
        return self.word_to_idx(self.sos_token)

    @property
    def eos_idx(self):
        return self.word_to_idx(self.eos_token)

    @property
    def size(self):
        return len(self._idx_to_word)

    @classmethod
    def load(cls, path: str):
        assert os.path.exists(path), "Trying to load a vocabulary from a non-existing file {}".format(path)
        with open(path, 'r') as infile:
            all_data = json.load(infile)
            sos_token = all_data["sos_token"]
            eos_token = all_data["eos_token"]
            pad_token = all_data["pad_token"]
            vocab = cls(sos_token=sos_token, eos_token=eos_token, pad_token=pad_token)
            vocab._idx_to_word = all_data["idx_to_word"]
            vocab._word_to_idx = defaultdict(int)
            for word, idx in all_data["word_to_idx"].items():
                vocab._word_to_idx[word] = idx
            vocab._word_frequencies = Counter(all_data["word_frequencies"])
        return vocab

    def to_dict(self) -> dict:
        return {
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "idx_to_word": self._idx_to_word,
            "word_to_idx": self._word_to_idx,
            "word_frequencies": self._word_frequencies
        }

    def save(self, path: str) -> str:
        with open(path, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)
        return path
    
class ReaSCANDataset(object):
    """
    Loads a GroundedScan instance from a specified location.
    """
    def __init__(self, data_json, save_directory: str, k: int, split="all", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False):
        if not generate_vocabulary:
            assert os.path.exists(os.path.join(save_directory, input_vocabulary_file)) and os.path.exists(
                os.path.join(save_directory, target_vocabulary_file)), \
                "Trying to load vocabularies from non-existing files."
        
        # we simply load the json file.
        logger.info(f"Formulating the dataset from the passed in json file...")
        self.data_json = data_json
        
        if split == "test" and generate_vocabulary:
            logger.warning("WARNING: generating a vocabulary from the test set.")
            
        # some helper initialization
        self.grid_size = self.data_json['grid_size']

        intransitive_verbs = ["walk"]
        transitive_verbs = ["push", "pull"]
        adverbs = ["while zigzagging", "while spinning", "cautiously", "hesitantly"]
        nouns = ["circle", "cylinder", "square", "box"]
        color_adjectives = ["red", "blue", "green", "yellow"]
        size_adjectives = ["big", "small"]
        relative_pronouns = ["that is"]
        relation_clauses = ["in the same row as", 
                            "in the same column as", 
                            "in the same color as", 
                            "in the same shape as", 
                            "in the same size as",
                            "inside of"]
        reaSCANVocabulary = ReaSCANVocabulary.initialize(intransitive_verbs=intransitive_verbs,
                                                   transitive_verbs=transitive_verbs, adverbs=adverbs, nouns=nouns,
                                                   color_adjectives=color_adjectives,
                                                   size_adjectives=size_adjectives, 
                                                   relative_pronouns=relative_pronouns, 
                                                   relation_clauses=relation_clauses)
        min_object_size = 1
        max_object_size = 4
        object_vocabulary = ObjectVocabulary(shapes=reaSCANVocabulary.get_semantic_shapes(),
                                             colors=reaSCANVocabulary.get_semantic_colors(),
                                             min_size=min_object_size, max_size=max_object_size)
        
        self._world = World(grid_size=self.grid_size, colors=reaSCANVocabulary.get_semantic_colors(),
                            object_vocabulary=object_vocabulary,
                            shapes=reaSCANVocabulary.get_semantic_shapes(),
                            save_directory=save_directory)
        self._world.clear_situation()
            
        self.image_dimensions = self._world.get_current_situation_image().shape[0] 
        self.image_channels = 3
        self.split = split
        self.directory = save_directory
        self.k = k

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
    
        self.k=k # this is for the few-shot splits only!

    @staticmethod
    def command_repr(command: List[str]) -> str:
        return ','.join(command)

    @staticmethod
    def parse_command_repr(command_repr: str) -> List[str]:
        return command_repr.split(',')
    
    def initialize_world(self, situation: Situation, mission="") -> {}:
        """
        Initializes the world with the passed situation.
        :param situation: class describing the current situation in the world, fully determined by a grid size,
        agent position, agent direction, list of placed objects, an optional target object and optional carrying object.
        :param mission: a string defining a command (e.g. "Walk to a green circle.")
        """
        objects = []
        for positioned_object in situation.placed_objects:
            objects.append((positioned_object.object, positioned_object.position))
        self._world.initialize(objects, agent_position=situation.agent_pos, agent_direction=situation.agent_direction,
                               target_object=situation.target_object, carrying=situation.carrying)
        if mission:
            self._world.set_mission(mission)
    
    def get_examples_with_image(self, split="train", simple_situation_representation=False) -> dict:
        """
        Get data pairs with images in the form of np.ndarray's with RGB values or with 1 pixel per grid cell
        (see encode in class Grid of minigrid.py for details on what such representation looks like).
        :param split: string specifying which split to load.
        :param simple_situation_representation:  whether to get the full RGB image or a simple representation.
        :return: data examples.
        """
        for example in self.data_json["examples"][split]:
            command = self.parse_command_repr(example["command"])
            if example.get("meaning"):
                meaning = example["meaning"]
            else:
                meaning = example["command"]
            meaning = self.parse_command_repr(meaning)
            situation = Situation.from_representation(example["situation"])
            self._world.clear_situation()
            self.initialize_world(situation)
            if simple_situation_representation:
                situation_image = self._world.get_current_situation_grid_repr()
            else:
                situation_image = self._world.get_current_situation_image()
            target_commands = self.parse_command_repr(example["target_commands"])
            yield {"input_command": command, "input_meaning": meaning,
                   "derivation_representation": example.get("derivation"),
                   "situation_image": situation_image, "situation_representation": example["situation"],
                   "target_command": target_commands}
    
    def read_vocabularies(self) -> {}:
        """
        Loop over all examples in the dataset and add the words in them to the vocabularies.
        """
        logger.info("Populating vocabulary...")
        for i, example in enumerate(self.get_examples_with_image(self.split)):
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

    def read_dataset(self, max_examples=None, simple_situation_representation=True) -> {}:
        """
        Loop over the data examples in GroundedScan and convert them to tensors, also save the lengths
        for input and target sequences that are needed for padding.
        :param max_examples: how many examples to read maximally, read all if None.
        :param simple_situation_representation: whether to read the full situation image in RGB or the simplified
        smaller representation.
        """
        few_shots_ids = []
        logger.info("Converting dataset to tensors...")
        if self.split == "few_shot_single_clause_logic" and self.k != 0:
            logger.info("Removing examples for few-shots training test set...")
            path_to_few_shot_data = os.path.join(self.directory, f"few-shot-inoculations-{self.k}.txt")
            logger.info(f"Reading few-shot inoculation from file: {path_to_few_shot_data}...")
            few_shots_ids = json.load(open(path_to_few_shot_data, "r"))
        
        for i, example in enumerate(self.get_examples_with_image(self.split, simple_situation_representation)):
            if i in few_shots_ids: # this is just for few-shot experiments.
                continue
            if max_examples:
                if len(self._examples) > max_examples - 1:
                    break
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
        
        # we also need to load few-shots examples in case k is not 0.
        if self.k != 0:
            logger.info("Loading few examples into the training set for few-shots learning...")
            # Let us also record the few shots examples index, so in evaluation,
            # we can move them out!
            few_shot_single_clause_logic = self.data_json["examples"]["few_shot_single_clause_logic"]
            few_shots_ids = [i for i in range(len(few_shot_single_clause_logic))]
            few_shots_ids = random.sample(few_shots_ids, self.k)
            logger.info("The following idx examples are selected for few-shot learning:")
            logger.info(few_shots_ids)
            with open(os.path.join(self.directory, f"few-shot-inoculations-{self.k}.txt"), "w") as fd:
                json.dump(few_shots_ids, fd, indent=4)
                
            all_examples_few_shots_selected = []
            for i, example in enumerate(
                self.get_examples_with_image(
                    "few_shot_single_clause_logic", simple_situation_representation
                )
            ):
                if i in few_shots_ids:
                    all_examples_few_shots_selected.append(example)
            for i, example in enumerate(all_examples_few_shots_selected):
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