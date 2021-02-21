#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import logging
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '../multimodal_seq2seq_gSCAN/'))
import random
import copy 

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
from experiments_utils import *

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


# In[5]:


def evaluate_syntactic_dependency(flags):
    for argument, value in flags.items():
        logger.info("{}: {}".format(argument, value))

    # 1. preparing datasets
    logger.info("Loading datasets.")
    compositional_splits_data_path = os.path.join(flags["data_directory"], "dataset.txt")
    compositional_splits_preprocessor = DummyGroundedScanDataset(compositional_splits_data_path, 
                                            flags["data_directory"], 
                                            input_vocabulary_file=flags["input_vocab_path"], 
                                            target_vocabulary_file=flags["target_vocab_path"],
                                            generate_vocabulary=False,
                                            k=flags["k"])
    compositional_splits_dataset =         GroundedScan.load_dataset_from_file(
            compositional_splits_data_path, 
            save_directory=flags["output_directory"], 
            k=flags["k"])

    logger.info("Loading models.")
    # 2. load up models
    raw_example = None
    for _, example in enumerate(compositional_splits_dataset.get_examples_with_image(flags["split"], True)):
        raw_example = example
        break
    single_example = compositional_splits_preprocessor.process(raw_example)
    model = Model(input_vocabulary_size=compositional_splits_preprocessor.input_vocabulary_size,
                  target_vocabulary_size=compositional_splits_preprocessor.target_vocabulary_size,
                  num_cnn_channels=compositional_splits_preprocessor.image_channels,
                  input_padding_idx=compositional_splits_preprocessor.input_vocabulary.pad_idx,
                  target_pad_idx=compositional_splits_preprocessor.target_vocabulary.pad_idx,
                  target_eos_idx=compositional_splits_preprocessor.target_vocabulary.eos_idx,
                  **input_flags)
    model = model.cuda() if use_cuda else model
    _ = model.load_model(flags["resume_from_file"])
    
    # TODO: let us enable multi-gpu settings here to save up times
    
    logger.info("Starting evaluations.")
    input_levDs = []
    pred_levDs = []
    accuracies = []
    corrupt_accuracies = []
    example_count = 0
    limit = flags["max_testing_examples"]
    split = flags["split"]
    dataloader = [example for example in compositional_splits_dataset.get_examples_with_image(split, True)]

    random.shuffle(dataloader) # shuffle this to get a unbiased estimate of accuracies

    dataloader = dataloader[:limit] if limit else dataloader
    
    for _, example in enumerate(tqdm(dataloader, desc="Iteration")):

        # non-corrupt
        single_example = compositional_splits_preprocessor.process(example)
        output = predict_single(single_example, model=model, 
                                max_decoding_steps=30, 
                                pad_idx=compositional_splits_preprocessor.target_vocabulary.pad_idx, 
                                sos_idx=compositional_splits_preprocessor.target_vocabulary.sos_idx,
                                eos_idx=compositional_splits_preprocessor.target_vocabulary.eos_idx, 
                                device=device)
        pred_command = compositional_splits_preprocessor.array_to_sentence(output[3], vocabulary="target")
        accuracy = sequence_accuracy(output[3], output[4][0].tolist()[1:-1])
        accuracies += [accuracy]

        # corrupt
        corrupt_example = make_corrupt_example(example, flags["corrupt_methods"])
        corrupt_single_example = compositional_splits_preprocessor.process(corrupt_example)
        corrupt_output = predict_single(corrupt_single_example, model=model, 
                                        max_decoding_steps=30, 
                                        pad_idx=compositional_splits_preprocessor.target_vocabulary.pad_idx, 
                                        sos_idx=compositional_splits_preprocessor.target_vocabulary.sos_idx,
                                        eos_idx=compositional_splits_preprocessor.target_vocabulary.eos_idx, 
                                        device=device)
        corrupt_pred_command = compositional_splits_preprocessor.array_to_sentence(corrupt_output[3], vocabulary="target")
        corrupt_accuracy = sequence_accuracy(corrupt_output[3], corrupt_output[4][0].tolist()[1:-1])
        corrupt_accuracies += [corrupt_accuracy]

        input_levD = levenshteinDistance(example['input_command'], corrupt_example['input_command'])
        pred_levD = levenshteinDistance(pred_command, corrupt_pred_command)
        input_levDs.append(input_levD)
        pred_levDs.append(pred_levD)
        example_count += 1

    exact_match = 0
    for acc in accuracies:
        if acc == 100:
            exact_match += 1
    exact_match = exact_match * 1.0 / len(accuracies)
    
    corrupt_exact_match = 0
    for acc in corrupt_accuracies:
        if acc == 100:
            corrupt_exact_match += 1
    corrupt_exact_match = corrupt_exact_match * 1.0 / len(corrupt_accuracies)
            
    logger.info("Eval Split={}, Original Exact Match %={}, Corrupt Exact Match %={}".format(split, exact_match, corrupt_exact_match))

    return {"input_levDs" : input_levDs, 
            "pred_levDs" : pred_levDs, 
            "accuracies" : accuracies, 
            "corrupt_accuracies" : corrupt_accuracies}
    


# In[ ]:


if __name__ == "__main__":
    input_flags = vars(get_gSCAN_parser().parse_args())
    
    saved_to_dict = evaluate_syntactic_dependency(flags=input_flags)
    split = input_flags["split"]
    corrupt_methods = input_flags["corrupt_methods"]
    if input_flags["save_eval_result_dict"]:
        torch.save(saved_to_dict, 
                   os.path.join(
                       input_flags["output_directory"],
                       f"eval_result_split_{split}_corrupt_{corrupt_methods}_dict.bin")
                  )
    else:
        logger.info("Skip saving results.")

