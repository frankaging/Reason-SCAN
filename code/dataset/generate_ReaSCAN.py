#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
from collections import namedtuple, OrderedDict
import itertools
import os
import numpy as np
from typing import Tuple
from typing import List
from typing import Dict
import random
from itertools import product
import copy
import re
import random
import hashlib
import pathlib
import json
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
os.environ['QT_QPA_PLATFORM']='offscreen'
plt.rcParams["font.family"] = "DejaVu Serif"
font = {'family' : 'DejaVu Serif',
        'size'   : 20}
plt.rc('font', **font)
import plotly.tools as tls

from utils import one_hot
from utils import generate_possible_object_names
from utils import numpy_array_to_image

from vocabulary import *
from object_vocabulary import *
from world import *
from grammer import *
from simulator import *
from relation_graph import *

import logging

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Helpers.
def get_relation_statistics(command_structs):
    """
    Return a dictionary, (relation, position) with counts
    """
    stats = {}
    for i in range(2): # at max 2!
        stats[f"position-{i}"] = {}
    for command in command_structs:
        pos_id = 0
        for k, v in command["rel_map"].items():
            if v in stats[f"position-{pos_id}"].keys():
                stats[f"position-{pos_id}"][v] += 1
            else:
                stats[f"position-{pos_id}"][v] = 1
            pos_id += 1
    return stats


def get_attribute_statistics(command_structs, include_keywords=["circle", "cylinder", "square", "box", "object"]):
    
    stats = {}
    # for k, v in command_structs[0]["obj_map"].items():
    #     stats[k] = {} # we can do it in object level!
    for i in range(3): # at max 2!
        stats[f"$OBJ_{i}"] = {}
        
    for command in command_structs:
        for k, v in command["obj_map"].items():
            for keyword in include_keywords:
                keyword_list = keyword.split(" ") # in case there are a couple!
                match = True
                for sub_k in keyword_list:
                    if sub_k not in v:
                        match = False
                        break
                if match:
                    if keyword in stats[k].keys():
                        stats[k][keyword] += 1
                    else:
                        stats[k][keyword] = 1
    return stats


def get_keyword_statistics(command_structs, include_keyword="adverb"):
    stats = {}
    for command in command_structs:
        keyword = command[include_keyword]
        if keyword in stats.keys():
            stats[keyword] += 1
        else:
            stats[keyword] = 1
    return stats

def flatten_dictionary(
    dictionary_in
):
    flat_dictionary = {}
    for k, v in dictionary_in.items():
        for kk, vv in v.items():
            if kk not in flat_dictionary:
                flat_dictionary[kk] = vv
            else:
                flat_dictionary[kk] += vv
    return flat_dictionary

def plot_dictionary(
    dictionary_in,
    y_label="Frequency",
    x_label="Conditions",
    title="Missing Title",
    save_file=None,
    is_plot=False,
    wandb=None,
):
    group_str = [k for k, _ in dictionary_in[0].items()]
    if len(group_str) > 8:
        rotate=90
        fontsize=10
    else:
        rotate=45
        fontsize=13
    all_stats = []
    for d in dictionary_in:
        group_stats = [d[k] for k in group_str]
        all_stats.append(group_stats)
    all_stats = np.array(all_stats)
    std = np.std(all_stats, axis=0)
    mean = np.mean(all_stats, axis=0)

    # input data
    mean_values = mean
    variance = std**2
    bar_labels = group_str
        
    # plot bars
    x_pos = list(range(len(bar_labels)))
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    g = ax.bar(x_pos, mean_values, yerr=variance, align='center', alpha=0.5)

    plt.grid()

    # set height of the y-axis
    max_y = max(zip(mean_values, variance)) # returns a tuple, here: (3, 5)
    plt.ylim([0, (max_y[0] + max_y[1]) * 1.1])

    # set axes labels and title
    plt.ylabel(y_label)
    
    plt.xticks(x_pos, bar_labels)
    plt.xticks(rotation = rotate, fontsize=fontsize)    
    plt.yticks(rotation = 45)
    plt.title(title, fontsize=10)
    if mean_values[0] > 10000:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    
    if wandb != None:
        # Let us also try to log this plot to wandb!
        wandb.log({title: wandb.Image(fig)})

    if save_file != None:
        plt.savefig(save_file, dpi=100, bbox_inches='tight')
        plt.close(fig)
    else:
        if is_plot:
            plt.show()
    
def get_command_struct_statistics(
    command_structs, run_name="ReaSCAN-Awesome", date="2021-05-06", 
    split="demo",
    compositional_split=False,
    n_sample=-1, n_runs=10,
    output_dir="../../data-files/ReaSCAN-compositional_splits/",
    save_to_disk=True,
    wandb=None
):
    statistics = OrderedDict({
        "run_name": run_name,
        "date": date,
        "splits": split,
        "number_of_these_examples_seen_in_training": -1 if not compositional_split else 0,
        "number_of_command_structs": len(command_structs),
    })
    if n_sample == -1:
        n_sample = len(command_structs)
    # If we are downsampling, we need to do more runs as well!
    random.shuffle(command_structs)
    
    patterns = set([])
    for command_s in command_structs:
        patterns.add(command_s["grammer_pattern"])
    statistics["command_patterns"] = list(patterns)
    
    pattern_stats = get_keyword_statistics(command_structs, include_keyword="grammer_pattern")
    statistics["pattern_stats"] = pattern_stats
    
    # verb
    verb_stats = get_keyword_statistics(command_structs, include_keyword="verb")
    statistics["verb_stats"] = verb_stats
    plot_dictionary(
        [verb_stats],
        title="Verbs",
        save_file=os.path.join(output_dir, f"verb_stats-{split}.png"),
        wandb=wandb,
    )
    
    # adverb
    adverb_stats = get_keyword_statistics(command_structs, include_keyword="adverb")
    # special handling for adverb for better readabilities
    adverb_stats_rebuild = {}
    for k, v in adverb_stats.items():
        if k == "":
            adverb_stats_rebuild["EMPTY"] = v
        else:
            adverb_stats_rebuild[k] = v
    statistics["adverb_stats"] = adverb_stats_rebuild
    plot_dictionary(
        [adverb_stats_rebuild],
        title="Adverbs",
        save_file=os.path.join(output_dir, f"adverb_stats-{split}.png"),
        wandb=wandb,
    )
    
    # relation
    relation_stats = get_relation_statistics(command_structs)
    if len(flatten_dictionary(relation_stats)) != 0:
        statistics["relation_stats"] = relation_stats
        plot_dictionary(
            [flatten_dictionary(relation_stats)],
            title="Relation-Types",
            save_file=os.path.join(output_dir, f"relation_type_stats-{split}.png"),
            wandb=wandb,
        )
    
    # attribute
    nouns = ["circle", "cylinder", "square", "box", "object"]
    n_stats = get_attribute_statistics(command_structs, include_keywords=nouns)
    statistics["shape_stats"] = n_stats
    plot_dictionary(
        [flatten_dictionary(n_stats)],
        title="Shapes",
        save_file=os.path.join(output_dir, f"shape_stats-{split}.png"),
        wandb=wandb,
    )
    
    color_adjectives = ["red", "blue", "green", "yellow"]
    c_stats = get_attribute_statistics(command_structs, include_keywords=color_adjectives)
    statistics["color_stats"] = c_stats
    if len(flatten_dictionary(c_stats)) != 0:
        plot_dictionary(
            [flatten_dictionary(c_stats)],
            title="Colors",
            save_file=os.path.join(output_dir, f"color_stats-{split}.png"),
            wandb=wandb,
        )

    size_adjectives = ["big", "small"]
    s_stats = get_attribute_statistics(command_structs, include_keywords=size_adjectives)
    if len(flatten_dictionary(s_stats)) != 0:
        statistics["size_stats"] = s_stats
        plot_dictionary(
            [flatten_dictionary(s_stats)],
            title="Sizes",
            save_file=os.path.join(output_dir, f"size_stats-{split}.png"),
            wandb=wandb,
        )
    
    # second order attribute
    color_adjectives = ["red", "blue", "green", "yellow"]
    nouns = ["circle", "cylinder", "square", "box", "object"]
    c_n_p = product(color_adjectives, nouns)
    include_keywords = [" ".join(c_n) for c_n in c_n_p]
    c_n_stats = get_attribute_statistics(command_structs, include_keywords=include_keywords)
    statistics["color_and_shape_stats"] = c_n_stats
    if len(flatten_dictionary(c_n_stats)) != 0:
        plot_dictionary(
            [flatten_dictionary(c_n_stats)],
            title="Colors-Shapes",
            save_file=os.path.join(output_dir, f"color+shape_stats-{split}.png"),
            wandb=wandb,
        )

    size_adjectives = ["big", "small"]
    nouns = ["circle", "cylinder", "square", "box", "object"]
    s_n_p = product(size_adjectives, nouns)
    include_keywords = [" ".join(s_n) for s_n in s_n_p]
    s_n_stats = get_attribute_statistics(command_structs, include_keywords=include_keywords)
    statistics["size_and_shape_stats"] = s_n_stats
    if len(flatten_dictionary(s_n_stats)) != 0:
        plot_dictionary(
            [flatten_dictionary(s_n_stats)],
            title="Sizes-Shapes",
            save_file=os.path.join(output_dir, f"size+shape_stats-{split}.png"),
            wandb=wandb,
        )
    
    # third order attribute
    size_adjectives = ["big", "small"]
    color_adjectives = ["red", "blue", "green", "yellow"]
    nouns = ["circle", "cylinder", "square", "box", "object"]
    all_p = product(size_adjectives, color_adjectives, nouns)
    include_keywords = [" ".join(a) for a in all_p]
    all_stats = get_attribute_statistics(command_structs, include_keywords=include_keywords)
    statistics["size_and_color_and_shape_stats"] = all_stats
    
    if save_to_disk:
        import yaml
        with open(os.path.join(output_dir, f"command_struct_only_stats-{split}.yml"), 'w') as yaml_file:
            yaml.dump(statistics, yaml_file, default_flow_style=False)
    
    return statistics

def arg_parse():
    
    # This is a single loop to generate the dataset.
    n_processes = 1
    mode = "all"
    n_command_struct = 10000
    grid_size = 6
    n_object_max = 10
    seed = 42
    date = "2021-05-07"
    per_command_world_retry_max = 200
    per_command_world_target_count = 10 # for each command, we target to have 50 shapeWorld!
    resumed_from_file_path = ""
    is_tensorboard = False
    
    parser = argparse.ArgumentParser(description='ReaSCAN argparse.')
    # Experiment management:
    parser.add_argument('--n_processes', type=int, default=1,
                        help='Number of process used to generate the dataset.')
    parser.add_argument('--index_start', type=int, default=-1,
                        help='Number of command sampled from the command population.')
    parser.add_argument('--index_end', type=int, default=-1,
                        help='Number of command sampled from the command population.')

    parser.add_argument('--mode', type=str, default="all",
                        help='mode')
    parser.add_argument('--n_command_struct', type=int, default=10000,
                        help='Number of command sampled from the command population.')
    parser.add_argument('--grid_size', type=int, default=6,
                        help='Grid size of the world.')
    parser.add_argument('--n_object_max', type=int, default=10,
                        help='Number of object at max in the shapeWorld (Note that you may still have more than this number!).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--date', type=str,
                        help='date')
    parser.add_argument('--per_command_world_retry_max', type=int, default=200,
                        help='How many times you can retry for each world generation.')
    parser.add_argument('--per_command_world_target_count', type=int, default=50,
                        help='The targeted number of world to have per command.')
    parser.add_argument("--is_tensorboard",
                        default=False,
                        action='store_true',
                        help="Whether to use tensorboard.")
    
    parser.add_argument("--include_relation_distractor",
                        default=False,
                        action='store_true',
                        help="Whether to use tensorboard.")
    parser.add_argument("--include_attribute_distractor",
                        default=False,
                        action='store_true',
                        help="Whether to use tensorboard.")
    parser.add_argument("--include_isomorphism_distractor",
                        default=False,
                        action='store_true',
                        help="Whether to use tensorboard.")
    parser.add_argument("--include_random_distractor",
                        default=False,
                        action='store_true',
                        help="Whether to use tensorboard.")
    parser.add_argument('--full_relation_probability', type=float, default=1.0,
                        help='Probability of including full relation distractors.')
    
    parser.add_argument('--save_interal', type=int, default=200,
                        help='Saving intervel in command count.')
    
    
    parser.add_argument('--command_pattern', type=str, default="p3",
                        help='What pattern to use, currently, we support p1-p4.')
    
    parser.add_argument('--resumed_from_file_path', type=str, default="",
                        help='Whether to resume for this file.')
    parser.add_argument('--output_dir', type=str, default="../../data-files/ReaSCAN-compositional_splits/",
                        help='Whether to resume for this file.')

    parser.set_defaults(
        # Exp management:
        n_processes=1,
        mode="all",
        n_command_struct=10000,
        grid_size=6,
        n_object_max=10,
        seed=42,
        date="2021-05-07",
        per_command_world_retry_max=200,
        per_command_world_target_count=50,
        resumed_from_file_path="",
        is_tensorboard=False,
        output_dir="../../data-files/ReaSCAN-compositional_splits/",
    )
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        args = parser.parse_args([])
    except:
        args = parser.parse_args()
    return args

def example_classifier(
    task_info,
    mode="demo",
    default_split_prob={
        "train": 0.9, 
        "dev": 0.01,
        "test": 0.09,
    },
):
    """
    This will return the split this data belongs to.
    """
    if mode == "demo" or mode == "all":
        if random.random() < default_split_prob["train"]:
            return "train"
        else:
            if random.random() < 0.9:
                return "test"
            else:
                return "dev"
    else:
        # We need to add here logics to determine
        # compositional splits!
        pass


# In[ ]:


# Some tips:
# Do not debug in this file, you can simply copy the questionable struct
# to the lightweight demo file, and you can debug there!


# In[ ]:


if __name__ == "__main__":
    
    # Loading arguments
    args = arg_parse()
    try:
#         get_ipython().run_line_magic('matplotlib', 'inline')
#         # Experiment management:
#         args.n_processes=1
#         args.mode="demo"
#         args.n_command_struct=20
#         args.grid_size=6
#         args.n_object_max=10
#         args.seed=42
#         args.date="2021-05-07"
#         args.per_command_world_retry_max=20
#         args.per_command_world_target_count=3
#         args.resumed_from_file_path=""
#         args.is_tensorboard=True # Let us try this!
#         args.output_dir="../../data-files/ReaSCAN-demo/"
#         is_jupyter = True
        
        get_ipython().run_line_magic('matplotlib', 'inline')
        # Experiment management:
        args.n_processes=1
        args.mode="train"
        args.n_command_struct=675*5
        args.grid_size=6
        args.n_object_max=10
        args.seed=42
        args.save_interal = 200
        args.date="2021-05-30"
        args.per_command_world_retry_max=1000
        args.per_command_world_target_count=180
        args.resumed_from_file_path=""
        args.is_tensorboard=True # Let us try this!
        args.output_dir="../../data-files/ReaSCAN-compositional-p3-full-relation/"
        is_jupyter = True
        
        args.index_start = -1
        args.index_end = -1
    except:
        is_jupyter = False
    
    loading_p1 = True if args.command_pattern == "p1" else False
    p1_exhaustive_verb_adverb = False
    loading_p2 = True if args.command_pattern == "p2" else False
    loading_p3 = True if args.command_pattern == "p3" else False
    loading_p4 = True if args.command_pattern == "p4" else False

    save_command_stats = False
    save_at_interval = True
    save_interal = args.save_interal

    # TODO: add these to args.
    logging_interval = 1000
    
    # Create output directory if not exists.
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True) 
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(levelname)-8s %(message)s', 
        datefmt='%a, %d %b %Y %H:%M:%S', 
        filename=os.path.join(args.output_dir, "generator.log"),
    )
    logger = logging.getLogger(__name__)
    logging.getLogger().addHandler(logging.StreamHandler(os.sys.stdout))
    
    logger.info("Generating ReaSCAN with following parameters: ")
    logger.info(args)
    
    # This is a single loop to generate the dataset.
    n_processes = args.n_processes
    mode = args.mode
    n_command_struct = args.n_command_struct
    grid_size = args.grid_size
    n_object_max = args.n_object_max
    seed = args.seed
    date = args.date
    per_command_world_retry_max = args.per_command_world_retry_max
    per_command_world_target_count = args.per_command_world_target_count # for each command, we target to have 50 shapeWorld!
    resumed_from_file_path = args.resumed_from_file_path
    output_dir = args.output_dir
    is_tensorboard = args.is_tensorboard
    
    if is_tensorboard:
        logger.warning("Enabling wandb for tensorboard logging...")
        import wandb
        run = wandb.init(project="ReaSCAN", entity="wuzhengx")
        run_name = wandb.run.name
        wandb.config.update(args)
    else:
        wandb = None

    random.seed(seed)
    np.random.seed(seed)
    
    # We also need something to generate generalization
    # splits!
    params = {
        "n_processes": n_processes,
        "mode": mode,
        "n_command_struct": n_command_struct,
        "grid_size": grid_size,
        "n_object_max": n_object_max,
        "seed": seed,
        "per_command_world_retry_max": per_command_world_retry_max,
        "per_command_world_target_count": per_command_world_target_count,
    }
    
    if mode == "all" or mode == "demo" or mode == "train":
        # Meaning we are generating the random ReaSCAN train + dev + test splits!
        logger.warning(f"You are generating data for {mode} splits only!")
        split_percentage = {
            "train": 0.9, 
        }
    elif mode == "all,noval_1,noval_2,noval_3,noval_4":
        # here we need to define how to check for noval_*
        pass
    elif mode == "compositional":
        # Meaning we are generating the random ReaSCAN train + dev + test splits!
        logger.warning("You are generating data for all compositional splits!")
    elif mode == "":
        pass # Not implemented!
        
    # Using the full vocabulary.
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
    vocabulary = Vocabulary.initialize(intransitive_verbs=intransitive_verbs,
                                       transitive_verbs=transitive_verbs, adverbs=adverbs, nouns=nouns,
                                       color_adjectives=color_adjectives,
                                       size_adjectives=size_adjectives, 
                                       relative_pronouns=relative_pronouns, 
                                       relation_clauses=relation_clauses)
    
    # test out the object vocab
    min_object_size = 1
    max_object_size = 4
    object_vocabulary = ObjectVocabulary(shapes=vocabulary.get_semantic_shapes(),
                                         colors=vocabulary.get_semantic_colors(),
                                         min_size=min_object_size, max_size=max_object_size)
    
    # Generating all the core command structs.
    grammer = Grammer(vocabulary)
    
    # Bootup our simulator.
    simulator = Simulator(
        object_vocabulary, vocabulary, 
        grid_size=grid_size, 
        n_object_max=n_object_max,
    )
    
    command_structs = []
    logger.info("Finished loading required modules...")
    # Sampling all the possible command score structs.
    
    if loading_p4:
        # Currently, we hard-code the pattern!
        grammer_pattern = '$OBJ_0 ^ $OBJ_1 & $OBJ_2 & $OBJ_3'
        logger.info(f"Including pattern:= {grammer_pattern}...")
        # Sampling relations
        relations = grammer.sample_object_relation_grammer(
            '$OBJ_0', 
            grammer.build_dependency_graph(grammer_pattern))
        for relation in relations:
            obj_pattern_map = relation[0]
            rel_map = relation[1]
            grammer_bindings = grammer.grounding_grammer_with_vocabulary(grammer_pattern, obj_pattern_map, rel_map)
            for obj_map in grammer_bindings:
                # here, we also sample the verb and adverb bindings!
                adverb_enhance_list = vocabulary.get_adverbs()
                adverb_enhance_list += [""]
                command_struct = {
                    "obj_pattern_map" : obj_pattern_map,
                    "rel_map" : rel_map,
                    "obj_map" : obj_map,
                    "grammer_pattern" : grammer_pattern,
                    "adverb" : random.choice(adverb_enhance_list),
                    "verb" : random.choice(vocabulary.get_transitive_verbs() + vocabulary.get_intransitive_verbs()),
                }
                command_structs += [command_struct]
                
    if loading_p3:
        # Currently, we hard-code the pattern!
        grammer_pattern = '$OBJ_0 ^ $OBJ_1 & $OBJ_2'
        logger.info(f"Including pattern:= {grammer_pattern}...")
        # Sampling relations
        relations = grammer.sample_object_relation_grammer(
            '$OBJ_0', 
            grammer.build_dependency_graph(grammer_pattern))
        for relation in relations:
            obj_pattern_map = relation[0]
            rel_map = relation[1]
            grammer_bindings = grammer.grounding_grammer_with_vocabulary(grammer_pattern, obj_pattern_map, rel_map)
            for obj_map in grammer_bindings:
                # here, we also sample the verb and adverb bindings!
                adverb_enhance_list = vocabulary.get_adverbs()
                adverb_enhance_list += [""]
                command_struct = {
                    "obj_pattern_map" : obj_pattern_map,
                    "rel_map" : rel_map,
                    "obj_map" : obj_map,
                    "grammer_pattern" : grammer_pattern,
                    "adverb" : random.choice(adverb_enhance_list),
                    "verb" : random.choice(vocabulary.get_transitive_verbs() + vocabulary.get_intransitive_verbs()),
                }
                command_structs += [command_struct]
    
    if loading_p2:
        grammer_pattern = '$OBJ_0 ^ $OBJ_1'
        logger.info(f"Including pattern:= {grammer_pattern}...")
        # Sampling relations
        relations = grammer.sample_object_relation_grammer(
            '$OBJ_0', 
            grammer.build_dependency_graph(grammer_pattern))
        for relation in relations:
            obj_pattern_map = relation[0]
            rel_map = relation[1]
            grammer_bindings = grammer.grounding_grammer_with_vocabulary(grammer_pattern, obj_pattern_map, rel_map)
            for obj_map in grammer_bindings:
                # here, we also sample the verb and adverb bindings!
                adverb_enhance_list = vocabulary.get_adverbs()
                adverb_enhance_list += [""]
                command_struct = {
                    "obj_pattern_map" : obj_pattern_map,
                    "rel_map" : rel_map,
                    "obj_map" : obj_map,
                    "grammer_pattern" : grammer_pattern,
                    "adverb" : random.choice(adverb_enhance_list),
                    "verb" : random.choice(vocabulary.get_transitive_verbs() + vocabulary.get_intransitive_verbs()),
                }
                command_structs += [command_struct]
    
    if loading_p1:
        p1_exhaustive_verb_adverb = True
        # for gSCAN command, we don't need to undersample, they are small!
        grammer_pattern = '$OBJ_0'
        logger.info(f"Including pattern:= {grammer_pattern}...")
        # Sampling relations
        relations = grammer.sample_object_relation_grammer(
            '$OBJ_0', 
            grammer.build_dependency_graph(grammer_pattern))
        for relation in relations:
            obj_pattern_map = relation[0]
            rel_map = relation[1]
            grammer_bindings = grammer.grounding_grammer_with_vocabulary(grammer_pattern, obj_pattern_map, rel_map)
            for obj_map in grammer_bindings:
                if p1_exhaustive_verb_adverb:
                    for adverb in vocabulary.get_adverbs() + [""]:
                        for verb in vocabulary.get_transitive_verbs() + vocabulary.get_intransitive_verbs():
                            # here, we also sample the verb and adverb bindings!
                            command_struct = {
                                "obj_pattern_map" : obj_pattern_map,
                                "rel_map" : rel_map,
                                "obj_map" : obj_map,
                                "grammer_pattern" : grammer_pattern,
                                "adverb" : adverb,
                                "verb" : verb,
                            }
                            command_structs += [command_struct]
            
    # We only sample these command!
    """
    WARNING: beaware that not all command struct can
    be sampled for world-command pair! They may or
    may not fail.
    """
    under_sample = True
    if under_sample:
        sampled_command_struct = []
        random.shuffle(command_structs)
        if n_command_struct != -1:
            sampled_command_struct = command_structs[:n_command_struct]
        if args.index_start == -1 or args.index_end == -1:
            pass
        else:
            # we only look at one shard! this is for multiprocess
            logger.info(f"WARNING: contine with sharding: start at {args.index_start}; end at {args.index_end}")
            sampled_command_struct = command_structs[args.index_start:args.index_end]
        logger.info(f"Sampled {len(sampled_command_struct)} from {len(command_structs)} core command structs for pattern={grammer_pattern}.")
            
    logger.info(f"Finished sampling core command structs with total {len(sampled_command_struct)}...")
    
    command_struct_file_path = os.path.join(args.output_dir, f"command_struct-{args.mode}.txt")
    formatted_sampled_command_struct = []
    for command_struct in sampled_command_struct:
        formatted_command_struct = {
            "obj_pattern_map" : command_struct["obj_pattern_map"],
            "rel_map" : [(k, v) for k, v in command_struct["rel_map"].items()],
            "obj_map" : command_struct["obj_map"],
            "grammer_pattern" : command_struct["grammer_pattern"],
            "adverb" : command_struct["adverb"],
            "verb" : command_struct["verb"],
        }
        formatted_sampled_command_struct += [formatted_command_struct]
    # dump to the disk.
    with open(command_struct_file_path, "w") as fd:
        json.dump(formatted_sampled_command_struct, fd, indent=4)
    logger.info(f"Saved command struct to {command_struct_file_path} for later use...")
                    
    # print out quick stats on how many command per pattern!
    per_pattern_command_count = {}
    for command_struct in sampled_command_struct:
        grammer_pattern = command_struct["grammer_pattern"]
        if grammer_pattern in per_pattern_command_count.keys():
            per_pattern_command_count[grammer_pattern] += 1
        else:
            per_pattern_command_count[grammer_pattern] = 1
    logger.info(f"Counts per command pattern: ")
    logger.info(per_pattern_command_count)

    # From the struct, let us sample shape world.
    """
    We just need a couple more steps beyond this point:
    (1) Sample a world
    (2) Making sure it is valid
    (3) Construct the command, providing determiners
    (4) Generate action sequences to the target
    (5) Get all the action related metadata as gSCAN
    (6) Save it to per command example
    """
    
    # We need a way to index the sampled command.
    sampled_command_struct_indexed = OrderedDict({})
    global_command_struct_index = 0
    for command_struct in sampled_command_struct:
        sampled_command_struct_indexed[global_command_struct_index] = command_struct
        global_command_struct_index += 1
    
    root = "$OBJ_0"
    per_command_world_counts = OrderedDict({})
    if mode == "demo" or mode == "all" or mode == "train":
        created_examples_by_splits = OrderedDict({
            "train" : [],
        })
    else:
        pass
    shaperized_command_struct = []
    per_command_world_unique_check = OrderedDict({})
    
    # Some global control for data quality control.
    global_step = 0
    success_step = 0
    
    # Distractor info logs.
    d_full_relation_count = 0
    d_relation_count = 0
    d_attribute_count = 0
    d_iso_count = 0
    d_random_count = 0
    
    logger.info(f"Started to generate the dataset...")
    for command_struct_index, command_struct in sampled_command_struct_indexed.items():
        logger.info(f"Generating for command struct (seed={seed}): {command_struct_index+1}/{len(sampled_command_struct_indexed)}...")
        per_command_world_counts[command_struct_index] = 0 # 0 world for each command in the beginning!
        per_command_world_unique_check[command_struct_index] = set([])
        obj_pattern_map = command_struct["obj_pattern_map"]
        rel_map = command_struct["rel_map"]
        obj_map = command_struct["obj_map"]
        grammer_pattern = command_struct["grammer_pattern"]
        verb = command_struct["verb"]
        adverb = command_struct["adverb"]
        # This is the target world number generated for this command
        for n_world_try in range(per_command_world_target_count):
            # How many time we need to retry before we give up?
            at_least_success = False
            for n_retry in range(per_command_world_retry_max):
                global_step += 1
                if success_step == 0:
                    denom = 1
                else:
                    denom = success_step
                d_full_relation_ratio = 1.0*d_full_relation_count/denom
                d_relation_ratio = 1.0*d_relation_count/denom
                d_attribute_ratio = 1.0*d_attribute_count/denom
                d_iso_ratio = 1.0*d_iso_count/denom
                d_random_ratio = 1.0*d_random_count/denom
                global_success_ratio = 1.0*success_step/global_step
                # logging some very useful information to wandb if avaliable!
                if is_tensorboard:
                    if (global_step%logging_interval) == 0:
                        wandb.log({'global_success_ratio': global_success_ratio, 'global_step': global_step})
                        wandb.log({'current_example_count': success_step, 'global_step': global_step})
                        wandb.log({'d_full_relation_ratio': d_full_relation_ratio, 'global_step': global_step})
                        wandb.log({'d_relation_ratio': d_relation_ratio, 'global_step': global_step})
                        wandb.log({'d_attribute_ratio': d_attribute_ratio, 'global_step': global_step})
                        wandb.log({'d_iso_ratio': d_iso_ratio, 'global_step': global_step})
                        wandb.log({'d_random_ratio': d_random_ratio, 'global_step': global_step})  
                else:
                    if (global_step%(logging_interval*10)) == 0:
                        logger.info({'global_success_ratio': global_success_ratio, 'global_step': global_step})
                        logger.info({'current_example_count': success_step, 'global_step': global_step})
                        logger.info({'d_full_relation_ratio': d_full_relation_ratio, 'global_step': global_step})
                        logger.info({'d_relation_ratio': d_relation_ratio, 'global_step': global_step})
                        logger.info({'d_attribute_ratio': d_attribute_ratio, 'global_step': global_step})
                        logger.info({'d_iso_ratio': d_iso_ratio, 'global_step': global_step})
                        logger.info({'d_random_ratio': d_random_ratio, 'global_step': global_step})
                    
                if mode == "demo":
                    sampled_world = simulator.sample_situations_from_grounded_grammer(
                        copy.deepcopy(grammer_pattern), 
                        copy.deepcopy(obj_pattern_map), 
                        copy.deepcopy(rel_map), 
                        copy.deepcopy(obj_map),
                        is_plot=False,
                        include_relation_distractor=args.include_relation_distractor, 
                        include_attribute_distractor=args.include_attribute_distractor, 
                        include_isomorphism_distractor=args.include_isomorphism_distractor, 
                        include_random_distractor=args.include_random_distractor,
                        full_relation_probability=args.full_relation_probability,
                        debug=False
                    ) # This is the minimum settings! You need to turn on attribute always!
                else:
                    # Sample a shapeWorld!
                    sampled_world = simulator.sample_situations_from_grounded_grammer(
                        copy.deepcopy(grammer_pattern), 
                        copy.deepcopy(obj_pattern_map), 
                        copy.deepcopy(rel_map), 
                        copy.deepcopy(obj_map),
                        is_plot=False,
                        include_relation_distractor=args.include_relation_distractor, 
                        include_attribute_distractor=args.include_attribute_distractor, 
                        include_isomorphism_distractor=args.include_isomorphism_distractor, 
                        include_random_distractor=args.include_random_distractor,
                        full_relation_probability=args.full_relation_probability, # ReaSCAN Special: 15 distractors!
                        debug=False
                    )

                # Validate the world is valid!
                graph = ReaSCANGraph(
                    objects=sampled_world["obj_map"], 
                    object_patterns=sampled_world["obj_pattern_map"], 
                    vocabulary=vocabulary,
                    positions=sampled_world["pos_map"], 
                    referred_object=sampled_world["referred_obj"],
                    debug=False
                )
                
                pattern_graph = ReaSCANGraph(
                    objects=obj_map, 
                    object_patterns=None,
                    vocabulary=vocabulary,
                    relations=rel_map, 
                    referred_object='$OBJ_0', 
                    debug=False
                )
                
                potential_referent_target = graph.find_referred_object_super_fast(
                    pattern_graph, referred_object='$OBJ_0', 
                    debug=False
                )

                # Save the result if the world is valid!
                
                # This may be to strict, but it ensures 100% correct!
                if len(potential_referent_target) == 1 and '$OBJ_0' in potential_referent_target:
                    # A quick world repeat check!
                    hash_world_str = hashlib.md5(str(sampled_world["situation"].to_representation()).encode('utf-8')).hexdigest()
                    if hash_world_str not in per_command_world_unique_check[command_struct_index]:
                        per_command_world_unique_check[command_struct_index].add(hash_world_str)
                    else:
                        continue # This is highly unlikely, but just to prevent!
                    
                    # Form the command with grounded determiners!
                    obj_determiner_map = graph.find_determiners(
                        pattern_graph, 
                        referred_object='$OBJ_0', 
                        debug=False,
                    )
                    
                    # we don't check this for P1 and P2?
                    
#                     valid_determiner = True
#                     for k, v in obj_determiner_map.items():
#                         if k != '$OBJ_0':
#                             if v != "a":
#                                 valid_determiner = False
#                                 break
#                     if not valid_determiner:
#                         continue # we should abort and resample!
                    
                    at_least_success = True
                    success_step += 1
                    
                    command_str = grammer.repre_str_command(
                        grammer_pattern, rel_map, obj_map, 
                        obj_determiner_map, 
                        verb,
                        adverb,
                    )

                    # Form the golden label for the action list!
                    is_transitive = False
                    if verb in simulator.vocabulary.get_transitive_verbs():
                        is_transitive = True
                    # Direct walk.
                    action = "walk" # this is definit!
                    primitive_command = simulator.vocabulary.translate_word(action)
                    target_position = sampled_world["situation"].target_object.position
                    simulator._world.go_to_position(
                        position=target_position, manner=adverb, 
                        primitive_command=primitive_command
                    )
                    # Object actions.
                    if is_transitive:
                        semantic_action = simulator.vocabulary.translate_word(verb)
                        simulator._world.move_object_to_wall(action=semantic_action, manner=adverb)
                    target_commands, _ = simulator._world.get_current_observations()
                    
                    has_relation_distractor = False
                    full_relation_distractor = True
                    for rel_bool in sampled_world["distractor_switch_map"]["relation"]:
                        if rel_bool:
                            has_relation_distractor = True
                        else:
                            full_relation_distractor = False
                    
                    # Save all relevant information for a task.
                    task_struct = OrderedDict({
                        "command": ",".join(command_str.split(" ")),
                        "grammer_pattern": grammer_pattern,
                        "meaning": ",".join(command_str.split(" ")),
                        "derivation": grammer_pattern,
                        "situation": sampled_world["situation"].to_representation(),
                        "target_commands": ",".join(target_commands),
                        "verb_in_command": verb,
                        "adverb_in_command": adverb,
                        "referred_target": obj_map["$OBJ_0"],
                        "object_pattern_map": obj_pattern_map,
                        "relation_map": [(k, v) for k, v in rel_map.items()],
                        "object_expression": obj_map,
                        "n_object": len(sampled_world["obj_map"]),
                        "n_distractor": len(sampled_world["obj_map"])-len(obj_map),
                        "full_relation_distractor": full_relation_distractor,
                        "has_relation_distractor": has_relation_distractor,
                        "has_attribute_distractor": sampled_world["distractor_switch_map"]["attribute"],
                        "has_isomorphism_distractor": sampled_world["distractor_switch_map"]["isomorphism"],
                        "has_random_distractor": True if sampled_world["n_random_distractor"] != -1 else False,
                        "n_random_distractor": sampled_world["n_random_distractor"] if sampled_world["n_random_distractor"] != -1 else 0,
                        "relation_distractor_metadata": sampled_world["relation_distractor_metadata"],
                        "attribute_distractor_metadata": sampled_world["attribute_distractor_metadata"],
                        "isomorphism_distractor_metadata": sampled_world["isomorphism_distractor_metadata"],
                        "random_distractor_metadata": sampled_world["random_distractor_metadata"],
                    })
                    
                    # Record distractor related info
                    if task_struct["full_relation_distractor"]:
                        d_full_relation_count += 1
                    if task_struct["has_relation_distractor"]:
                        d_relation_count += 1
                    if task_struct["has_attribute_distractor"]:
                        d_attribute_count += 1
                    if task_struct["has_isomorphism_distractor"]:
                        d_iso_count += 1
                    if task_struct["n_random_distractor"]:
                        d_random_count += 1
                    
                    # Here, we decide which split we put the example into!
                    split = args.mode
                    created_examples_by_splits[split].append(task_struct)
                    per_command_world_counts[command_struct_index] += 1
                    break # break the retry loop!
            if not at_least_success:
                logger.info(f"WARNING: the success rate for this command is close to 0.0%, skipping...")
                break # success rate for this comman is ~= 0.0%, let us directly skip
        if save_at_interval and (command_struct_index+1)% save_interal == 0:
            logger.info(f"Saving data files and statistics to {args.output_dir} for checkpoints...")
            # Now, we need to save data into the folder
            # along with possible statistics.
            to_save_command_struct = []
            per_command_count = []
            for command_struct_index, count in per_command_world_counts.items():
                per_command_count += [count]
                if count >= 1:
                    to_save_command_struct.append(sampled_command_struct_indexed[command_struct_index])
            if save_command_stats:
                _ = get_command_struct_statistics(
                    to_save_command_struct, run_name=f"ReaSCAN-{mode}", date=args.date, 
                    split=mode,
                    compositional_split=False,
                    n_sample=-1,
                    output_dir=args.output_dir,
                    save_to_disk=True if args.output_dir != "" else False,
                    wandb=wandb
                )
            
            # wandb.log({"per_command_world_count": wandb.Histogram(per_command_count)})
            
            data_file_path = os.path.join(args.output_dir, f"data-{args.mode}.txt")
            
            if mode == "demo" or mode == "all" or mode == "train":
                logger.info(f"total example count={success_step}...")
                dataset_representation = {
                    "grid_size": args.grid_size,
                    "type_grammar": "ReaSCAN-Grammer",
                    "min_object_size": 1,
                    "max_object_size": 4,
                    "percentage_train": split_percentage["train"],
                    "examples": created_examples_by_splits,
                    "intransitive_verbs": intransitive_verbs,
                    "transitive_verbs": transitive_verbs,
                    "adverbs": adverbs,
                    "nouns": nouns,
                    "color_adjectives": color_adjectives,
                    "size_adjectives": size_adjectives,
                    "relative_pronouns": relative_pronouns,
                    "relation_clauses": relation_clauses,
                }
                # dump to the disk.
                with open(data_file_path, "w") as fd:
                    json.dump(dataset_representation, fd, indent=4)
            else:
                pass
            
    # Last round of saving!
    logger.info(f"Saving FINAL data files and statistics to {args.output_dir}...")
    # Now, we need to save data into the folder
    # along with possible statistics.
    to_save_command_struct = []
    per_command_count = []
    for command_struct_index, count in per_command_world_counts.items():
        per_command_count += [count]
        if count >= 1:
            to_save_command_struct.append(sampled_command_struct_indexed[command_struct_index])
    if save_command_stats:
        _ = get_command_struct_statistics(
            to_save_command_struct, run_name=f"ReaSCAN-{mode}", date=args.date, 
            split=mode,
            compositional_split=False,
            n_sample=-1,
            output_dir=args.output_dir,
            save_to_disk=True if args.output_dir != "" else False,
            wandb=wandb
        )

    # wandb.log({"per_command_world_count": wandb.Histogram(per_command_count)})

    data_file_path = os.path.join(args.output_dir, f"data-{args.mode}.txt")

    if mode == "demo" or mode == "all" or mode == "train":
        logger.info(f"total example count={success_step}...")
        dataset_representation = {
            "grid_size": args.grid_size,
            "type_grammar": "ReaSCAN-Grammer",
            "min_object_size": 1,
            "max_object_size": 4,
            "percentage_train": split_percentage["train"],
            "examples": created_examples_by_splits,
            "intransitive_verbs": intransitive_verbs,
            "transitive_verbs": transitive_verbs,
            "adverbs": adverbs,
            "nouns": nouns,
            "color_adjectives": color_adjectives,
            "size_adjectives": size_adjectives,
            "relative_pronouns": relative_pronouns,
            "relation_clauses": relation_clauses,
        }
        # dump to the disk.
        with open(data_file_path, "w") as fd:
            json.dump(dataset_representation, fd, indent=4)
    else:
        pass
    
    logger.info("==FINISH==")
            
    if args.is_tensorboard:
        # end wandb
        wandb.finish()

