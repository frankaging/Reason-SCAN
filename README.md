# ReaSCAN: Compositional Reasoning in Language Grounding
ReaSCAN is a synthetic navigation task that requires models to reason about surroundings over syntactically difficult languages.

## Contents

* [Citation](#Citation)
* [Example](#Example)
* [Dataset](#Dataset)
* [Data format](#data-format)
* [ReaSCAN as an Abstract Reasoning Challenge](#reascan-as-an-abstract-reasoning-challenge)
* [Dataset Artifacts](#Dataset-artifacts)
* [Models](#models)
* [Other files](#other-files)
* [License](#license)

## Citation

[Zhengxuan Wu](http://zen-wu.social), [Elisa Kreiss](https://www.elisakreiss.com/), [Desmond C. Ong](https://web.stanford.edu/~dco/), and [Christopher Potts](http://web.stanford.edu/~cgpotts/). 2020. [ReaSCAN: Compositional Reasoning in Language Grounding](http://zen-wu.social). Ms., Stanford University.

```stex
  @article{wu-etal-2021-reascan,
    title={{ReaSCAN}: Compositional Reasoning in Language Grounding},
    author={Wu, Zhengxuan and Kreiss, Elisa and Ong, Desmond C. and Potts, Christopher},
    journal={},
    url={},
    year={2021}}
```

## Example
Four command-world pairs for different command patterns. Our simple command is equivalent to [gSCAN](https://arxiv.org/abs/2003.05161). **RD** means distractors are randomly sampled. Referent targets shaded in red with distractors are shaded in blue, and are highlighted by green dash lines.

<img src="https://i.ibb.co/Yfx5WTj/overview.png" width="800">

## Dataset

### Off-the-shelf ReaSCAN

We generated ReaSCAN using our pipeline with fixed random seeds. You can reproduce the version of ReaSCAN we use in the paper by running the pipeline. Additionally, we also update the version we use to a online folder where you can directly download and use as-it-is. Note that, the dataset files are really large. It may take a while to download them.

Our generated data is in [ReaSCAN-v1.0.zip](https://drive.google.com/file/d/1quUyPHTRdsfdZ80hrGX9p7o-TMdEGJtj/view?usp=sharing), which is saved in a shared drive. The dataset consists subsets generated for different patterns (P1: Simple (similar to gSCAN), P2: 1-relative-clause, P3: 2-relative-clauses, P4: 3-relative-clauses) and different compositional splits (see [our paper]() for details about each split).

By patterns,
* `ReaSCAN-compositional`: ReaSCAN all commands, containing train, dev and test sets.
* `ReaSCAN-compositional-p1`: ReaSCAN Simple set, containing train, dev and test sets.
* `ReaSCAN-compositional-p2`: ReaSCAN 1-relative-clause set, containing train, dev and test sets.
* `ReaSCAN-compositional-p3`: ReaSCAN 2-relative-clauses set, containing train, dev and test sets.
* `ReaSCAN-compositional-p1-test`: ReaSCAN Simple set, containing test set only.
* `ReaSCAN-compositional-p2-test`: ReaSCAN 1-relative-clause set, containing test set only.
* `ReaSCAN-compositional-p3-test`: ReaSCAN 2-relative-clauses set, containing test set only.

By splits,
* `ReaSCAN-compositional-a1`: ReaSCAN A1 compositional split, containing test set only.
* `ReaSCAN-compositional-a2`: ReaSCAN A2 compositional split, containing test set only.
* `ReaSCAN-compositional-a3`: ReaSCAN A3 compositional split, containing test set only.
* `ReaSCAN-compositional-b1`: [WARNING] This split is deprecated! Do not use!
* `ReaSCAN-compositional-b2`: ReaSCAN B compositional split, containing test set only.
* `ReaSCAN-compositional-c`: ReaSCAN C compositional split, containing test set only.

Note that our A1 and A2 is similar to gSCAN's split B and C in the setup but different in what they are testing. We plan to support the splits in gSCAN in future releases as well. In fact, you can also generate your own compositional splits by modifying couple lines in `code/dataset/generate_ReaSCAN_splits.ipynb`.

Special split,
* `ReaSCAN-compositional-p3-rd`: ReaSCAN 2-relative-clauses set with random distractors, containing train, dev and test sets.
* `ReaSCAN-compositional-p4` or `ReaSCAN-compositional-p4-test`: ReaSCAN 3-relative-clauses set only, containing test set only.

### Regenerate ReaSCAN

You can recreate ReaSCAN shared above using provided scripts. Since generating a full-fleged dataset can take long, you can use our multi-process generator which can generate any subset included in our paper within 20 mininutes with 50 processes. Here are some example code we used to generate 2-relative-clauses set dataset. For exact scripts we use to generate our dataset used in the paper, you can refer to ``code/experiments.sh``.

Single process generation,
```bash
cd code/dataset

python generate_ReaSCAN.py \
--mode train \
--n_command_struct 100 \
--date 2021-05-30 \
--grid_size 6 \
--n_object_max 13 \
--per_command_world_retry_max 500 \
--per_command_world_target_count 3 \
--output_dir ./ReaSCAN-compositional-demo/ \
--include_relation_distractor \
--include_attribute_distractor \
--include_isomorphism_distractor \
--include_random_distractor \
--full_relation_probability 1.0 \
--command_pattern p3 \
--save_interal 200
```

Multi-process generation,
```bash
cd code/dataset

python generate_ReaSCAN_batch.py
```
Note that you need to go into the file and modify some variables to generate the dataset you want.

## Dataset format

### Loading ReaSCAN

Once you generate the dataset ``.txt`` file (in ``json`` format), you can simply load any dataset as,
```python
import json

path_to_data = "data-compositional-splits.txt"
logger.info(f"Reading dataset from file: {p1_path_to_data}...")
data_json = json.load(open(path_to_data, "r"))

print(data_json["examples"].keys())
```

We keep our format the same as gSCAN. For each example, we provide the command and the world representation. Additionally, we provide ReaSCAN specific metadata,

<details open>
<summary>The first data example in the split called ReaSCAN-compositional-p3-test set. Click to open/close.</summary>
<p>
 
```javascript
{
                "command": "pull,a,small,object,that,is,in,the,same,column,as,a,green,cylinder,and,in,the,same,shape,as,a,small,red,object,cautiously",
                "grammer_pattern": "$OBJ_0 ^ $OBJ_1 & $OBJ_2",
                "meaning": "pull,a,small,object,that,is,in,the,same,column,as,a,green,cylinder,and,in,the,same,shape,as,a,small,red,object,cautiously",
                "derivation": "$OBJ_0 ^ $OBJ_1 & $OBJ_2",
                "situation": {
                    "grid_size": 6,
                    "agent_position": {
                        "row": "1",
                        "column": "1"
                    },
                    "agent_direction": 0,
                    "target_object": {
                        "vector": "010010000001",
                        "position": {
                            "row": "2",
                            "column": "3"
                        },
                        "object": {
                            "shape": "circle",
                            "color": "yellow",
                            "size": "2"
                        }
                    },
                    "distance_to_target": "3",
                    "direction_to_target": "se",
                    "placed_objects": {
                        "0": {
                            "vector": "010010000001",
                            "position": {
                                "row": "2",
                                "column": "3"
                            },
                            "object": {
                                "shape": "circle",
                                "color": "yellow",
                                "size": "2"
                            }
                        },
                        "1": {
                            "vector": "001001000010",
                            "position": {
                                "row": "0",
                                "column": "3"
                            },
                            "object": {
                                "shape": "cylinder",
                                "color": "green",
                                "size": "3"
                            }
                        },
                        "2": {
                            "vector": "010010001000",
                            "position": {
                                "row": "3",
                                "column": "0"
                            },
                            "object": {
                                "shape": "circle",
                                "color": "red",
                                "size": "2"
                            }
                        },
                        "3": {
                            "vector": "100000100100",
                            "position": {
                                "row": "3",
                                "column": "2"
                            },
                            "object": {
                                "shape": "square",
                                "color": "blue",
                                "size": "1"
                            }
                        },
                        "4": {
                            "vector": "010010001000",
                            "position": {
                                "row": "5",
                                "column": "5"
                            },
                            "object": {
                                "shape": "circle",
                                "color": "red",
                                "size": "2"
                            }
                        },
                        "5": {
                            "vector": "100001001000",
                            "position": {
                                "row": "3",
                                "column": "4"
                            },
                            "object": {
                                "shape": "cylinder",
                                "color": "red",
                                "size": "1"
                            }
                        },
                        "6": {
                            "vector": "001001000010",
                            "position": {
                                "row": "0",
                                "column": "4"
                            },
                            "object": {
                                "shape": "cylinder",
                                "color": "green",
                                "size": "3"
                            }
                        },
                        "7": {
                            "vector": "010000101000",
                            "position": {
                                "row": "4",
                                "column": "3"
                            },
                            "object": {
                                "shape": "square",
                                "color": "red",
                                "size": "2"
                            }
                        },
                        "8": {
                            "vector": "010001000001",
                            "position": {
                                "row": "1",
                                "column": "3"
                            },
                            "object": {
                                "shape": "cylinder",
                                "color": "yellow",
                                "size": "2"
                            }
                        },
                        "9": {
                            "vector": "100001001000",
                            "position": {
                                "row": "1",
                                "column": "5"
                            },
                            "object": {
                                "shape": "cylinder",
                                "color": "red",
                                "size": "1"
                            }
                        },
                        "10": {
                            "vector": "001010001000",
                            "position": {
                                "row": "3",
                                "column": "5"
                            },
                            "object": {
                                "shape": "circle",
                                "color": "red",
                                "size": "3"
                            }
                        },
                        "11": {
                            "vector": "001010001000",
                            "position": {
                                "row": "0",
                                "column": "1"
                            },
                            "object": {
                                "shape": "circle",
                                "color": "red",
                                "size": "3"
                            }
                        },
                        "12": {
                            "vector": "001001000001",
                            "position": {
                                "row": "5",
                                "column": "0"
                            },
                            "object": {
                                "shape": "cylinder",
                                "color": "yellow",
                                "size": "3"
                            }
                        },
                        "13": {
                            "vector": "001000100010",
                            "position": {
                                "row": "0",
                                "column": "0"
                            },
                            "object": {
                                "shape": "square",
                                "color": "green",
                                "size": "3"
                            }
                        },
                        "14": {
                            "vector": "100001001000",
                            "position": {
                                "row": "5",
                                "column": "1"
                            },
                            "object": {
                                "shape": "cylinder",
                                "color": "red",
                                "size": "1"
                            }
                        }
                    },
                    "carrying_object": null
                },
                "target_commands": "turn left,turn right,turn right,turn left,walk,turn left,turn right,turn right,turn left,walk,turn right,turn left,turn right,turn right,turn left,walk",
                "verb_in_command": "pull",
                "adverb_in_command": "cautiously",
                "referred_target": "small object",
                "object_pattern_map": {
                    "$OBJ_0": "$SIZE $ABS_SHAPE",
                    "$OBJ_1": "$COLOR $SHAPE",
                    "$OBJ_2": "$SIZE $COLOR $ABS_SHAPE"
                },
                "relation_map": [
                    [
                        [
                            "$OBJ_0",
                            "$OBJ_1"
                        ],
                        "$SAME_COLUMN"
                    ],
                    [
                        [
                            "$OBJ_0",
                            "$OBJ_2"
                        ],
                        "$SAME_SHAPE"
                    ]
                ],
                "object_expression": {
                    "$OBJ_0": "small object",
                    "$OBJ_1": "green cylinder",
                    "$OBJ_2": "small red object"
                },
                "n_object": 15,
                "n_distractor": 12,
                "full_relation_distractor": true,
                "has_relation_distractor": true,
                "has_attribute_distractor": true,
                "has_isomorphism_distractor": true,
                "has_random_distractor": false,
                "n_random_distractor": 0,
                "relation_distractor_metadata": [
                    {
                        "distractor_metadata": {
                            "edge": [
                                "$OBJ_0",
                                "$OBJ_1"
                            ],
                            "relation_old_type": "$SAME_COLUMN",
                            "full_set": true
                        }
                    },
                    {
                        "distractor_metadata": {
                            "edge": [
                                "$OBJ_0",
                                "$OBJ_2"
                            ],
                            "relation_old_type": "$SAME_SHAPE",
                            "full_set": true
                        }
                    }
                ],
                "attribute_distractor_metadata": [
                    {
                        "distractor_metadata": [
                            {
                                "modified_obj": "$OBJ_1",
                                "modified_attribute": "$COLOR"
                            }
                        ]
                    }
                ],
                "isomorphism_distractor_metadata": [
                    {
                        "distractor_metadata": [
                            {
                                "swapped_pair": [
                                    "$OBJ_1",
                                    "$OBJ_2"
                                ],
                                "before_pair_obj_str": [
                                    "green cylinder",
                                    "small red object"
                                ],
                                "after_pair_obj_str": [
                                    "small green object",
                                    "red cylinder"
                                ],
                                "size_shuffled": true,
                                "color_shuffled": false,
                                "shape_shuffled": true
                            }
                        ]
                    }
                ],
                "random_distractor_metadata": [
                    {}
                ]
            }
```
</p>
</details>

This is one example from this dataset. It contains the *"command"*, or input  instruction, 'pull,a,small,object,that,is,in,the,same,column,as,a,green,cylinder,and,in,the,same,shape,as,a,small,red,object,cautiously' separated by `,`, which for the specified world state (i.e., *"situation"*) maps to the *"target_commands"*: "turn left,turn right,turn right,turn left,walk,turn left,turn right,turn right,turn left,walk,turn right,turn left,turn right,turn right,turn left,walk". The example contains the situation representation, or world state, at the key *"situation"*, and also contains additional information that is needed in generating the world for example what are our distractors made of, such as fields in the `relation_distractor_metadata`.

To be more compatiable with other models, we also provide a translation script that can translate each exmaple into a compressed dictionary containing all the information needed to train a neural model (i.e., input: a command sequence + tensor representation of a shape world, output: an output action sequence are all you need.). To convert, you can refer the following script,
```bash
cd code/models/gSCAN_with_language_conditioned_embedding

jupyter notebook

# open this file: read_reascan.ipynb
```

Following steps in this script, each example will be translated to a data structure like,

<details open>
<summary>Compact version of ReaSCAN that is ready-to-use by any neural models. Click to open/close.</summary>
<p>
 
```javascript
{"input": ["pull", "a", "big", "yellow", "square", "that", "is", "in", "the", "same", "row", "as", "a", "small", "blue", "circle", "and", "in", "the", "same", "column", "as", "a", "big", "green", "cylinder"], "target": ["turn left", "turn left", "walk", "walk", "walk", "walk", "turn left", "walk", "walk", "pull", "pull", "pull", "pull"], "situation": [[[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]]}
```
</p>
</details>

Note that the situation is a tensor representation of the shape world. Each sub-list is the representation of each cell in the world. It encodes what object is in what position based on the following information,
```python
"""
Each grid cell in a situation is fully specified by a vector:
[_ _ _ _ _ _ _   _       _      _      _    _   _ _ _ _]
 1 2 3 4 r g b circle square cylinder box agent E S W N
 _______ _____ ______________________ _____ _______
   size  color        shape           agent agent dir.
:param situation_representation: data from dataset.txt at key "situation".
:param grid_size: int determining row/column number.
:return: grid to be parsed by computational models.
"""
```
In case, if there are overlayed objects in a single cell, we add them together. This is only for a object that is inside of the box if the object is at the upper left corner. There are many other ways to represent this situation, but we take the simplest approach.


## ReaSCAN as an Abstract Reasoning Challenge

Two simplified abstract reasoning challenges with ReaSCAN. The task mimics human reasoning test where giving a set of input-output (input on the left and output on the right) pairs, the task taker needs to guess the output for the last input. For each task, we provide one potential abstract reasoning to solve the task.

<img src="https://i.ibb.co/0J4n24c/Rea-SCAN-ARC.png" width="800">

You can generate such tasks using the script provided in `code/dataset/future-looking-demo.ipynb`.

## Dataset Artifacts

ReaSCAN in not perfect. In fact, we document a list of artifacts in our paper. Please see our **Appendix B** for details. Please read this before you use ReaSCAN. Here is a short summary of that section in bullet points:

* **Non-comprehensive Linguistic Structures**: Commands from ReaSCAN follow a specific linguistic template and are non-comprehensive in covering all linguistic structures. 
* **Non-comprehensive Distractors**: ReaSCAN is not able to cover all possible distractors to make sure every part of the command is necessary to resolve the referring expression.
* **Shapes and Relations Biases**: The frequency distributions of shapes and relations may be biased due to the generation program.
* **Self-exclusiveness**: We assume every object mention in the command matches a unique object in the world.
* **Other Induced Artifacts**: We also discuss frequency distributions of verbs, adverbs, agent facing directions, agent-target relative directions, etc.


## Models

We use two existing models, and adapt their codes to benchmark ReaSCAN. Both models are published and experimented on gSCAN. Other than hyperparameter tunning, we are not changing model architectures.

### Multimodal LSTM

This model is published with gSCAN [in this paper](https://arxiv.org/abs/2003.05161) from [this repo](https://github.com/LauraRuis/multimodal_seq2seq_gSCAN). You can refer to their repo for details about the model. Here, we already adapt interface changes that are needed to run with ReaSCAN, you can simply run training with following lines,

```bash
cd code/models/seq2seq

CUDA_VISIBLE_DEVICES=0 python run_reascan.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=ReaSCAN-compositional-p1 \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=./training_logs/p1-random-seed-44 \
--training_batch_size=2000 \
--max_training_iterations=200000 \
--seed=44
```

Note that this requires you generate the vocabulary file before hand to save time. You can do so by following scripts provided in the notebook ``ReaSCAN-vocab-generator.ipynb`` in the same folder.

To evaluate this model, you need to run evaluation script and generate all predictions. Note that we follow the original repo, and you can refer to their code for your own implementations. This is the script we run,

```bash
cd code/models/seq2seq

CUDA_VISIBLE_DEVICES=0 python run_reascan.py \
 --mode=test \
 --data_directory=../../../data-files-updated/ReaSCAN-compositional-p1/ \
 --input_vocab_path=input_vocabulary.txt \
 --target_vocab_path=target_vocabulary.txt \
 --attention_type=bahdanau \
 --no_auxiliary_task \
 --conditional_attention \
 --output_directory=../../../testing_logs/p1-random-seed-44/  \
 --resume_from_file=../../../training_logs/p1-random-seed-44/model_best.pth.tar \
 --splits=dev \
 --output_file_name=p1-random-seed-44.json \
 --max_decoding_steps=120
```
Note that this is for ``--splits=dev``, you can change to ``--splits=test`` if you want to evaluate with test splits.

After this script, it will generate predictions in the file in the output directory. Then, you can use our notebook to analyze the results by running the notebook ``performance-analysis.ipynb`` in the model folder!


### GCN + LSTM

This model is published with gSCAN [in this paper](https://arxiv.org/pdf/2009.05552.pdf) from [this repo](https://github.com/HQ01/gSCAN_with_language_conditioned_embedding). You can refer to their repo for details about the model. Here, we already adapt interface changes that are needed to run with ReaSCAN, you can simply run training with following lines,

```bash
cd code/models/gSCAN_with_language_conditioned_embedding

CUDA_VISIBLE_DEVICES=0 python main_model.py \
--run p1-random-seed-66 \
--data_dir ./parsed_dataset-p1/ \
--seed 44 \
--txt
```

Note that the script above assumed that you already parse the dataset following the parsing helpers provided in the notebook ``read_reascan.ipynb``.

After running this script, all models will be saved in the directory folder. Then, you can evaluate performance of this model using scripts as,
```bash
cd code/models/gSCAN_with_language_conditioned_embedding

CUDA_VISIBLE_DEVICES=0 python eval_best_model.py \
--load ./output/p1-random-seed-44/model_best.pth.tar \
--data_dir ./parsed_dataset-p1/ \
--seed 44 \
--test_split dev
```
Note that this is for ``--test_split=dev``, you can change to ``--test_split=test`` if you want to evaluate with test splits.


## Other files

In this repo, we also provide a lot of useful scripts to analyze ReaSCAN in various ways. Here are a non-comprehensive list of them with their purposes,

* `code/models/seq2seq/performance-analysis.ipynb`: evaluate model performance.
* `code/models/seq2seq/ReaSCAN-vocab-generator.ipynb`: generate required vocab files.
* `code/models/gSCAN_with_language_conditioned_embedding/read_reascan.ipynb`: helper to parse the dataset into model readable format.
* `code/experiments.sh`: all bash scripts we run for our experiment results.
* `code/dataset/demo.ipynb`: demo file for all components involved in ReaSCAN data generation process.
* `code/dataset/unit_tests.ipynb`: unit tests for ReaSCAN. If you want to customized ReaSCAN, please run this unit test before changing anything.
* `code/dataset/generate_ReaSCAN_splits.ipynb`: generate splits for ReaSCAN.
* `code/dataset/ReaSCAN-analysis.ipynb`: some analyses we conduct in the paper.


## License

ReaSCAN has a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
