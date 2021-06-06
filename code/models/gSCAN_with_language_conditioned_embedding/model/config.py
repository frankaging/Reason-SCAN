from ast import literal_eval
import copy
import yaml
import numpy as np
import os
import argparse

class AttrDict(dict):

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

__C = AttrDict()
cfg = __C




# --------------------------------------------------------------------------- #
# general options
# --------------------------------------------------------------------------- #
##OS options
__C.DATA_DIRECTORY = "parsed_dataset-p1"
__C.OUTPUT_DIRECTORY = "output"
__C.EXP_NAME = 'p1-run-1'
__C.OUTPUT_FILE_NAME = "predict.json"
__C.MODE = "train"

## Dataset options
__C.SPLIT = "test" #NOT USE
__C.GENERATE_VOCABULARIES = False
__C.LOAD_VOCABULARIES = False
__C.INPUT_VOCAB_PATH = ""
__C.TARGET_VOCAB_PATH = ""

__C.INIT_WRD_EMB_FROM_FILE = False
__C.WRD_EMB_INIT_FILE = ''

# --------------------------------------------------------------------------- #
# model options
# --------------------------------------------------------------------------- #
## Command Encoder
__C.CMD_D_EMBED = 32
__C.CMD_D_ENC = 64
__C.CMD_D_H = 64 # Same as ENC_DIM

## Situation Encoder (LGCN)
__C.SITU_D_FEAT = 64 # 3 * D_CNN_OUTPUT if CNN then LGCN
__C.SITU_D_CTX = 64 # 512
__C.SITU_D_CMD = 64 # 512
__C.SITU_D_CNN_OUTPUT = 64
#1

## Decoder
__C.DEC_D_H = 64
__C.DEC_NUM_LAYER = 1
__C.DEC_CONDITIONAL_ATTENTION = True




__C.H_FEAT = 14
__C.W_FEAT = 14
__C.D_FEAT = 1152  # 1024+128
__C.T_ENCODER = 45
__C.ADD_POS_ENC = True
__C.PE_DIM = 128
__C.PE_SCALE = 1.

__C.MSG_ITER_NUM = 4

__C.STEM_NORMALIZE = True
__C.STEM_LINEAR = True
__C.STEM_CNN = False
__C.STEM_CNN_DIM = 512
__C.STEM_RENORMALIZE = False
# __C.WRD_EMB_DIM = 300
__C.WRD_EMB_FIXED = False
# __C.ENC_DIM = 512
__C.CMD_DIM = 512
__C.CMD_INPUT_ACT = 'ELU'
__C.CTX_DIM = 512
__C.OUT_QUESTION_MUL = True
__C.OUT_CLASSIFIER_DIM = 512

__C.USE_EMA = True
__C.EMA_DECAY_RATE = 0.999

# Dropouts
__C.encInputDropout = 0.8
__C.locDropout = 1.
__C.cmdDropout = 0.92
__C.memoryDropout = 0.85
__C.readDropout = 0.85
__C.outputDropout = 0.85
__C.decoderDropout = 0.85

__C.MASK_PADUNK_IN_LOGITS = True

__C.BUILD_VQA = True
__C.BUILD_REF = False

# CLEVR-Ref configs
__C.BBOX_IOU_THRESH = .5
__C.IMG_H = 320  # size in loc
__C.IMG_W = 480  # size in loc

# Loss option
__C.AUXILIARY_TASK = False

# --------------------------------------------------------------------------- #
# training options
# --------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()
__C.TRAIN.BATCH_SIZE = 200
__C.VAL_BATCH_SIZE = 4000
__C.TRAIN.START_EPOCH = 0

__C.TRAIN.CLIP_GRADIENTS = True
__C.TRAIN.GRAD_MAX_NORM = 8.

__C.TRAIN.SOLVER = AttrDict()
# __C.TRAIN.SOLVER.LR = 3e-4
__C.TRAIN.SOLVER.LR = 8e-4
__C.TRAIN.SOLVER.LR_DECAY = 0.9
__C.TRAIN.SOLVER.ADAM_BETA1 = 0.9
__C.TRAIN.SOLVER.ADAM_BETA2 = 0.999
__C.TRAIN.SOLVER.LR_DECAY_STEP = 20000
__C.TRAIN.MAX_EPOCH = 100
__C.TRAIN.RUN_EVAL = True
__C.TRAIN.USE_MULTI_GPU = True
__C.PRINT_EVERY = 100
__C.EVALUATE_EVERY = 10
__C.SAVE_EVERY = 20

#GSCAN Specific
__C.TRAIN.K = 0
__C.TRAIN.WEIGHT_TARGET_LOSS = 0.3 # change only when auxiliary is used



# --------------------------------------------------------------------------- #
# test options
# --------------------------------------------------------------------------- #
__C.TEST = AttrDict()
__C.TEST.SPLIT = "" #\TODO test split not initialized yet
__C.TEST.MAX_DECODING_STEP = 30

__C.TEST.BATCH_SIZE = 1
__C.TEST.EPOCH = -1  # Needs to be supplied
__C.TEST.DUMP_PRED = False
__C.TEST.RESULT_DIR = './exp_clevr/results/%s/%04d'

__C.TEST.NUM_VIS = 0
__C.TEST.VIS_DIR_PREFIX = 'vis'
__C.TEST.VIS_FILTER_EDGE = True
__C.TEST.VIS_EDGE_SCALE = 1.
__C.TEST.VIS_FINAL_REL_TH = .025
__C.TEST.VIS_FINAL_ABS_TH = .025
__C.TEST.VIS_MSG_TH = .1

# --------------------------------------------------------------------------- #
# post-processing configs after loading
# --------------------------------------------------------------------------- #
def _postprocess_cfg():  # NoQA
    __C.GPUS = __C.GPUS.replace(' ', '').replace('(', '').replace(')', '')
    assert __C.EXP_NAME != '<fill-with-filename>', 'EXP_NAME must be specified'

# --------------------------------------------------------------------------- #


def build_cfg_from_argparse(args_list=None):
    """Load config with command line options (`--cfg` and a list of options)"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args(args_list)
    if args.cfg:
        _merge_cfg_from_file(args.cfg)
    if args.opts:
        _merge_cfg_from_list(args.opts)
    _postprocess_cfg()
    return __C


def _merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = yaml.load(f)
    if yaml_cfg is not None:
        _merge_a_into_b(AttrDict(yaml_cfg), __C)
    if __C.EXP_NAME == '<fill-with-filename>':
        __C.EXP_NAME = os.path.basename(cfg_filename).replace('.yaml', '')


def _merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def _merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
