import argparse
import os

from torch.optim.lr_scheduler import LambdaLR

from dataloader import dataloader
from model.config import cfg
from model.model import GSCAN_model
from model.utils import *


def evaluate(data_iterator, model, max_decoding_steps, pad_idx, sos_idx, eos_idx, max_examples_to_evaluate=None):
    target_accuracies = []
    exact_match = 0
    num_examples = 0
    correct_terms = 0
    total_terms = 0
    for input_sequence, output_sequence, target_sequence, _, _, aux_acc_target in predict(
            data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
            sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate):
        num_examples += output_sequence.shape[0]
        seq_eq = torch.eq(output_sequence, target_sequence)
        mask = torch.eq(target_sequence, pad_idx) + torch.eq(target_sequence, sos_idx)
        seq_eq.masked_fill_(mask, 0)
        total = (~mask).sum(-1).float()
        accuracy = seq_eq.sum(-1) / total
        total_terms += total.sum().data.item()
        correct_terms += seq_eq.sum().data.item()
        exact_match += accuracy.eq(1.).sum().data.item()
        target_accuracies.append(aux_acc_target)
    return (float(correct_terms) / total_terms) * 100, (exact_match / num_examples) * 100, \
           float(np.mean(np.array(target_accuracies))) * 100


def train(train_data_path: str, val_data_paths: dict, use_cuda: bool, resume_from_file: str, is_baseline: bool):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')

    logger.info("Loading Training set...")
    train_iter, train_input_vocab, train_target_vocab = dataloader(train_data_path,
                                                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                                                   use_cuda=use_cuda)
    val_iters = {}
    for split_name, path in val_data_paths.items():
        val_iters[split_name], _, _ = dataloader(path, batch_size=cfg.VAL_BATCH_SIZE, use_cuda=use_cuda,
                                                 input_vocab=train_input_vocab, target_vocab=train_target_vocab)

    pad_idx, sos_idx, eos_idx = train_target_vocab.stoi['<pad>'], train_target_vocab.stoi['<sos>'], \
                                train_target_vocab.stoi['<eos>']

    train_input_vocab_size, train_target_vocab_size = len(train_input_vocab.itos), len(train_target_vocab.itos)

    logger.info("Loading Dev. set...")

    val_input_vocab_size, val_target_vocab_size = train_input_vocab_size, train_target_vocab_size
    logger.info("Done Loading Dev. set.")

    model = GSCAN_model(pad_idx, eos_idx, train_input_vocab_size, train_target_vocab_size, is_baseline=is_baseline)

    model = model.cuda() if use_cuda else model

    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.TRAIN.SOLVER.LR,
                                 betas=(cfg.TRAIN.SOLVER.ADAM_BETA1, cfg.TRAIN.SOLVER.ADAM_BETA2))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: cfg.TRAIN.SOLVER.LR_DECAY ** (t / cfg.TRAIN.SOLVER.LR_DECAY_STEP))

    cfg.RESUME_FROM_FILE = resume_from_file
    assert os.path.isfile(cfg.RESUME_FROM_FILE), "No checkpoint found at {}".format(cfg.RESUME_FROM_FILE)
    logger.info("Loading checkpoint from file at '{}'".format(cfg.RESUME_FROM_FILE))
    optimizer_state_dict = model.load_model(cfg.RESUME_FROM_FILE)
    optimizer.load_state_dict(optimizer_state_dict)
    start_iteration = model.trained_iterations
    print("start iteration is .. ", start_iteration)
    logger.info("Loaded checkpoint '{}' (iter {})".format(cfg.RESUME_FROM_FILE, start_iteration))

    logger.info("Training starts..")
    with torch.no_grad():
        model.eval()
        logger.info("Evaluating..")
        print(val_iters)
        for split_name, val_iter in val_iters.items():
            accuracy, exact_match, target_accuracy = evaluate(
                val_iter, model=model,
                max_decoding_steps=30, pad_idx=pad_idx,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_examples_to_evaluate=None)
            logger.info(" %s Accuracy: %5.2f Exact Match: %5.2f "
                        " Target Accuracy: %5.2f " % (split_name, accuracy, exact_match, target_accuracy))


def main(flags, use_cuda):
    train_data_path = os.path.join(cfg.DATA_DIRECTORY, "train.json")

    test_splits = [
        'situational_1',
        'situational_2',
        'test',
        'visual',
        'visual_easier',
        'dev',
        'adverb_1',
        'adverb_2',
        'contextual',
    ]
    val_data_paths = {split_name: os.path.join(cfg.DATA_DIRECTORY, split_name + '.json') for split_name in test_splits}

    if cfg.MODE == "train":
        train(train_data_path=train_data_path, val_data_paths=val_data_paths, use_cuda=use_cuda,
              resume_from_file=flags.load, is_baseline=flags.is_baseline)

    elif cfg.MODE == "predict":
        raise NotImplementedError()

    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(cfg.MODE))


if __name__ == "__main__":
    # torch.manual_seed(cfg.SEED)
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                        datefmt="%Y-%m-%d %H:%M")
    logger = logging.getLogger(__name__)
    use_cuda = True if torch.cuda.is_available() else False
    logger.info("Initialize logger")

    if use_cuda:
        logger.info("Using CUDA.")
        logger.info("Cuda version: {}".format(torch.version.cuda))

    parser = argparse.ArgumentParser(description="LGCN models for GSCAN")
    parser.add_argument('--load', type=str, help='Path to model')
    parser.add_argument('--baseline', dest='is_baseline', action='store_true')
    parser.set_defaults(is_baseline=False)
    args = parser.parse_args()

    main(args, use_cuda)
