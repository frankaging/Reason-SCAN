import argparse
import os
import sys

import torch
import numpy as np
import random
from torch.optim.lr_scheduler import LambdaLR

from dataloader import dataloader
from model.config import cfg
from model.model import GSCAN_model
from model.utils import *


def train(train_data_path: str, val_data_paths: dict, use_cuda: bool, model_name: str, is_baseline: bool,
          resume_from_file=None):
    logger.info("Loading Training set...")
    logger.info(model_name)
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

    '''
    Input (command) [0]: batch_size x max_cmd_len       [1]: batch_size x 0 (len for each cmd)
    Situation: batch_size x grid x grid x feat_size
    Target (action) [0]: batch_size x max_action_len    [1]: batch_size x 0 (len for each action sequence)

    max_cmd_len = 6, max_action_len = 16
    '''
    logger.info("Done Loading Training set.")

    # if generate_vocabularies:
    #     training_set.save_vocabularies(input_vocab_path, target_vocab_path)
    #     logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Dev. set...")

    # val_input_vocab_size, val_target_vocab_size = train_input_vocab_size, train_target_vocab_size

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.

    # val_set.shuffle_data()
    logger.info("Done Loading Dev. set.")

    model = GSCAN_model(pad_idx, eos_idx, train_input_vocab_size, train_target_vocab_size, is_baseline=is_baseline,
                        output_directory=os.path.join(os.getcwd(), cfg.OUTPUT_DIRECTORY, model_name))

    model = model.cuda() if use_cuda else model

    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.TRAIN.SOLVER.LR,
                                 betas=(cfg.TRAIN.SOLVER.ADAM_BETA1, cfg.TRAIN.SOLVER.ADAM_BETA2))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: cfg.TRAIN.SOLVER.LR_DECAY ** (t / cfg.TRAIN.SOLVER.LR_DECAY_STEP))

    start_iteration = 1
    best_exact_match = 0

    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = model.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file, start_iteration))

    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < cfg.TRAIN.MAX_EPOCH:  # iterations here actually means "epoch"

        # Shuffle the dataset and loop over it.
        # training_set.shuffle_data()
        num_batch = 0
        for x in train_iter:
            is_best = False
            model.train()
            target_scores, target_position_scores = model(x.input, x.situation,
                                                          x.target)

            loss = model.get_loss(target_scores, x.target[0])

            target_loss = 0
            if cfg.AUXILIARY_TASK:
                target_loss = model.get_auxiliary_loss(target_position_scores,
                                                       x.target)
            loss += cfg.TRAIN.WEIGHT_TARGET_LOSS * target_loss

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.update_state(is_best=is_best)

            # Print current metrics.
            if num_batch % cfg.PRINT_EVERY == 0:
                accuracy, exact_match = model.get_metrics(target_scores, x.target[0])
                if cfg.AUXILIARY_TASK:
                    auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores,
                                                                             x.target)
                else:
                    auxiliary_accuracy_target = 0.
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f,"
                            " aux. accuracy target pos %5.2f" % (training_iteration, loss, accuracy, exact_match,
                                                                 learning_rate, auxiliary_accuracy_target))

            num_batch += 1

        if training_iteration % cfg.EVALUATE_EVERY == 0:
            with torch.no_grad():
                model.eval()
                logger.info("Evaluating..")
                test_exact_match = 0
                test_accuracy = 0
                try:
                    for split_name, val_iter in val_iters.items():
                        accuracy, exact_match, target_accuracy = evaluate(
                            val_iter, model=model,
                            max_decoding_steps=30, pad_idx=pad_idx,
                            sos_idx=sos_idx,
                            eos_idx=eos_idx,
                            max_examples_to_evaluate=None)
                        if split_name == 'dev':
                            test_exact_match = exact_match
                            test_accuracy = accuracy

                        logger.info(" %s Accuracy: %5.2f Exact Match: %5.2f "
                                    " Target Accuracy: %5.2f " % (split_name, accuracy, exact_match, target_accuracy))
                except:
                    print("Exception!")

                if test_exact_match > best_exact_match:
                    is_best = True
                    best_accuracy = test_accuracy
                    best_exact_match = test_exact_match
                    model.update_state(accuracy=test_accuracy, exact_match=test_exact_match, is_best=is_best)
                file_name = model_name + "checkpoint.{}th.tar".format(str(training_iteration))
                # file_name = os.path.join(os.getcwd(), cfg.OUTPUT_DIRECTORY, model_name, file_name)
                if is_best:
                    logger.info("saving best model...")
                    model.save_checkpoint(file_name=file_name, is_best=is_best,
                                          optimizer_state_dict=optimizer.state_dict())

        if training_iteration % cfg.SAVE_EVERY == 0:
            logger.info("forcing to save model every several epochs...")
            file_name = model_name + " checkpoint_force.{}th.tar".format(str(training_iteration))
            # file_name = os.path.join(os.getcwd(), cfg.OUTPUT_DIRECTORY, model_name, file_name)
            model.save_checkpoint(file_name=file_name, is_best=False, optimizer_state_dict=optimizer.state_dict())

        training_iteration += 1  # warning: iteratin represents epochs here
    logger.info("Finished training.")


def main(flags, use_cuda):
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIRECTORY, flags.run)):
        os.mkdir(os.path.join(os.getcwd(), cfg.OUTPUT_DIRECTORY, flags.run))

    # Some checks on the flags
    if cfg.GENERATE_VOCABULARIES:
        assert cfg.INPUT_VOCAB_PATH and cfg.TARGET_VOCAB_PATH, "Please specify paths to vocabularies to save."

    train_data_path = os.path.join(flags.data_dir, "train.json")

    test_splits = [
        'test',
        'dev',
    ]
    val_data_paths = {split_name: os.path.join(flags.data_dir, split_name + '.json') for split_name in test_splits}

    if cfg.MODE == "train":
        if flags.is_baseline:
            logger.info("Running baseline + embedding...")
        else:
            logger.info("Running full model...")
        train(train_data_path=train_data_path, val_data_paths=val_data_paths, use_cuda=use_cuda, model_name=flags.run,
              resume_from_file=flags.load, is_baseline=flags.is_baseline)

    elif cfg.MODE == "predict":
        raise NotImplementedError()


    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(cfg.MODE))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LGCN models for GSCAN")
    parser.add_argument('--run', type=str, help='Define the run name')
    parser.add_argument('--txt', dest='redirect_output', action='store_true')
    parser.add_argument('--baseline', dest='is_baseline', action='store_true')
    parser.add_argument('--load', type=str, help='Path to model')
    parser.add_argument('--data_dir', type=str, help='Path to dataset')
    parser.add_argument('--seed', type=int, help='random seeds')
    parser.set_defaults(redirect_output=False, is_baseline=False)
    args = parser.parse_args()
    FORMAT = "%(asctime)-15s %(message)s"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.redirect_output:
        output_file = open(os.path.join('exp/', args.run + '.txt'), 'w')
        sys.stdout = output_file
        sys.stderr = sys.stdout
        logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                            datefmt="%Y-%m-%d %H:%M", filename=os.path.join('exp/', args.run + '.txt'))
    else:
        logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                            datefmt="%Y-%m-%d %H:%M")

    logger = logging.getLogger(__name__)
    use_cuda = True
    logger.info("Initialize logger")

    if use_cuda:
        logger.info("Using CUDA.")
        logger.info("Cuda version: {}".format(torch.version.cuda))
    
    logger.info("Training arguments:")
    logger.info(args)
    
    main(args, use_cuda)
