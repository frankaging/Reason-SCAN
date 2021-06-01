import argparse
import os

from GroundedScan.dataset import GroundedScan
from dataloader import dataloader
from model.config import cfg
from model.model import GSCAN_model
from model.utils import *

model_file = "output/full3_8/model_best.pth.tar"
baseline_file = "output/baseline3/model_best.pth.tar"


def exact_match_indicator(data_iterator, model, max_decoding_steps, pad_idx, sos_idx, eos_idx,
                          max_examples_to_evaluate=None):
    exact_match_terms = []
    with torch.no_grad():
        for batch, output_sequence, target_sequence, _, _, aux_acc_target in predict(
                data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
                sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate):
            seq_eq = torch.eq(output_sequence, target_sequence)
            mask = torch.eq(target_sequence, pad_idx) + torch.eq(target_sequence, sos_idx)
            seq_eq.masked_fill_(mask, 0)
            total = (~mask).sum(-1).float()
            accuracy = seq_eq.sum(-1) / total
            exact_match_terms.append(accuracy.eq(1.))
    return torch.cat(exact_match_terms, dim=0)


def predict_and_write(data_iterator, model, example_indicator, max_decoding_steps, input_vocab, target_vocab,
                      max_examples_to_output=None, original_dataset=None, out='predict.json', split_name=None):
    # example_indicator: [datasetsize,] bool tensor indicating which example should be saved
    indicator_idx = 0
    pad_idx, sos_idx, eos_idx = target_vocab.stoi['<pad>'], target_vocab.stoi['<sos>'], \
                                target_vocab.stoi['<eos>']
    predict_output = []
    accu_ex_sum = 0
    with torch.no_grad():
        for batch, output_sequence, target_sequence, _, attention_weights_situations, \
            aux_acc_target in predict(data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps,
                                      pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx,
                                      max_examples_to_evaluate=None):
            # output_sequence: bs x max_decoding_steps
            batchsize = batch.situation.shape[0]
            batch_indicator = example_indicator[indicator_idx:indicator_idx + batchsize]
            situation_idx_offsets = batch_indicator.nonzero()
            if batch_indicator.sum() == 0:
                continue
            select_and_convert = lambda x: x[batch_indicator].cpu().numpy().astype(int)

            selected_input = select_and_convert(batch.input[0])
            input_tokens = translate_sequence(selected_input, input_vocab.itos, eos_idx=input_vocab.stoi['<eos>'])

            selected_output = select_and_convert(output_sequence[:, 1:])
            output_tokens = translate_sequence(selected_output, target_vocab.itos, eos_idx=target_vocab.stoi['<eos>'])

            selected_target = select_and_convert(target_sequence[:, 1:])
            target_tokens = translate_sequence(selected_target, target_vocab.itos, eos_idx=target_vocab.stoi['<eos>'])

            selected_attn_sit = np.array(attention_weights_situations)[:, batch_indicator.cpu().numpy(), :]

            for i in range(len(selected_input)):
                situation_idx = indicator_idx + situation_idx_offsets[i]
                predict_output.append({"input": input_tokens[i], "prediction": output_tokens[i],
                                       "target": target_tokens[i], "situation":
                                           original_dataset._data_pairs[split_name][situation_idx]['situation'],
                                       "attention_weights_situation": selected_attn_sit[:, i:i + 1, :].tolist()})
            if max_examples_to_output is not None:
                accu_ex_sum += batch_indicator.sum()
                if accu_ex_sum > max_examples_to_output:
                    break
            indicator_idx += batchsize

    with open(out, 'w') as f:
        json.dump(predict_output, f)


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
        # torch.eq(target_sequence, eos_idx)
        seq_eq.masked_fill_(mask, 0)
        total = (~mask).sum(-1).float()
        accuracy = seq_eq.sum(-1) / total
        total_terms += total.sum().data.item()
        correct_terms += seq_eq.sum().data.item()
        exact_match += accuracy.eq(1.).sum().data.item()
        target_accuracies.append(aux_acc_target)
    return (float(correct_terms) / total_terms) * 100, (exact_match / num_examples) * 100, \
           float(np.mean(np.array(target_accuracies))) * 100


def train(train_data_path: str, val_data_paths: dict, use_cuda: bool):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')

    logger.info("Loading Training set...")
    train_iter, train_input_vocab, train_target_vocab = dataloader(train_data_path,
                                                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                                                   use_cuda=use_cuda)
    val_iters = {}
    for split_name, path in val_data_paths.items():
        val_iters[split_name], _, _ = dataloader(path, batch_size=cfg.VAL_BATCH_SIZE, use_cuda=use_cuda,
                                                 input_vocab=train_input_vocab, target_vocab=train_target_vocab,
                                                 random_shuffle=False)

    pad_idx, sos_idx, eos_idx = train_target_vocab.stoi['<pad>'], train_target_vocab.stoi['<sos>'], \
                                train_target_vocab.stoi['<eos>']

    train_input_vocab_size, train_target_vocab_size = len(train_input_vocab.itos), len(train_target_vocab.itos)

    logger.info("Loading Dev. set...")

    val_input_vocab_size, val_target_vocab_size = train_input_vocab_size, train_target_vocab_size
    logger.info("Done Loading Dev. set.")

    model = GSCAN_model(pad_idx, eos_idx, train_input_vocab_size, train_target_vocab_size, is_baseline=False)
    model = model.cuda() if use_cuda else model
    assert os.path.isfile(model_file), "No model checkpoint found at {}".format(model_file)
    logger.info("Loading model checkpoint from file at '{}'".format(model_file))
    _ = model.load_model(model_file)

    baseline = GSCAN_model(pad_idx, eos_idx, train_input_vocab_size, train_target_vocab_size, is_baseline=True)
    baseline = baseline.cuda() if use_cuda else baseline
    assert os.path.isfile(baseline_file), "No baseline checkpoint found at {}".format(baseline_file)
    logger.info("Loading model checkpoint from file at '{}'".format(baseline_file))
    _ = baseline.load_model(baseline_file)

    original_dataset = GroundedScan.load_dataset_from_file(
        "/root/multimodal_seq2seq_gSCAN/data/compositional_splits/dataset.txt",
        save_directory="stat/", k=10)
    # original_dataset = dill.load(open('original_dataset.p', 'rb'))

    with torch.no_grad():
        model.eval()
        logger.info("Evaluating..")
        print(val_iters)
        for split_name, val_iter in val_iters.items():
            model_exact_match = exact_match_indicator(
                val_iter, model=model,
                max_decoding_steps=30, pad_idx=pad_idx,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_examples_to_evaluate=None)
            baseline_exact_match = exact_match_indicator(
                val_iter, model=baseline,
                max_decoding_steps=30, pad_idx=pad_idx,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_examples_to_evaluate=None)
            model_diff = torch.bitwise_xor(model_exact_match, baseline_exact_match)
            model_better_exs = torch.bitwise_and(model_diff, model_exact_match)
            # predict_and_write(val_iter, model, model_exact_match, 30, input_vocab=train_input_vocab,
            #                   target_vocab=train_target_vocab, out='model_good/'+split_name + '_predict.json',
            #                   split_name=split_name, original_dataset=original_dataset, max_examples_to_output=20)
            predict_and_write(val_iter, baseline, model_better_exs, 30, input_vocab=train_input_vocab,
                              target_vocab=train_target_vocab, out='model_good_bl_fail/' + split_name + '_predict.json',
                              split_name=split_name, original_dataset=original_dataset, max_examples_to_output=200)
            predict_and_write(val_iter, model, ~model_exact_match, 30, input_vocab=train_input_vocab,
                              target_vocab=train_target_vocab, out='model_bad/' + split_name + '_predict.json',
                              split_name=split_name, original_dataset=original_dataset, max_examples_to_output=200)


def main(flags, use_cuda):
    if not os.path.exists(cfg.OUTPUT_DIRECTORY):
        os.mkdir(os.path.join(os.getcwd(), cfg.OUTPUT_DIRECTORY))

    train_data_path = os.path.join(cfg.DATA_DIRECTORY, "train.json")

    test_splits = [
        'adverb_2',
    ]
    val_data_paths = {split_name: os.path.join(cfg.DATA_DIRECTORY, split_name + '.json') for split_name in test_splits}

    if cfg.MODE == "train":
        train(train_data_path=train_data_path, val_data_paths=val_data_paths, use_cuda=use_cuda)

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
    args = parser.parse_args()

    main(args, use_cuda)
