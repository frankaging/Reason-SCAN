from seq2seq.predict import predict
from seq2seq.helpers import sequence_accuracy

import torch.nn as nn
from typing import Iterator
from typing import Tuple
import numpy as np


def evaluate(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
             eos_idx: int, max_examples_to_evaluate=None) -> Tuple[float, float, float]:
    accuracies = []
    target_accuracies = []
    exact_match = 0
    for input_sequence, _, _, output_sequence, target_sequence, _, _, aux_acc_target in predict(
            data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
            sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate):
        accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
        if accuracy == 100:
            exact_match += 1
        accuracies.append(accuracy)
        target_accuracies.append(aux_acc_target)
    return (float(np.mean(np.array(accuracies))), (exact_match / len(accuracies)) * 100,
            float(np.mean(np.array(target_accuracies))))
