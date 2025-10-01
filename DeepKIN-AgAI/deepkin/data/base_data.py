import random
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence


def generate_square_subsequent_mask(sz: int, dtype=torch.float32, device=None) -> torch.Tensor:
    if device is None:
        device = torch.device('cpu')
    return torch.triu(torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),diagonal=1,)

def generate_input_key_padding_mask(input_lengths: List[int], ignore_last=False, device=None) -> torch.Tensor:
    if device is None:
        device = torch.device('cpu')
    input_masks = [torch.zeros(length, dtype=torch.bool, device=device) for length in input_lengths]
    if ignore_last:
        for i in range(len(input_masks)):
            if len(input_masks[i]) > 0:
                input_masks[i][-1] = True
    input_masks_padded = pad_sequence(input_masks, batch_first=True, padding_value=1)  # Shape: N x S
    return input_masks_padded


def get_random_start_line_in_document(doc_ends: List[int], corpus_lines: List[str]) -> int:
    # Go to the next starting line
    dcx = random.randint(0, len(doc_ends) - 1) % len(doc_ends)
    start_line = (doc_ends[dcx] + 1) % len(corpus_lines)
    if random.random() < 0.8:  # 80% of the time, start from anywhere within the corpus.
        end_line = (doc_ends[(dcx + 1) % len(doc_ends)] - 1)
        if end_line <= start_line:
            end_line = len(corpus_lines) - 1
        if end_line > start_line:
            if random.random() < 0.6:
                # Start from within 2/3 of the beginning of chosen the document
                max_line = start_line + int(2.0 * (end_line - start_line) / 3.0)
                start_line = random.randint(start_line, max_line)
            else:
                # Start from within 3/4 of the beginning of chosen the document
                max_line = start_line + int(3.0 * (end_line - start_line) / 4.0)
                start_line = random.randint(start_line, max_line) % len(corpus_lines)

    return start_line


def single_collate_fun(batches: List):
    return batches[0]
