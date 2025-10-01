from __future__ import annotations

import gc
import math
import os
import random
import sys
from typing import List, Set, Union, Tuple, Any, Dict

import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from progressbar import ProgressBar
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, IterableDataset

from deepkin.clib.libkinlp.kinlpy import ParsedFlexSentence, BOS_ID, EOS_ID, MSK_ID, NUM_SPECIAL_TOKENS
from deepkin.data.morpho_data import MorphoDataItem
from deepkin.modules.flex_modules import FlexConfig
from deepkin.utils.arguments import FlexArguments
from deepkin.utils.misc_functions import time_now


def prepare_morpho_mlm_data_from_sentence(device: str, sentence: ParsedFlexSentence, add_cls: bool, cfg:FlexConfig, add_eos: bool = True) -> MorphoDataItem:
    lm_morphs: List[int] = []
    pos_tags: List[int] = []
    stems: List[int] = []
    affixes: List[int] = []
    affix_lengths: List[int] = []

    predicted_tokens_idx: List[int] = []
    predicted_tokens_affixes_idx: List[int] = []
    predicted_stems: List[int] = []
    predicted_pos_tags: List[int] = []
    predicted_lm_morphs: List[int] = []
    predicted_affixes: List[int] = []
    predicted_affix_lengths: List[int] = []

    # Add <CLS> Token
    token_idx = 0
    if add_cls:
        lm_morphs.append(BOS_ID)
        pos_tags.append(BOS_ID)
        stems.append(BOS_ID)
        affix_lengths.append(0)
        token_idx += 1

    if (len(sentence.tokens) == 0) and add_eos: # New document
        lm_morphs.append(EOS_ID)
        pos_tags.append(EOS_ID)
        stems.append(EOS_ID)
        affix_lengths.append(0)
        token_idx += 1
    else:
        for token in sentence.tokens:
            # Whole word masking, decide on masking per whole word
            unchanged_morpho = True
            predict_morpho = False
            rped_morpho = random.random()
            if (rped_morpho <= 0.15): # 15% of tokens/words are predicted
                predict_morpho = True
                rval = rped_morpho / 0.15
                if(rval < 0.8): # 80% of predicted tokens are masked
                    unchanged_morpho = False
                    # Whole word masking
                    for _ in token.stems_ids:
                        lm_morphs.append(MSK_ID)
                        pos_tags.append(MSK_ID)
                        stems.append(MSK_ID)
                        v_afx = random.random()
                        if v_afx < 0.15: # Include Affixes for 15% of the time to enforce morphology learning
                            affixes.extend(token.affixes)
                            affix_lengths.append(len(token.affixes))
                        else:
                            affix_lengths.append(0)

                elif (rval < 0.9): # 10% are replaced by random tokens, 10% are left unchanged
                    unchanged_morpho = False
                    # Whole word masking
                    for _ in token.stems_ids:
                        lm_morphs.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_lm_morphs - 1))
                        pos_tags.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_pos_tags - 1))
                        stems.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_stems - 1))

                        v_afx = random.random()
                        if v_afx < 0.15: # Include Affixes for 15% of the time to enforce morphology learning
                            affixes.extend(token.affixes)
                            affix_lengths.append(len(token.affixes))
                        else:
                            num_afx = len((token.affixes))
                            affix_lengths.append(num_afx)
                            for i in range(num_afx):
                                affixes.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_affixes - 1))
            # When no re-sampling happens
            if(unchanged_morpho):
                # Whole word masking
                for sid in token.stems_ids:
                    lm_morphs.append(token.lm_morph_id)
                    pos_tags.append(token.pos_tag_id)
                    stems.append(sid)
                    affixes.extend(token.affixes)
                    affix_lengths.append(len(token.affixes))

            # For prediction tokens
            if(predict_morpho):
                for index,sid in enumerate(token.stems_ids):
                    predicted_stems.append(sid)
                    predicted_pos_tags.append(token.pos_tag_id)
                    predicted_lm_morphs.append(token.lm_morph_id)
                    predicted_tokens_idx.append(token_idx+index)
                    if (len(token.affixes) > 0):
                        predicted_affixes.extend(token.affixes)
                        predicted_tokens_affixes_idx.append(len(predicted_tokens_idx) - 1)
                        predicted_affix_lengths.append(len(token.affixes))
            token_idx += len(token.stems_ids)
    return MorphoDataItem(device,
                        stems = stems,
                        pos_tags = pos_tags,
                        lm_morphs = lm_morphs,
                        affixes = affixes,
                        affix_lengths = affix_lengths,
                        predicted_tokens_idx = predicted_tokens_idx,
                        predicted_tokens_affixes_idx = predicted_tokens_affixes_idx,
                        predicted_stems = predicted_stems,
                        predicted_pos_tags = predicted_pos_tags,
                        predicted_lm_morphs = predicted_lm_morphs,
                        predicted_affixes = predicted_affixes,
                        predicted_affix_lengths = predicted_affix_lengths)


def initial_mlm_sentence(contained_indices: Set[int], parsed_sentences: List[str], doc_ends: List[int]) -> int:
    while True:
        dcx = random.randint(0, len(doc_ends) - 1) % len(doc_ends)
        start_line = (doc_ends[dcx] + 1) % len(parsed_sentences)
        if random.random() < 0.8:  # 80% of the time, start from anywhere within the corpus.
            end_line = (doc_ends[(dcx + 1) % len(doc_ends)] - 1)
            if end_line <= start_line:
                end_line = len(parsed_sentences) - 1
            if end_line > start_line:
                # Note 23/10/2024: Made starting positions more at the top to have more long documents
                if random.random() < 0.666:
                    # Start from within 1/2 of the beginning of chosen the document
                    max_line = start_line + int(1.0 * (end_line - start_line) / 2.0)
                    start_line = random.randint(start_line, max_line)
                else:
                    # Start from within 2/3 of the beginning of chosen the document
                    max_line = start_line + int(2.0 * (end_line - start_line) / 3.0)
                    start_line = random.randint(start_line, max_line) % len(parsed_sentences)
        if start_line not in contained_indices:
            contained_indices.add(start_line)
            break
    return start_line

def add_sequence(num_tokens, device, ret_list, seq, bar, contained_indices, parsed_sentences, doc_ends):
    if len(seq) > 0:
        ret_list.append(seq)
        num_tokens += len(seq)
        if bar is not None:
            bar.update(len(ret_list))
            sys.stdout.flush()
    seq = MorphoDataItem(device)
    sentence_line_idx = initial_mlm_sentence(contained_indices, parsed_sentences, doc_ends)
    return num_tokens, seq, sentence_line_idx, 1

def extend_single_mlm_doc(num_docs,seq: MorphoDataItem,contained_indices,device,parsed_sentences,doc_ends,cfg,max_seq_len) -> Tuple[int,int]:
    init_len = len(seq)
    sentence_line_idx = initial_mlm_sentence(contained_indices, parsed_sentences, doc_ends)
    sentence = ParsedFlexSentence(parsed_sentences[sentence_line_idx])
    tmp = prepare_morpho_mlm_data_from_sentence(device, sentence, (len(seq) == 0), cfg)
    # Add EOS
    if ((len(seq) + len(tmp) + 1) <= max_seq_len):
        if (len(seq)>0):
            eos_sent = ParsedFlexSentence('')
            eos_tmp = prepare_morpho_mlm_data_from_sentence(device, eos_sent, (len(seq) == 0), cfg)
            seq.extend(eos_tmp)
            num_docs += 1
    else:
        return len(seq)-init_len,num_docs
    while (len(seq) + len(tmp)) <= max_seq_len:
        seq.extend(tmp)
        sentence_line_idx = (sentence_line_idx + 1)
        sentence = ParsedFlexSentence(parsed_sentences[sentence_line_idx])
        if (len(sentence.tokens) == 0):
            break
        tmp = prepare_morpho_mlm_data_from_sentence(device, sentence, (len(seq) == 0), cfg)
    return len(seq)-init_len,num_docs

def gather_itemized_morpho_mlm_data(device: str, parsed_sentences: List[str], doc_ends: List[int], max_seq_len: int, max_mlm_documents:int, max_batch_items: int, cfg:FlexConfig, bar: Union[ProgressBar,None] = None) -> List[MorphoDataItem]:
    contained_indices:Set[int] = set()
    ret_list:List[MorphoDataItem] = []
    seq: MorphoDataItem = MorphoDataItem(device)
    num_docs = 1
    trials = 0
    while (len(ret_list) < max_batch_items):
        extension,num_docs = extend_single_mlm_doc(num_docs,seq,contained_indices,device,parsed_sentences,doc_ends,cfg,max_seq_len)
        # print(f'len(seq): {len(seq)}, extension: {extension}, num_docs: {num_docs} @ {device}', flush=True)
        if (len(seq) > 0) and ((extension==0) or (num_docs >= max_mlm_documents)):
            ret_list.append(seq)
            if bar is not None:
                bar.update(len(ret_list))
                sys.stdout.flush()
            seq: MorphoDataItem = MorphoDataItem(device)
            num_docs = 1
            trials = 0
        elif (extension==0):
            trials += 1
        if trials > max_mlm_documents:
            break
    return ret_list

def batch_size(args:FlexArguments, x, align=1):
    b = (args.max_gpu_memory - args.min_gpu_memory) / ((args.mlm_batch_C1*x*x)+(args.mlm_batch_C2*x)+args.mlm_batch_C3)
    return min(int(math.ceil(b/align)*align),args.max_mlm_batch_size)

def get_bucket_keys(args: FlexArguments, batch_step, L=512):
    start = 0
    end = batch_step
    buckets = []
    while end <= 512:
        bs = batch_size(args,end)
        buckets.append((start,end,bs))
        start = end
        end += batch_step
    return buckets

def get_bucket_key(buckets, nseq):
    for (start,end,bs) in buckets:
        if (nseq > start) and (nseq <= end):
            return (start,end,bs)
    return None

@dataclass
class MLMSequenceBucket:
    key: Tuple[int,int,int]
    num_batches:int = 0
    num_sequences:int = 0
    parsed_sentences: List[str] = field(default_factory=lambda: [])
    doc_ends: List[int] = field(default_factory=lambda: [])

def get_bucketed_data(args: FlexArguments, corpus_file, file_size, mlm_batching_step, max_batch_items, device):
    # 1. Load max_batch_items sequences from file to memory
    mlm_sequence_buckets: Dict[Tuple[int, int, int], MLMSequenceBucket] = {k: MLMSequenceBucket(key=k) for k in get_bucket_keys(args, mlm_batching_step)}
    buckets_bs = [f'{e}: {bs}' for s, e, bs in mlm_sequence_buckets.keys()]
    print(f'{time_now()} {device}: Data Buckets-batch-size:>> ', '  '.join(buckets_bs), flush=True)
    print(f'{time_now()} {device}: Median Bucket-batch-size:>> ', buckets_bs[int(math.floor(len(buckets_bs)/2))], flush=True)
    total_seq = 0
    while total_seq < max_batch_items:
        try:
            corpus_file.seek(random.randint(0, file_size - args.max_dataset_chunk_size))
            lines = [l.rstrip('\n') for l in corpus_file.readlines(args.max_dataset_chunk_size)]
            ends = [i for i, l in enumerate(lines) if (len(l) == 0)]
            # Skip extreme ends because the file was loaded from random file positions
            lines = lines[(ends[0] + 1):(ends[-1] + 1)]
            start = 0
            nseq = 0
            docs = []
            for idx, line in enumerate(lines):
                if len(line) > 4:
                    s = sum([(int(t.split(' ')[0].split(',')[4]) + 1) for t in line.split('\t')])
                    nseq += s
                else:
                    end = idx + 1
                    if (nseq > 0) and (end > start):
                        docs.append((start, end, nseq))
                    nseq = 0
                    start = end
            for (start, end, nseq) in docs:
                key = get_bucket_key(mlm_sequence_buckets.keys(), nseq)
                if key is not None:
                    total_seq += 1
                    mlm_sequence_buckets[key].parsed_sentences.extend(lines[start:end].copy())
                    mlm_sequence_buckets[key].num_sequences += 1
        except:
            pass
    print(f'{time_now()} {device}: Loaded {total_seq:,} sequences/documents!', flush=True)
    bucket_keys = [key for key in mlm_sequence_buckets.keys()]

    prev_working_key = None
    for key in sorted(bucket_keys, key=lambda x: x[1], reverse=True):
        (start_size, end_size, batch_size_sequences) = key
        mlm_sequence_buckets[key].doc_ends = [i for i, l in enumerate(mlm_sequence_buckets[key].parsed_sentences) if
                                              (len(l) == 0)]
        mlm_sequence_buckets[key].num_batches = len(mlm_sequence_buckets[key].doc_ends) // batch_size_sequences
        # print(f' ==> Bucket: {(start_size, end_size, batch_size_sequences)} ==> {mlm_sequence_buckets[key].num_sequences} sequences, {mlm_sequence_buckets[key].num_batches} batches, {len(mlm_sequence_buckets[key].doc_ends)} end_docs', flush=True)
        if mlm_sequence_buckets[key].num_batches > 0:
            prev_working_key = key
        elif prev_working_key is not None:
            # Merge insufficient buckets into previous ones
            mlm_sequence_buckets[prev_working_key].parsed_sentences.extend(
                mlm_sequence_buckets[key].parsed_sentences.copy())
            mlm_sequence_buckets[prev_working_key].num_sequences += mlm_sequence_buckets[key].num_sequences
            mlm_sequence_buckets[prev_working_key].doc_ends = [i for i, l in enumerate(
                mlm_sequence_buckets[prev_working_key].parsed_sentences) if (len(l) == 0)]
            mlm_sequence_buckets[prev_working_key].num_batches = len(
                mlm_sequence_buckets[prev_working_key].doc_ends) // batch_size_sequences
            print(
                f' *REPLACE* Bucket: {(start_size, end_size, batch_size_sequences)} ==> {mlm_sequence_buckets[key].num_sequences} sequences, {mlm_sequence_buckets[key].num_batches} batches, {len(mlm_sequence_buckets[key].doc_ends)} end_docs',
                flush=True)

    # Only sample those buckets with enough batches
    bucket_keys = [key for key in mlm_sequence_buckets.keys() if mlm_sequence_buckets[key].num_batches > 0]
    # Find borderline batch sizes
    key_lengths = sorted([end_size for (start_size, end_size, batch_size_sequences) in bucket_keys])
    prefix_keys = {key_lengths[0], key_lengths[len(key_lengths) // 4], key_lengths[len(key_lengths) // 2],
                   key_lengths[3 * len(key_lengths) // 4], key_lengths[-1]}
    prefix = [i for i, (s, e, b) in enumerate(bucket_keys) if (e in prefix_keys)]
    extension = [i for i, _ in enumerate(bucket_keys)]
    random.shuffle(extension)
    prefix += extension
    key_ids = []
    total_batches = 0
    for id, key in enumerate(bucket_keys):
        total_batches += mlm_sequence_buckets[key].num_batches
        key_ids.extend([id for _ in range(mlm_sequence_buckets[key].num_batches)])
    random.shuffle(key_ids)
    random.shuffle(key_ids)
    # Sample from largest seq len bucket N times first
    key_ids = prefix + key_ids
    print(f'{time_now()} {device}: Sampling {len(prefix):,} + {total_batches:,} batches (i.e. {len(key_ids):,} ids) from {len(bucket_keys):,}/{len(mlm_sequence_buckets):,} buckets!',
        flush=True)
    return mlm_sequence_buckets, bucket_keys, key_ids

class KinyaMLMDataset(Dataset):

    def __init__(self, args: FlexArguments,
                 parsed_corpus_file_path: str,
                 cfg: FlexConfig,
                 device: str,
                 max_batch_items:int,
                 mlm_batching_step: int = 16,
                 validation=False):
        self.args = args
        self.cfg = cfg
        self.device = device
        self.max_batch_items = max_batch_items
        self.mlm_batching_step = mlm_batching_step
        self.parsed_corpus_file_path = parsed_corpus_file_path
        self.validation = validation
        if validation:
            self.itemized_data: List[List[MorphoDataItem]] = []
            with open(self.parsed_corpus_file_path, 'r', encoding='utf-8') as f:
                parsed_sentences = [l.rstrip('\n') for l in f]
                doc_ends = [i for i, l in enumerate(parsed_sentences) if (len(l) == 0)]
                batches = int(math.ceil(len(doc_ends)/args.batch_size))
                print(f'{time_now()} {self.device}: Loading {batches} validation data batches ...', flush=True)
                with ProgressBar(min_value=0, max_value=batches, redirect_stderr=True,redirect_stdout=True) as pbar:
                    for itr in range(batches):
                        pbar.update(itr)
                        self.itemized_data.append(gather_itemized_morpho_mlm_data(self.device, parsed_sentences, doc_ends, self.args.dataset_max_seq_len, self.args.max_mlm_documents, args.batch_size, self.cfg))
                print(f'{time_now()} {self.device}: Loaded {len(self.itemized_data):,} validation batches!', flush=True)
        else:
            self.itemized_data: List[List[MorphoDataItem]] = []

    def load(self):
        if len(self.itemized_data) == 0:
            print(f'{time_now()} {self.device}: Reading data from {self.parsed_corpus_file_path} ...')
            corpus_file = open(self.parsed_corpus_file_path, 'r', encoding='utf-8')
            file_size = os.path.getsize(self.parsed_corpus_file_path)
            mlm_sequence_buckets, bucket_keys, key_ids = get_bucketed_data(self.args, corpus_file, file_size, self.mlm_batching_step, self.max_batch_items, self.device)
            corpus_file.close()
            gc.collect()
            for id in key_ids:
                key = bucket_keys[id]
                (start_size, end_size, batch_size_sequences) = key
                self.itemized_data.append(gather_itemized_morpho_mlm_data(self.device, mlm_sequence_buckets[key].parsed_sentences, mlm_sequence_buckets[key].doc_ends, end_size, self.args.max_mlm_documents, batch_size_sequences, self.cfg))
            del mlm_sequence_buckets
            del bucket_keys
            gc.collect()

    def __len__(self):
        self.load()
        return len(self.itemized_data)

    def __getitem__(self, idx):
        self.load()
        return self.itemized_data[idx]

class KinyaMLMIterableDataset(IterableDataset):

    def __init__(self, args: FlexArguments,
                 parsed_corpus_file_path: str,
                 cfg: FlexConfig,
                 device: str,
                 max_batch_items:int,
                 mlm_batching_step: int = 16):
        self.args = args
        self.cfg = cfg
        self.device = device
        self.max_batch_items = max_batch_items
        self.mlm_batching_step = mlm_batching_step
        self.parsed_corpus_file_path = parsed_corpus_file_path

    def generate(self):
        while True:
            corpus_file = open(self.parsed_corpus_file_path)
            file_size = os.path.getsize(self.parsed_corpus_file_path)
            print(f'{time_now()} {self.device}: Reading data from {self.parsed_corpus_file_path} ...')
            mlm_sequence_buckets, bucket_keys, key_ids = get_bucketed_data(self.args, corpus_file, file_size, self.mlm_batching_step, self.max_batch_items,self.device)
            corpus_file.close()
            gc.collect()
            for id in key_ids:
                key = bucket_keys[id]
                (start_size, end_size, batch_size_sequences) = key
                batch_items = gather_itemized_morpho_mlm_data(self.device, mlm_sequence_buckets[key].parsed_sentences, mlm_sequence_buckets[key].doc_ends, end_size, self.args.max_mlm_documents, batch_size_sequences, self.cfg)
                if len(batch_items) > 0:
                    yield morpho_mlm_data_collate_wrapper([batch_items])
            del mlm_sequence_buckets
            del bucket_keys
            gc.collect()


    def __iter__(self):
        return iter(self.generate())

def fetch_new_mlm_doc_seq(device, contained_indices:Set[int], parsed_corpus_lines, parsed_corpus_doc_ends, dataset_max_seq_len, cfg):
    doc_seq = MorphoDataItem(device)
    line_idx = initial_mlm_sentence(contained_indices, parsed_corpus_lines, parsed_corpus_doc_ends)
    sent = ParsedFlexSentence(parsed_corpus_lines[line_idx])
    while len(sent.tokens) > 0:
        seq = prepare_morpho_mlm_data_from_sentence(device, sent, False, cfg)
        if (len(doc_seq) + len(seq) + 2) < dataset_max_seq_len:
            doc_seq.extend(seq)
        else:
            break
        line_idx += 1
        sent = ParsedFlexSentence(parsed_corpus_lines[line_idx])
    return doc_seq


class SimpleMorphoMLMIterableDataset(IterableDataset):

    def __init__(self, args: FlexArguments,
                 parsed_corpus_file_path: str,
                 device: str):
        self.args = args
        self.cfg = FlexConfig()
        self.device = device
        self.batch_size = batch_size
        self.parsed_corpus_file_path = parsed_corpus_file_path

    def generate(self):
        print(f'{time_now()} {self.device}: Reading data from {self.parsed_corpus_file_path} ...', flush=True)
        parsed_corpus_lines = []
        parsed_corpus_doc_ends = []
        idx = 0
        with open(self.parsed_corpus_file_path, 'r', encoding='utf-8') as f:
            for ln in f:
                line = ln.rstrip('\n')
                parsed_corpus_lines.append(line)
                if len(line) == 0:
                    parsed_corpus_doc_ends.append(idx)
                idx+= 1
        print(f'{time_now()} {self.device}: Got {len(parsed_corpus_lines)} lines / {len(parsed_corpus_doc_ends)} docs from corpus ...', flush=True)
        while True:
            batch_items:List[MorphoDataItem] = []
            contained_indices:Set[int] = set()
            # BOS/CLS
            mlm_seq = prepare_morpho_mlm_data_from_sentence(self.device, ParsedFlexSentence(''), True, self.cfg, add_eos=False)
            mlm_seq.extend(fetch_new_mlm_doc_seq(self.device, contained_indices, parsed_corpus_lines, parsed_corpus_doc_ends, self.args.dataset_max_seq_len, self.cfg))
            while len(batch_items) < self.args.batch_size:
                doc_seq = fetch_new_mlm_doc_seq(self.device, contained_indices, parsed_corpus_lines, parsed_corpus_doc_ends, self.args.dataset_max_seq_len, self.cfg)
                if (len(mlm_seq) + len(doc_seq) + 1) <= self.args.dataset_max_seq_len:
                    # EOS
                    mlm_seq.extend(prepare_morpho_mlm_data_from_sentence(self.device, ParsedFlexSentence(''), False, self.cfg, add_eos=True))
                    mlm_seq.extend(doc_seq)
                else:
                    batch_items.append(mlm_seq)
                    # BOS/CLS
                    mlm_seq = prepare_morpho_mlm_data_from_sentence(self.device, ParsedFlexSentence(''), True, self.cfg, add_eos=False)
                    mlm_seq.extend(doc_seq)
            yield morpho_mlm_data_collate_wrapper([batch_items])


    def __iter__(self):
        return iter(self.generate())


def morpho_mlm_data_collate_wrapper(batched: List):
    if len(batched) == 1:
        batch_items = batched[0]
        if type(batch_items) is tuple:
            return batch_items
    else:
        batch_items = batched
    batch: MorphoDataItem = MorphoDataItem(batch_items[0].device)
    batch_input_sequence_lengths: List[int] = []
    batch_predicted_tokens_idx = []
    batch_predicted_tokens_affixes_idx = []
    for (bidx, seq) in enumerate(batch_items):
        batch.extend(seq)
        batch_predicted_tokens_affixes_idx.extend([(len(batch_predicted_tokens_idx) + t) for t in seq.predicted_tokens_affixes_idx])
        batch_predicted_tokens_idx.extend([(t, len(batch_input_sequence_lengths)) for t in seq.predicted_tokens_idx])
        batch_input_sequence_lengths.append(len(seq))
    batch.predicted_tokens_idx = [s * max(batch_input_sequence_lengths) + t for t, s in batch_predicted_tokens_idx]
    batch.predicted_tokens_affixes_idx = batch_predicted_tokens_affixes_idx
    return batch.to_mlm_training_tuple(batch_input_sequence_lengths)

def create_morpho_mlm_dataset(device: torch.device, args: FlexArguments, data_cache, dataset, data_loader, validation: bool=False) -> Tuple[Any,Any,Any]:
    if validation:
        if dataset is not None:
            del dataset
        if data_loader is not None:
            del data_loader
        max_batch_items = args.number_of_load_batches // dist.get_world_size()
        dataset = KinyaMLMDataset(args, args.dev_parsed_corpus, FlexConfig(), f'{device}', max_batch_items, validation=True)
        data_loader = DataLoader(dataset, batch_size=1, collate_fn=morpho_mlm_data_collate_wrapper,
                                 drop_last=False, shuffle=False, pin_memory=args.dataloader_pin_memory,
                                 num_workers=1, persistent_workers=False)
    elif args.use_iterable_dataset and (not validation):
        if dataset is None:
            dataset = SimpleMorphoMLMIterableDataset(args, args.dev_parsed_corpus if validation else args.train_parsed_corpus, f'{device}')
        if data_loader is None:
            data_loader = DataLoader(dataset, batch_size=1, collate_fn=morpho_mlm_data_collate_wrapper,
                                     drop_last=False, shuffle=False, pin_memory=args.dataloader_pin_memory,
                                     num_workers=args.dataloader_num_workers, persistent_workers=args.dataloader_persistent_workers)
    else:
        if dataset is not None:
            del dataset
        if data_loader is not None:
            del data_loader
        max_batch_items = args.number_of_load_batches // dist.get_world_size()
        dataset = KinyaMLMDataset(args, args.dev_parsed_corpus if validation else args.train_parsed_corpus, FlexConfig(), f'{device}', max_batch_items, mlm_batching_step=(128 if validation else 16))
        data_loader = DataLoader(dataset, batch_size=1, collate_fn=morpho_mlm_data_collate_wrapper,
                                 drop_last=False, shuffle=False, pin_memory=args.dataloader_pin_memory,
                                 num_workers=args.dataloader_num_workers, persistent_workers=False)
    return data_cache, dataset, data_loader
