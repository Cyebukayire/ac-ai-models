import random
from typing import List, Tuple, Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset

from deepkin.clib.libkinlp.kinlpy import ParsedFlexSentence
from deepkin.data.morpho_data import MorphoDataItem, prepare_morpho_data_from_sentence
from deepkin.modules.flex_modules import FlexConfig
from deepkin.utils.arguments import FlexArguments
from deepkin.utils.misc_functions import time_now, read_lines

QA_NUM_SPECIALS = 2
QUESTION_TYPE_ID = 0
DOCUMENT_TYPE_ID = 1

def qpn_collate_fn(qpn: List[Tuple]) -> Tuple:
    device = 'cpu'
    query = MorphoDataItem(device)
    query_lengths = []

    pos = MorphoDataItem(device)
    pos_lengths = []

    neg = MorphoDataItem(device)
    neg_lengths = []

    for q,p,n in qpn:
        query.extend(q)
        query_lengths.append(len(q))

        pos.extend(p)
        pos_lengths.append(len(p))

        neg.extend(n)
        neg_lengths.append(len(n))

    docs = pos.extend(neg)
    docs_lengths = pos_lengths + neg_lengths
    return (query.to_simple_inputs(query_lengths),
            docs.to_simple_inputs(docs_lengths))

class KinyaQATripleDataset(Dataset):
    def __init__(self, args: FlexArguments, device:str, validation:bool):
        self.args: FlexArguments = args
        self.device = device
        self.cfg = FlexConfig()
        print(time_now(), f'@{self.device}', f'Reading data inputs ...', flush=True)
        # [CLS] [Q] Text...
        self.queries: Dict[str, MorphoDataItem] = {
            id: prepare_morpho_data_from_sentence(self.cfg, self.device, ParsedFlexSentence(txt)).prepend_special_token(QUESTION_TYPE_ID).add_bos() for id, txt in
            zip(read_lines(self.args.qa_dev_query_id if validation else self.args.qa_train_query_id), read_lines(self.args.qa_dev_query_text if validation else self.args.qa_train_query_text))}
        print(time_now(), f'@{self.device}', f'Got {len(self.queries)} queries!', flush=True)

        # [CLS] [D] Text...
        self.passages: Dict[str, MorphoDataItem] = {
            id: prepare_morpho_data_from_sentence(self.cfg, self.device, ParsedFlexSentence(txt)).prepend_special_token(DOCUMENT_TYPE_ID).add_bos() for id, txt in
            zip(read_lines(self.args.qa_dev_passage_id if validation else self.args.qa_train_passage_id), read_lines(self.args.qa_dev_passage_text if validation else self.args.qa_train_passage_text))}
        print(time_now(), f'@{self.device}', f'Got {len(self.passages)} passages!', flush=True)

        self.triples: List[Tuple] = []
        raw_count = 0
        for l in read_lines(self.args.qa_dev_qpn_triples if validation else self.args.qa_train_qpn_triples):
            raw_count += 1
            qid, pid, nid = tuple(l.split('\t'))
            q = self.queries[qid]
            p = self.passages[pid]
            n = self.passages[nid]
            if (len(q) < 508) and (len(p) < 508) and (len(n) < 508):
                self.triples.append((qid, pid, nid))
        print(time_now(), f'@{self.device}', f'Got {len(self.triples)}/{raw_count} triples!', flush=True)
        random.shuffle(self.triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        qid, pid, nid = self.triples[idx]
        q = self.queries[qid]
        p = self.passages[pid]
        n = self.passages[nid]
        return (q,p,n)

def create_morpho_qa_triple_dataset(device: torch.device, args: FlexArguments, data_cache, dataset, data_loader, validation) -> Tuple[Any,Any,Any]:
    if dataset is None:
        dataset = KinyaQATripleDataset(args, f'{device}', validation)
    random.shuffle(dataset.triples)

    if data_loader is None:
        data_loader = DataLoader(dataset, batch_size=(1 if validation else args.batch_size), collate_fn=qpn_collate_fn, drop_last=True, shuffle=True,
                                 pin_memory=args.dataloader_pin_memory, persistent_workers=args.dataloader_persistent_workers,
                                 num_workers=args.dataloader_num_workers)
    return data_cache, dataset, data_loader
