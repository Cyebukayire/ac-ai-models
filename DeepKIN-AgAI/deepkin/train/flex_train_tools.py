from __future__ import print_function, division

import gc
import os
# Ignore warnings
import warnings

from torch.optim import AdamW
from torch.utils.data import DataLoader

from deepkin.data.morpho_classification_data import create_cls_data, create_tag_data
from deepkin.data.morpho_mlm_data import create_morpho_mlm_dataset
from deepkin.data.morpho_qa_triple_data import create_morpho_qa_triple_dataset
from deepkin.models.flex_tts import FlexTTSTrainer
from deepkin.models.kinyabert import KinyaBERT, KinyaBERT_Classifier, KinyaColBERT
from deepkin.modules.param_init import init_bert_params
from deepkin.modules.tts_arguments import tts_base_args
from deepkin.modules.tts_data import TextAudioSpeakerLoader, TextAudioSpeakerCollate, DistributedBucketSampler

warnings.filterwarnings("ignore")

from typing import Tuple, Any

import torch
try:
    import apex
    from apex.optimizers import FusedLAMB, FusedAdam
except:
    from deepkin.optim.lamb import Lamb as FusedLAMB

from deepkin.modules.flex_modules import FlexConfig, MORPHO_NUM_LOSSES
from deepkin.optim.gradvac import GradVacc
from deepkin.optim.learning_rates import AnnealingLR, InverseSQRT_LRScheduler
from deepkin.utils.arguments import FlexArguments, morpho_large_args, morpho_base_args


def set_bert_accumulation_steps(args: FlexArguments, model_type):
    pass
def init_global_data_b4mp(args: FlexArguments):
    return None

def train_args_init_b4mp(args: FlexArguments) -> FlexArguments:
    model_type = args.model_variant.split(':')[0]
    if model_type == 'flex_tts':
        ret_args = FlexArguments().parse_args()
        return ret_args
    elif (args.model_variant == 'kinyabert:large'):
        ret_args: FlexArguments = morpho_large_args()
        ret_args.num_losses = MORPHO_NUM_LOSSES
        set_bert_accumulation_steps(ret_args, model_type)
        return ret_args
    elif ((args.model_variant == 'kinyabert_cls:large') or
          (args.model_variant == 'kinyabert_tag:large') or
          (args.model_variant == 'kinya_colbert:large')):
        ret_args: FlexArguments = morpho_large_args()
        ret_args.num_losses = 1
        return ret_args
    elif (args.model_variant == 'kinyabert:base'):
        ret_args: FlexArguments = morpho_base_args()
        ret_args.num_losses = MORPHO_NUM_LOSSES
        set_bert_accumulation_steps(ret_args, model_type)
        return ret_args
    elif ((args.model_variant == 'kinyabert_cls:base') or
          (args.model_variant == 'kinyabert_tag:base') or
          (args.model_variant == 'kinya_colbert:base')):
        ret_args: FlexArguments = morpho_base_args()
        ret_args.num_losses = 1
        return ret_args
    return args

def create_model(dist_rank, device, args: FlexArguments) -> Tuple[Any,Any]:
    model, model_cache = None, None
    model_type = args.model_variant.split(':')[0]
    if model_type == 'flex_tts':
        tts_args = tts_base_args(list_args=[])
        model = FlexTTSTrainer(tts_args, dist_rank, device)
    elif ('kinyabert_cls' in model_type) or ('kinyabert_tag' in model_type):
        bert = KinyaBERT.from_pretrained(device, args.pretrained_bert_model_file)
        model = KinyaBERT_Classifier(args, FlexConfig()).to(device)
        model.cls_head.out_proj.weight.data.normal_(mean=0.0, std=args.ft_proj_init_stdev)
        if model.cls_head.out_proj.bias is not None:
            model.cls_head.out_proj.bias.data.zero_()
        model.encoder.load_state_dict(bert.encoder.state_dict())
        for layer_idx in range(args.ft_reinit_layers):
            model.encoder.main_encoder.layers[-(layer_idx + 1)].apply(init_bert_params)
        del bert
        gc.collect()
    elif 'kinya_colbert' in model_type:
        bert = KinyaBERT.from_pretrained(device, args.pretrained_bert_model_file)
        # Model init
        model = KinyaColBERT(args).to(device)
        # Load BERT state
        model.bert_encoder.load_state_dict(bert.encoder.state_dict())
        model.bert_encoder.morpho_encoder.extend_with_special_tokens(2, init_range=0.02)
        del bert
        gc.collect()
    elif 'kinyabert' in model_type:
        model = KinyaBERT(args, FlexConfig()).to(device)
    return model, model_cache

def next_validation_dataset(dist_rank, global_data, device: torch.device, args: FlexArguments, data_cache, dataset, data_loader) -> Tuple[Any,Any,Any]:
    model_type = args.model_variant.split(':')[0]
    if 'kinya_colbert' in model_type:
        return create_morpho_qa_triple_dataset(device, args, data_cache, dataset, data_loader, True)
    elif 'kinyabert_cls' in model_type:
        if data_loader is None:
            dataset, data_loader = create_cls_data(args, validation=True)
    elif 'kinyabert_tag' in model_type:
        if data_loader is None:
            dataset, data_loader = create_tag_data(args, validation=True)
    elif 'kinyabert' in model_type:
        if (args.dev_parsed_corpus is not None) and os.path.exists(args.dev_parsed_corpus):
            return create_morpho_mlm_dataset(device, args, data_cache, dataset, data_loader, validation=True)
    return data_cache, dataset, data_loader

def next_train_dataset(dist_rank, global_data, device: torch.device, args: FlexArguments, train_data_cache, train_dataset, train_data_loader) -> Tuple[Any,Any,Any]:
    model_type = args.model_variant.split(':')[0]
    if model_type == 'flex_tts':
        if train_dataset is None:
            train_dataset = TextAudioSpeakerLoader(args.tts_data_dir, args.tts_train_data_file, tts_base_args(list_args=[]))
            train_sampler = DistributedBucketSampler(train_dataset, args.batch_size, [32, 300, 400, 500, 600, 700, 800, 900, 1000], num_replicas=args.gpus, rank=dist_rank, shuffle=True)
            collate_fn = TextAudioSpeakerCollate()
            train_data_loader = DataLoader(train_dataset, num_workers=args.dataloader_num_workers, shuffle=False, pin_memory=args.dataloader_pin_memory, persistent_workers=args.dataloader_persistent_workers, collate_fn=collate_fn, batch_sampler=train_sampler)
    elif 'kinya_colbert' in model_type:
        return create_morpho_qa_triple_dataset(device, args, train_data_cache, train_dataset, train_data_loader, False)
    elif 'kinyabert_cls' in model_type:
        if train_data_loader is None:
            train_dataset, train_data_loader = create_cls_data(args, validation=False)
    elif 'kinyabert_tag' in model_type:
        if train_data_loader is None:
            train_dataset, train_data_loader = create_tag_data(args, validation=False)
    elif 'kinyabert' in model_type:
        return create_morpho_mlm_dataset(device, args, train_data_cache, train_dataset, train_data_loader, validation=False)
    return train_data_cache, train_dataset, train_data_loader

def create_optimizer_and_lr_scheduler(dist_rank, args: FlexArguments, model, device: torch.device) -> Tuple[Any,Any,Any]:
    optimizer, scaler, lr_scheduler = None, None, None
    model_type = args.model_variant.split(':')[0]
    if ('kinyabert_cls' in model_type) or ('kinyabert_tag' in model_type) or ('kinya_colbert' in model_type):
        optimizer = AdamW(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.wd)
        lr_scheduler = AnnealingLR(optimizer, start_lr=args.peak_lr, warmup_iter=args.warmup_iter, num_iters=args.num_iters, decay_style=args.lr_decay_style, last_iter=-1)
    elif model_type == 'mamba_asr':
        optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.98), eps=1e-08)
        lr_scheduler = InverseSQRT_LRScheduler(optimizer, start_lr=args.peak_lr, warmup_iter=args.warmup_iter, num_iters=args.num_iters,last_iter=-1)
    elif (model_type == 'flexbert') or (model_type == 'flexgpt') or ('kinyabert' in model_type) or ('morphogpt' in model_type):
        if args.use_mtl_optimizer:
            opt = FusedLAMB(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.98), eps=1e-07, weight_decay=args.wd)
            optimizer = GradVacc(args.num_losses, opt, device, scaler=None, beta=1e-2, cpu_offload=False)
            lr_scheduler = AnnealingLR(opt, start_lr=args.peak_lr, warmup_iter=args.warmup_iter, num_iters=args.num_iters, decay_style=args.lr_decay_style, last_iter=-1)
        else:
            optimizer = FusedLAMB(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.98), eps=1e-07, weight_decay=args.wd)
            lr_scheduler = AnnealingLR(optimizer, start_lr=args.peak_lr, warmup_iter=args.warmup_iter, num_iters=args.num_iters, decay_style=args.lr_decay_style, last_iter=-1)
    return optimizer, scaler, lr_scheduler
