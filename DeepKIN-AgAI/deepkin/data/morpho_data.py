from __future__ import annotations

import gc
from dataclasses import dataclass, field
from itertools import accumulate
from typing import List, Tuple, Union

import torch
from deepkin.clib.libkinlp.kinlpy import ParsedFlexSentence, BOS_ID, EOS_ID, ParsedSentenceMulti
from deepkin.data.base_data import generate_square_subsequent_mask, generate_input_key_padding_mask
from deepkin.modules.flex_modules import FlexConfig, MORPHO_AFFIX_CONTEXT_ITEMS
from torch.nn.utils.rnn import pad_sequence


@dataclass
class SharedMorphoDataItem:
    stems: torch.Tensor
    lm_morphs: torch.Tensor
    pos_tags: torch.Tensor
    affixes: torch.Tensor
    affix_lengths: torch.Tensor

    def reset(self):
        del self.stems
        del self.lm_morphs
        del self.pos_tags
        del self.affixes
        del self.affix_lengths

    def to_kin2en_train_tuple(self, device:str,
                             shared_english_tokens: torch.Tensor,
                             shared_metadata: torch.Tensor,
                             meta_start: int,
                             meta_end: int) -> Tuple:
        stems  = []
        pos_tags = []
        lm_morphs = []
        affixes = []
        affix_lengths: List[int] = []

        input_sequence_lengths = []
        english_sequence_lengths = []

        english_input_ids = []

        for list_item in shared_metadata[meta_start:meta_end].tolist():
            (avg_len, start_en, end_en, start_rw, end_rw, affixes_start, affixes_end) = tuple(list_item)
            stems.append(self.stems[start_rw:end_rw])
            pos_tags.append(self.pos_tags[start_rw:end_rw])
            lm_morphs.append(self.lm_morphs[start_rw:end_rw])
            affixes.append(self.affixes[affixes_start:affixes_end])
            affix_lengths.extend(self.affix_lengths[start_rw:end_rw].tolist())
            english_input_ids.append(shared_english_tokens[start_en:end_en])
            input_sequence_lengths.append(end_rw - start_rw)
            english_sequence_lengths.append(end_en - start_en)

        # device = torch.device(device)
        device = torch.device('cpu')
        stems = torch.cat(stems).to(device)
        pos_tags = torch.cat(pos_tags).to(device)
        lm_morphs = torch.cat(lm_morphs).to(device)
        affixes = torch.cat(affixes)#.to(device)

        english_input_ids = torch.cat(english_input_ids).to(device)

        predicted_english_tokens = english_input_ids.split(english_sequence_lengths)
        predicted_english_tokens = [tns[1:length] for length, tns in zip(english_sequence_lengths, predicted_english_tokens)]
        predicted_english_tokens = torch.cat(predicted_english_tokens, dim=0).to(device, dtype=torch.int64)

        affixes_padded = affixes.split(affix_lengths)
        affixes_padded = pad_sequence(affixes_padded, batch_first=False)
        affixes_padded = affixes_padded.to(device, dtype=torch.long)

        morpho_masks_padded = generate_input_key_padding_mask([(x + MORPHO_AFFIX_CONTEXT_ITEMS) for x in affix_lengths], device=device)
        main_masks_padded = generate_input_key_padding_mask(input_sequence_lengths, device=device)
        seq_len = max(english_sequence_lengths)
        src_key_padding_mask = generate_input_key_padding_mask(input_sequence_lengths, device=device, ignore_last=False)
        tgt_key_padding_mask = generate_input_key_padding_mask(english_sequence_lengths, device=device, ignore_last=True)
        decoder_mask = generate_square_subsequent_mask(seq_len, device=device)

        return (english_input_ids, english_sequence_lengths, predicted_english_tokens,
                 stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded, input_sequence_lengths, # number of words per each sequence in the batch
                 main_masks_padded,
                 src_key_padding_mask, tgt_key_padding_mask, decoder_mask)

    def to_en2kin_train_tuple(self, device:str,
                             shared_english_tokens: torch.Tensor,
                             shared_metadata: torch.Tensor,
                             meta_start: int,
                             meta_end: int) -> Tuple:
        stems  = []
        lm_morphs = []
        pos_tags = []
        affixes = []
        affix_lengths: List[int] = []

        input_sequence_lengths = []
        english_sequence_lengths = []

        english_input_ids = []

        for list_item in shared_metadata[meta_start:meta_end].tolist():
            (avg_len, start_en, end_en, start_rw, end_rw, affixes_start, affixes_end) = tuple(list_item)
            stems.append(self.stems[start_rw:end_rw])
            pos_tags.append(self.pos_tags[start_rw:end_rw])
            lm_morphs.append(self.lm_morphs[start_rw:end_rw])
            affixes.append(self.affixes[affixes_start:affixes_end])
            affix_lengths.extend(self.affix_lengths[start_rw:end_rw].tolist())

            english_input_ids.append(shared_english_tokens[start_en:end_en])

            input_sequence_lengths.append(end_rw - start_rw)
            english_sequence_lengths.append(end_en - start_en)

        # device = torch.device(device)
        device = torch.device('cpu')
        lm_morphs = torch.cat(lm_morphs) # .to(device)
        pos_tags = torch.cat(pos_tags) # .to(device)
        stems = torch.cat(stems) # .to(device)

        affixes = torch.cat(affixes)#.to(device)

        english_input_ids = torch.cat(english_input_ids).to(device)

        cfg: FlexConfig = FlexConfig()
        pred_affixes_list = [affixes[x - y: x] for x, y in zip(accumulate(affix_lengths), affix_lengths)]
        afx_prob = torch.zeros(len(pred_affixes_list), cfg.tot_num_affixes)
        for i, lst in enumerate(pred_affixes_list):
            if (len(lst) > 0):
                afx_prob[i, lst] = 1.0
        affixes_prob = afx_prob # .to(device, dtype=torch.float)

        affixes_padded = affixes.split(affix_lengths)
        affixes_padded = pad_sequence(affixes_padded, batch_first=False)
        affixes_padded = affixes_padded.to(device, dtype=torch.long)

        morpho_masks_padded = generate_input_key_padding_mask([(x + MORPHO_AFFIX_CONTEXT_ITEMS) for x in affix_lengths], device=device)
        input_masks_padded = generate_input_key_padding_mask(input_sequence_lengths, device=device, ignore_last=True)
        seq_len = max(input_sequence_lengths)
        src_key_padding_mask = generate_input_key_padding_mask(english_sequence_lengths, device=device, ignore_last=False)
        tgt_key_padding_mask = generate_input_key_padding_mask(input_sequence_lengths, device=device, ignore_last=True)
        decoder_mask = generate_square_subsequent_mask(seq_len, device=device)

        predicted_stems = stems.split(input_sequence_lengths)
        predicted_stems = [tns[1:length] for length, tns in zip(input_sequence_lengths, predicted_stems)]
        predicted_stems = torch.cat(predicted_stems, dim=0).to(device, dtype=torch.int64)

        predicted_pos_tags = pos_tags.split(input_sequence_lengths)
        predicted_pos_tags = [tns[1:length] for length, tns in zip(input_sequence_lengths, predicted_pos_tags)]
        predicted_pos_tags = torch.cat(predicted_pos_tags, dim=0).to(device, dtype=torch.int64)

        predicted_lm_morphs = lm_morphs.split(input_sequence_lengths)
        predicted_lm_morphs = [tns[1:length] for length, tns in zip(input_sequence_lengths, predicted_lm_morphs)]
        predicted_lm_morphs = torch.cat(predicted_lm_morphs, dim=0).to(device, dtype=torch.int64)

        predicted_affixes_prob = affixes_prob.split(input_sequence_lengths)
        predicted_affixes_prob = [tns[1:length, :] for length, tns in zip(input_sequence_lengths, predicted_affixes_prob)]
        predicted_affixes_prob = torch.cat(predicted_affixes_prob, dim=0).to(device, dtype=torch.float32)

        pos_tags = pos_tags.to(device, dtype=torch.int64)
        lm_morphs = lm_morphs.to(device, dtype=torch.int64)

        return (english_input_ids, english_sequence_lengths,
                 stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                 input_sequence_lengths,  # number of words per each sequence in the batch
                 input_masks_padded,
                 predicted_stems, predicted_pos_tags, predicted_lm_morphs, predicted_affixes_prob,
                 src_key_padding_mask, tgt_key_padding_mask, decoder_mask)

@dataclass
class MorphoDataItem:
    device: str
    stems: List[int] = field(default_factory=lambda: [])
    pos_tags: List[int] = field(default_factory=lambda: [])
    lm_morphs: List[int] = field(default_factory=lambda: [])
    affixes: List[int] = field(default_factory=lambda: [])
    affix_lengths: List[int] = field(default_factory=lambda: [])
    predicted_tokens_idx: List[int] = field(default_factory=lambda: [])
    predicted_tokens_affixes_idx: List[int] = field(default_factory=lambda: [])
    predicted_stems: List[int] = field(default_factory=lambda: [])
    predicted_pos_tags: List[int] = field(default_factory=lambda: [])
    predicted_lm_morphs: List[int] = field(default_factory=lambda: [])
    predicted_affixes: List[int] = field(default_factory=lambda: [])
    predicted_affix_lengths: List[int] = field(default_factory=lambda: [])
    surface_forms: List[str] = field(default_factory=lambda: [])
    stem_probs: List[float] = field(default_factory=lambda: [])
    stem_mixer_counts: List[int] = field(default_factory=lambda: [])

    def __len__(self):
        return len(self.stems)

    def append(self, stem: int, pos_tag: int, lm_morph: int, affixes: List[int], surface_form: str) -> MorphoDataItem:
        self.stems.append(stem)
        self.lm_morphs.append(lm_morph)
        self.pos_tags.append(pos_tag)
        self.affixes.extend(affixes)
        self.affix_lengths.append(len(affixes))
        self.surface_forms.append(surface_form)
        return self

    def append_mixer_probs(self, probs: List[float], mixer_counts: List[int]):
        self.stem_probs.extend(probs)
        self.stem_mixer_counts.extend(mixer_counts)

    def extend(self, other: MorphoDataItem) -> MorphoDataItem:
        self.predicted_tokens_affixes_idx.extend([i+len(self.predicted_tokens_idx) for i in other.predicted_tokens_affixes_idx])
        # This need to be after pred_affix_idx extension
        self.predicted_tokens_idx.extend([i+len(self.affix_lengths) for i in other.predicted_tokens_idx])

        self.stems.extend(other.stems)
        self.lm_morphs.extend(other.lm_morphs)
        self.pos_tags.extend(other.pos_tags)
        self.affixes.extend(other.affixes)
        self.affix_lengths.extend(other.affix_lengths)

        self.predicted_stems.extend(other.predicted_stems)
        self.predicted_pos_tags.extend(other.predicted_pos_tags)
        self.predicted_lm_morphs.extend(other.predicted_lm_morphs)
        self.predicted_affixes.extend(other.predicted_affixes)
        self.predicted_affix_lengths.extend(other.predicted_affix_lengths)
        self.surface_forms.extend(other.surface_forms)
        return self

    def to_shared_data_item(self, share_memory=True) -> SharedMorphoDataItem:
        global_stems = torch.tensor(self.stems, dtype=torch.int32)
        if share_memory:
            global_stems.share_memory_()
        self.stems = []
        gc.collect()

        global_lm_morphs = torch.tensor(self.lm_morphs, dtype=torch.int32)
        if share_memory:
            global_lm_morphs.share_memory_()
        self.lm_morphs = []
        gc.collect()

        global_pos_tags = torch.tensor(self.pos_tags, dtype=torch.int32)
        if share_memory:
            global_pos_tags.share_memory_()
        self.pos_tags = []
        gc.collect()

        global_affixes = torch.tensor(self.affixes, dtype=torch.int32)
        if share_memory:
            global_affixes.share_memory_()
        self.affixes = []
        gc.collect()

        global_affix_lengths = torch.tensor(self.affix_lengths, dtype=torch.int32)
        if share_memory:
            global_affix_lengths.share_memory_()
        self.affix_lengths = []
        gc.collect()

        return SharedMorphoDataItem(stems = global_stems,
                                  lm_morphs = global_lm_morphs,
                                  pos_tags = global_pos_tags,
                                  affixes = global_affixes,
                                  affix_lengths = global_affix_lengths)

    def to_mlm_training_tuple(self, input_sequence_lengths: List[int]) -> Tuple:
        assert len(self.predicted_stems) == len(set(self.predicted_tokens_idx)), f"Mismatch between predicted_stems and predicted_tokens_idx: {len(self.predicted_stems)} ~ {len(set(self.predicted_tokens_idx))} <== {self.predicted_tokens_idx}"
        assert len(self.predicted_pos_tags) == len(set(self.predicted_tokens_idx)), f"Mismatch between predicted_pos_tags and predicted_tokens_idx: {len(self.predicted_pos_tags)} ~ {len(set(self.predicted_tokens_idx))} <== {self.predicted_tokens_idx}"

        # device = torch.device(self.device)
        device = torch.device('cpu')
        stems = torch.tensor(self.stems).to(device)
        lm_morphs = torch.tensor(self.lm_morphs).to(device)
        pos_tags = torch.tensor(self.pos_tags).to(device)

        predicted_tokens_idx = torch.tensor(self.predicted_tokens_idx).to(device)
        predicted_tokens_affixes_idx = torch.tensor(self.predicted_tokens_affixes_idx).to(device)

        predicted_stems = torch.tensor(self.predicted_stems).to(device)
        predicted_pos_tags = torch.tensor(self.predicted_pos_tags).to(device)
        predicted_lm_morphs = torch.tensor(self.predicted_lm_morphs).to(device)

        cfg: FlexConfig = FlexConfig()
        affixes = torch.tensor(self.affixes)
        pred_affixes_list = [self.predicted_affixes[x - y: x] for x, y in zip(accumulate(self.predicted_affix_lengths), self.predicted_affix_lengths)]
        afx_prob = torch.zeros(len(pred_affixes_list), cfg.tot_num_affixes)
        for i, lst in enumerate(pred_affixes_list):
            assert (len(lst) > 0)
            afx_prob[i, lst] = 1.0
        predicted_affixes_prob = afx_prob.to(device, dtype=torch.float)

        affixes_padded = affixes.split(self.affix_lengths)
        affixes_padded = pad_sequence(affixes_padded, batch_first=False)
        affixes_padded = affixes_padded.to(device, dtype=torch.long)

        morpho_masks_padded = generate_input_key_padding_mask([(x + MORPHO_AFFIX_CONTEXT_ITEMS) for x in self.affix_lengths], device=device)
        main_masks_padded = generate_input_key_padding_mask(input_sequence_lengths, device=device)

        return (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                 input_sequence_lengths, main_masks_padded,
                 predicted_tokens_idx,
                 predicted_tokens_affixes_idx,
                 predicted_stems,
                 predicted_pos_tags,
                 predicted_lm_morphs,
                 predicted_affixes_prob)

    def to_kinya_src_inference_tuple(self, args, input_sequence_lengths: List[int], multi_morph=False) -> Tuple:
        device = torch.device(self.device)
        # device = torch.device('cpu')
        stems = torch.tensor(self.stems).to(device)
        lm_morphs = torch.tensor(self.lm_morphs).to(device)
        pos_tags = torch.tensor(self.pos_tags).to(device)

        affixes = torch.tensor(self.affixes)
        affixes_padded = affixes.split(self.affix_lengths)
        affixes_padded = pad_sequence(affixes_padded, batch_first=False)
        affixes_padded = affixes_padded.to(device, dtype=torch.long)

        morpho_masks_padded = generate_input_key_padding_mask([(x + MORPHO_AFFIX_CONTEXT_ITEMS) for x in self.affix_lengths], device=device)
        main_masks_padded = generate_input_key_padding_mask(input_sequence_lengths, device=device)
        mixed_morphs_masks_padded, mixed_morphs_probs_attn_bias, input_stem_sequence_lengths = None,None,None
        if multi_morph:
            input_stem_sequence_lengths = self.stem_mixer_counts
            mixed_morphs_masks_padded = generate_input_key_padding_mask(input_stem_sequence_lengths, device=device)
            mixed_morphs_probs_attn_bias = pad_sequence(torch.tensor(self.stem_probs).split(input_stem_sequence_lengths, 0), batch_first=True)
            (_, S) = mixed_morphs_probs_attn_bias.shape
            mixed_morphs_probs_attn_bias = mixed_morphs_probs_attn_bias.unsqueeze(-1).repeat(args.main_sequence_encoder_num_heads, 1, S).to(device=device)
            # (N,S)

        return (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded, input_sequence_lengths,
                main_masks_padded, main_masks_padded,
                mixed_morphs_masks_padded, mixed_morphs_probs_attn_bias, input_stem_sequence_lengths)

    def add_bos_and_eos(self) -> MorphoDataItem:
        self.stems = [BOS_ID] + self.stems + [EOS_ID]
        self.lm_morphs = [BOS_ID] + self.lm_morphs + [EOS_ID]
        self.pos_tags = [BOS_ID] + self.pos_tags + [EOS_ID]
        self.affix_lengths = [0] + self.affix_lengths + [0]
        self.stem_probs = [1.0] + self.stem_probs + [1.0]
        self.stem_mixer_counts = [1] + self.stem_mixer_counts + [1]
        return self

    def add_bos(self) -> MorphoDataItem:
        self.stems = [BOS_ID] + self.stems
        self.lm_morphs = [BOS_ID] + self.lm_morphs
        self.pos_tags = [BOS_ID] + self.pos_tags
        self.affix_lengths = [0] + self.affix_lengths
        self.stem_probs = [1.0] + self.stem_probs
        self.stem_mixer_counts = [1] + self.stem_mixer_counts
        return self

    def add_eos(self) -> MorphoDataItem:
        self.stems = self.stems + [EOS_ID]
        self.lm_morphs = self.lm_morphs + [EOS_ID]
        self.pos_tags = self.pos_tags + [EOS_ID]
        self.affix_lengths = self.affix_lengths + [0]
        self.stem_probs = self.stem_probs + [1.0]
        self.stem_mixer_counts = self.stem_mixer_counts + [1]
        return self

    def prepend_special_token(self, special_id) -> MorphoDataItem:
        cfg = FlexConfig()
        (st,ps,af) = cfg.special_token(special_id)
        self.stems = [st] + self.stems
        self.lm_morphs = [af] + self.lm_morphs
        self.pos_tags = [ps] + self.pos_tags
        self.affix_lengths = [0] + self.affix_lengths
        self.stem_probs = [1.0] + self.stem_probs
        self.stem_mixer_counts = [1] + self.stem_mixer_counts
        return self

    def append_special_token(self, special_id) -> MorphoDataItem:
        cfg = FlexConfig()
        (st,ps,af) = cfg.special_token(special_id)
        self.stems = self.stems + [st]
        self.lm_morphs = self.lm_morphs + [ps]
        self.pos_tags = self.pos_tags + [af]
        self.affix_lengths = self.affix_lengths + [0]
        self.stem_probs = self.stem_probs + [1.0]
        self.stem_mixer_counts = self.stem_mixer_counts + [1]
        return self

    def to_simple_inputs(self, input_sequence_lengths: List[int]) -> Tuple:
        # device = torch.device(self.device)
        device = torch.device('cpu')
        lm_morphs = torch.tensor(self.lm_morphs).to(device, dtype=torch.int64)
        pos_tags = torch.tensor(self.pos_tags).to(device, dtype=torch.int64)
        stems = torch.tensor(self.stems).to(device, dtype=torch.int64)
        affixes_padded = pad_sequence(torch.tensor(self.affixes).split(self.affix_lengths), batch_first=False).to(device, dtype=torch.long)
        morpho_masks_padded = generate_input_key_padding_mask([(x + MORPHO_AFFIX_CONTEXT_ITEMS) for x in self.affix_lengths], device=device)
        main_masks_padded = generate_input_key_padding_mask(input_sequence_lengths, device=device, ignore_last=True)
        return (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                input_sequence_lengths, main_masks_padded)

    def to_gpt_training_tuple(self, input_sequence_lengths: List[int]) -> Tuple:
        #device = torch.device(self.device)
        device = torch.device('cpu')
        lm_morphs = torch.tensor(self.lm_morphs)#.to(device)
        pos_tags = torch.tensor(self.pos_tags)#.to(device)
        stems = torch.tensor(self.stems)#.to(device)

        cfg: FlexConfig = FlexConfig()
        affixes = torch.tensor(self.affixes)
        pred_affixes_list = [affixes[x - y: x] for x, y in zip(accumulate(self.affix_lengths), self.affix_lengths)]
        afx_prob = torch.zeros(len(pred_affixes_list), cfg.tot_num_affixes)
        for i, lst in enumerate(pred_affixes_list):
            if (len(lst) > 0):
                afx_prob[i, lst] = 1.0
        affixes_prob = afx_prob#.to(device, dtype=torch.float)

        affixes_padded = affixes.split(self.affix_lengths)
        affixes_padded = pad_sequence(affixes_padded, batch_first=False)
        affixes_padded = affixes_padded.to(device, dtype=torch.long)

        morpho_masks_padded = generate_input_key_padding_mask([(x + MORPHO_AFFIX_CONTEXT_ITEMS) for x in self.affix_lengths], device=device)

        input_masks_padded = generate_input_key_padding_mask(input_sequence_lengths, device=device, ignore_last=True)
        decoder_mask = generate_square_subsequent_mask(max(input_sequence_lengths), device=device)

        predicted_stems = stems.split(input_sequence_lengths)
        predicted_stems = [tns[1:length] for length, tns in zip(input_sequence_lengths, predicted_stems)]
        predicted_stems = torch.cat(predicted_stems, dim=0).to(device, dtype=torch.int64)

        predicted_pos_tags = pos_tags.split(input_sequence_lengths)
        predicted_pos_tags = [tns[1:length] for length, tns in zip(input_sequence_lengths, predicted_pos_tags)]
        predicted_pos_tags = torch.cat(predicted_pos_tags, dim=0).to(device, dtype=torch.int64)

        predicted_lm_morphs = lm_morphs.split(input_sequence_lengths)
        predicted_lm_morphs = [tns[1:length] for length, tns in zip(input_sequence_lengths, predicted_lm_morphs)]
        predicted_lm_morphs = torch.cat(predicted_lm_morphs, dim=0).to(device, dtype=torch.int64)

        predicted_affixes_prob = affixes_prob.split(input_sequence_lengths)
        predicted_affixes_prob = [tns[1:length, :] for length, tns in zip(input_sequence_lengths, predicted_affixes_prob)]
        predicted_affixes_prob = torch.cat(predicted_affixes_prob, dim=0).to(device, dtype=torch.float32)

        stems = stems.to(device, dtype=torch.int64)
        pos_tags = pos_tags.to(device, dtype=torch.int64)
        lm_morphs = lm_morphs.to(device, dtype=torch.int64)

        return (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                 input_sequence_lengths,
                 input_masks_padded, decoder_mask,
                 predicted_stems, predicted_pos_tags, predicted_lm_morphs, predicted_affixes_prob)

    def to_kinya_tgt_inference_tuple(self, input_sequence_lengths: List[int]) -> Tuple:
        # For inference, send data directly to GPU
        device = torch.device(self.device)
        #device = torch.device('cpu')
        lm_morphs = torch.tensor(self.lm_morphs)#.to(device)
        pos_tags = torch.tensor(self.pos_tags)#.to(device)
        stems = torch.tensor(self.stems)#.to(device)

        affixes = torch.tensor(self.affixes)
        affixes_padded = affixes.split(self.affix_lengths)
        affixes_padded = pad_sequence(affixes_padded, batch_first=False)
        affixes_padded = affixes_padded.to(device, dtype=torch.long)

        morpho_masks_padded = generate_input_key_padding_mask([(x + MORPHO_AFFIX_CONTEXT_ITEMS) for x in self.affix_lengths], device=device)

        input_masks_padded = generate_input_key_padding_mask(input_sequence_lengths, device=device, ignore_last=False)
        decoder_mask = generate_square_subsequent_mask(max(input_sequence_lengths), device=device)

        stems = stems.to(device, dtype=torch.int64)
        pos_tags = pos_tags.to(device, dtype=torch.int64)
        lm_morphs = lm_morphs.to(device, dtype=torch.int64)

        return (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded, input_sequence_lengths,
                input_masks_padded, decoder_mask)

    def to_classification_tuple(self, input_sequence_lengths: List[int], labels: Union[None,List[Union[int,float]]]) -> Tuple:
        device = torch.device(self.device)
        lm_morphs = torch.tensor(self.lm_morphs)
        pos_tags = torch.tensor(self.pos_tags)
        stems = torch.tensor(self.stems)

        predicted_targets = None if (labels is None) else torch.tensor(labels)

        affixes = torch.tensor(self.affixes)
        affixes_padded = affixes.split(self.affix_lengths)
        affixes_padded = pad_sequence(affixes_padded, batch_first=False)
        affixes_padded = affixes_padded.to(device, dtype=torch.long)

        morpho_masks_padded = generate_input_key_padding_mask([(x + MORPHO_AFFIX_CONTEXT_ITEMS) for x in self.affix_lengths], device=device)

        main_masks_padded = generate_input_key_padding_mask(input_sequence_lengths, device=device, ignore_last=False)

        stems = stems.to(device, dtype=torch.int64)
        pos_tags = pos_tags.to(device, dtype=torch.int64)
        lm_morphs = lm_morphs.to(device, dtype=torch.int64)

        return (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                input_sequence_lengths, main_masks_padded, predicted_targets)

    def to_kin2en_train_tuple(self, device: torch.device, input_sequence_lengths: List[int], english_input_ids: List[int], english_sequence_lengths: List[int]) -> Tuple:
        stems = torch.tensor(self.stems, dtype=torch.long).to(device)
        pos_tags = torch.tensor(self.pos_tags, dtype=torch.long).to(device)
        lm_morphs = torch.tensor(self.lm_morphs, dtype=torch.long).to(device)
        affixes = torch.tensor(self.affixes, dtype=torch.long)#.to(device)

        english_input_ids = torch.tensor(english_input_ids, dtype=torch.long).to(device)

        predicted_english_tokens = english_input_ids.split(english_sequence_lengths)
        predicted_english_tokens = [tns[1:length] for length, tns in zip(english_sequence_lengths, predicted_english_tokens)]
        predicted_english_tokens = torch.cat(predicted_english_tokens, dim=0).to(device, dtype=torch.int64)

        affixes_padded = affixes.split(self.affix_lengths)
        affixes_padded = pad_sequence(affixes_padded, batch_first=False)
        affixes_padded = affixes_padded.to(device, dtype=torch.long)

        morpho_masks_padded = generate_input_key_padding_mask([(x + MORPHO_AFFIX_CONTEXT_ITEMS) for x in self.affix_lengths], device=device)
        main_masks_padded = generate_input_key_padding_mask(input_sequence_lengths, device=device)
        seq_len = max(english_sequence_lengths)
        src_key_padding_mask = generate_input_key_padding_mask(input_sequence_lengths, device=device, ignore_last=False)
        tgt_key_padding_mask = generate_input_key_padding_mask(english_sequence_lengths, device=device, ignore_last=True)
        decoder_mask = generate_square_subsequent_mask(seq_len, device=device)

        return (english_input_ids, english_sequence_lengths, predicted_english_tokens,
                 stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded, input_sequence_lengths, # number of words per each sequence in the batch
                 main_masks_padded,
                 src_key_padding_mask, tgt_key_padding_mask, decoder_mask)

    def to_en2kin_train_tuple(self, device: torch.device, input_sequence_lengths: List[int], english_input_ids: List[int], english_sequence_lengths: List[int]) -> Tuple:
        stems = torch.tensor(self.stems, dtype=torch.long).to(device)
        pos_tags = torch.tensor(self.pos_tags, dtype=torch.long).to(device)
        lm_morphs = torch.tensor(self.lm_morphs, dtype=torch.long).to(device)
        affixes = torch.tensor(self.affixes, dtype=torch.long)#.to(device)

        english_input_ids = torch.tensor(english_input_ids, dtype=torch.long).to(device)

        cfg: FlexConfig = FlexConfig()
        pred_affixes_list = [affixes[x - y: x] for x, y in zip(accumulate(self.affix_lengths), self.affix_lengths)]
        afx_prob = torch.zeros(len(pred_affixes_list), cfg.tot_num_affixes)
        for i, lst in enumerate(pred_affixes_list):
            if (len(lst) > 0):
                afx_prob[i, lst] = 1.0
        affixes_prob = afx_prob # .to(device, dtype=torch.float)

        affixes_padded = affixes.split(self.affix_lengths)
        affixes_padded = pad_sequence(affixes_padded, batch_first=False)
        affixes_padded = affixes_padded.to(device, dtype=torch.long)

        morpho_masks_padded = generate_input_key_padding_mask([(x + MORPHO_AFFIX_CONTEXT_ITEMS) for x in self.affix_lengths], device=device)
        input_masks_padded = generate_input_key_padding_mask(input_sequence_lengths, device=device, ignore_last=True)
        seq_len = max(input_sequence_lengths)
        src_key_padding_mask = generate_input_key_padding_mask(english_sequence_lengths, device=device, ignore_last=False)
        tgt_key_padding_mask = generate_input_key_padding_mask(input_sequence_lengths, device=device, ignore_last=True)
        decoder_mask = generate_square_subsequent_mask(seq_len, device=device)

        predicted_stems = stems.split(input_sequence_lengths)
        predicted_stems = [tns[1:length] for length, tns in zip(input_sequence_lengths, predicted_stems)]
        predicted_stems = torch.cat(predicted_stems, dim=0).to(device, dtype=torch.int64)

        predicted_pos_tags = pos_tags.split(input_sequence_lengths)
        predicted_pos_tags = [tns[1:length] for length, tns in zip(input_sequence_lengths, predicted_pos_tags)]
        predicted_pos_tags = torch.cat(predicted_pos_tags, dim=0).to(device, dtype=torch.int64)

        predicted_lm_morphs = lm_morphs.split(input_sequence_lengths)
        predicted_lm_morphs = [tns[1:length] for length, tns in zip(input_sequence_lengths, predicted_lm_morphs)]
        predicted_lm_morphs = torch.cat(predicted_lm_morphs, dim=0).to(device, dtype=torch.int64)

        predicted_affixes_prob = affixes_prob.split(input_sequence_lengths)
        predicted_affixes_prob = [tns[1:length, :] for length, tns in zip(input_sequence_lengths, predicted_affixes_prob)]
        predicted_affixes_prob = torch.cat(predicted_affixes_prob, dim=0).to(device, dtype=torch.float32)

        pos_tags = pos_tags.to(device, dtype=torch.int64)
        lm_morphs = lm_morphs.to(device, dtype=torch.int64)

        return (english_input_ids, english_sequence_lengths,
                 stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                 input_sequence_lengths,  # number of words per each sequence in the batch
                 input_masks_padded,
                 predicted_stems, predicted_pos_tags, predicted_lm_morphs, predicted_affixes_prob,
                 src_key_padding_mask, tgt_key_padding_mask, decoder_mask)

def prepare_morpho_data_from_sentence(cfg: FlexConfig, device: str, sentence: ParsedFlexSentence, max_len=505) -> MorphoDataItem:
    lm_morphs: List[int] = []
    pos_tags: List[int] = []
    affixes: List[int] = []
    affix_lengths: List[int] = []
    stems: List[int] = []

    sentence.trim(max_len)

    for token in sentence.tokens:
        assert ((len(token.affixes) == 0) or (len(token.stems_ids)==1)), f"Extra tokens with affixes: {token.to_parsed_format()}"
        affixes.extend(token.affixes)
        for sid in token.stems_ids:
            lm_morphs.append(token.lm_morph_id)
            pos_tags.append(token.pos_tag_id)
            stems.append(sid)
            affix_lengths.append(len(token.affixes)) # If extra stem id, then there are no affixes

    return MorphoDataItem(device,
                        stems = stems,
                        lm_morphs = lm_morphs,
                        pos_tags = pos_tags,
                        affixes = affixes,
                        affix_lengths = affix_lengths)

def prepare_morpho_data_from_sentence_multi_morph(cfg: FlexConfig, device: str, sentence: ParsedSentenceMulti) -> MorphoDataItem:
    lm_morphs: List[int] = []
    pos_tags: List[int] = []
    affixes: List[int] = []
    affix_lengths: List[int] = []
    stems: List[int] = []
    stem_probs: List[float] = []
    stem_mixer_counts: List[int] = []

    for multi_token in sentence.multi_tokens:
        if len(multi_token.alt_tokens) > 1:
            stem_mixer_counts.append(len(multi_token.alt_tokens))
        for token in multi_token.alt_tokens:
            assert ((len(token.affixes) == 0) or (len(token.stems_ids)==1)), f"Extra tokens with affixes: {token.to_parsed_format()}"
            affixes.extend(token.affixes)
            for sid in token.stems_ids:
                if len(multi_token.alt_tokens) == 1:
                    stem_mixer_counts.append(1)
                stem_probs.append(token.prob)
                lm_morphs.append(token.lm_morph_id)
                pos_tags.append(token.pos_tag_id)
                stems.append(sid)
                affix_lengths.append(len(token.affixes)) # If extra stem id, then there are no affixes

    return MorphoDataItem(device,
                          stems = stems,
                          lm_morphs = lm_morphs,
                          pos_tags = pos_tags,
                          affixes = affixes,
                          affix_lengths = affix_lengths,
                          stem_probs=stem_probs,
                          stem_mixer_counts=stem_mixer_counts)

def send_tuple_to(batch_item:Tuple, device: torch.device, non_blocking=True, top=0, args=None):
    if (args is not None) and (top == 0):
        model_type = args.model_variant.split(':')[0]
        if 'kinya_colbert' in model_type:
            batch_item = tuple(batch_item[0]), tuple(batch_item[1])
    return tuple([it.to(device, non_blocking=non_blocking) if torch.is_tensor(it) else (send_tuple_to(it, device, non_blocking=non_blocking, top=(top+1), args=args) if (type(it) is tuple) else it) for it in batch_item])

def print_batch_tuple(batch_item:Tuple):
    out = []
    for it in batch_item:
        if torch.is_tensor(it):
            out.append(it.shape)
        elif isinstance(it, list):
            out.append(f':{len(it)}')
        else:
            out.append(it)
    print('Batch items =>', out, flush=True)
