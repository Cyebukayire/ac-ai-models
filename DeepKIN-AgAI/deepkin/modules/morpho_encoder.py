from __future__ import print_function, division

# Ignore warnings
import warnings

from torch.amp import custom_fwd

from deepkin.modules.flex_modules import FlexConfig
from deepkin.modules.flex_transformers import TransformerEncoderLayer, TransformerEncoder
from deepkin.utils.arguments import FlexArguments

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def extend_embedding_with_special_tokens(emb, num_specials, init_range=0.02):
    ext = torch.randn(num_specials, emb.weight.size(1), device=emb.weight.device)
    ext.normal_(mean=0.0, std=init_range)
    emb.weight = nn.Parameter(torch.cat((emb.weight, ext)))
    return emb

class KinyaEncoder(nn.Module):

    def __init__(self, cfg:FlexConfig, flex_hidden_dim:int = 192, flex_dim_feedforward:int = 512, flex_num_heads=4, flex_num_layers:int = 4, flex_dropout:float = 0.1, sentence_stem_mult=3, KIN_PAD_IDX:int = 0):
        super(KinyaEncoder, self).__init__()
        self.flex_hidden_dim = flex_hidden_dim
        self.flex_dim_feedforward=flex_dim_feedforward
        self.flex_num_layers=flex_num_layers
        self.flex_num_heads=flex_num_heads
        self.main_hidden_dim = (flex_hidden_dim * 3) + (flex_hidden_dim * sentence_stem_mult)
        self.pos_tag_embedding = nn.Embedding(cfg.tot_num_pos_tags, flex_hidden_dim, padding_idx=KIN_PAD_IDX)
        self.lm_morph_embedding = nn.Embedding(cfg.tot_num_lm_morphs, flex_hidden_dim, padding_idx=KIN_PAD_IDX)
        self.affixes_embedding = nn.Embedding(cfg.tot_num_affixes, flex_hidden_dim, padding_idx=KIN_PAD_IDX)
        self.word_stem_embedding = nn.Embedding(cfg.tot_num_stems, flex_hidden_dim, padding_idx=KIN_PAD_IDX)
        self.sentence_stem_embedding = nn.Embedding(cfg.tot_num_stems, (flex_hidden_dim * sentence_stem_mult), padding_idx=KIN_PAD_IDX)
        encoder_layers = TransformerEncoderLayer(flex_hidden_dim, flex_num_heads,
                                                dim_feedforward=flex_dim_feedforward,
                                                dropout=flex_dropout,
                                                use_rms_norm=True)
        self.morpho_encoder = TransformerEncoder(encoder_layers, flex_num_layers)

    def extend_with_special_tokens(self, num_specials, init_range=0.02):
        extend_embedding_with_special_tokens(self.pos_tag_embedding, num_specials, init_range=init_range)
        extend_embedding_with_special_tokens(self.lm_morph_embedding, num_specials, init_range=init_range)
        extend_embedding_with_special_tokens(self.affixes_embedding, num_specials, init_range=init_range)
        extend_embedding_with_special_tokens(self.word_stem_embedding, num_specials, init_range=init_range)
        extend_embedding_with_special_tokens(self.sentence_stem_embedding, num_specials, init_range=init_range)

    @custom_fwd(device_type='cuda')
    def forward(self, stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded, input_sequence_lengths, mixed_morphs_masks_padded=None, mixed_morphs_probs_attn_bias=None, input_stem_sequence_lengths=None): # number of words per each sequence in the batch
        word_stems = self.word_stem_embedding(stems) # (B) -> (B,D)
        word_stems = torch.unsqueeze(word_stems, 0) # (1,B,D)

        pos_tags = self.pos_tag_embedding(pos_tags) # (B) -> (B,D)
        pos_tags = torch.unsqueeze(pos_tags, 0) # (1,B,D)

        lm_morphs = self.lm_morph_embedding(lm_morphs) # (B) -> (B,D)
        lm_morphs = torch.unsqueeze(lm_morphs, 0) # (1,B,D)

        morpho_embed = torch.cat((word_stems, pos_tags, lm_morphs), 0) # (3,B,D)

        if affixes_padded.nelement() > 0:
            morpho_affix = self.affixes_embedding(affixes_padded)
            morpho_embed = torch.cat((morpho_embed, morpho_affix), 0)

        morpho_out = self.morpho_encoder(morpho_embed, src_key_padding_mask=morpho_masks_padded)  # (3+A,B,D), A = afx_padded.size(0)

        # Pooling
        morpho_out = morpho_out[:3, :, :] # (3,L,D)
        morpho_out = morpho_out.permute(1, 0, 2) # ==> (L,3,D)
        L = morpho_out.size(0)
        morpho_out = morpho_out.contiguous().view(L, -1)  # (L, 3D)

        sent_stems = self.sentence_stem_embedding(stems) # (L,3D)

        seq_input = torch.cat((morpho_out, sent_stems), 1) # (L,E=3D+3D)
        seq_input = pad_sequence(seq_input.split(input_sequence_lengths, 0), batch_first=False) # (L,E) -> (S,N,E); L=SN; S=max(input_sequence_lengths)

        return seq_input # shape: (S,N,E)


class MultiMorphoEncoder(nn.Module):

    def __init__(self, args: FlexArguments, cfg:FlexConfig, flex_hidden_dim:int = 192, flex_dim_feedforward:int = 512, flex_num_heads=4, flex_num_layers:int = 4, flex_dropout:float = 0.1, sentence_stem_mult=3, KIN_PAD_IDX:int = 0):
        super(MultiMorphoEncoder, self).__init__()
        self.flex_hidden_dim = flex_hidden_dim
        self.flex_dim_feedforward=flex_dim_feedforward
        self.flex_num_layers=flex_num_layers
        self.flex_num_heads=flex_num_heads
        self.main_hidden_dim = (flex_hidden_dim * 3) + (flex_hidden_dim * sentence_stem_mult)
        self.pos_tag_embedding = nn.Embedding(cfg.tot_num_pos_tags, flex_hidden_dim, padding_idx=KIN_PAD_IDX)
        self.lm_morph_embedding = nn.Embedding(cfg.tot_num_lm_morphs, flex_hidden_dim, padding_idx=KIN_PAD_IDX)
        self.affixes_embedding = nn.Embedding(cfg.tot_num_affixes, flex_hidden_dim, padding_idx=KIN_PAD_IDX)
        self.word_stem_embedding = nn.Embedding(cfg.tot_num_stems, flex_hidden_dim, padding_idx=KIN_PAD_IDX)
        self.sentence_stem_embedding = nn.Embedding(cfg.tot_num_stems, (flex_hidden_dim * sentence_stem_mult), padding_idx=KIN_PAD_IDX)

        encoder_layers = TransformerEncoderLayer(flex_hidden_dim, flex_num_heads,
                                                dim_feedforward=flex_dim_feedforward,
                                                dropout=flex_dropout,
                                                use_rms_norm=True)
        self.morpho_encoder = TransformerEncoder(encoder_layers, flex_num_layers)

        mixer_layers = TransformerEncoderLayer(self.main_hidden_dim, args.main_sequence_encoder_num_heads,
                                                 dim_feedforward=args.main_sequence_encoder_dim_ffn,
                                                 dropout=args.main_sequence_encoder_dropout,
                                                 use_rms_norm=True)
        self.morpho_mixer = TransformerEncoder(mixer_layers, flex_num_layers)


    def extend_with_special_tokens(self, num_specials, init_range=0.02):
        extend_embedding_with_special_tokens(self.pos_tag_embedding, num_specials, init_range=init_range)
        extend_embedding_with_special_tokens(self.lm_morph_embedding, num_specials, init_range=init_range)
        extend_embedding_with_special_tokens(self.affixes_embedding, num_specials, init_range=init_range)
        extend_embedding_with_special_tokens(self.word_stem_embedding, num_specials, init_range=init_range)
        extend_embedding_with_special_tokens(self.sentence_stem_embedding, num_specials, init_range=init_range)

    @custom_fwd(device_type='cuda')
    def forward(self, stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded, input_sequence_lengths, mixed_morphs_masks_padded=None, mixed_morphs_probs_attn_bias=None, input_stem_sequence_lengths=None): # number of words per each sequence in the batch
        word_stems = self.word_stem_embedding(stems) # (B) -> (B,D)
        word_stems = torch.unsqueeze(word_stems, 0) # (1,B,D)

        pos_tags = self.pos_tag_embedding(pos_tags) # (B) -> (B,D)
        pos_tags = torch.unsqueeze(pos_tags, 0) # (1,B,D)

        lm_morphs = self.lm_morph_embedding(lm_morphs) # (B) -> (B,D)
        lm_morphs = torch.unsqueeze(lm_morphs, 0) # (1,B,D)

        morpho_embed = torch.cat((word_stems, pos_tags, lm_morphs), 0) # (3,B,D)

        if affixes_padded.nelement() > 0:
            morpho_affix = self.affixes_embedding(affixes_padded)
            morpho_embed = torch.cat((morpho_embed, morpho_affix), 0)

        morpho_out = self.morpho_encoder(morpho_embed, src_key_padding_mask=morpho_masks_padded)  # (3+A,B,D), A = afx_padded.size(0)

        # Pooling
        morpho_out = morpho_out[:3, :, :] # (3,L,D)
        morpho_out = morpho_out.permute(1, 0, 2) # ==> (L,3,D)
        L = morpho_out.size(0)
        morpho_out = morpho_out.contiguous().view(L, -1)  # (L, 3D)

        sent_stems = self.sentence_stem_embedding(stems) # (L,3D)
        mix_input = torch.cat((morpho_out, sent_stems), 1) # (L,E=3D+3D)
        mix_input = pad_sequence(mix_input.split(input_stem_sequence_lengths, 0), batch_first=False) # (L,E) -> (A,B,E)
        # A ==> max alt morphs
        # B ==> batch

        mix_input = self.morpho_mixer(mix_input, attn_bias=mixed_morphs_probs_attn_bias, src_key_padding_mask=mixed_morphs_masks_padded)
        # Pick the first alt-morph with the highest parsed probability
        seq_input = pad_sequence(mix_input[0, :, :].split(input_sequence_lengths, 0), batch_first=False) # (B,E) -> (S,N,E)

        return seq_input # shape: (S,N,E)
