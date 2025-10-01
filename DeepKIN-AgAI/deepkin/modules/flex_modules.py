from __future__ import print_function, division

# Ignore warnings
import warnings

from torch.amp import custom_fwd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear

from deepkin.modules.layer_norm import FusedRMSNorm

# Ignore warnings
import warnings
from typing import Tuple

warnings.filterwarnings("ignore")

# 4 items attend to affixes: (pos_tags, lm_morphs, word_stems)
NUM_AFFIX_CONTEXT_ITEMS = 4
FLEX_NUM_LOSSES = 3

MORPHO_AFFIX_CONTEXT_ITEMS = 3
MORPHO_NUM_LOSSES = 4

class FlexConfig:
    def __init__(self, cfg = None):
        if cfg is not None:
            self.tot_num_lm_morphs = cfg.tot_num_lm_morphs
            self.tot_num_pos_tags = cfg.tot_num_pos_tags
            self.tot_num_stems = cfg.tot_num_stems
            self.tot_num_affixes = cfg.tot_num_affixes
        else:
            self.tot_num_lm_morphs = 24122
            self.tot_num_pos_tags = 157
            self.tot_num_stems = 35497
            self.tot_num_affixes = 407

    def special_token(self, special_id) -> Tuple[int,int,int]:
        return (self.tot_num_stems+special_id,
                self.tot_num_pos_tags+special_id,
                self.tot_num_lm_morphs+special_id)

# tot_num_affixes = 407
# tot_num_lm_morphs = 24122
# tot_num_lm_stems = 12350
# tot_num_pos_tags = 157
# tot_num_stems = 35497


class FlexHeadTransform(nn.Module):
    def __init__(self, tr_d_model, cls_ctxt_size, layernorm_epsilon, dropout=0.0):
        super(FlexHeadTransform, self).__init__()
        self.norm = FusedRMSNorm(tr_d_model, eps=layernorm_epsilon)
        self.dense = Linear(tr_d_model, cls_ctxt_size)
        self.dropout = Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class FlexTokenHead(nn.Module):
    def __init__(self, embedding_weights, tr_d_model, layernorm_epsilon, dropout=0.0):
        super(FlexTokenHead, self).__init__()
        self.token_transform = FlexHeadTransform(tr_d_model, embedding_weights.size(1), layernorm_epsilon, dropout=dropout)
        self.token_decoder = Linear(embedding_weights.size(1), embedding_weights.size(0), bias=False)
        self.token_decoder.weight = embedding_weights
        self.token_decoder_bias = nn.Parameter(torch.zeros(embedding_weights.size(0)))

    def get_logits(self, hidden_state):
        logits = self.token_transform(hidden_state)
        logits = self.token_decoder(logits) + self.token_decoder_bias
        return logits

    @custom_fwd(device_type='cuda')
    def forward(self, hidden_state,):
        return self.get_logits(hidden_state)
