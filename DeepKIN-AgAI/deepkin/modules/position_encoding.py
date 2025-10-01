import math

import torch
import torch.nn.functional as F
from deepkin.modules.layer_norm import FusedLayerNorm
from torch import nn
from torch.amp import custom_fwd

from torch.nn import Linear

# From: https://github.com/guolinke/TUPE/blob/master/fairseq/modules/transformer_sentence_encoder.py
# this is from T5
def tupe_relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret

# TUPE: https://arxiv.org/abs/2006.15595
# https://github.com/guolinke/TUPE/blob/master/fairseq/modules/transformer_sentence_encoder.py
class PositionEncoding(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_seq_len, rel_pos_bins, max_rel_pos, separate_cls_sos):
        super(PositionEncoding, self).__init__()
        self.separate_cls_sos = separate_cls_sos
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.attn_scale_factor = 2
        self.pos = nn.Embedding(self.max_seq_len + 1, self.hidden_dim)
        self.pos_q_linear = Linear(self.hidden_dim, self.hidden_dim)
        self.pos_k_linear = Linear(self.hidden_dim, self.hidden_dim)
        self.pos_scaling = float(self.hidden_dim / self.num_heads * self.attn_scale_factor) ** -0.5
        self.pos_ln = FusedLayerNorm(self.hidden_dim)
        self.tupe_rel_pos_bins = rel_pos_bins
        self.tupe_max_rel_pos = max_rel_pos
        self.relative_attention_bias = nn.Embedding(self.tupe_rel_pos_bins + 1, self.num_heads)
        seq_len = self.max_seq_len
        context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        self.rp_bucket = tupe_relative_position_bucket(
            relative_position,
            num_buckets=self.tupe_rel_pos_bins,
            max_distance=self.tupe_max_rel_pos
        )
        # others to [CLS]
        self.rp_bucket[:, 0] = self.tupe_rel_pos_bins
        # [CLS] to others, Note: self.tupe_rel_pos_bins // 2 is not used in relative_position_bucket
        self.rp_bucket[0, :] = self.tupe_rel_pos_bins // 2

    def get_tupe_rel_pos_bias(self, seq_len, device):
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        if self.rp_bucket.device != device:
            self.rp_bucket = self.rp_bucket.to(device)
        # Adjusted because final x's shape is L x B X E
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def get_position_attn_bias(self, seq_len, batch_size, device):
        tupe_rel_pos_bias = self.get_tupe_rel_pos_bias(seq_len, device)
        if self.separate_cls_sos:
            # 0 is for other-to-cls 1 is for cls-to-other
            # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
            weight = self.pos_ln(self.pos.weight[:seq_len + 1, :])
            pos_q =  self.pos_q_linear(weight).view(seq_len + 1, self.num_heads, -1).transpose(0, 1) * self.pos_scaling
            pos_k =  self.pos_k_linear(weight).view(seq_len + 1, self.num_heads, -1).transpose(0, 1)
            abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
            # p_0 \dot p_0 is cls to others
            cls_2_other = abs_pos_bias[:, 0, 0]
            # p_1 \dot p_1 is others to cls
            other_2_cls = abs_pos_bias[:, 1, 1]
            # offset
            abs_pos_bias = abs_pos_bias[:, 1:, 1:]
            abs_pos_bias[:, :, 0] = other_2_cls.view(-1, 1)
            abs_pos_bias[:, 0, :] = cls_2_other.view(-1, 1)
            tupe_rel_pos_bias += abs_pos_bias
        else:
            weight = self.pos_ln(self.pos.weight[:seq_len, :])
            pos_q = self.pos_q_linear(weight).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.pos_scaling
            pos_k = self.pos_k_linear(weight).view(seq_len, self.num_heads, -1).transpose(0, 1)
            abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
            tupe_rel_pos_bias += abs_pos_bias

        tupe_rel_pos_bias = tupe_rel_pos_bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(-1, seq_len, seq_len)
        # Final shape: [batch_size x num_heads, from_seq_length, to_seq_length]
        return tupe_rel_pos_bias

    @custom_fwd(device_type='cuda')
    def forward(self, hidden):
        # hidden shape for our Transformer models: (Len, Batch, Embed)
        device = hidden.device
        seq_len = hidden.size(0)
        batch_size = hidden.size(1)
        return self.get_position_attn_bias(seq_len, batch_size, device) #(BH,L,L)

class CrossAttentionPositionalEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 num_attn_heads,
                 mask_src_cls_rel_pos = False,
                 max_seq_len = 512,
                 use_tupe_rel_pos_bias = True,
                 tupe_rel_pos_bins: int = 64,
                 tupe_max_rel_pos: int = 256):
        super(CrossAttentionPositionalEncoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.seq_tr_d_model = d_model
        self.seq_tr_nhead = num_attn_heads
        self.attn_scale_factor = 2.0
        self.mask_src_cls_rel_pos = mask_src_cls_rel_pos

        # This is from TUPE
        self.pos_tgt = nn.Embedding(self.max_seq_len + 1, self.seq_tr_d_model)
        self.pos_src = nn.Embedding(self.max_seq_len + 1, self.seq_tr_d_model)
        self.pos_q_linear = Linear(self.seq_tr_d_model, self.seq_tr_d_model)
        self.pos_k_linear = Linear(self.seq_tr_d_model, self.seq_tr_d_model)
        self.pos_scaling = float(self.seq_tr_d_model / self.seq_tr_nhead * self.attn_scale_factor) ** -0.5
        self.pos_tgt_ln = FusedLayerNorm(self.seq_tr_d_model)
        self.pos_src_ln = FusedLayerNorm(self.seq_tr_d_model)

        self.use_tupe_rel_pos_bias = use_tupe_rel_pos_bias
        if self.use_tupe_rel_pos_bias:
            assert tupe_rel_pos_bins % 2 == 0
            self.tupe_rel_pos_bins = tupe_rel_pos_bins
            self.tupe_max_rel_pos = tupe_max_rel_pos
            self.relative_attention_bias = nn.Embedding(self.tupe_rel_pos_bins + 1, self.seq_tr_nhead)
            seq_len = self.max_seq_len
            context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = tupe_relative_position_bucket(
                relative_position,
                num_buckets=self.tupe_rel_pos_bins,
                max_distance=self.tupe_max_rel_pos
            )
            if self.mask_src_cls_rel_pos:
                self.rp_bucket[:, 0] = self.tupe_rel_pos_bins
                self.cls_pos_embed = nn.Embedding(2, self.seq_tr_nhead)

    def get_tupe_rel_pos_bias(self, src_len, tgt_len, device):
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        if self.rp_bucket.device != device:
            self.rp_bucket = self.rp_bucket.to(device)
        # Adjusted because final x's shape is L x B X E
        rp_bucket = self.rp_bucket[:tgt_len, :src_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous() # (nhead, tgt_len, src_len)

    def get_position_attn_bias(self, src_len, tgt_len, batch_size, device):
        tupe_rel_pos_bias = self.get_tupe_rel_pos_bias(src_len, tgt_len, device) if self.use_tupe_rel_pos_bias else None

        weight_q = self.pos_tgt_ln(self.pos_tgt.weight[:tgt_len, :])
        weight_k = self.pos_src_ln(self.pos_src.weight[:src_len, :])
        pos_q = self.pos_q_linear(weight_q).view(tgt_len, self.seq_tr_nhead, -1).transpose(0, 1) * self.pos_scaling
        pos_k = self.pos_k_linear(weight_k).view(src_len, self.seq_tr_nhead, -1).transpose(0, 1)
        abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
        if self.mask_src_cls_rel_pos:
            abs_pos_bias[:, :, 0] = self.cls_pos_embed(torch.tensor([0], device=device)).view(-1, 1)

        if tupe_rel_pos_bias is not None:
            abs_pos_bias += tupe_rel_pos_bias

        abs_pos_bias = abs_pos_bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(-1, tgt_len, src_len)

        return abs_pos_bias


    @custom_fwd(device_type='cuda')
    def forward(self, tgt, src): # L,N,E
        src_len = src.size(0)
        tgt_len = tgt.size(0)
        batch_size = src.size(1)
        device = src.device
        return self.get_position_attn_bias(src_len, tgt_len, batch_size, device)
