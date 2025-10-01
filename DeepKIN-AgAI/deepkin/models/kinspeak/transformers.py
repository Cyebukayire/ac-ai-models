import copy
import math
from typing import Optional, List
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_fwd
from torch.nn import Module, Linear, Dropout, ModuleList, Embedding

from deepkin.models.kinspeak.layer_norm import FusedLayerNorm
from deepkin.modules.flex_transformers import MultiheadAttention, MultiheadAttentionBias


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, attn_scale_factor=1, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, scale_factor=attn_scale_factor, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Implemented Pre-LN Architecture for better efficiency

        # Self-Attention
        src1 = self.norm1(src)
        src1 = self.self_attn(src1, src1, src1, attn_mask=src_mask, attn_bias=attn_bias, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src1)

        # FFN
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)

        return src

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, attn_bias=attn_bias, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def embeddings(self, src: Tensor, mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> List[Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        embeds = [src]
        for mod in self.layers:
            output = mod(embeds[-1], src_mask=mask, attn_bias=attn_bias, src_key_padding_mask=src_key_padding_mask)
            embeds.append(output)
        return embeds

class TransformerDecoderLayer(Module):

    def __init__(self, d_model, nhead, tgt_attn_scale_factor=1,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.tgt_attn = MultiheadAttention(d_model, nhead, scale_factor=tgt_attn_scale_factor, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = dropout
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        # Self-Attention
        tgt1 = self.norm1(tgt)
        tgt1 = self.tgt_attn(tgt1, tgt1, tgt1, attn_mask=tgt_mask, attn_bias=tgt_attn_bias, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt +  F.dropout(tgt1, p=self.dropout, training=self.training)

        # FFN
        tgt1 = self.norm2(tgt)
        tgt1 = self.activation(self.linear1(tgt1))
        tgt1 = F.dropout(tgt1, p=self.dropout, training=self.training)
        tgt1 = self.linear2(tgt1)
        tgt = tgt + F.dropout(tgt1, p=self.dropout, training=self.training)

        return tgt

class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)

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

class CrossAttentionPositionalEncoder(Module):
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
        self.pos_tgt = Embedding(self.max_seq_len + 1, self.seq_tr_d_model)
        self.pos_src = Embedding(self.max_seq_len + 1, self.seq_tr_d_model)
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
            self.relative_attention_bias = Embedding(self.tupe_rel_pos_bins + 1, self.seq_tr_nhead)
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
                self.cls_pos_embed = Embedding(2, self.seq_tr_nhead)

        self.apply(init_bert_params)

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


class KINMT_TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", bert=True):
        super(KINMT_TransformerEncoderLayer, self).__init__()
        self.d_model=d_model
        if bert:
            self.self_attn = MultiheadAttention(d_model, nhead, scale_factor=3, dropout=dropout)
            self.bert_attn_bias = MultiheadAttentionBias(d_model, nhead, scale_factor=3)
        else:
            self.self_attn = MultiheadAttention(d_model, nhead, scale_factor=2, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(KINMT_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Implemented Pre-LN Architecture for better efficiency

        # Self-Attention
        src1 = self.norm1(src)
        if src_bert is not None:
            if src_attn_bias is None:
                src_attn_bias = self.bert_attn_bias(src1, src_bert, key_padding_mask=src_key_padding_mask, attn_mask = src_mask)
            else:
                src_attn_bias = src_attn_bias + self.bert_attn_bias(src1, src_bert, key_padding_mask=src_key_padding_mask, attn_mask = src_mask)
        src1 = self.self_attn(src1, src1, src1, attn_mask=src_mask, attn_bias=src_attn_bias, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src1)

        # FFN
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

class KINMT_TransformerDecoderLayer(Module):

    def __init__(self, d_model, nhead,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 bert=True,
                 gpt=True):
        super(KINMT_TransformerDecoderLayer, self).__init__()
        self.d_model=d_model


        if bert:
            self.src_attn = MultiheadAttention(d_model, nhead, scale_factor=3, dropout=dropout)
            self.bert_attn_bias = MultiheadAttentionBias(d_model, nhead, scale_factor=3)
        else:
            self.src_attn = MultiheadAttention(d_model, nhead, scale_factor=2, dropout=dropout)
        if gpt:
            self.tgt_attn = MultiheadAttention(d_model, nhead, scale_factor=3, dropout=dropout)
            self.gpt_attn_bias = MultiheadAttentionBias(d_model, nhead, scale_factor=3)
        else:
            self.tgt_attn = MultiheadAttention(d_model, nhead, scale_factor=2, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = dropout
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)
        self.norm3 = FusedLayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(KINMT_TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                src_attn_bias: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                tgt_attn_bias: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None, tgt_gpt: Optional[Tensor] = None, decoding=False) -> Tuple[Tensor,Tensor]:

        # Self-Attention
        tgt1 = self.norm1(tgt)
        if tgt_gpt is not None:
            if tgt_attn_bias is None:
                tgt_attn_bias = self.gpt_attn_bias(tgt1, tgt_gpt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            else:
                tgt_attn_bias = tgt_attn_bias + self.gpt_attn_bias(tgt1, tgt_gpt, attn_mask=tgt_mask, attn_bias=tgt_attn_bias, key_padding_mask=tgt_key_padding_mask)
        tgt1 = self.tgt_attn(tgt1, tgt1, tgt1, attn_mask=tgt_mask, attn_bias=tgt_attn_bias, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt +  F.dropout(tgt1, p=self.dropout, training=self.training)

        # Source Attention (i.e. Encoder-Decoder Attention)
        tgt1 = self.norm2(tgt)
        if src_bert is not None:
            if src_attn_bias is None:
                src_attn_bias = self.bert_attn_bias(tgt1, src_bert, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            else:
                src_attn_bias = src_attn_bias + self.bert_attn_bias(tgt1, src_bert, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        tgt1, src_attn_weights = self.src_attn(tgt1, src, src, attn_mask=src_mask, attn_bias=src_attn_bias,
                                               key_padding_mask=src_key_padding_mask, need_weights=decoding)
        tgt = tgt +  F.dropout(tgt1, p=self.dropout, training=self.training)

        # FFN
        tgt1 = self.norm3(tgt)
        tgt1 = self.activation(self.linear1(tgt1))
        tgt1 = F.dropout(tgt1, p=self.dropout, training=self.training)
        tgt1 = self.linear2(tgt1)
        tgt = tgt + F.dropout(tgt1, p=self.dropout, training=self.training)

        return tgt, src_attn_weights

class KINMT_TransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer: KINMT_TransformerEncoderLayer, num_layers, norm=None):
        super(KINMT_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None) -> Tensor:

        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                         src_bert=src_bert)

        if self.norm is not None:
            output = self.norm(output)

        return output

class KINMT_TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer: KINMT_TransformerDecoderLayer, num_layers, norm=None):
        super(KINMT_TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, src_attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None,
                tgt_gpt: Optional[Tensor] = None,
                decoding=False) -> Tuple[Tensor, Tensor]:
        output = tgt
        src_attn_weights = None
        for mod in self.layers:
            (output, src_attn_weights) = mod(src, output,
                                         src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                                         tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask,
                                         src_bert=src_bert,
                                         tgt_gpt=tgt_gpt,
                                         decoding=decoding)

        if self.norm is not None:
            output = self.norm(output)

        return (output, src_attn_weights)

class KINMT_Transformer(Module):

    def __init__(self, encoder_layer: KINMT_TransformerEncoderLayer, decoder_layer: KINMT_TransformerDecoderLayer, num_encoder_layers, num_decoder_layers,
                 encoder_norm=None, decoder_norm=None,
                 bert=True, gpt=True, use_cross_pos_attn=True):
        super(KINMT_Transformer, self).__init__()
        self.encoder = KINMT_TransformerEncoder(encoder_layer, num_encoder_layers, norm=encoder_norm)
        self.decoder = KINMT_TransformerDecoder(decoder_layer, num_decoder_layers, norm=decoder_norm)
        self.use_cross_pos_attn = use_cross_pos_attn
        if self.use_cross_pos_attn:
            self.tgt_to_src_pos_attn = CrossAttentionPositionalEncoder(decoder_layer.tgt_attn.embed_dim,
                                                                       decoder_layer.tgt_attn.num_heads)
        if bert:
            self.bert_norm = FusedLayerNorm(decoder_layer.d_model)
        if gpt:
            self.gpt_norm = FusedLayerNorm(decoder_layer.d_model)
    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, src_attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None,
                tgt_gpt: Optional[Tensor] = None) -> Tensor:
        if src_bert is not None:
            src_bert = self.bert_norm(src_bert)
        if tgt_gpt is not None:
            tgt_gpt = self.gpt_norm(tgt_gpt)
        src = self.encoder(src, src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                src_bert=src_bert)
        if self.use_cross_pos_attn:
            src_attn_bias = self.tgt_to_src_pos_attn(tgt, src)
        else:
            src_attn_bias = None
        return self.decoder(src, tgt,
                           src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                           tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask,
                           src_bert=src_bert,
                           tgt_gpt=tgt_gpt, decoding=False)[0]


    def encode(self, src: Tensor,
                src_mask: Optional[Tensor] = None, src_attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None) -> Tensor:
        if src_bert is not None:
            src_bert = self.bert_norm(src_bert)
        return self.encoder(src, src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask, src_bert=src_bert)

    def decode(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None,
                tgt_gpt: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.use_cross_pos_attn:
            src_attn_bias = self.tgt_to_src_pos_attn(tgt, src)
        else:
            src_attn_bias = None
        if src_bert is not None:
            src_bert = self.bert_norm(src_bert)
        if tgt_gpt is not None:
            tgt_gpt = self.gpt_norm(tgt_gpt)
        return self.decoder(src, tgt,
                            src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                            tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask,
                            src_bert=src_bert,
                            tgt_gpt=tgt_gpt,
                            decoding=True)

class ASR_Transformer(Module):

    def __init__(self, decoder_layer: KINMT_TransformerDecoderLayer, num_decoder_layers, decoder_norm=None,
                 gpt=True, use_cross_pos_attn=True,
                 max_seq_len = 1024,
                 tupe_rel_pos_bins = 256,
                 tupe_max_rel_pos = 256):
        super(ASR_Transformer, self).__init__()
        self.decoder = KINMT_TransformerDecoder(decoder_layer, num_decoder_layers, norm=decoder_norm)
        self.use_cross_pos_attn = use_cross_pos_attn
        if self.use_cross_pos_attn:
            self.tgt_to_src_pos_attn = CrossAttentionPositionalEncoder(decoder_layer.tgt_attn.embed_dim,
                                                                       decoder_layer.tgt_attn.num_heads,
                                                                       max_seq_len = max_seq_len,
                                                                       tupe_rel_pos_bins = tupe_rel_pos_bins,
                                                                       tupe_max_rel_pos = tupe_max_rel_pos)
        if gpt:
            self.gpt_norm = FusedLayerNorm(decoder_layer.d_model)
    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_gpt: Optional[Tensor] = None) -> Tensor:
        if tgt_gpt is not None:
            tgt_gpt = self.gpt_norm(tgt_gpt)
        if self.use_cross_pos_attn:
            src_attn_bias = self.tgt_to_src_pos_attn(tgt, src)
        else:
            src_attn_bias = None
        return self.decoder(src, tgt,
                           src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                           tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask,
                           tgt_gpt=tgt_gpt, decoding=False)[0]

    def decode(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_gpt: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.use_cross_pos_attn:
            src_attn_bias = self.tgt_to_src_pos_attn(tgt, src)
        else:
            src_attn_bias = None
        if tgt_gpt is not None:
            tgt_gpt = self.gpt_norm(tgt_gpt)
        return self.decoder(src, tgt,
                            src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                            tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask,
                            tgt_gpt=tgt_gpt,
                            decoding=True)