from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from torch.amp import custom_fwd
from torch.nn import Linear

from deepkin.models.kinspeak.layer_norm import FusedLayerNorm

class BaseHeadTransform(nn.Module):
    def __init__(self, tr_d_model: int, cls_ctxt_size: int, layernorm_epsilon: float):
        super(BaseHeadTransform, self).__init__()
        self.dense = Linear(tr_d_model, cls_ctxt_size)
        self.layerNorm = FusedLayerNorm(cls_ctxt_size, eps=layernorm_epsilon)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layerNorm(hidden_states)
        return hidden_states

class KINMT_HeadTransform(nn.Module):
    def __init__(self, tr_d_model: int, cls_ctxt_size: int, layernorm_epsilon: float, dropout: float = 0.3):
        super(KINMT_HeadTransform, self).__init__()
        self.dense = Linear(tr_d_model, cls_ctxt_size)
        self.layerNorm = FusedLayerNorm(cls_ctxt_size, eps=layernorm_epsilon)
        self.in_dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_dropout(x)
        x = self.dense(x)
        x = self.layerNorm(x)
        x = torch.nn.functional.relu(x)
        return x

class TokenClassificationHead(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float = 0.3, head_trunk: bool = False):
        super(TokenClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.head_trunk = head_trunk
        if self.head_trunk:
            self.trunk_dense = Linear(input_dim, inner_dim)
            self.trunk_layerNorm = FusedLayerNorm(inner_dim)
            self.trunk_activation_fn = torch.tanh
            self.trunk_dropout = nn.Dropout(p=pooler_dropout)
        self.out_dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = Linear(inner_dim if self.head_trunk else input_dim, num_classes)

    @custom_fwd(device_type='cuda')
    def forward(self, features: Tensor, input_sequence_lengths: List[int]) -> Tensor:
        # features.shape = S x N x E
        # Remove [CLS]
        # len already includes [CLS] in the sequence length count, so number of normal tokens here is (len-1)
        inputs = [features[1:len, i, :].contiguous().view(-1, self.input_dim) for i, len in
                  enumerate(input_sequence_lengths)]
        x = torch.cat(inputs, 0) #  B x E
        if self.head_trunk:
            x = self.trunk_dropout(x)
            x = self.trunk_dense(x)
            x = self.trunk_layerNorm(x)
            x = self.trunk_activation_fn(x)
        x = self.out_dropout(x)
        x = self.out_proj(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float = 0.0, head_trunk: bool = False):
        super(ClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.head_trunk = head_trunk
        if self.head_trunk:
            self.trunk_dense = Linear(input_dim, inner_dim)
            self.trunk_layerNorm = FusedLayerNorm(inner_dim)
            self.trunk_activation_fn = torch.tanh
            self.trunk_dropout = nn.Dropout(p=pooler_dropout)
        self.out_dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = Linear(inner_dim if self.head_trunk else input_dim, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        # features.shape = S x N x E
        x = features[0, :, :]  # Take [CLS]
        if self.head_trunk:
            x = self.trunk_dropout(x)
            x = self.trunk_dense(x)
            x = self.trunk_layerNorm(x)
            x = self.trunk_activation_fn(x)
        x = self.out_dropout(x)
        x = self.out_proj(x)
        return x

class GPTClassificationHead(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float = 0.0, head_trunk: bool = False):
        super(GPTClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.head_trunk = head_trunk
        if self.head_trunk:
            self.trunk_dense = Linear(input_dim, inner_dim)
            self.trunk_layerNorm = FusedLayerNorm(inner_dim)
            self.trunk_activation_fn = torch.tanh
            self.trunk_dropout = nn.Dropout(p=pooler_dropout)
        self.out_dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = Linear(inner_dim if self.head_trunk else input_dim, num_classes)

    def forward(self, features: Tensor, input_sequence_lengths: List[int]) -> Tensor:
        # features.shape = L x N x E
        x = torch.cat([features[(ln-1):ln, n, :] for n,ln in enumerate(input_sequence_lengths)], 0)
        if self.head_trunk:
            x = self.trunk_dropout(x)
            x = self.trunk_dense(x)
            x = self.trunk_layerNorm(x)
            x = self.trunk_activation_fn(x)
        x = self.out_dropout(x)
        x = self.out_proj(x)
        return x


class Engl_TokenGenerator(nn.Module):
    def __init__(self, token_embedding_weights: Tensor, tr_d_model: int, layernorm_epsilon: float,
                 copy_tokens_embedding_weights: Optional[Tensor] = None):
        super(Engl_TokenGenerator, self).__init__()

        self.token_transform = BaseHeadTransform(tr_d_model, token_embedding_weights.size(1), layernorm_epsilon)
        self.token_decoder = Linear(token_embedding_weights.size(1), token_embedding_weights.size(0), bias=False)
        self.token_decoder.weight = token_embedding_weights
        self.token_decoder_bias = nn.Parameter(torch.zeros(token_embedding_weights.size(0)))

        if copy_tokens_embedding_weights is None:
            self.copy_tokens = False
        else:
            self.copy_tokens = True
            self.copy_tokens_transform = BaseHeadTransform(tr_d_model, copy_tokens_embedding_weights.size(1), layernorm_epsilon)
            self.copy_tokens_decoder = Linear(copy_tokens_embedding_weights.size(1), copy_tokens_embedding_weights.size(0), bias=False)
            self.copy_tokens_decoder.weight = copy_tokens_embedding_weights
            self.copy_tokens_decoder_bias = nn.Parameter(torch.zeros(copy_tokens_embedding_weights.size(0)))


    @torch.jit.script
    def label_smoothed_nll_loss(self, lprobs: Tensor, target: Tensor, epsilon: float,
                                ignore_index: Union[Tensor, None] = None, reduce: bool = True) -> Tuple[Tensor, Tensor]:
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()
        eps_i = epsilon / (lprobs.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    @custom_fwd(device_type='cuda')
    def forward(self, tr_hidden_state: Tensor, tokens: Tensor, input_sequence_lengths: List[int],
                copy_tokens_prob: Optional[Tensor] = None,
                epsilon_ls: float = 0.1):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # Last <EOS> token not processed
        sub = 1
        hidden_states = [token_hidden_state[i, :(input_sequence_lengths[i] - sub), :] for i in range(N)]

        batch_logits = torch.cat(hidden_states, dim=0)
        # (B,E), B=Batch Size = sum([l-1 for l in input_sequence_lengths])

        target_tokens = tokens.split(input_sequence_lengths)
        target_tokens = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_tokens)]
        target_tokens = torch.cat(target_tokens, dim=0)

        token_scores = self.token_transform(batch_logits)
        token_scores = self.token_decoder(token_scores) + self.token_decoder_bias
        token_scores = F.log_softmax(token_scores, dim=1)
        # token_loss_avg = F.nll_loss(token_scores, target_tokens)
        token_loss_avg, token_nll_loss_avg = self.label_smoothed_nll_loss(token_scores, target_tokens, epsilon_ls)

        if self.copy_tokens:
            target_copy_tokens_prob = copy_tokens_prob.split(input_sequence_lengths)
            target_copy_tokens_prob = [tns[1:length,:] for length, tns in zip(input_sequence_lengths, target_copy_tokens_prob)]
            target_copy_tokens_prob = torch.cat(target_copy_tokens_prob, dim=0)

            copy_tokens_predicted_state = self.copy_tokens_transform(batch_logits)
            copy_tokens_scores = self.copy_tokens_decoder(copy_tokens_predicted_state) + self.copy_tokens_decoder_bias
            copy_tokens_loss_avg = F.binary_cross_entropy_with_logits(copy_tokens_scores, target_copy_tokens_prob)

            losses = [token_loss_avg, copy_tokens_loss_avg]
            nll_losses = [token_nll_loss_avg, copy_tokens_loss_avg]
        else:
            losses = [token_loss_avg]
            nll_losses = [token_nll_loss_avg]

        return losses, nll_losses

    def predict(self, tr_hidden_state: Tensor, input_sequence_lengths: List[int]) -> Tuple[Tensor,Union[Tensor,None]]:
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # No <EOS> at end, so no sub
        sub = 0
        hidden_states = [token_hidden_state[i, (input_sequence_lengths[i] - sub - 1):(input_sequence_lengths[i] - sub), :] for i in range(N)]
        batch_logits = torch.cat(hidden_states, dim=0)

        token_scores = self.token_transform(batch_logits)
        token_scores = self.token_decoder(token_scores) + self.token_decoder_bias
        # next_tokens = F.softmax(token_scores, dim=1)

        if self.copy_tokens:
            copy_tokens_scores = self.copy_tokens_transform(batch_logits)
            copy_tokens_scores = self.copy_tokens_decoder(copy_tokens_scores) + self.copy_tokens_decoder_bias
            next_copy_prob = F.sigmoid(copy_tokens_scores)

            return token_scores, next_copy_prob
        else:
            return token_scores, None
