from __future__ import print_function, division

# Ignore warnings
import warnings

from deepkin.modules.flex_modules import FlexTokenHead
from deepkin.modules.losses import label_smoothed_nll_loss
from torch.amp import custom_fwd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTokenGenerator(nn.Module):
    def __init__(self, token_embedding_weights, tr_d_model, layernorm_epsilon, dropout=0.0):
        super(SingleTokenGenerator, self).__init__()
        self.token_head = FlexTokenHead(token_embedding_weights, token_embedding_weights, layernorm_epsilon, dropout=dropout)

    @custom_fwd(device_type='cuda')
    def forward(self, tr_hidden_state, tokens, input_sequence_lengths,
                epsilon_ls=0.1):
        tr_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = tr_hidden_state.size(0)
        # Last <EOS> token not processed
        sub = 1
        tr_hidden_state = [tr_hidden_state[i, :(input_sequence_lengths[i] - sub), :] for i in range(N)]

        tr_hidden_state = torch.cat(tr_hidden_state, dim=0)
        # (B,E), B=Batch Size = sum([l-1 for l in input_sequence_lengths])

        target_tokens = tokens.split(input_sequence_lengths)
        target_tokens = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_tokens)]
        target_tokens = torch.cat(target_tokens, dim=0)

        token_loss_avg, token_nll_loss_avg = label_smoothed_nll_loss(self.token_head(tr_hidden_state), target_tokens, epsilon_ls)

        losses = [token_loss_avg]

        return losses

    def predict(self, tr_hidden_state, input_sequence_lengths):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # No <EOS> at end, so no sub
        sub = 0
        hidden_states = [
            token_hidden_state[i, (input_sequence_lengths[i] - sub - 1):(input_sequence_lengths[i] - sub), :] for i in
            range(N)]
        batch_logits = torch.cat(hidden_states, dim=0)

        next_tokens = F.softmax(self.token_head(batch_logits), dim=1)

        return next_tokens
