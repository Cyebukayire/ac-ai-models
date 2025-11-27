from __future__ import print_function, division, annotations

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from deepkin.modules.flex_modules import FlexTokenHead
from deepkin.modules.losses import label_smoothed_nll_loss
from torch.amp import custom_fwd

import torch
import torch.nn as nn
import torch.nn.functional as F


class KinyaGenerator(nn.Module):
    def __init__(self, stem_embedding_weights,
                 pos_tag_embedding_weights,
                 lm_morph_embedding_weights,
                 affixes_embedding_weights,
                 main_d_model,
                 layernorm_epsilon,
                 dropout=0.1):
        super(KinyaGenerator, self).__init__()
        self.stem_head = FlexTokenHead(stem_embedding_weights, main_d_model, layernorm_epsilon, dropout=dropout)
        self.pos_tag_head = FlexTokenHead(pos_tag_embedding_weights, main_d_model, layernorm_epsilon, dropout=dropout)
        self.lm_morph_head = FlexTokenHead(lm_morph_embedding_weights, main_d_model, layernorm_epsilon, dropout=dropout)
        self.affixes_head = FlexTokenHead(affixes_embedding_weights, main_d_model, layernorm_epsilon, dropout=dropout)

    @custom_fwd(device_type='cuda')
    def forward(self, hidden_state, input_sequence_lengths,
                predicted_stems, predicted_pos_tags, predicted_lm_morphs, predicted_affixes_prob,
                epsilon_ls=0.1):
        hidden_state = hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = hidden_state.size(0)

        # Last <EOS> token not used for prediction
        sub = 1
        hidden_state = [hidden_state[i, :(input_sequence_lengths[i] - sub), :] for i in range(N)]

        hidden_state = torch.cat(hidden_state, dim=0)

        stem_loss_avg, _ = label_smoothed_nll_loss(F.log_softmax(self.stem_head(hidden_state), dim=1), predicted_stems, epsilon_ls)
        pos_tag_loss_avg, _ = label_smoothed_nll_loss(F.log_softmax(self.pos_tag_head(hidden_state), dim=1), predicted_pos_tags, epsilon_ls)
        lm_morph_loss_avg, _ = label_smoothed_nll_loss(F.log_softmax(self.lm_morph_head(hidden_state), dim=1), predicted_lm_morphs, epsilon_ls)
        affixes_loss_avg = F.binary_cross_entropy_with_logits(self.affixes_head(hidden_state), predicted_affixes_prob)

        losses = [stem_loss_avg, pos_tag_loss_avg, lm_morph_loss_avg, affixes_loss_avg]
        return losses

    def predict(self, tr_hidden_state, input_sequence_lengths):
        tr_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = tr_hidden_state.size(0)
        # No <EOS> at end, so no sub
        sub = 0
        hidden_states = [tr_hidden_state[i, (input_sequence_lengths[i] - sub - 1):(input_sequence_lengths[i] - sub), :] for i in range(N)]
        batch_logits = torch.cat(hidden_states, dim=0)  # (B,E)

        next_stems = F.softmax(self.stem_head(batch_logits), dim=1)
        next_pos_tags = F.softmax(self.pos_tag_head(batch_logits), dim=1)
        next_lm_morphs = F.softmax(self.lm_morph_head(batch_logits), dim=1)
        next_affixes = F.sigmoid(self.affixes_head(batch_logits))
        return (next_stems, next_pos_tags, next_lm_morphs, next_affixes)

    def score_kin(self, hidden_state, input_sequence_lengths,
                predicted_stems, predicted_pos_tags, predicted_lm_morphs, predicted_affixes_prob):
        hidden_state = hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = hidden_state.size(0)

        # Last <EOS> token not used for prediction
        sub = 1
        hidden_state = [hidden_state[i, :(input_sequence_lengths[i] - sub), :] for i in range(N)]

        hidden_state = torch.cat(hidden_state, dim=0)

        stem_loss_avg = F.nll_loss(F.log_softmax(self.stem_head(hidden_state), dim=1), predicted_stems)
        pos_tag_loss_avg = F.nll_loss(F.log_softmax(self.pos_tag_head(hidden_state), dim=1), predicted_pos_tags)
        lm_morph_loss_avg = F.nll_loss(F.log_softmax(self.lm_morph_head(hidden_state), dim=1), predicted_lm_morphs)
        affixes_loss_avg = F.binary_cross_entropy_with_logits(self.affixes_head(hidden_state), predicted_affixes_prob)

        losses = [stem_loss_avg, pos_tag_loss_avg, lm_morph_loss_avg, affixes_loss_avg]

        return (-sum(losses)).cpu().item()

