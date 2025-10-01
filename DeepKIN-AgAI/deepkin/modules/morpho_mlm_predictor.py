from __future__ import print_function, division

# Ignore warnings
import warnings

from deepkin.modules.flex_modules import FlexTokenHead
from torch.amp import custom_fwd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F


class KinyaMLMPredictor(nn.Module):
    def __init__(self, stem_embedding_weights,
                 pos_tag_embedding_weights,
                 lm_morph_embedding_weights,
                 affixes_embedding_weights,
                 main_d_model,
                 layernorm_epsilon,
                 dropout=0.1):
        super(KinyaMLMPredictor, self).__init__()
        self.stem_head = FlexTokenHead(stem_embedding_weights, main_d_model, layernorm_epsilon, dropout=dropout)
        self.pos_tag_head = FlexTokenHead(pos_tag_embedding_weights, main_d_model, layernorm_epsilon, dropout=dropout)
        self.lm_morph_head = FlexTokenHead(lm_morph_embedding_weights, main_d_model, layernorm_epsilon, dropout=dropout)
        self.affixes_head = FlexTokenHead(affixes_embedding_weights, main_d_model, layernorm_epsilon, dropout=dropout)

    @custom_fwd(device_type='cuda')
    def forward(self, main_hidden_state, #(L, N, E)
                predicted_tokens_idx,
                predicted_tokens_affixes_idx,
                predicted_stems,
                predicted_pos_tags,
                predicted_lm_morphs,
                predicted_affixes_prob):
        # main_hidden_state: (L, N, E)
        main_hidden_state = main_hidden_state.permute(1, 0, 2).reshape(-1, main_hidden_state.shape[2])  # (L, N, E) -> (N, L, E) -> (NL, E)
        mopho_hidden_state = torch.index_select(main_hidden_state, 0, index=predicted_tokens_idx)
        stem_loss_avg = F.cross_entropy(self.stem_head(mopho_hidden_state), predicted_stems)
        pos_tag_loss_avg = F.cross_entropy(self.pos_tag_head(mopho_hidden_state), predicted_pos_tags)
        lm_morph_loss_avg = F.cross_entropy(self.lm_morph_head(mopho_hidden_state), predicted_lm_morphs)
        affixes_loss_avg = torch.tensor(0.0, device=main_hidden_state.device)
        if predicted_tokens_affixes_idx.nelement() > 0:
            affixes_hidden_state = torch.index_select(mopho_hidden_state, 0, index=predicted_tokens_affixes_idx)
            affixes_loss_avg = F.binary_cross_entropy_with_logits(self.affixes_head(affixes_hidden_state), predicted_affixes_prob)
        losses = [stem_loss_avg, pos_tag_loss_avg, lm_morph_loss_avg, affixes_loss_avg]
        return losses

    def logits_to_predictions(self, logits, max_top_predictions):
        top_preds_prob, top_preds = torch.topk(F.softmax(logits, dim=-1), max_top_predictions, dim=1)
        predictions = []
        predictions_prob = []
        for batch in range(top_preds.shape[0]):
            predictions.append(top_preds[batch, :].cpu().tolist())
            predictions_prob.append(top_preds_prob[batch, :].cpu().tolist())
        return predictions, predictions_prob

    def logits_to_multilabel_predictions(self, logits, max_top_predictions):
        top_preds_prob, top_preds = torch.topk(F.sigmoid(logits), max_top_predictions, dim=1)
        predictions = []
        predictions_prob = []
        for batch in range(top_preds.shape[0]):
            predictions.append(top_preds[batch, :].cpu().tolist())
            predictions_prob.append(top_preds_prob[batch, :].cpu().tolist())
        return predictions, predictions_prob

    def predict(self, main_hidden_state,
                predicted_tokens_idx,
                max_prediction_affixes=24,
                max_top_predictions=8):
        # main_hidden_state: (L, N, E)
        main_hidden_state = main_hidden_state.permute(1, 0, 2).reshape(-1, main_hidden_state.shape[2])  # (L, N, E) -> (N, L, E) -> (NL, E)
        mopho_hidden_state = torch.index_select(main_hidden_state, 0, index=predicted_tokens_idx)

        stem_predictions,stem_predictions_prob = self.logits_to_predictions(self.stem_head(mopho_hidden_state), max_top_predictions)
        pos_tag_predictions,pos_tag_predictions_prob = self.logits_to_predictions(self.pos_tag_head(mopho_hidden_state), max_top_predictions)
        lm_morph_predictions,lm_morph_predictions_prob = self.logits_to_predictions(self.lm_morph_head(mopho_hidden_state), max_top_predictions)
        affixes_predictions,affixes_predictions_prob = self.logits_to_multilabel_predictions(self.affixes_head(mopho_hidden_state), max_prediction_affixes)

        return ((stem_predictions, pos_tag_predictions, lm_morph_predictions, affixes_predictions),
                (stem_predictions_prob, pos_tag_predictions_prob, lm_morph_predictions_prob, affixes_predictions_prob))
