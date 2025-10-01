import math
from typing import List, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from apex.normalization import FusedLayerNorm
from torch.amp import custom_fwd
from torch.nn import Linear
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence

from deepkin.clib.libkinlp.kinlpy import BOS_ID, EOS_ID
from deepkin.models.kinspeak.data import SampleConfig
from deepkin.models.kinspeak.utils import Wav2Vec2Config, compute_mask_indices, index_put, compute_padding_mask, \
    buffered_arange
from deepkin.models.kinspeak.base_modules import BaseHeadTransform
from deepkin.models.kinspeak.conformer import ConformerEncoderLayer
from deepkin.modules.losses import label_smoothed_nll_loss
from deepkin.modules.param_init import init_bert_params
from deepkin.modules.position_encoding import PositionEncoding


class ContextEncoder(torch.nn.Module):

    def __init__(self, cfg:Wav2Vec2Config,
                 log_mel_spectrogram_dim = 80,
                 channel_multiple = 32,
                 embed_dim = 768,
                 encoder_ffn_dim = 3072,
                 attention_heads = 8,
                 dropout = 0.1,
                 num_conformer_layers=16):
        """
        Args:
        """
        super(ContextEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.conv_sub_sampling_module = nn.Sequential(
            nn.Conv2d(1, channel_multiple, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.SiLU(),
            nn.Conv2d(channel_multiple, channel_multiple, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.SiLU())
        self.features_dim = channel_multiple * self.compute_subsampling_output_sizes([log_mel_spectrogram_dim])[0]

        # print('features_dim:', self.features_dim)
        # print('embed_dim:', self.embed_dim)

        self.conformer_layers = nn.ModuleList(
            [ConformerEncoderLayer(self.embed_dim, encoder_ffn_dim, attention_heads, dropout)
             for _ in range(num_conformer_layers)]
        )

        self.pos_encoder = PositionEncoding(self.embed_dim, attention_heads, 10240, 512, 1024, False)

        self.cfg = cfg

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(self.features_dim).uniform_()
        )
        self.layer_norm = FusedLayerNorm(self.features_dim)
        self.project_input = Linear(self.features_dim, self.embed_dim)
        self.project_target = Linear(self.features_dim, self.embed_dim)
        self.dropout_input = nn.Dropout(dropout)

    def compute_subsampling_output_sizes(self, input_sizes):
        kernel = 3
        stride = 2
        padding = 1
        output_sizes = []
        for l in input_sizes:
            l1 = int(math.floor(((l + (2 * padding) - (kernel - 1) - 1) / stride) + 1))
            l2 = int(math.floor(((l1 + (2 * padding) - (kernel - 1) - 1) / stride) + 1))
            output_sizes.append(l2)
        return output_sizes

    def apply_mask(
            self,
            x,
            padding_mask,
            mask_indices=None,
    ):
        B, T, C = x.shape
        # print('B={}, T={}, C={}'.format(B, T, C))

        if self.cfg.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.cfg.mask_prob,
                    self.cfg.mask_length,
                    self.cfg.mask_selection,
                    self.cfg.mask_other,
                    min_masks=2,
                    no_overlap=self.cfg.no_mask_overlap,
                    min_space=self.cfg.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        return x, mask_indices

    def forward(self, log_mel_spectrograms: torch.Tensor,
                log_mel_spectrogram_lengths: List[int], pretraining: bool = False):
        """
        Args:
            log_mel_spectrograms: Tensor of shape (N,F,L)
            log_mel_spectrogram_lengths: List[int]
            pretraining: Optional flag indicating whether we are pretraining the model
        Returns:
            Tensor of shape T,N,F
        """

        # nonfinite = (~(log_mel_spectrograms.isfinite())).sum().item()
        # finite = ((log_mel_spectrograms.isfinite())).sum().item()
        # if nonfinite > 0:
        #     print("Finite:", finite, "Non-finite:", nonfinite, "{:.1f} %".format(100.0 * nonfinite / (finite + nonfinite)), flush=True)

        # Convolutional subsampling
        x = log_mel_spectrograms.unsqueeze(1)  # (N,1,F,L)
        x = self.conv_sub_sampling_module(x)  # (N,C',F',L')

        N, C, F, L = x.shape
        # print('N={}, C={}, F={}, L={}'.format(N, C, F, L), flush=True)
        x = x.reshape(N, -1, L)  # (N,CF,L)
        x = x.transpose(-2, -1)  # (N,L,CF) or (B,T,C)

        x = self.layer_norm(x)  # (B,T,C)
        padding_mask = compute_padding_mask(self.compute_subsampling_output_sizes(log_mel_spectrogram_lengths),
                                            x.device)

        mask_indices = None
        y = None
        if pretraining:
            unmasked_features = x.clone()
            x, mask_indices = self.apply_mask(x, padding_mask)
            y = unmasked_features[mask_indices].view(
                unmasked_features.size(0), -1, unmasked_features.size(-1)
            )
            y = self.project_target(y) # (N,M,F)
        x = self.project_input(x)  # (B,T,E)

        if not pretraining:
            x = self.dropout_input(x)

        x = x.transpose(0, 1)  # (B,T,E) --> (T,B,E)
        pos_bias = self.pos_encoder(x)
        (attn, layer_result) = (None, None)
        for i, layer in enumerate(self.conformer_layers):
            x, (attn, layer_result) = layer(x, encoder_padding_mask=padding_mask, attn_bias=pos_bias)

        return (x,  # (T,N,F)
                (attn, layer_result),
                mask_indices,  # bool (B,T)
                y)  # (N,M,F)
    def sample_negatives(self, y, num_negatives=100, padding_count=None):
        num = y.size(1)
        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"
            tszs = (
                buffered_arange(num)
                .unsqueeze(-1)
                .expand(-1, num_negatives)
                .flatten()
            )
            neg_idxs = torch.randint(
                low=0, high=high - 1, size=(bsz, num_negatives * num)
            )
            neg_idxs[neg_idxs >= tszs] += 1
        neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, num_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC; i.e. (K,N,M,F)
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives, logit_temp=0.1):
        # x: N,M,F
        # y: N,M,F
        # negatives: K,N,M,F
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0) # 1,N,M,F
        targets = torch.cat([y, negatives], dim=0) # 1+K,N,M,F

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1) # 1+K,N,M
        logits = logits / logit_temp
        logits = logits.type_as(x)

        if neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                self._inftensor = torch.tensor(float("-inf"))
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits
    def get_logits(self, output):
        logits = output # 1+K,N,M
        logits = logits.transpose(0, 2) # 1+K,N,M -> M,N,1+K
        logits = logits.reshape(-1, logits.size(-1)) # MN,1+K
        return logits #

    def get_targets(self, output): # 1+K,N,M
        return output.new_zeros(output.size(1) * output.size(2), dtype=torch.long) # (MN) i.e. the target label index is always at position 0, thus all zeroes

    def contrastive_loss(self, x, y, mask_indices, num_negatives=100, logit_temp=0.1, padding_count=None):
        x = x.transpose(0, 1) # (T,N,F) -> (N,T,F)
        x = x[mask_indices].view(x.size(0), -1, x.size(-1)) # (N,M,F); M: masked indices; y: (N,M,F)
        negs, _ = self.sample_negatives(y, num_negatives=num_negatives, padding_count=padding_count)
        output = self.compute_preds(x, y, negs, logit_temp=logit_temp) # 1+K,N,M
        logits = self.get_logits(output) # (MN,1+K)
        target = self.get_targets(output) # (MN)
        sample_size = mask_indices.sum()
        loss = F.cross_entropy(logits, target, reduction="sum") / sample_size
        return loss

class ASR_Syllabe_Predictor(nn.Module):
    def __init__(self, syllabe_embedding_weights,
                 tr_d_model,
                 layernorm_epsilon,
                 target_blank_id):
        super(ASR_Syllabe_Predictor, self).__init__()

        self.target_blank_id = target_blank_id

        self.acoustic_transform = BaseHeadTransform(tr_d_model, syllabe_embedding_weights.size(1), layernorm_epsilon)
        self.acoustic_decoder = Linear(syllabe_embedding_weights.size(1), syllabe_embedding_weights.size(0), bias=False)
        self.acoustic_decoder.weight = syllabe_embedding_weights
        self.acoustic_decoder_bias = nn.Parameter(torch.zeros(syllabe_embedding_weights.size(0)))

        self.syllabe_transform = BaseHeadTransform(tr_d_model, syllabe_embedding_weights.size(1), layernorm_epsilon)
        self.syllabe_decoder = Linear(syllabe_embedding_weights.size(1), syllabe_embedding_weights.size(0), bias=False)
        self.syllabe_decoder.weight = syllabe_embedding_weights
        self.syllabe_decoder_bias = nn.Parameter(torch.zeros(syllabe_embedding_weights.size(0)))

        self.apply(init_bert_params)

    @custom_fwd(device_type='cuda')
    def forward(self, source_encoder_output, source_encoder_output_lengths,
                target_decoder_hidden_state,
                syllabe_ids, syllabe_id_lengths,
                epsilon_ls=0.1):
        encoder_scores = self.acoustic_transform(source_encoder_output)
        encoder_scores = self.acoustic_decoder(encoder_scores) + self.acoustic_decoder_bias
        log_probs = F.log_softmax(encoder_scores, dim=-1)
        ctc_loss = F.ctc_loss(log_probs, syllabe_ids.to(dtype=torch.int32),
                              torch.tensor(source_encoder_output_lengths, dtype=torch.int32),
                              torch.tensor(syllabe_id_lengths, dtype=torch.int32),
                              blank=self.target_blank_id, reduction='mean', zero_infinity=True)

        # print('source_encoder_output_lengths:', source_encoder_output_lengths, flush=True)
        # print('syllabe_id_lengths:', syllabe_id_lengths, flush=True)
        # print('ctc_loss:', ctc_loss, flush=True)

        syllabe_hidden_state = target_decoder_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = syllabe_hidden_state.size(0)
        # Last <EOS> syllabe not processed
        sub = 1
        hidden_states = [syllabe_hidden_state[i, :(syllabe_id_lengths[i] - sub), :] for i in range(N)]

        batch_logits = torch.cat(hidden_states, dim=0)
        # (B,E), B=Batch Size = sum([l-1 for l in input_sequence_lengths])

        target_syllabes = syllabe_ids.split(syllabe_id_lengths)
        target_syllabes = [tns[1:length] for length, tns in zip(syllabe_id_lengths, target_syllabes)]
        target_syllabes = torch.cat(target_syllabes, dim=0)

        syllabe_scores = self.syllabe_transform(batch_logits)
        syllabe_scores = self.syllabe_decoder(syllabe_scores) + self.syllabe_decoder_bias
        syllabe_scores = F.log_softmax(syllabe_scores, dim=-1)
        # syllabe_loss_avg = F.nll_loss(syllabe_scores, target_syllabes)
        syllabe_loss_avg, syllabe_nll_loss_avg = label_smoothed_nll_loss(syllabe_scores, target_syllabes, epsilon_ls)

        # print('syllabe_loss_avg:', syllabe_loss_avg, flush=True)
        # print('syllabe_nll_loss_avg:', syllabe_nll_loss_avg, flush=True)

        return ctc_loss, syllabe_loss_avg, syllabe_nll_loss_avg

    def predict_acoustic(self, source_encoder_output):
        encoder_scores = self.acoustic_transform(source_encoder_output)
        encoder_scores = self.acoustic_decoder(encoder_scores) + self.acoustic_decoder_bias
        acoustic_log_probs = F.log_softmax(encoder_scores, dim=-1)
        return acoustic_log_probs

    def predict_syllabe_lm(self, target_decoder_hidden_state):
        syllabe_scores = self.syllabe_transform(target_decoder_hidden_state)
        syllabe_scores = self.syllabe_decoder(syllabe_scores) + self.syllabe_decoder_bias
        syllabe_log_probs = F.log_softmax(syllabe_scores, dim=-1)
        return syllabe_log_probs

    def score_syllabe_lm(self, target_decoder_hidden_state,
                syllabe_ids, syllabe_id_lengths):
        syllabe_hidden_state = target_decoder_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = syllabe_hidden_state.size(0)
        # Last <EOS> syllabe not processed
        sub = 1
        hidden_states = [syllabe_hidden_state[i, :(syllabe_id_lengths[i] - sub), :] for i in range(N)]

        batch_logits = torch.cat(hidden_states, dim=0)
        # (B,E), B=Batch Size = sum([l-1 for l in input_sequence_lengths])

        target_syllabes = syllabe_ids.split(syllabe_id_lengths)
        target_syllabes = [tns[1:length] for length, tns in zip(syllabe_id_lengths, target_syllabes)]
        target_syllabes = torch.cat(target_syllabes, dim=0) # (C)

        syllabe_scores = self.syllabe_transform(batch_logits)
        syllabe_scores = self.syllabe_decoder(syllabe_scores) + self.syllabe_decoder_bias
        # (N,C)
        scores = -F.nll_loss(syllabe_scores, target_syllabes, reduction='none')
        # (N)
        return scores

class KinspeakASRModel(torch.nn.Module):
    def __init__(self, cfg:Wav2Vec2Config,
                 target_vocab_size,
                 target_blank_id,
                 use_syllabe_gpt=False,
                 pretrained_ctxt_encoder_file=None):
        super(KinspeakASRModel, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.target_blank_id = target_blank_id
        self.use_syllabe_gpt = use_syllabe_gpt
        self.ctxt_encoder = ContextEncoder(cfg)
        if pretrained_ctxt_encoder_file is not None:
            state_dict = torch.load(pretrained_ctxt_encoder_file, map_location=torch.device('cpu'))
            self.ctxt_encoder.load_state_dict(state_dict['model_state_dict'])
        del self.ctxt_encoder.project_target
        self.encoder_projection = Linear(self.ctxt_encoder.embed_dim, self.ctxt_encoder.embed_dim)
        self.encoder_projection.apply(init_bert_params)
        self.acoustic_transform = BaseHeadTransform(self.ctxt_encoder.embed_dim,
                                                    self.ctxt_encoder.embed_dim,
                                                    1e-6)
        self.acoustic_decoder = Linear(self.ctxt_encoder.embed_dim, self.target_vocab_size)
        self.acoustic_transform.apply(init_bert_params)
        self.acoustic_decoder.apply(init_bert_params)

    def encoder_parameters(self) -> Iterator[Parameter]:
        for mod in [self.ctxt_encoder, self.encoder_projection]:
            for param in mod.parameters():
                yield param

    def forward(self, log_mel_spectrograms: torch.Tensor, #log_mel_spectrograms: (N,F,L)
                log_mel_spectrogram_lengths: List[int],
                target_syllabe_ids:torch.Tensor, target_syllabe_id_lengths:List[int],
                target_syllabe_ids_with_eos=True,
                target_syllabe_gpt_output = None):
        source_encoder_output_lengths = self.ctxt_encoder.compute_subsampling_output_sizes(log_mel_spectrogram_lengths)
        (source_encoder_output, (attn, layer_result), mask_indices, y) = self.ctxt_encoder(log_mel_spectrograms, log_mel_spectrogram_lengths, pretraining = False) # (B,T,C)
        source_encoder_output = self.encoder_projection(source_encoder_output)  # (T,N,C) (N:batch_size,T:input_lengths,C:pred_classes)
        encoder_scores = self.acoustic_transform(source_encoder_output)
        encoder_scores = self.acoustic_decoder(encoder_scores)
        log_probs = F.log_softmax(encoder_scores, dim=-1)
        ctc_loss = F.ctc_loss(log_probs, target_syllabe_ids.to(dtype=torch.int32),
                              torch.tensor(source_encoder_output_lengths, dtype=torch.int32),
                              torch.tensor(target_syllabe_id_lengths, dtype=torch.int32),
                              blank=self.target_blank_id, reduction='mean', zero_infinity=True)
        return ctc_loss

    def encode(self, log_mel_spectrograms: torch.Tensor, #log_mel_spectrograms: (N,F,L)
                log_mel_spectrogram_lengths: List[int]):
        source_encoder_output_lengths = self.ctxt_encoder.compute_subsampling_output_sizes(log_mel_spectrogram_lengths)
        (source_encoder_output, (attn, layer_result), mask_indices, y) = self.ctxt_encoder(log_mel_spectrograms, log_mel_spectrogram_lengths, pretraining = False) # (B,T,C)
        source_encoder_output = self.encoder_projection(source_encoder_output) # (T,N,C) (N:batch_size,T:input_lengths,C:pred_classes)
        return source_encoder_output, source_encoder_output_lengths

    def predict_acoustic(self, source_encoder_output):
        encoder_scores = self.acoustic_transform(source_encoder_output)
        encoder_scores = self.acoustic_decoder(encoder_scores)
        acoustic_log_probs = F.log_softmax(encoder_scores, dim=-1)
        return acoustic_log_probs

from deepkin.data.syllabe_vocab import build_kinyarwanda_dictionary_trie, id_sequence_to_text, syllbe_vocab_size, BLANK_ID
from deepkin.models.kinspeak.fast_ctc_beam_search import ctc_beam_search

class KinspeakASRInferenceModel(torch.nn.Module):
    def __init__(self, trained_model_file):
        import torchaudio.transforms as T
        super(KinspeakASRInferenceModel, self).__init__()
        cfg = SampleConfig()
        win_length = cfg.resample_rate * 25 // 1000  # 25ms
        hop_length = cfg.resample_rate * 10 // 1000  # 10ms
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=cfg.resample_rate, n_fft=cfg.n_fft,
                                           win_length=win_length,
                                           hop_length=hop_length, center=True, pad_mode="reflect", power=2.0,
                                           norm="slaney", onesided=True, n_mels=cfg.n_mels,
                                           mel_scale="htk", )
        self.asr_model = KinspeakASRModel(Wav2Vec2Config(), syllbe_vocab_size(), BLANK_ID,
                                          use_syllabe_gpt=False,
                                          pretrained_ctxt_encoder_file=None)
        state_dict = torch.load(trained_model_file, map_location='cpu')
        self.asr_model.load_state_dict(state_dict['model_state_dict'])
        self.dictionary = build_kinyarwanda_dictionary_trie(filename='KINLP/data/KinTokens.tsv')

    def forward(self, waveform) -> str:
        log_eps = 1e-36
        x = self.mel_spectrogram(waveform) # (1,F,L)
        x = torch.log(x + log_eps) # (1,F,L)
        x_lengths = [x.size(-1)]
        # print('x:',x.shape,flush=True)
        y, y_lengths = self.asr_model.encode(x, x_lengths)
        log_probs = self.asr_model.predict_acoustic(y)  # (T,N,C)
        log_probs = log_probs.squeeze() # (T,C)
        labels = ctc_beam_search(log_probs, beam_width = 24, blank_idx = BLANK_ID)
        KINSPEAK_PREDICTION = id_sequence_to_text([t for t in labels if (t != BOS_ID) and (t != EOS_ID)])
        KINSPEAK_PREDICTION = KINSPEAK_PREDICTION.replace('?', ' ? ').replace('.', ' . ').replace(',', ' , ').replace(
            '!',
            ' ! ').replace(
            ':', ' : ').replace(';', ' ; ')
        return KINSPEAK_PREDICTION




class KinspeakEmformerRNNT(torch.nn.Module):
    def __init__(self, target_vocab_size,
                 target_blank_id):
        super(KinspeakEmformerRNNT, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.target_blank_id = target_blank_id
        self.rnnt = torchaudio.models.emformer_rnnt_base(self.target_vocab_size)
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum", clamp=1.0)

    def forward(self, log_mel_spectrograms: torch.Tensor, #log_mel_spectrograms: (N,F,L)
                log_mel_spectrogram_lengths: List[int],
                target_syllabe_ids:torch.Tensor, target_syllabe_id_lengths:List[int],
                target_syllabe_ids_with_eos=True,
                target_syllabe_gpt_output = None):
        sources = log_mel_spectrograms.transpose(1,2)
        source_lengths = torch.tensor(log_mel_spectrogram_lengths).to(sources.device, dtype=torch.int32)
        target_syllabe_ids = target_syllabe_ids.to(dtype=torch.int32)
        targets = target_syllabe_ids.split(target_syllabe_id_lengths)
        targets = pad_sequence(targets, batch_first=True)
        target_lengths = torch.tensor(target_syllabe_id_lengths).to(targets.device, dtype=torch.int32)
        prepended_targets = targets.new_empty([targets.size(0), targets.size(1) + 1])
        prepended_targets[:, 1:] = targets
        prepended_targets[:, 0] = self.target_blank_id
        prepended_target_lengths = target_lengths + 1
        (output, src_lengths, _, __) = self.rnnt(sources, source_lengths, prepended_targets, prepended_target_lengths)
        loss = torchaudio.functional.rnnt_loss(output, targets, src_lengths-1, target_lengths, blank=self.target_blank_id, reduction = 'mean', clamp=1.0)
        return loss
