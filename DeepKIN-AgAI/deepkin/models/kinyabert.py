from __future__ import print_function, division, annotations

import gc
from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd

from deepkin.clib.libkinlp.kinlpy import KIN_PAD_IDX, ParsedFlexSentence, NUM_SPECIAL_TOKENS
from deepkin.clib.libkinlp.language_data import all_pos_tags
from deepkin.data.morpho_data import prepare_morpho_data_from_sentence, send_tuple_to, MorphoDataItem
from deepkin.data.morpho_qa_triple_data import DOCUMENT_TYPE_ID
from deepkin.modules.flex_modules import FlexConfig
from deepkin.modules.flex_transformers import TransformerEncoderLayer, TransformerEncoder
from deepkin.modules.morpho_encoder import KinyaEncoder
from deepkin.modules.morpho_mlm_predictor import KinyaMLMPredictor
from deepkin.utils.arguments import FlexArguments
from deepkin.modules.layer_norm import FusedRMSNorm
from deepkin.modules.param_init import init_bert_params
from deepkin.modules.position_encoding import PositionEncoding


class KinyaBERT_ClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.1, head_trunk=False):
        super(KinyaBERT_ClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.head_trunk = head_trunk
        if self.head_trunk:
            self.trunk_dense = nn.Linear(input_dim, inner_dim)
            self.trunk_layerNorm = FusedRMSNorm(inner_dim)
            self.trunk_activation_fn = torch.tanh
            self.trunk_dropout = nn.Dropout(p=pooler_dropout)
        self.out_dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim if self.head_trunk else input_dim, num_classes)

    def forward(self, features):
        # features.shape = S x N x E
        x = features[0, :, :]  # Take [CLS]
        if self.head_trunk:
            x = self.trunk_dropout(x)
            x = self.trunk_dense(x)
            x = self.trunk_layerNorm(x)
            x = self.trunk_activation_fn(x)
        x = self.out_dropout(x)
        x = self.out_proj(x)
        return x  # (B,V)


class KinyaBERT_TokenClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.3, head_trunk=False):
        super(KinyaBERT_TokenClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.head_trunk = head_trunk
        if self.head_trunk:
            self.trunk_dense = nn.Linear(input_dim, inner_dim)
            self.trunk_layerNorm = FusedRMSNorm(inner_dim)
            self.trunk_activation_fn = torch.tanh
            self.trunk_dropout = nn.Dropout(p=pooler_dropout)
        self.out_dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim if self.head_trunk else input_dim, num_classes)

    @custom_fwd(device_type='cuda')
    def forward(self, features, input_sequence_lengths):
        # features.shape = S x N x E
        # Remove [CLS]
        # len already includes [CLS] in the sequence length count, so number of normal tokens here is (len-1)
        inputs = [features[1:len, i, :].contiguous().view(-1, self.input_dim) for i, len in
                  enumerate(input_sequence_lengths)]
        x = torch.cat(inputs, 0)  # B x E
        if self.head_trunk:
            x = self.trunk_dropout(x)
            x = self.trunk_dense(x)
            x = self.trunk_layerNorm(x)
            x = self.trunk_activation_fn(x)
        x = self.out_dropout(x)
        x = self.out_proj(x)
        return x  # (B,V)


class KinyaBERTEncoder(nn.Module):
    def __init__(self, args: FlexArguments, cfg: FlexConfig):
        super(KinyaBERTEncoder, self).__init__()
        self.morpho_encoder: KinyaEncoder = KinyaEncoder(cfg, flex_hidden_dim=args.morpho_dim_hidden,
                                                         flex_dim_feedforward=args.morpho_dim_ffn,
                                                         flex_num_heads=args.morpho_num_heads,
                                                         flex_num_layers=args.morpho_num_layers,
                                                         flex_dropout=args.morpho_dropout,
                                                         sentence_stem_mult=args.sentence_stem_mult,
                                                         KIN_PAD_IDX=KIN_PAD_IDX)
        self.hidden_dim = self.morpho_encoder.main_hidden_dim
        self.num_heads = args.main_sequence_encoder_num_heads
        self.pos_encoder = PositionEncoding(self.hidden_dim,
                                            self.num_heads,
                                            args.main_sequence_encoder_max_seq_len,
                                            args.main_sequence_encoder_rel_pos_bins,
                                            args.main_sequence_encoder_max_rel_pos,
                                            True)
        encoder_layers = TransformerEncoderLayer(self.hidden_dim, args.main_sequence_encoder_num_heads,
                                                 dim_feedforward=args.main_sequence_encoder_dim_ffn,
                                                 dropout=args.main_sequence_encoder_dropout,
                                                 use_rms_norm=True)
        self.main_encoder = TransformerEncoder(encoder_layers, args.main_sequence_encoder_num_layers)
        self.apply(init_bert_params)

    @custom_fwd(device_type='cuda')
    def forward(self, stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded, input_sequence_lengths,
                main_masks_padded):
        seq_input = self.morpho_encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                        input_sequence_lengths)  # shape: (L,B,E)
        abs_pos_bias = self.pos_encoder(seq_input)
        output = self.main_encoder(seq_input, attn_bias=abs_pos_bias,
                                   src_key_padding_mask=main_masks_padded)  # Shape: L x N x E, with L = max sequence length
        return output  # (L, N, E)

    @custom_fwd(device_type='cuda')
    def embeddings(self, stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                   input_sequence_lengths, main_masks_padded):
        seq_input = self.morpho_encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                        input_sequence_lengths)  # shape: (L,B,E)
        abs_pos_bias = self.pos_encoder(seq_input)
        embed = self.main_encoder.embeddings(seq_input, attn_bias=abs_pos_bias, src_key_padding_mask=main_masks_padded)
        return embed  # List of Tensors of shape: L x N x E, with L = max sequence length


class KinyaBERT(nn.Module):
    def __init__(self, args: FlexArguments, cfg: FlexConfig):
        super(KinyaBERT, self).__init__()
        self.encoder = KinyaBERTEncoder(args, cfg)
        self.predictor = KinyaMLMPredictor(self.encoder.morpho_encoder.sentence_stem_embedding.weight,
                                           self.encoder.morpho_encoder.pos_tag_embedding.weight,
                                           self.encoder.morpho_encoder.lm_morph_embedding.weight,
                                           self.encoder.morpho_encoder.affixes_embedding.weight,
                                           self.encoder.morpho_encoder.main_hidden_dim,
                                           args.layernorm_epsilon,
                                           dropout=args.morpho_dropout)
        self.apply(init_bert_params)

    @custom_fwd(device_type='cuda')
    def forward(self, batch_idx: int, args: FlexArguments, model_cache, training_item: Tuple):
        (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
         input_sequence_lengths, main_masks_padded,
         predicted_tokens_idx,
         predicted_tokens_affixes_idx,
         predicted_stems,
         predicted_pos_tags,
         predicted_lm_morphs,
         predicted_affixes_prob) = training_item

        main_hidden_state = self.encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                         input_sequence_lengths, main_masks_padded)

        return self.predictor(main_hidden_state,  # (L, N, E)
                              predicted_tokens_idx,
                              predicted_tokens_affixes_idx,
                              predicted_stems,
                              predicted_pos_tags,
                              predicted_lm_morphs,
                              predicted_affixes_prob)

    def predict(self, stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                input_sequence_lengths, main_masks_padded, predicted_tokens_idx,
                max_prediction_affixes=24, max_top_predictions=8):
        main_hidden_state = self.encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                         input_sequence_lengths, main_masks_padded)

        return self.predictor.predict(main_hidden_state,
                                      predicted_tokens_idx,
                                      max_prediction_affixes=max_prediction_affixes,
                                      max_top_predictions=max_top_predictions)

    @custom_fwd(device_type='cuda')
    def get_embeddings(self, parsed_sentences: List[ParsedFlexSentence]):
        # Get device we are on
        device = self.encoder.morpho_encoder.word_stem_embedding.weight.device
        data_item = MorphoDataItem(f'{device}')
        input_sequence_lengths = []
        for sent in parsed_sentences:
            it = prepare_morpho_data_from_sentence(FlexConfig(), f'{device}', sent)
            data_item.add_bos().extend(it).add_eos()
            input_sequence_lengths.append(len(it) + 2)
        (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
         input_sequence_lengths, main_masks_padded) = send_tuple_to(data_item.to_simple_inputs(input_sequence_lengths),
                                                                    device)
        with torch.no_grad():
            output = self.encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                  input_sequence_lengths, main_masks_padded)
        # (L,N,E)
        # Return [CLS] embedding
        embed = output[0, :, :]
        return embed

    @staticmethod
    def from_pretrained(device: torch.device, pretrained_model_file: str, ret_args: bool = False) -> Union[
        KinyaBERT, Tuple[KinyaBERT, FlexArguments]]:
        cfg = FlexConfig()
        print(f'Loading pre-trained KinyaBERT model from {pretrained_model_file} ...')
        kb_state_dict = torch.load(pretrained_model_file, map_location=device)
        args = FlexArguments().from_dict(kb_state_dict['args'])
        pretrained_model = KinyaBERT(args, cfg).to(device)
        print(f"Pre-training steps: {kb_state_dict['lr_scheduler_state_dict']['num_iters'] // 1000}K")
        pretrained_model.load_state_dict(kb_state_dict['model_state_dict'])
        del kb_state_dict
        gc.collect()
        pretrained_model.float()
        pretrained_model.eval()
        if ret_args:
            return pretrained_model, args
        return pretrained_model


def PairwiseCELoss(scores):
    CELoss = nn.CrossEntropyLoss()
    logits = scores.view(2, -1).permute(1, 0)  # (B*2 1) -> (B 2)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return CELoss(logits, labels)


class KinyaColBERT(nn.Module):
    def __init__(self, args: FlexArguments, extend_embeddings: bool = False):
        super(KinyaColBERT, self).__init__()
        self.bert_encoder = KinyaBERTEncoder(args, FlexConfig())
        self.linear = nn.Linear(self.bert_encoder.hidden_dim, args.colbert_embedding_dim)
        self.apply(init_bert_params)
        # Skip punctuation marks
        self.similarity_metric = args.colbert_similarity_metric
        self.excluded_pos = {'PT', 'CJ', 'CA', 'CO', 'CP', 'IJ', 'PR', 'QU', 'RE', 'VC'}
        self.skip_pos_tag_ids = set([(i + NUM_SPECIAL_TOKENS) for i in range(len(all_pos_tags)) if
                                     (all_pos_tags[i]['cls_str'] in self.excluded_pos)])
        if extend_embeddings:
            self.bert_encoder.morpho_encoder.extend_with_special_tokens(2)

    @custom_fwd(device_type='cuda')
    def forward(self, batch_idx: int, args: FlexArguments, model_cache, training_item: Tuple):
        query_tuple, pos_neg_docs_tuple = training_item
        (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
         input_sequence_lengths, main_masks_padded) = query_tuple
        (doc_stems, doc_pos_tags, doc_lm_morphs, doc_affixes_padded,
         doc_morpho_masks_padded, doc_input_sequence_lengths, doc_main_masks_padded) = pos_neg_docs_tuple
        Query = self.bert_encoder(stems, pos_tags, lm_morphs, affixes_padded,
                                  morpho_masks_padded, input_sequence_lengths, main_masks_padded)
        # (Lq, B, E)
        Docs = self.bert_encoder(doc_stems, doc_pos_tags, doc_lm_morphs, doc_affixes_padded,
                                 doc_morpho_masks_padded, doc_input_sequence_lengths, doc_main_masks_padded)
        # (Ld, 2B, E)

        Query = Query.transpose(0, 1)  # (Lq,N,E) -> (N,Lq,E)
        Docs = Docs.transpose(0, 1)  # (Ld,2N,E) -> (2N,Ld,E)

        keep_d_dims = True
        d_mask = self.mask(doc_pos_tags, doc_input_sequence_lengths)

        Query = self.colbert_pooler(Query)
        Docs = self.colbert_pooler(Docs, mask=d_mask, keep_dims=keep_d_dims)

        scores_colbert = self.pairwise_score(Query.repeat(2, 1, 1), Docs)  # (2*B 1)
        loss = PairwiseCELoss(scores_colbert)
        return [loss]

    def get_colbert_embeddings(self, sequences: List[ParsedFlexSentence], seq_type: int):
        cfg = FlexConfig()
        device = self.linear.weight.device
        valid_sequences = []
        for sent in sequences:
            sent.trim(508)
            valid_sequences.append(sent)
        items = [prepare_morpho_data_from_sentence(cfg, f'{device}', seq).prepend_special_token(seq_type).add_bos() for
                 seq in sequences]
        data = MorphoDataItem(f'{device}')
        data_lengths = []
        for it in items:
            data.extend(it)
            data_lengths.append(len(it))
        data = data.to_simple_inputs(data_lengths)
        (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded, input_sequence_lengths,
         main_masks_padded) = send_tuple_to(data, device)
        data = self.bert_encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                 input_sequence_lengths, main_masks_padded)
        data = data.transpose(0, 1)  # (Lq,N,E) -> (N,Lq,E)
        keep_d_dims = True
        mask = self.mask(pos_tags, data_lengths) if (seq_type == DOCUMENT_TYPE_ID) else 1
        data = self.colbert_pooler(data, mask=mask, keep_dims=keep_d_dims) if (
                    seq_type == DOCUMENT_TYPE_ID) else self.colbert_pooler(data)
        return data

    def pairwise_score(self, Q, D):
        """ Max sim operator for pairwise loss
        1. tokens cos-sim: (B Lq H) X (B H Ld) = (B Lq Ld)
        2. max token-token cos-sim: (B Lq), the last dim indicates max of qd-cos-sim of q
        3. sum by batch: (B 1)
        """
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)
        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, pos_tags, input_sequence_lengths):
        L, B = max(input_sequence_lengths), len(input_sequence_lengths)
        mask = [[False for _ in range(L)] for __ in range(B)]
        idx = 0
        for i, ln in enumerate(input_sequence_lengths):
            for j in range(ln):
                x = pos_tags[idx]
                mask[i][j] = (x not in self.skip_pos_tag_ids) and (x != 0)
                idx += 1
        # mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in pos_tags.cpu().tolist()]
        mask = torch.tensor(mask, device=self.linear.weight.device).unsqueeze(2).float()
        return mask  # B L 1

    def colbert_pooler(self, tokens_last_hidden, mask=1, keep_dims=True):
        X = self.linear(tokens_last_hidden)
        X = X * mask  # for d
        X = F.normalize(X, p=2, dim=2)

        if not keep_dims:  # for d
            X, mask = X.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            X = [d[mask[idx]] for idx, d in enumerate(X)]
        return X

    @staticmethod
    def from_pretrained(device: torch.device, pretrained_model_file: str, ret_args=False) -> Union[
        KinyaColBERT, Tuple[KinyaColBERT, FlexArguments]]:
        print(f'Loading pre-trained KinyaColBERT model from {pretrained_model_file} ...')
        kb_state_dict = torch.load(pretrained_model_file, map_location=device)
        args = FlexArguments().from_dict(kb_state_dict['args'])
        pretrained_model = KinyaColBERT(args, extend_embeddings=True).to(device)
        print(f"Pre-training steps: {kb_state_dict['lr_scheduler_state_dict']['num_iters']:,}")
        pretrained_model.load_state_dict(kb_state_dict['model_state_dict'])
        del kb_state_dict
        gc.collect()
        pretrained_model.float()
        pretrained_model.eval()
        if ret_args:
            return pretrained_model, args
        return pretrained_model


class KinyaBERT_Classifier(nn.Module):

    def __init__(self, args: FlexArguments, cfg: FlexConfig, cls_num_inner_mult: int = 8,
                 encoder: KinyaBERTEncoder = None):
        super(KinyaBERT_Classifier, self).__init__()
        self.token_tagger = ('_tag:' in args.model_variant)
        self.encoder_fine_tune = args.encoder_fine_tune
        self.num_classes = len(args.cls_labels.split(','))
        self.is_regression = (self.num_classes == 1)
        labels = args.cls_labels.split(',')
        self.label_dict = {label.strip(): id for id, label in enumerate(labels)}
        self.inverse_label_dict = {id: label for label, id in self.label_dict.items()}
        self.args = args
        self.cfg = cfg
        if (encoder is not None):
            self.encoder = encoder
        else:
            if self.encoder_fine_tune:
                self.encoder = KinyaBERTEncoder(args, cfg)
        if self.token_tagger:
            self.cls_head = KinyaBERT_TokenClassificationHead(self.encoder.morpho_encoder.main_hidden_dim,
                                                              self.num_classes * cls_num_inner_mult, self.num_classes,
                                                              pooler_dropout=args.pooler_dropout,
                                                              head_trunk=args.head_trunk)
        else:
            self.cls_head = KinyaBERT_ClassificationHead(self.encoder.morpho_encoder.main_hidden_dim,
                                                         self.num_classes * cls_num_inner_mult, self.num_classes,
                                                         pooler_dropout=args.pooler_dropout,
                                                         head_trunk=args.head_trunk)

    @custom_fwd(device_type='cuda')
    def forward(self, batch_idx: int, args: FlexArguments, model_cache, training_item: Tuple, return_logits=False):
        (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
         input_sequence_lengths, main_masks_padded,
         predicted_targets) = training_item
        shared_encoder = model_cache
        if shared_encoder is not None:
            if self.encoder_fine_tune:
                tr_hidden_state = shared_encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                                 input_sequence_lengths, main_masks_padded)
            else:
                shared_encoder.eval()
                with torch.no_grad():
                    tr_hidden_state = shared_encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                                     input_sequence_lengths, main_masks_padded)
        else:
            if not self.encoder_fine_tune:
                raise RuntimeError("No shared encoder provided when encoder_fine_tune=True")
            tr_hidden_state = self.encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                           input_sequence_lengths, main_masks_padded)
        if return_logits:
            if self.token_tagger:
                return self.cls_head(tr_hidden_state, input_sequence_lengths)
            elif self.is_regression:
                return self.cls_head(tr_hidden_state).view(-1).float()
            else:
                return self.cls_head(tr_hidden_state)
        else:
            if self.token_tagger:
                output_scores = F.log_softmax(self.cls_head(tr_hidden_state, input_sequence_lengths), dim=1)
                loss = F.nll_loss(output_scores, predicted_targets)
            elif self.is_regression:
                loss = F.mse_loss(self.cls_head(tr_hidden_state).view(-1).float(), predicted_targets)
            else:
                output_scores = F.log_softmax(self.cls_head(tr_hidden_state), dim=1)
                loss = F.nll_loss(output_scores, predicted_targets)
            return [loss]

    def predict_classic_seq_cls_reg(self, device: torch.device, data_item_tuple, shared_encoder=None):
        (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded, input_sequence_lengths, main_masks_padded,
         fake_targets) = send_tuple_to(data_item_tuple, device)
        if shared_encoder is not None:
            if self.encoder_fine_tune:
                tr_hidden_state = shared_encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                                 input_sequence_lengths, main_masks_padded)
            else:
                shared_encoder.eval()
                with torch.no_grad():
                    tr_hidden_state = shared_encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                                     input_sequence_lengths, main_masks_padded)
        else:
            if not self.encoder_fine_tune:
                raise RuntimeError("No shared encoder provided when encoder_fine_tune=True")
            tr_hidden_state = self.encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                           input_sequence_lengths, main_masks_padded)
        if self.is_regression:
            logits = self.cls_head(tr_hidden_state)
            label = logits.item()
        else:
            logits = self.cls_head(tr_hidden_state)
            label = logits.argmax(-1).item()
        return label

    def predict(self, device: torch.device, input0: ParsedFlexSentence, input1: Union[None, ParsedFlexSentence] = None,
                shared_encoder=None):
        item = prepare_morpho_data_from_sentence(self.cfg, f'{device}', input0).add_bos()
        if input1 is not None:
            item = item.add_eos().extend(prepare_morpho_data_from_sentence(self.cfg, f'{device}', input1))
        (stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded, input_sequence_lengths, main_masks_padded,
         fake_targets) = send_tuple_to(item.to_classification_tuple([len(item)], None), device)
        if shared_encoder is not None:
            if self.encoder_fine_tune:
                tr_hidden_state = shared_encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                                 input_sequence_lengths, main_masks_padded)
            else:
                shared_encoder.eval()
                with torch.no_grad():
                    tr_hidden_state = shared_encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                                     input_sequence_lengths, main_masks_padded)
        else:
            if not self.encoder_fine_tune:
                raise RuntimeError("No shared encoder provided when encoder_fine_tune=True")
            tr_hidden_state = self.encoder(stems, pos_tags, lm_morphs, affixes_padded, morpho_masks_padded,
                                           input_sequence_lengths, main_masks_padded)
        output = 0.0
        if self.is_regression:
            logits = self.cls_head(tr_hidden_state)
            output = self.args.regression_scale_factor * logits.item()
        elif self.token_tagger:
            logits = self.cls_head(tr_hidden_state, input_sequence_lengths)
            labels = logits.argmax(-1).tolist()
            output = []
            idx = 0
            for i in range(len(input0.tokens)):
                output.append(self.inverse_label_dict[labels[idx]])
                idx += (len(input0.tokens[i].id_extra_tokens) + 1)
            if input1 is not None:
                idx += 1  # skip EOS tag if more than one input
                for i in range(len(input1.tokens)):
                    output.append(self.inverse_label_dict[labels[idx]])
                    idx += (len(input0.tokens[i].id_extra_tokens) + 1)
        else:
            logits = self.cls_head(tr_hidden_state)
            output = self.inverse_label_dict[logits.argmax(-1).item()]
        return output

    @staticmethod
    def from_pretrained(device: torch.device, pretrained_model_file: str, ret_args=False) -> Union[
        KinyaBERT_Classifier, Tuple[KinyaBERT_Classifier, FlexArguments]]:
        cfg = FlexConfig()
        print(f'Loading fine-tuned KinyaBERT_Classifier model from {pretrained_model_file} ...')
        kb_state_dict = torch.load(pretrained_model_file, map_location=device)
        args = FlexArguments().from_dict(kb_state_dict['args'])
        pretrained_model = KinyaBERT_Classifier(args, cfg).to(device)
        print(f"Fine-tuning steps: {kb_state_dict['lr_scheduler_state_dict']['num_iters']}")
        pretrained_model.load_state_dict(kb_state_dict['model_state_dict'])
        del kb_state_dict
        gc.collect()
        pretrained_model.float()
        pretrained_model.eval()
        if ret_args:
            return pretrained_model, args
        return pretrained_model

    @staticmethod
    def new_classifier_from_pretrained_kinyabert(device: torch.device,
                                                 pretrained_model_file: str,
                                                 ft_reinit_layers=0,
                                                 pooler_dropout=0.0,
                                                 head_trunk=False,
                                                 cls_labels='0',
                                                 model_variant='',
                                                 ret_args=False) -> Union[
        KinyaBERT_Classifier, Tuple[KinyaBERT_Classifier, FlexArguments]]:
        cfg = FlexConfig()
        print(f'Loading pre-trained KinyaBERT model from {pretrained_model_file} ...')
        kb_state_dict = torch.load(pretrained_model_file, map_location=device)
        args = FlexArguments().from_dict(kb_state_dict['args'])

        args.model_variant = model_variant
        args.ft_reinit_layers = ft_reinit_layers
        args.pooler_dropout = pooler_dropout
        args.head_trunk = head_trunk
        args.cls_labels = cls_labels
        args.encoder_fine_tune = True

        bert = KinyaBERT(args, cfg).to(device)
        print(f"Pre-training steps: {kb_state_dict['lr_scheduler_state_dict']['num_iters'] // 1000}K")
        bert.load_state_dict(kb_state_dict['model_state_dict'])
        del kb_state_dict
        gc.collect()
        cls_model: KinyaBERT_Classifier = KinyaBERT_Classifier(args, FlexConfig()).to(device)
        cls_model.encoder.load_state_dict(bert.encoder.state_dict())
        for layer_idx in range(args.ft_reinit_layers):
            cls_model.encoder.main_encoder.layers[-(layer_idx + 1)].apply(init_bert_params)
        del bert
        gc.collect()
        if ret_args:
            return cls_model, args
        return cls_model
