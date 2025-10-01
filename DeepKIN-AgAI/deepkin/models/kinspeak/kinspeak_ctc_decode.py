import time
from operator import itemgetter

import numpy as np
import torch
import torchmetrics
from cffi import FFI

from deepkin.clib.libkinlp.kinlpy import BOS_ID, EOS_ID
from deepkin.data.base_data import generate_input_key_padding_mask, generate_square_subsequent_mask
from deepkin.data.syllabe_vocab import id_sequence_to_text, text_to_id_sequence
from deepkin.utils.misc_functions import time_now


def build_kinspeak_inference_lib():
    ffibuilder = FFI()

    ffibuilder.cdef("""
        void init_syllabe_vocab(const char * lexicon_file);
        void destroy_syllabe_vocab(void);
        void syllabe_ctc_beam_search(float * log_probs, int beam_width, int blank_idx, int max_T[], int dim_B, int dim_T, int dim_C,
                int * ret_labels, int * ret_labels_lengths, float * ret_labels_scores, int ret_labels_count[], int ret_labels_max[1]);
    
        void init_char_vocab(const char * lexicon_file);
        void destroy_char_vocab(void);
        void char_ctc_beam_search(float * log_probs, int beam_width, int blank_idx, int max_T[], int dim_B, int dim_T, int dim_C,
                int * ret_labels, int * ret_labels_lengths, float * ret_labels_scores, int ret_labels_count[], int ret_labels_max[1]);
    """)

    ffibuilder.set_source("ksp_inference",
                          """
                               #include "kinspeak-inference/ksp_lib.h"
                          """,
                          extra_compile_args=['-fopenmp', '-D use_openmp', '-O3', '-march=native', '-ffast-math',
                                              '-Wall', '-Werror'],
                          extra_link_args=['-fopenmp'],
                          libraries=['kinspeak-inference'])  # library name, for the linker

    ffibuilder.compile(verbose=True)


def form_txt_from_sylabe_ids(ids):
    from deepkin.data.syllabe_vocab import id_sequence_to_text
    txt = id_sequence_to_text([t for t in ids if (t != BOS_ID) and (t != EOS_ID)]).replace('  ', ' ').replace('  ',
                                                                                                              ' ').strip()
    return txt


def kinya_gpt_score(kinya_gpt_ffi_lib_bcfg_device, txt):
    kinya_gpt, ffi, lib, bcfg, device = kinya_gpt_ffi_lib_bcfg_device
    with torch.no_grad():
        kg_score = kinya_gpt.score_all_tokens_log10(ffi, lib, txt, bcfg, device)
    return kg_score


CTC_BS_ALLOCATION_TIME = 0.0
CTC_BS_LIB_SEARCH = 0.0
CTC_BS_SCORING = 0.0
CTC_BS_LM_EVAL = 0.0


def reset_and_print_ctc_beam_search_latency_info(TIME_COUNT, TOTAL_TIME):
    global CTC_BS_ALLOCATION_TIME
    global CTC_BS_LIB_SEARCH
    global CTC_BS_SCORING
    global CTC_BS_LM_EVAL

    if (TIME_COUNT > 0):
        print(time_now(), 'CTC BEAM SEARCH STATS PER 1 SEC INPUT:')
        print(
            'CTC_BS_ALLOCATION_TIME/1SEC:   {:.3f} msec - {:.0f} %'.format(1000.0 * CTC_BS_ALLOCATION_TIME / TIME_COUNT,
                                                                           100.0 * CTC_BS_ALLOCATION_TIME / TOTAL_TIME))
        print('CTC_BS_LIB_SEARCH/1SEC:        {:.3f} msec - {:.0f} %'.format(1000.0 * CTC_BS_LIB_SEARCH / TIME_COUNT,
                                                                             100.0 * CTC_BS_LIB_SEARCH / TOTAL_TIME))
        print('CTC_BS_SCORING/1SEC:           {:.3f} msec - {:.0f} %'.format(1000.0 * CTC_BS_SCORING / TIME_COUNT,
                                                                             100.0 * CTC_BS_SCORING / TOTAL_TIME))
        print('CTC_BS_LM_EVAL/1SEC:           {:.3f} msec - {:.0f} %'.format(1000.0 * CTC_BS_LM_EVAL / TIME_COUNT,
                                                                             100.0 * CTC_BS_LM_EVAL / TOTAL_TIME))

    CTC_BS_ALLOCATION_TIME = 0.0
    CTC_BS_LIB_SEARCH = 0.0
    CTC_BS_SCORING = 0.0
    CTC_BS_LM_EVAL = 0.0


def cpp_ctc_beam_search(log_probs_table: np.ndarray,
                        source_encoder_output_lengths,
                        ffi, lib,
                        char=False,
                        beam_width: int = 16,
                        blank_idx=6,
                        gpt_lm=None,
                        device=None,
                        return_scores=False):
    global CTC_BS_ALLOCATION_TIME
    global CTC_BS_LIB_SEARCH
    global CTC_BS_SCORING
    global CTC_BS_LM_EVAL

    start_time = time.time()
    TOP = 40
    dim_T, dim_B, dim_C = log_probs_table.shape

    log_probs = ffi.cast("float *", log_probs_table.ctypes.data)
    ret_labels = ffi.new("int[]", dim_B * TOP * dim_T)
    ret_labels_lengths = ffi.new("int[]", dim_B * TOP)
    ret_labels_scores = ffi.new("float[]", dim_B * TOP)
    ret_labels_count = ffi.new("int[]", dim_B)
    max_T = ffi.new("int[]", dim_B)
    ret_labels_max = ffi.new("int[1]")
    for b in range(dim_B):
        ret_labels_count[b] = 0
        max_T[b] = source_encoder_output_lengths[b]
    ret_labels_max[0] = TOP

    CTC_BS_ALLOCATION_TIME += (time.time() - start_time)
    start_time = time.time()

    if char:
        lib.char_ctc_beam_search(log_probs, beam_width, blank_idx, max_T, dim_B, dim_T, dim_C,
                                 ret_labels, ret_labels_lengths, ret_labels_scores, ret_labels_count, ret_labels_max)
    else:
        lib.syllabe_ctc_beam_search(log_probs, beam_width, blank_idx, max_T, dim_B, dim_T, dim_C,
                                    ret_labels, ret_labels_lengths, ret_labels_scores, ret_labels_count, ret_labels_max)

    CTC_BS_LIB_SEARCH += (time.time() - start_time)
    start_time = time.time()

    ret_list = []
    for bidx in range(dim_B):
        ret_idx = 0
        syllabe_ids = []
        batch_ids = []
        am_scores = []
        syllabe_id_lengths = []
        for ix in range(ret_labels_count[bidx]):
            if ret_labels_lengths[(bidx * TOP) + ix] > 0:
                am_scores.append(ret_labels_scores[(bidx * TOP) + ix])
                ids = []
                for _ in range(ret_labels_lengths[(bidx * TOP) + ix]):
                    ids.append(ret_labels[(bidx * TOP * dim_T) + ret_idx])
                    ret_idx += 1
                syllabe_ids.extend([i for i in ids if i != blank_idx])
                syllabe_id_lengths.append(len([i for i in ids if i != blank_idx]))
                batch_ids.append(ids)

        best_labels = batch_ids[0]
        max_score = am_scores[0]
        max_lm_score = 0.0
        max_am_score = am_scores[0]

        if (gpt_lm is not None) and (device is not None) and (len(syllabe_ids) > 0):
            lm_start = time.time()
            syllabe_ids = torch.tensor(syllabe_ids).to(device)
            with torch.no_grad():
                gpt_lm.eval()
                tgt_key_padding_mask = generate_input_key_padding_mask(syllabe_id_lengths, ignore_last=True).to(
                    syllabe_ids.device)
                tgt_decoder_mask = generate_square_subsequent_mask(max(syllabe_id_lengths)).to(syllabe_ids.device)
                lm_scores = gpt_lm.batched_nll_losses(syllabe_ids, syllabe_id_lengths, tgt_key_padding_mask,
                                                      tgt_decoder_mask)

            CTC_BS_LM_EVAL += (time.time() - lm_start)

            assert len(lm_scores) == len(am_scores), "Mismatch am lm scores!"

            scores = [((-lm * 15.0) + (am * 0.5)) for lm, am in zip(lm_scores, am_scores)]
            max_index, max_score = max(enumerate(scores), key=itemgetter(1))
            best_labels = batch_ids[max_index]
            max_lm_score = lm_scores[max_index]
            max_am_score = am_scores[max_index]

        if return_scores:
            ret_list.append((best_labels, max_score, max_am_score, max_lm_score, len(best_labels)))
        else:
            ret_list.append(best_labels)

    CTC_BS_SCORING += (time.time() - start_time)

    return ret_list


def norm_text_eval(pred_txt, target_txt):
    pred_txt = id_sequence_to_text(text_to_id_sequence(pred_txt))
    target_txt = id_sequence_to_text(text_to_id_sequence(target_txt))

    pred_txt = pred_txt.lower().replace('\' ', '\'').replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('!',
                                                                                                                   ' ').replace(
        '-', ' ').replace(':', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace(
        '  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').strip()
    target_txt = target_txt.lower().replace('\' ', '\'').replace(',', ' ').replace('.', ' ').replace('?', ' ').replace(
        '!', ' ').replace('-', ' ').replace(':', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace(
        '  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ',
                                                                                                       ' ').strip()
    wer = torchmetrics.functional.word_error_rate(preds=pred_txt, target=target_txt)
    cer = torchmetrics.functional.char_error_rate(preds=pred_txt, target=target_txt)
    return cer, wer, pred_txt, target_txt
