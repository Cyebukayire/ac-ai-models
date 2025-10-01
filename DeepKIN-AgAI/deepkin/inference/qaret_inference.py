from __future__ import annotations

from typing import Tuple

import torch

from deepkin.clib.libkinlp.kinlpy import build_kinlpy_lib
from deepkin.models.morpho_bert import KinyaColBERT_from_pretrained


def init_qaret_inference_setup(pretrained_colbert_model_file: str,
                               rank=0,
                               sock_file="KINLP/data/run/deepkin.sock") -> Tuple:
    build_kinlpy_lib()
    from kinlpy import ffi, lib
    kinlp_conf = 'KINLP/data/config_deepkin.conf'
    lib.init_kinlp_socket(kinlp_conf.encode('utf-8'), sock_file.encode('utf-8'))

    device = torch.device('cuda:%d' % rank)
    torch.cuda.set_device(rank)

    ColBERT = KinyaColBERT_from_pretrained(device, pretrained_colbert_model_file)
    ColBERT.float()
    ColBERT.eval()

    return (ColBERT, device, lib, ffi)
