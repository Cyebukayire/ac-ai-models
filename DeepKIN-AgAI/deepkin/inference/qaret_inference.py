from __future__ import annotations

from typing import Tuple

import torch

from deepkin.clib.libkinlp.uds_client import UnixSocketClient
from deepkin.models.kinyabert import KinyaColBERT


def init_retrieval_inference_setup(pretrained_colbert_model_file: str,
                                   rank=0,
                                   sock_file="/home/ubuntu/MORPHODATA/run/morpho.sock",
                                   max_retries=5,
                                   retry_delay_seconds=2.0) -> Tuple[KinyaColBERT, torch.device, UnixSocketClient]:
    uds_client = UnixSocketClient(sock_file, max_retries=max_retries, retry_delay=retry_delay_seconds)
    device = torch.device('cuda:%d' % rank)
    torch.cuda.set_device(rank)

    kinya_colbert_model = KinyaColBERT.from_pretrained(device, pretrained_colbert_model_file)
    kinya_colbert_model.float()
    kinya_colbert_model.eval()

    return (kinya_colbert_model, device, uds_client)
