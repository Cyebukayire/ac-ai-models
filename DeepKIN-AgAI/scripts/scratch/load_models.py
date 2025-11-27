import torch

from deepkin.models.flex_tts import FlexTTSTrainer, FlexKinyaTTS
from deepkin.models.kinyabert import KinyaBERT, KinyaColBERT

if __name__ == "__main__":
    MODELS_DIR = '/home/nzeyi/Desktop/C4IR_AGRI_AI/Models_Release'

    device: torch.device= torch.device('cpu')

    kinyabert = KinyaBERT.from_pretrained(device, f'{MODELS_DIR}/kinyabert_base_pretrained.pt')
    kinyabert.eval()

    kinyacolbert = KinyaColBERT.from_pretrained(device, f'{MODELS_DIR}/kinya_colbert_large_rw_ag_retrieval_finetuned_512D.pt')
    kinyacolbert.eval()

    kinya_tts = FlexKinyaTTS.from_pretrained(device, f'{MODELS_DIR}/kinya_flex_tts_base_trained.pt')
    kinya_tts.eval()

