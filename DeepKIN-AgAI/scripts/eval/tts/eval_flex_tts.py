import sys

import torch
import torchaudio

from deepkin.data.kinya_norm import text_to_sequence
from deepkin.models.flex_tts import FlexKinyaTTS
from deepkin.modules.tts_commons import intersperse

if __name__ == '__main__':
    DATA_DIR = sys.argv[1] #'/home/ubuntu/DATA'

    rank = 0
    device = torch.device('cuda:%d' % rank) if torch.cuda.is_available() else torch.device('cpu')

    kinya_tts = FlexKinyaTTS.from_pretrained(device, f'{DATA_DIR}/kinya_flex_tts_base_trained.pt')
    kinya_tts.eval()

    texts = ["Iri shoramari ryo ku rwego rwo hejuru mu mikino no kwidagadura riherereye mu Murenge wa Kagarama, Akarere ka Kicukiro, mu Mujyi wa Kigali.",
             "Bimwe mu byagaragajwe nk’ibikizitiye umuryango ni ingo zibana mu makimbirane zitamaze kabiri zirimo n’izisenyuka zigishingwa."
             ]
    for i,text in enumerate(texts):
        text_id_sequence = text_to_sequence(text, norm=True)
        text_id_sequence = intersperse(text_id_sequence, 0)
        for sid in range(3):
            torchaudio.save(f"{DATA_DIR}/flex_tts_sample_Speaker.{sid}_Example.{i}.wav", kinya_tts(text_id_sequence, 0), 24000)
            print("Generated", f"{DATA_DIR}/flex_tts_sample_Speaker.{sid}_Example.{i}.wav")
