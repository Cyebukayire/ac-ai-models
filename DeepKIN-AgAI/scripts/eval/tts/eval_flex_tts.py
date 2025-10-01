import torchaudio

from deepkin.data.kinya_norm import text_to_sequence
from deepkin.models.flex_tts import FlexKinyaTTS, FlexTTSTrainer
from deepkin.modules.tts_commons import intersperse

if __name__ == '__main__':

    trained_model = FlexTTSTrainer.from_pretrained('KINLP/data/flex_tts_base_agktjw_trainer_2025-06-01.pt')
    trained_model.flex_tts.eval()
    tts = FlexKinyaTTS(trained_model.flex_tts)
    tts.flex_tts.eval()
    tts.flex_tts.zero_grad()
    tts.flex_tts.remove_weight_norm()
    del tts.flex_tts.enc_q

    texts = ["Igifenesi (Artocarpus heterophyllus) ni urubuto rw'ingirakamaro ku buzima, rukungahaye ku ntungamubiri nka vitamine A, C, B1, B2, B3, B6, n'imyunyu ngugu irimo ubutare, potasiyumu, n'izindi. Igikomeza ubudahingarwa, imbaraga, ndetse gifasha mu by’ubuzima bitandukanye. Igifenesi gikunda ikirere gishyushye, ubuhitisha amazi, n’ubutumburuke kugeza kuri m 1600, imvura iri hagati ya mm 1000-2400.",
             "Mu buhinzi bw'ibirayi, ifumbire y’imborera iboze neza iterwa mu gihe cy’itabira ingana n'ibiro 300 kuri are imwe. Mu gihe cyo gutera, wongeraho ikiro n’igice (kg 1.5) k’ifumbire mvaruganda ya NPK 17.17.17 kuri are imwe, ishyirwa mu myobo cyangwa mu tugende. Ubwo ubutaka busharira, mbere yo gutera, ushyiraho hagati ya kg 25 na 30 by’ishwagara idatwitswe kuri are imwe bikamara ibihe by’ihinga bine.",
             "Bavanga ml 4.5 z’umuti na litiro 15 z’amazi maze iyo mvange igaterwa ku biti hagati ya 40 na 45."
             ]
    for i,text in enumerate(texts):
        text_id_sequence = text_to_sequence(text, norm=True)
        text_id_sequence = intersperse(text_id_sequence, 0)

        torchaudio.save(f"KINLP/data/flex_tts_sample_0_{i}.wav", tts(text_id_sequence, 0), 24000)
        torchaudio.save(f"KINLP/data/flex_tts_sample_1_{i}.wav", tts(text_id_sequence, 1), 24000)
        torchaudio.save(f"KINLP/data/flex_tts_sample_2_{i}.wav", tts(text_id_sequence, 2), 24000)
