import sys
import wave
import random
import os

import progressbar

from deepkin.data.kinya_norm import norm_text
from deepkin.utils.misc_functions import time_now


def get_wav_duration_seconds(filename):
    with wave.open(filename, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration


def read_file_data(fn, norm=True):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = [tuple(line.rstrip('\n').split('\t')) for line in f.readlines()]
    data = {k: (norm_text(v) if norm else v) for k, v in lines}
    print(time_now(), f'Read {len(data)} lines from {fn}', flush=True)
    return data


if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    PSV_FILE = sys.argv[2]

    ag_tts_female_norm_text = read_file_data(f'{DATA_DIR}/kinya-ag-tts/rw_ag_tts_female.tsv')
    ag_tts_male_norm_text = read_file_data(f'{DATA_DIR}/kinya-ag-tts/rw_ag_tts_male.tsv')

    MIN_TTS_AUDIO_SECONDS = 2.0
    MIN_CHARS = 5

    final_data = []

    with progressbar.ProgressBar(initial_value=0, max_value=len(ag_tts_female_norm_text), redirect_stdout=True) as bar:
        for itr,(code,text) in enumerate(ag_tts_female_norm_text.items()):
            bar.update(itr)
            if os.path.isfile(f"{DATA_DIR}/kinya-ag-tts/rw_ag_tts_female/{code}.wav"):
                if (get_wav_duration_seconds(f"{DATA_DIR}/kinya-ag-tts/rw_ag_tts_female/{code}.wav") >= MIN_TTS_AUDIO_SECONDS) and (len(text) >= MIN_CHARS):
                    final_data.append(f"kinya-ag-tts/rw_ag_tts_female/{code}.wav|0|{text}")

    with progressbar.ProgressBar(initial_value=0, max_value=len(ag_tts_female_norm_text), redirect_stdout=True) as bar:
        for itr,(code,text) in enumerate(ag_tts_male_norm_text.items()):
            bar.update(itr)
            if os.path.isfile(f"{DATA_DIR}/kinya-ag-tts/rw_ag_tts_male/{code}.wav"):
                if (get_wav_duration_seconds(f"{DATA_DIR}/kinya-ag-tts/rw_ag_tts_male/{code}.wav") >= MIN_TTS_AUDIO_SECONDS) and (len(text) >= MIN_CHARS):
                    final_data.append(f"kinya-ag-tts/rw_ag_tts_male/{code}.wav|1|{text}")

    random.shuffle(final_data)
    random.shuffle(final_data)
    random.shuffle(final_data)

    with open(f'{DATA_DIR}/{PSV_FILE}', 'w', encoding='utf-8') as fout:
        for line in final_data:
            fout.write(line + '\n')
