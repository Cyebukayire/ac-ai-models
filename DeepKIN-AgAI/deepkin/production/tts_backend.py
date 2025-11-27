from __future__ import print_function, division

# Ignore warnings
import warnings

from deepkin.data.kinya_norm import text_to_sequence
from deepkin.models.flex_tts import FlexKinyaTTS
from deepkin.modules.tts_commons import intersperse

warnings.filterwarnings("ignore")
import uuid
import time

import torchaudio

from flask import request
import os
from flask import Flask

import torch

rank = 0
device = torch.device('cuda:%d' % rank)

DATA_DIR = '/home/ubuntu/DATA'

kinya_tts = FlexKinyaTTS.from_pretrained(device, f'{DATA_DIR}/kinya_flex_tts_base_trained.pt')
kinya_tts.eval()

inference_engine = (kinya_tts, device)

def kinya_tts(text, sid = 1,  output_wav_file = 'kinspeak_output.wav') -> str:
    global inference_engine
    (kinya_tts, device) = inference_engine
    text_id_sequence = text_to_sequence(text, norm=True)
    text_id_sequence = intersperse(text_id_sequence, 0)
    audio_stream = kinya_tts(text_id_sequence, sid, speed = 1.0)
    sampling_rate = 24_000
    torchaudio.save(output_wav_file, audio_stream, sampling_rate)
    return output_wav_file

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"

@app.route('/tts', methods=['POST'])
def tts_task():
    content_type = request.headers.get('Content-Type')
    if ('application/json' in content_type):
        json = request.get_json()
        input = json['input']
        sid = 0
        filename = f'{str(uuid.uuid4())}_{int(round(time.time() * 1000))}.wav'
        output_wav_file = f'/tmp/{filename}'
        if 'sid' in json:
            sid = int(json['sid'])
        if 'output_resample_rate' in json:
            kinya_tts(input, sid=sid, output_wav_file=output_wav_file,
                                          output_resample_rate=int(json['output_resample_rate']))
        else:
            kinya_tts(input, sid=sid, output_wav_file=output_wav_file)

        def stream_and_remove_file():
            try:
                with open(output_wav_file, 'rb') as file_handle:
                    yield from file_handle
            finally:
                try:
                    os.remove(output_wav_file)
                except Exception as error:
                    app.logger.error("Error removing downloaded file", error)

        r = app.response_class(stream_and_remove_file(), mimetype="audio/wav")
        r.headers.set('Content-Disposition', 'attachment', filename=filename)
        return r
    else:
        return 'Content-Type not supported!'

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=9595)
