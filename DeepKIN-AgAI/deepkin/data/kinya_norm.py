from deepkin.data.kinyarwanda import Kinyarwanda

_kinyarwanda = Kinyarwanda()
tts_symbols = _kinyarwanda.tts_symbols

def text_to_sequence(text, norm=False):
    if norm:
        text = _kinyarwanda.norm_text(text)
    return _kinyarwanda.text_to_sequence(text)


def sequence_to_text(seq):
    return _kinyarwanda.sequence_to_text(seq)


def norm_text(text, encoding="utf-8", skip_enumerations=False, timing=None, debug=False) -> str:
    return _kinyarwanda.norm_text(text, encoding=encoding, skip_enumerations=skip_enumerations)
