from typing import List, Tuple, Union

from deepkin.clib.libkinlp.uds_client import UnixSocketClient
from deepkin.data.syllabe_vocab import text_to_id_sequence, KINSPEAK_VOCAB_IDX

PAD_ID = 0
UNK_ID = 1
MSK_ID = 2
BOS_ID = 3
EOS_ID = 4

english_BOS_idx = 0
english_EOS_idx = 2

KIN_PAD_IDX = 0
EN_PAD_IDX = 1

NUM_SPECIAL_TOKENS = 5

MY_PRINTABLE = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


class ParsedToken:
    def __init__(self, w, ffi):
        # POS Info
        self.lm_stem_id = w.lm_stem_id
        self.lm_morph_id = w.lm_morph_id
        self.pos_tag_id = w.pos_tag_id
        self.valid_orthography = w.surface_form_has_valid_orthography

        # Morphology
        self.stem_id = w.stem_id
        self.affix_ids = [w.affix_ids[i] for i in range(w.len_affix_ids)]
        self.extra_stem_token_ids = [w.extra_stem_token_ids[i] for i in range(w.len_extra_stem_token_ids)]

        # Text
        self.uses_bpe = (w.uses_bpe == 1)
        self.is_apostrophed = w.is_apostrophed
        self.surface_form = ffi.string(w.surface_form).decode("utf-8") if (w.len_surface_form > 0) else ''
        self.raw_surface_form = ffi.string(w.raw_surface_form).decode("utf-8") if (w.len_raw_surface_form > 0) else ''
        if (self.is_apostrophed != 0) and (len(self.raw_surface_form) > 0) and (
                (self.raw_surface_form[-1] == 'a') or (self.raw_surface_form[-1] == 'A')):
            self.raw_surface_form = self.raw_surface_form[:-1] + "\'"
        self.syllables: List[Tuple[str, int]] = []
        if len(self.raw_surface_form) > 0:
            seq = text_to_id_sequence(self.raw_surface_form)
            self.syllables.extend([(KINSPEAK_VOCAB_IDX[i], i) for i in seq])
        self.tones: List[Tuple[int, int, int]] = []

    def to_parsed_format(self) -> str:
        word_list = [self.lm_stem_id, self.lm_morph_id, self.pos_tag_id, self.stem_id] + [
            len(self.extra_stem_token_ids)] + self.extra_stem_token_ids + [len(self.affix_ids)] + self.affix_ids
        return ','.join([str(x) for x in word_list])

    def adapted_raw_surface_form(self):
        if self.is_apostrophed:
            return self.raw_surface_form[:-1] + 'a'
        return self.raw_surface_form


COMMA_REPL = 'QWLTIO44O2TORIFHEWEIHGWEIOR094320'
COLON_REPL = '3489YJ12O3H92O20135ITRLW0UTTLEJWL'


class ParsedFlexToken:
    def __init__(self, parsed_token: str, real_parsed_token: ParsedToken = None, syllabify=False):
        self.audio_tokens = []
        if real_parsed_token is not None:
            self.lm_stem_id: int = real_parsed_token.lm_stem_id
            self.lm_morph_id: int = real_parsed_token.lm_morph_id
            self.pos_tag_id: int = real_parsed_token.pos_tag_id
            self.id_stem: int = real_parsed_token.stem_id
            self.id_extra_tokens: List[int] = real_parsed_token.extra_stem_token_ids
            self.stems_ids: List[int] = [self.id_stem] + self.id_extra_tokens
            self.affixes: List[int] = real_parsed_token.affix_ids
            self.is_apostrophed: bool = real_parsed_token.is_apostrophed
            self.surface_form: str = real_parsed_token.surface_form
            self.raw_surface_form: str = real_parsed_token.raw_surface_form
            self.syllables: List[Tuple[str, int]] = real_parsed_token.syllables
            self.tones: List[Tuple[int, int, int]] = real_parsed_token.tones
        else:
            self.surface_form: str = '_'
            self.raw_surface_form: str = '_'
            self.syllables: List[Tuple[str, int]] = []
            self.tones: List[Tuple[int, int, int]] = []
            pieces = parsed_token.split(' ')
            morpho_items = pieces[0].split(',')
            if len(pieces) > 1:
                self.raw_surface_form = pieces[1]
                self.surface_form = self.raw_surface_form.lower()
                if len(pieces) > 2:
                    syls: List[List[str]] = [tks.replace('::', COLON_REPL + ':').split(':') for tks in
                                             pieces[2].replace(',:', COMMA_REPL + ':').split(',')]
                    self.syllables.extend(
                        [(':' if (sl[0] == COLON_REPL) else (',' if (sl[0] == COMMA_REPL) else sl[0]), int(sl[1])) for
                         sl in syls])
                    if len(pieces) > 3:
                        tns: List[List[str]] = [tks.split(':') for tks in pieces[3].split(',')]
                        self.tones.extend([(int(tl[0]), int(tl[1]), int(tl[2])) for tl in tns])
                elif syllabify:
                    seq = text_to_id_sequence(self.raw_surface_form.lower())
                    self.syllables.extend([(KINSPEAK_VOCAB_IDX[i], i) for i in seq])
            self.lm_stem_id: int = int(morpho_items[0])
            self.lm_morph_id: int = int(morpho_items[1])
            self.pos_tag_id: int = int(morpho_items[2])
            self.id_stem: int = int(morpho_items[3])
            num_ext = int(morpho_items[4])
            self.id_extra_tokens: List[int] = [int(v) for v in morpho_items[5:(5 + num_ext)]]
            # This is to cap too long tokens for position encoding
            self.id_extra_tokens: List[int] = self.id_extra_tokens[:60]
            self.stems_ids: List[int] = [self.id_stem] + self.id_extra_tokens
            num_afx = int(morpho_items[(5 + num_ext)])
            self.affixes: List[int] = [int(v) for v in morpho_items[(6 + num_ext):(6 + num_ext + num_afx)]]
            self.is_apostrophed: bool = False
        assert ((len(self.affixes) == 0) or (
                    len(self.stems_ids) == 1)), f"Extra tokens with affixes: {self.to_parsed_format()}"

    def adapted_raw_surface_form(self):
        if self.is_apostrophed:
            if str(self.raw_surface_form[-1]) == "'":
                return self.raw_surface_form[:-1] + 'a'
        return self.raw_surface_form

    def noun_class_prefix(self, all_affixes):
        from deepkin.clib.libkinlp.kinlp_model import affix_view
        for i in self.affixes:
            v = affix_view(i, all_affixes)
            if v.startswith('N:1:'):
                prefix = v.split(':')[-1]
                if prefix == 'n':
                    prefix = 'zi'
                # ["mu", "ba", "mu", "mi", "ri", "ma", "ki", "bi", "n", "n", "ru", "ka", "tu", "bu", "ku", "ha"]
                return prefix
            elif v.startswith('QA:1:'):
                prefix = v.split(':')[-1]
                if prefix == 'n':
                    prefix = 'zi'
                # ["mu", "ba", "mu", "mi", "ri", "ma", "ki", "bi", "n", "n", "ru", "ka", "tu", "bu", "ku", "ha"]
                return prefix
        return None

    def to_parsed_format(self) -> str:
        word_list = [self.lm_stem_id, self.lm_morph_id, self.pos_tag_id, self.id_stem] + [
            len(self.id_extra_tokens)] + self.id_extra_tokens + [len(self.affixes)] + self.affixes
        ret = (','.join([str(x) for x in word_list])) + ' ' + self.raw_surface_form
        if len(self.syllables) > 0:
            ret += (' ' + (','.join([f'{s}:{i}' for s, i in self.syllables])))
            if len(self.tones) > 0:
                ret += (' ' + (','.join([f'{a}:{b}:{c}' for a, b, c in self.tones])))
        return ret


class ParsedAltToken:
    def __init__(self, indices_csv: str, prob: float, surface_form: str):
        self.surface_form: str = surface_form.lower() if (surface_form is not None) else '_'
        self.raw_surface_form: str = surface_form if (surface_form is not None) else '_'
        self.prob: float = prob
        morpho_items = indices_csv.split(',')
        self.lm_stem_id: int = int(morpho_items[0])
        self.lm_morph_id: int = int(morpho_items[1])
        self.pos_tag_id: int = int(morpho_items[2])
        self.id_stem: int = int(morpho_items[3])
        num_ext = int(morpho_items[4])
        self.id_extra_tokens: List[int] = [int(v) for v in morpho_items[5:(5 + num_ext)]]
        # This is to cap too long tokens for position encoding
        self.id_extra_tokens: List[int] = self.id_extra_tokens[:60]
        self.stems_ids: List[int] = [self.id_stem] + self.id_extra_tokens
        num_afx = int(morpho_items[(5 + num_ext)])
        self.affixes: List[int] = [int(v) for v in morpho_items[(6 + num_ext):(6 + num_ext + num_afx)]]
    def __str__(self):
        word_list = [self.lm_stem_id, self.lm_morph_id, self.pos_tag_id, self.id_stem] + [
            len(self.id_extra_tokens)] + self.id_extra_tokens + [len(self.affixes)] + self.affixes
        return (','.join([str(x) for x in word_list])) + f':{self.prob:.4f}'
    def is_multi_token(self) -> bool:
        return len(self.stems_ids) > 1


class ParsedMultiToken:
    def __init__(self, parsed_token_str: str):
        pieces = parsed_token_str.split(' ')
        self.raw_surface_form: str = pieces[1]
        self.alt_tokens: List[ParsedAltToken] = sorted(
            [ParsedAltToken(ip.split(':')[0], float(ip.split(':')[1]), self.raw_surface_form) for ip in
             pieces[0].split('|')], key=lambda x: x.prob, reverse=True)
        # Filter/Remove improbable tokens
        self.alt_tokens = [self.alt_tokens[0]] if self.alt_tokens[0].is_multi_token() else [t for t in self.alt_tokens if ((not t.is_multi_token()) and (t.prob>0.0))]
    def __str__(self):
        return ('|'.join([str(t) for t in self.alt_tokens])) + ' ' + self.raw_surface_form
    def __len__(self):
        return len(self.alt_tokens)


class ParsedSentenceMulti:
    def __init__(self, parsed_sentence_line: str, delimiter='\t'):
        self.multi_tokens: List[ParsedMultiToken] = [ParsedMultiToken(v) for v in parsed_sentence_line.split(delimiter)
                                                     if len(v) > 0]
    def __str__(self):
        return '\t'.join([str(t) for t in self.multi_tokens])
    def __len__(self):
        return len(self.multi_tokens)


class ParsedFlexSentence:
    def __init__(self, parsed_sentence_line: Union[str, None], parsed_tokens: List[ParsedToken] = None,
                 single_flex_parsed_token: Union[ParsedFlexToken, None] = None, delimiter='\t'):
        if single_flex_parsed_token is not None:
            self.tokens: List[ParsedFlexToken] = [single_flex_parsed_token]
        elif parsed_tokens is not None:
            self.tokens: List[ParsedFlexToken] = [ParsedFlexToken('_', real_parsed_token=token) for token in
                                                  parsed_tokens]
        else:
            self.tokens: List[ParsedFlexToken] = [ParsedFlexToken(v) for v in parsed_sentence_line.split(delimiter) if
                                                  len(v) > 0]

    def to_parsed_format(self) -> str:
        return '\t'.join([tk.to_parsed_format() for tk in self.tokens])

    def num_stems(self):
        return len([i for t in self.tokens for i in t.stems_ids])

    def trim(self, max_len):
        while self.num_stems() > max_len:
            self.tokens = self.tokens[:-1]
        return self

    def __len__(self):
        return self.num_stems()


def parse_text_to_morpho_sentence(uds_client: UnixSocketClient, txt: str) -> ParsedFlexSentence:
    parsed_sentence_line = ''
    success = uds_client.send_line('\t' + txt.strip())
    if success:
        parsed_sentence_line = uds_client.read_line()
    return ParsedFlexSentence(parsed_sentence_line)


def parse_document_to_morpho_sentence(uds_client: UnixSocketClient, text_lines: List[str]) -> ParsedFlexSentence:
    ret = ParsedFlexSentence(None, parsed_tokens=[], delimiter='\t')
    for txt in text_lines:
        it = parse_text_to_morpho_sentence(uds_client, txt)
        ret.tokens = ret.tokens + it.tokens
    return ret
