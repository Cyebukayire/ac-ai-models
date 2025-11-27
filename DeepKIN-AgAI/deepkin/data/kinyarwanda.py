from __future__ import print_function, division, annotations

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os
import unicodedata
import re
from typing import List, Dict

from deepkin.data.kinya_number_speller import rw_spell_number


class Kinyarwanda:
    def __init__(self):
        # ########################################################################
        # From symbols data
        self._pad = "_"
        self._punctuation = ';:,.!?¡¿—…"«»“”\' '
        self._letters = ['i', 'u', 'o', 'a', 'e', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'r', 'l', 's',
                         't', 'v', 'y', 'w', 'z', 'bw', 'by', 'cw', 'cy', 'dw', 'fw', 'gw', 'hw', 'kw', 'jw', 'jy',
                         'ny', 'mw', 'my', 'nw', 'pw', 'py', 'rw', 'ry', 'sw', 'sy', 'tw', 'ty', 'vw', 'vy', 'zw', 'pf',
                         'ts', 'sh', 'shy', 'mp', 'mb', 'mf', 'mv', 'nc', 'nj', 'nk', 'ng', 'nt', 'nd', 'ns', 'nz',
                         'nny', 'nyw', 'byw', 'ryw', 'shw', 'tsw', 'pfy', 'mbw', 'mby', 'mfw', 'mpw', 'mpy', 'mvw',
                         'mvy', 'myw', 'ncw', 'ncy', 'nsh', 'ndw', 'ndy', 'njw', 'njy', 'nkw', 'ngw', 'nsw', 'nsy',
                         'ntw', 'nty', 'nzw', 'shyw', 'mbyw', 'mvyw', 'nshy', 'nshw', 'nshyw', 'bg', 'pfw', 'pfyw',
                         'vyw', 'njyw', 'x', 'q']

        # Export all symbols:
        self.tts_symbols = [self._pad] + list(self._punctuation) + self._letters

        # Special symbol ids
        self.SPACE_ID = self.tts_symbols.index(" ")

        # ########################################################################

        # Mappings from symbol to numeric ID and vice versa:
        self._symbol_to_id = {s: i for i, s in enumerate(self.tts_symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.tts_symbols)}

        self._digits_map = {0: 'zeru',
                            1: 'rimwe',
                            2: 'kabiri',
                            3: 'gatatu',
                            4: 'kane',
                            5: 'gatanu',
                            6: 'gatandatu',
                            7: 'karindwi',
                            8: 'umunani',
                            9: 'icyenda'}

        self._rw_phone_pattern = r"(((\+250)|(250)|(0))?7[0-9]\d{7})"

        self._VOWELS = {'i', 'u', 'o', 'a', 'e'}
        self._CONSONANTS = {'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'r', 'l', 's', 't', 'v', 'y', 'w',
                            'z', 'bw', 'by', 'cw', 'cy', 'dw', 'fw', 'gw', 'hw', 'kw', 'jw', 'jy', 'ny', 'mw', 'my',
                            'nw', 'pw', 'py', 'rw', 'ry', 'sw', 'sy', 'tw', 'ty', 'vw', 'vy', 'zw', 'pf', 'ts', 'sh',
                            'shy', 'mp', 'mb', 'mf', 'mv', 'nc', 'nj', 'nk', 'ng', 'nt', 'nd', 'ns', 'nz', 'nny', 'nyw',
                            'byw', 'ryw', 'shw', 'tsw', 'pfy', 'mbw', 'mby', 'mfw', 'mpw', 'mpy', 'mvw', 'mvy', 'myw',
                            'ncw', 'ncy', 'nsh', 'ndw', 'ndy', 'njw', 'njy', 'nkw', 'ngw', 'nsw', 'nsy', 'ntw', 'nty',
                            'nzw', 'shyw', 'mbyw', 'mvyw', 'nshy', 'nshw', 'nshyw', 'bg', 'pfw', 'pfyw', 'vyw', 'njyw'}

        self.hour_map = {0: "sita z'ijoro", 1: "saba", 2: "munani", 3: "cyenda", 4: "kumi", 5: "kumi n'imwe",
                         6: "kumi n'ebyiri",
                         7: "moya", 8: "mbiri", 9: "tatu", 10: "yine", 11: "tanu", 12: "sita", 13: "saba", 14: "munani",
                         15: "cyenda", 16: "kumi", 17: "kumi n'imwe", 18: "kumi n'ebyiri", 19: "moya", 20: "mbiri",
                         21: "tatu",
                         22: "yine", 23: "tanu", 24: "sita z'ijoro"}

        # mu ba mu mi ri ma ki bi n n ru ka tu bu ku ha ku
        self.Rwanda_Months = {'mutarama', 'gashyantare', 'werurwe', 'mata', 'gicurasi', 'kamena', 'nyakanga', 'kanama',
                              'nzeli', 'nzeri', 'ukwakira', 'ugushyingo', 'ukuboza'}

        self.time_regex = "^([01]?[0-9]|2[0-3])(:[0-5]?[0-9]){1,2}$"
        self.time_pattern = re.compile(self.time_regex)

        self.class_prefixes = {'mu', 'ba', 'mi', 'ri', 'ma', 'ki', 'bi', 'n', 'ru', 'ka', 'tu', 'bu', 'ku', 'ha'}

        self.token_map: Dict[str, List[str]] = {
            '%': ['ku', 'ijana'],
            'cm': ['santimetero'],
            'dm': ['desimetero'],
            'kg': ['ibiro'],
            'canada': ['kanada'],
            'congo': ['kongo'],
            'america': ['amerika'],
            'france': ['faranse'],
            'antrachnose': ['antarakinoze'],
            'ph': ['pe', 'hashi'],
            'rab': ['rabu'],
            'managri': ['minagiri']
        }
        if os.path.exists('agai_pronunciation_adapter.tsv'):
            with open('agai_pronunciation_adapter.tsv', 'r') as tsv:
                for line in tsv:
                    tokens = line.rstrip('\n').split('\t')
                    if len(tokens) == 2:
                        self.token_map[tokens[0]] = ' '.join(tokens[1].split()).split()

    def process_cons(self, cons, seq):
        if cons in self._symbol_to_id:
            seq.append(self._symbol_to_id[cons])
        else:
            for c in cons:
                if c in self._symbol_to_id:
                    seq.append(self._symbol_to_id[c])

    def rw_prefix(self, word: str, n: int):
        if (len(word) < 4) or (word.lower() in self.Rwanda_Months):
            return None
        if len(word) == 0:
            return 'bi'
        if word[0] in self._VOWELS:
            return self.rw_prefix(word[1:], n)
        if (word[0] == 'n') or (word == 'hegitari') or (word == 'santimetero') or (word == 'kilometero') or (
                word == 'dogere') or (word == 'kirometero') or word.endswith('litiro') or word.endswith('metero') or (word == 'ha') or (
                word == 'cm') or (word == 'm'):
            return 'zi'
        if (word[0:2] == 'cy') and (word != 'cyangwa'):
            return 'ki'
        if word[1] == 'w':
            return word[0:1] + 'u'
        if (word[1] == 'y') and (word != 'cyangwa'):
            return word[0:1] + 'i'
        pr = word[0:2]
        if pr in self.class_prefixes:
            return pr
        return 'ri' if (n==1) else ('ka' if (n<8) else None)

    def is_time(self, tok):
        if tok == "":
            return False
        return re.search(self.time_pattern, tok) is not None

    def norm_text(self, string, encoding="utf-8", skip_enumerations=False) -> str:
        import re
        string = string.decode(encoding) if isinstance(string, type(b'')) else string
        string = string.replace('`', '\'')
        string = string.replace("'", "\'")
        string = string.replace("‘", "\'")
        string = string.replace("’", "\'")
        string = string.replace("‚", "\'")
        string = string.replace("‛", "\'")
        string = string.replace("–", "-")
        string = string.replace("—", "-")
        string = string.replace("−", "-")
        string = string.replace(u"æ", u"ae").replace(u"Æ", u"AE")
        string = string.replace(u"œ", u"oe").replace(u"Œ", u"OE")
        string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode("ascii").lower()

        if skip_enumerations:
            string = re.sub('^\d\.?\s', '', string)

        string = re.sub('([~!@#$%^&*()_+{}|"<>?`\-=\[\];\'/])', r' \1 ', string)

        # Add spaces to split special tokens of interest
        string = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", string)

        string = re.sub('\s{2,}', ' ', string)
        tokens = ' '.join(string.split()).split()

        # print('Tokens:', '\n'.join(tokens))

        final_tokens = []
        for i, tok in enumerate(tokens):
            if re.match(self._rw_phone_pattern, tok):
                phones = [x for q in re.findall(self._rw_phone_pattern, tok) for x in q if len(x)>8]
                for phone in phones:
                    for ttii,t in enumerate(list(phone)):
                        final_tokens.append('guteranya' if (t == '+') else self._digits_map[int(t)])
                        if ttii < (len(list(phone))-1):
                            final_tokens.append(',')
            elif (tok == '+') and (i < (len(tokens) - 1)):
                if re.match(self._rw_phone_pattern, tokens[i+1]):
                    final_tokens.append('guteranya')
                else:
                    final_tokens.append('guteranyaho')
            elif tok.isnumeric():
                if i > 0:
                    final_tokens.append(rw_spell_number(self.rw_prefix(tokens[i - 1], int(tok)), int(tok)).strip())
                else:
                    final_tokens.append(rw_spell_number(None, int(tok)).strip())
            elif (i > 0) and self.is_time(tok):
                if tokens[i - 1].lower() == 'saa':
                    final_tokens.extend(self.spell_time(tok))
                else:
                    final_tokens.extend(self.adapt_final_token(tokens[i - 1] if (i > 0) else None, tok, next_tok=(
                        tokens[i + 1] if (i < (len(tokens) - 1)) else None)))
            elif ((tok == '-') or tok == '+') and (i > 0) and (i < len(tokens) - 1):
                if tokens[i - 1].isnumeric() and tokens[i + 1].isnumeric():
                    final_tokens.extend(['kugeza', 'kuri'] if (tok == '-') else ['kongeraho'])
                else:
                    final_tokens.extend(self.adapt_final_token(tokens[i - 1] if (i > 0) else None, tok, next_tok=(
                        tokens[i + 1] if (i < (len(tokens) - 1)) else None)))
            else:
                final_tokens.extend(self.adapt_final_token(tokens[i - 1] if (i > 0) else None, tok, next_tok=(
                    tokens[i + 1] if (i < (len(tokens) - 1)) else None)))
        return ' '.join(final_tokens)

    def spell_time(self, time) -> List[str]:
        try:
            times = [int(t) for t in time.split(':')]
            hr = 0 if len(times) < 1 else times[0]
            mn = 0 if len(times) < 2 else times[1]
            sec = 0 if len(times) < 3 else times[2]
            minutes_list = [] if (mn == 0) else (
                ["n'umunota", "umwe"] if (mn == 1) else ["n'iminota", rw_spell_number('mi', mn)])
            seconds_list = [] if (sec == 0) else (
                ["n'isegonda", "rimwe"] if (sec == 1) else ["n'amasegonda", rw_spell_number('ma', sec)])
            return [self.hour_map[hr]] + minutes_list + seconds_list
        except:
            pass
        return [time]

    def text_to_sequence(self, text):
        seq = []
        txt = self.norm_text(self.norm_text(text))
        # print('==> norm_text:', text)
        txt = re.sub(r"\s+", '|', txt).strip()
        start = 0
        end = 0
        while end < len(txt):
            if (txt[end] in self._VOWELS) or (txt[end] == '|'):
                if end > start:
                    self.process_cons(txt[start:end], seq)
                if txt[end] == '|':
                    seq.append(self._symbol_to_id[' '])
                else:
                    seq.append(self._symbol_to_id[txt[end]])
                end += 1
                start = end
            else:
                end += 1
        if end > start:
            self.process_cons(txt[start:end], seq)
        return seq

    def sequence_to_text(self, sequence):
        """Converts a sequence of IDs back to a string"""
        result = ""
        for symbol_id in sequence:
            s = self._id_to_symbol[symbol_id]
            result += s
        return result

    def adapt_final_token(self, prev_token, tok: str, next_tok=None) -> List[str]:
        if ((tok[-1] == '.') or (tok[-1] == ',')) and (len(tok) > 1):
            return self.adapt_final_token(prev_token, tok[:-1], next_tok=next_tok) + [tok[-1:]]

        if ((tok[0] == '.') or (tok[0] == ',')) and (len(tok) > 1):
            return [tok[:1]] + self.adapt_final_token(None, tok[1:], next_tok=next_tok)

        if (tok.lower() in self.token_map) and (next_tok != '\''):
            return self.token_map[tok.lower()]
        numer_pieces = [t for k in tok.split(',') for t in k.split('.')]
        numbers = sum([t.isnumeric() for t in numer_pieces])
        if numbers == len(numer_pieces):
            num_prefix = None if prev_token is None else self.rw_prefix(prev_token, int(numer_pieces[0][-1:]))
            if (tok.count(',') > 0) and (tok.count('.') == 1):
                if tok.rindex('.') > tok.rindex(','):
                    pieces = tok.replace(',', '').split('.')
                    num = int(pieces[0])
                    dec = int(pieces[1])
                    if dec == 1:
                        return [rw_spell_number(num_prefix, num), "n'igice", "kimwe"]
                    else:
                        return [rw_spell_number(num_prefix, num), "n'ibice", rw_spell_number('bi', dec)]
                else:
                    return [rw_spell_number(None, int(t)) for t in numer_pieces]  # Invalid number
            elif (tok.count(',') == 0) and (tok.count('.') == 1):
                pieces = tok.split('.')
                num = int(pieces[0])
                dec = int(pieces[1])
                if dec == 1:
                    return [rw_spell_number(num_prefix, num), "n'igice", "kimwe"]
                else:
                    return [rw_spell_number(num_prefix, num), "n'ibice", rw_spell_number('bi', dec)]
            elif tok.count('.') == 0:
                return [rw_spell_number(num_prefix, int(tok.replace(',', '')))]
            else:
                return [rw_spell_number(None, int(t)) for t in numer_pieces]  # Invalid number
        new_pieces = re.sub('([~!@#$%^&*()_+{}|:"<>?`\-=\[\];\',./])', r' \1 ', tok).split()
        if new_pieces[0].isnumeric():
            num = int(new_pieces[0])
            num_prefix = None if prev_token is None else self.rw_prefix(prev_token, int(new_pieces[0][-1:]))
            return [rw_spell_number(num_prefix, num)] + [rw_spell_number(None, int(t)) if t.isnumeric() else t for t in
                                                         new_pieces[1:]]
        return [rw_spell_number(None, int(t)) if t.isnumeric() else t for t in new_pieces]
