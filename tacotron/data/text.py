import re
import logging
import random

logger = logging.getLogger(__name__)


def text_has_no_digits(text):
    return re.search(rf"\d", text) is None


def unpack_mixed(transcript):
    repr = []
    pos = 0
    for m in re.finditer(r"{([^}]*)\|([^}]*)}", transcript):
        if m.start() > pos:
            repr.append((transcript[pos : m.start()], None))
        repr.append((m.group(1), m.group(2)))
        pos = m.end()
    if pos < len(transcript):
        repr.append((transcript[pos:], None))
    return repr


class TextEncoder:
    def __init__(self, alphabet, char_map=None, bos=None, eos=None, base_index=1):
        self.char_map = dict(char_map) if char_map else dict()
        self.bos = bos
        self.eos = eos
        self.alphabet = alphabet
        self.lookup = {c: (i + base_index) for i, c in enumerate(alphabet)}
        self.unk_chars = set()

    def prepare(self, text):
        text = text.lower()
        for key, value in self.char_map.items():
            text = re.sub(key, value, text)
        if self.bos:
            text = self.bos + text
        if self.eos:
            text = text + self.eos
        return text

    def encode(self, text, encode_unk=None):
        # text_orig = text
        text = self.prepare(text)
        # if text != text_orig:
        #     logger.debug(f"Transformed [{text_orig}] to [{text}]")
        if encode_unk:
            encoded = [self.lookup[c] if c in self.lookup else encode_unk for c in text]
        else:
            encoded = [self.lookup[c] for c in text if c in self.lookup]
            for c in text:
                if not c in self.lookup:
                    if not c in self.unk_chars:
                        self.unk_chars.add(c)
                        logger.warning(f"Unknown character: [{c}]")
        return encoded

    def decode(self, enc, decode_unk=None):
        if decode_unk:
            return [
                (
                    self.alphabet[i - 1]
                    if i > 0 and i <= len(self.alphabet)
                    else decode_unk
                )
                for i in enc
            ]
        else:
            return [
                self.alphabet[i - 1] for i in enc if i > 0 and i <= len(self.alphabet)
            ]


class MixedTextEncoder:
    def __init__(
        self, graphemes, phonemes, char_map=None, bos=None, eos=None, p_graphemes=0.3
    ):
        self.g_encoder = TextEncoder(graphemes, char_map, base_index=1)
        self.p_encoder = TextEncoder(phonemes, char_map, base_index=1 + len(graphemes))
        self.bos = bos
        self.eos = eos
        self.alphabet = graphemes + phonemes
        self.p_graphemes = p_graphemes

    def encode(self, text, encode_unk=None):
        encoded = []
        if self.bos:
            encoded.append(self.bos)

        for g, p in unpack_mixed(text):
            if random.rand() < self.p_graphemes:
                encoded.extend(self.g_encoder.encode(g))
            else:
                encoded.extend(self.p_encoder.encode(p))

        if self.eos:
            encoded.append(self.eos)

        return encoded

    def decode(self, enc, decode_unk=None):
        if decode_unk:
            return [
                (
                    self.alphabet[i - 1]
                    if i > 0 and i <= len(self.alphabet)
                    else decode_unk
                )
                for i in enc
            ]
        else:
            return [
                self.alphabet[i - 1] for i in enc if i > 0 and i <= len(self.alphabet)
            ]
