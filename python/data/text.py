import re
import logging

logger = logging.getLogger(__name__)


def text_has_no_digits(text):
    return re.search(rf"\d", text) is None


class TextEncoder:
    def __init__(self, alphabet, char_map=None, bos=None, eos=None):
        self.char_map = dict(char_map) if char_map else dict()
        self.bos = bos
        self.eos = eos
        self.alphabet = alphabet
        self.lookup = {c: (i + 1) for i, c in enumerate(alphabet)}

    def encode(self, text, encode_unk=None):
        # text_orig = text
        text = text.lower()
        for key, value in self.char_map.items():
            text = re.sub(key, value, text)
        if self.bos:
            text = self.bos + text
        if self.eos:
            text = text + self.eos
        # if text != text_orig:
        #     logger.debug(f"Transformed [{text_orig}] to [{text}]")
        if encode_unk:
            encoded = [self.lookup[c] if c in self.lookup else encode_unk for c in text]
        else:
            encoded = [self.lookup[c] for c in text if c in self.lookup]
            unk_chars = ""
            for c in text:
                if not c in self.lookup:
                    unk_chars += c
            if unk_chars:
                logger.warning(f"Unknown characters: {unk_chars}")
        return encoded

    def decode(self, enc, decode_unk=None):
        if decode_unk:
            return [
                self.alphabet[i - 1] if i > 0 and i <= len(self.alphabet) else decode_unk
                for i in enc
            ]
        else:
            return [self.alphabet[i - 1] for i in enc if i > 0 and i <= len(self.alphabet)]