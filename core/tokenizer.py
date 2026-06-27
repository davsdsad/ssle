"""Tokenizer for SSLE-1.

Handles text normalization, vocabulary management and encode/decode.

Normalization pipeline:
    uppercase -> remove accents (NFD) -> remove punctuation -> collapse spaces -> split
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List

# Special tokens (fixed ids, see documentation 4.1).
PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
SEP = "<SEP>"

SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, SEP]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Apply the normalization pipeline to raw text."""
    text = text.upper()
    # Remove accents via NFD decomposition then drop combining marks.
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # Remove punctuation (keep word chars and whitespace).
    text = _PUNCT_RE.sub(" ", text)
    # Collapse whitespace.
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Normalize then split into word tokens."""
    norm = normalize(text)
    if not norm:
        return []
    return norm.split(" ")


class Tokenizer:
    """Vocabulary holder with frequency-based vocab capping."""

    def __init__(self, max_vocab: int | None = None):
        self.max_vocab = max_vocab
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.freq: Dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            self._add_token(tok)

    # ------------------------------------------------------------------ #
    # Vocabulary building
    # ------------------------------------------------------------------ #
    def _add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def count(self, texts: Iterable[str]) -> None:
        """Accumulate token frequencies from an iterable of raw texts."""
        for text in texts:
            for tok in tokenize(text):
                self.freq[tok] = self.freq.get(tok, 0) + 1

    def build(self) -> None:
        """Build the final vocabulary from accumulated frequencies.

        The most frequent tokens are kept up to ``max_vocab`` (special tokens
        always included). This is what controls the Nano vs Base size.
        """
        # Reset to specials only.
        self.token_to_id = {}
        self.id_to_token = {}
        for tok in SPECIAL_TOKENS:
            self._add_token(tok)

        ordered = sorted(self.freq.items(), key=lambda kv: (-kv[1], kv[0]))
        budget = None
        if self.max_vocab is not None:
            budget = max(0, self.max_vocab - len(SPECIAL_TOKENS))
        for i, (tok, _f) in enumerate(ordered):
            if budget is not None and i >= budget:
                break
            self._add_token(tok)

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def token_id(self, token: str) -> int:
        return self.token_to_id.get(token, UNK_ID)

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids = [self.token_id(t) for t in tokenize(text)]
        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def encode_tokens(self, tokens: List[str], add_special: bool = True) -> List[int]:
        ids = [self.token_id(t) for t in tokens]
        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        toks: List[str] = []
        for i in ids:
            tok = self.id_to_token.get(int(i), UNK)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            toks.append(tok)
        return " ".join(toks)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {"vocab": self.token_to_id, "next_id": self.vocab_size}

    @classmethod
    def from_dict(cls, data: dict) -> "Tokenizer":
        tok = cls()
        tok.token_to_id = {k: int(v) for k, v in data["vocab"].items()}
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        return tok
