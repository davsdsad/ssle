"""From-scratch BPE subword tokenizer (no external tokenizer libraries).

Special tokens occupy fixed low ids; learned BPE tokens follow. Theme tokens are
appended after BPE so a sample can be conditioned as:

    <bos> <thm:NAME> kw1 kw2 <sep> title-subwords... <eos>
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

PAD, BOS, EOS, UNK, SEP, KW = "<pad>", "<bos>", "<eos>", "<unk>", "<sep>", "<kw>"
SPECIALS = [PAD, BOS, EOS, UNK, SEP, KW]
PAD_ID, BOS_ID, EOS_ID, UNK_ID, SEP_ID, KW_ID = range(6)

END = "</w>"
_KEEP = re.compile(r"[^a-záàâãéêíóôõúüçñ0-9 .,!?:%\-]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = _KEEP.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _words(text: str) -> List[str]:
    return normalize(text).split()


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: List[Tuple[str, str]] = []
        self.ranks: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}        # token string -> id
        self.inv: Dict[int, str] = {}          # id -> token string
        self.themes: Dict[str, int] = {}       # theme name -> token id
        self._cache: Dict[str, List[str]] = {}

    # --- training --------------------------------------------------------
    def train(self, texts: List[str], vocab_size: int = 8000,
              min_freq: int = 2, themes: List[str] | None = None) -> None:
        """Learn merges with incremental pair counts (only re-scan the words
        touched by each merge), so training stays fast on large corpora."""
        word_freq: Counter[str] = Counter()
        for t in texts:
            word_freq.update(_words(t))

        word_syms: Dict[str, List[str]] = {w: list(w) + [END] for w in word_freq}
        alphabet = set()
        for w in word_freq:
            alphabet.update(w)
        alphabet.add(END)

        # Initial pair statistics + inverted index pair -> words containing it.
        pair_counts: Counter[Tuple[str, str]] = Counter()
        pair_words: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for w, freq in word_freq.items():
            syms = word_syms[w]
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                pair_counts[p] += freq
                pair_words[p].add(w)

        base_tokens = SPECIALS + sorted(alphabet)
        target_merges = max(0, vocab_size - len(base_tokens) -
                            (len(themes) if themes else 0))

        for _ in range(target_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best] < min_freq:
                break
            self.merges.append(best)
            a, b = best
            merged = a + b
            for w in list(pair_words[best]):
                syms = word_syms[w]
                freq = word_freq[w]
                # Retract this word's current pair contributions.
                for i in range(len(syms) - 1):
                    p = (syms[i], syms[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_words[p].discard(w)
                # Apply the merge.
                out: List[str] = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                word_syms[w] = out
                # Re-add new pair contributions.
                for i in range(len(out) - 1):
                    p = (out[i], out[i + 1])
                    pair_counts[p] += freq
                    pair_words[p].add(w)
            pair_counts.pop(best, None)
            pair_words.pop(best, None)

        self.ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Build final vocab: specials, then alphabet, then merged tokens.
        tokens = list(base_tokens)
        seen = set(tokens)
        for a, b in self.merges:
            tok = a + b
            if tok not in seen:
                seen.add(tok)
                tokens.append(tok)
        self.vocab = {t: i for i, t in enumerate(tokens)}

        # Theme tokens at the end.
        self.themes = {}
        for th in (themes or []):
            name = f"<thm:{th.upper()}>"
            self.vocab[name] = len(self.vocab)
            self.themes[th.upper()] = self.vocab[name]

        self.inv = {i: t for t, i in self.vocab.items()}
        self._cache = {}

    # --- encoding --------------------------------------------------------
    def _bpe_word(self, word: str) -> List[str]:
        if word in self._cache:
            return self._cache[word]
        syms = list(word) + [END]
        while len(syms) >= 2:
            best_rank = None
            best_i = -1
            for i in range(len(syms) - 1):
                rank = self.ranks.get((syms[i], syms[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i < 0:
                break
            syms = syms[:best_i] + [syms[best_i] + syms[best_i + 1]] + syms[best_i + 2:]
        self._cache[word] = syms
        return syms

    def encode_text(self, text: str) -> List[int]:
        ids: List[int] = []
        for w in _words(text):
            for sym in self._bpe_word(w):
                ids.append(self.vocab.get(sym, UNK_ID))
        return ids

    def decode(self, ids: List[int]) -> str:
        out: List[str] = []
        for i in ids:
            tok = self.inv.get(int(i), "")
            if tok in SPECIALS or tok.startswith("<thm:"):
                continue
            out.append(tok)
        text = "".join(out).replace(END, " ")
        return re.sub(r"\s+", " ", text).strip()

    def theme_id(self, theme: str) -> int:
        return self.themes.get(theme.upper(), UNK_ID)

    @property
    def size(self) -> int:
        return len(self.vocab)

    # --- persistence -----------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "merges": self.merges,
            "vocab": self.vocab,
            "themes": self.themes,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @staticmethod
    def from_dict(d: Dict) -> "BPETokenizer":
        tk = BPETokenizer()
        tk.merges = [tuple(m) for m in d["merges"]]
        tk.ranks = {pair: i for i, pair in enumerate(tk.merges)}
        tk.vocab = {k: int(v) for k, v in d["vocab"].items()}
        tk.themes = {k: int(v) for k, v in d["themes"].items()}
        tk.inv = {i: t for t, i in tk.vocab.items()}
        return tk

    @staticmethod
    def load(path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            return BPETokenizer.from_dict(json.load(f))
