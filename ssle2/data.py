"""Corpus loading, theme labelling and training-sample construction.

A sample sequence is:
    <bos> <thm:THEME> [kw subwords] <sep> [title subwords] <eos>
Loss is applied only from <sep> onward (the model is scored on generating the
title given the theme + keywords), so the prompt tokens are masked out.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np

from .tokenizer import BOS_ID, EOS_ID, KW_ID, PAD_ID, SEP_ID, BPETokenizer, normalize

# Source file -> theme label.
SOURCE_THEME = {
    "turismo": "TURISMO",
    "esporte": "ESPORTE",
    "cultura": "CULTURA",
    "anvisa": "SAUDE",
    "clickbait": "GERAL",
}

_STOP = {
    "para", "como", "mais", "está", "esta", "pela", "pelo", "dos", "das", "com",
    "que", "uma", "uns", "umas", "por", "sobre", "ser", "sua", "seu", "suas",
    "seus", "the", "and", "ate", "até", "nos", "nas", "aos", "este", "esse",
}


def load_corpus(raw_dir: str, min_words: int = 3, max_words: int = 18,
                max_chars: int = 110) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".txt"):
            continue
        theme = SOURCE_THEME.get(fname[:-4], "GERAL")
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
            for line in f:
                title = line.strip()
                norm = normalize(title)
                wc = len(norm.split())
                if min_words <= wc <= max_words and len(norm) <= max_chars:
                    pairs.append((theme, title))
    return pairs


def theme_list(pairs: List[Tuple[str, str]]) -> List[str]:
    return sorted({t for t, _ in pairs})


def _keywords(title: str, k: int, rng: random.Random) -> List[str]:
    words = [w for w in normalize(title).split() if len(w) >= 4 and w not in _STOP]
    if not words or k == 0:
        return []
    rng.shuffle(words)
    return words[:k]


def build_sample(tk: BPETokenizer, theme: str, title: str,
                 rng: random.Random, max_len: int) -> Tuple[List[int], List[int]]:
    """Return (token_ids, loss_mask) for one sample (mask aligns with tokens)."""
    # 40% no keywords, else 1-3 keywords drawn from the title.
    k = 0 if rng.random() < 0.4 else rng.randint(1, 3)
    kws = _keywords(title, k, rng)

    prompt = [BOS_ID, tk.theme_id(theme)]
    if kws:
        prompt.append(KW_ID)
        for w in kws:
            prompt.extend(tk.encode_text(w))
    prompt.append(SEP_ID)

    body = tk.encode_text(title) + [EOS_ID]
    ids = prompt + body
    # mask: 0 on prompt positions, 1 on body positions.
    mask = [0] * len(prompt) + [1] * len(body)
    if len(ids) > max_len:
        ids = ids[:max_len]
        mask = mask[:max_len]
    return ids, mask


def make_batches(samples: List[Tuple[List[int], List[int]]], batch_size: int,
                 max_len: int, rng: random.Random, shuffle: bool = True):
    order = list(range(len(samples)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        if len(chunk) < 2:
            continue
        T = max(len(samples[i][0]) for i in chunk)
        T = min(T, max_len)
        B = len(chunk)
        idx = np.full((B, T), PAD_ID, dtype=np.int64)
        mask = np.zeros((B, T), dtype=np.float32)
        for r, i in enumerate(chunk):
            ids, m = samples[i]
            ids = ids[:T]
            m = m[:T]
            idx[r, :len(ids)] = ids
            mask[r, :len(m)] = m
        yield idx, mask
