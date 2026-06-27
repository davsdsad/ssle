"""Autoregressive generation with temperature / top-k / top-p sampling."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .data import _keywords  # noqa: F401  (kept for API parity / reuse)
from .model import NexusLM
from .tokenizer import (
    BOS_ID,
    EOS_ID,
    KW_ID,
    PAD_ID,
    SEP_ID,
    UNK_ID,
    BPETokenizer,
)


def _filter_logits(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    logits = logits.copy()
    if top_k > 0:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    if 0.0 < top_p < 1.0:
        order = np.argsort(logits)[::-1]
        probs = np.exp(logits[order] - np.max(logits[order]))
        probs /= probs.sum()
        cum = np.cumsum(probs)
        cut = np.searchsorted(cum, top_p) + 1
        keep = order[:cut]
        mask = np.ones_like(logits, dtype=bool)
        mask[keep] = False
        logits[mask] = -np.inf
    return logits


def generate(model: NexusLM, tk: BPETokenizer, theme: str,
             keywords: Optional[List[str]] = None, max_new: int = 24,
             temperature: float = 0.9, top_k: int = 40, top_p: float = 0.92,
             repetition_penalty: float = 1.3, no_repeat_bigram: bool = True,
             seed: Optional[int] = None) -> str:
    rng = np.random.default_rng(seed)
    seq: List[int] = [BOS_ID, tk.theme_id(theme)]
    if keywords:
        seq.append(KW_ID)
        for w in keywords:
            seq.extend(tk.encode_text(w))
    seq.append(SEP_ID)
    prompt_len = len(seq)  # body (the title) is everything generated after this

    banned = {PAD_ID, BOS_ID, SEP_ID, KW_ID, UNK_ID}
    banned.update(tk.themes.values())

    for _ in range(max_new):
        idx = np.array([seq[-model.cfg.max_len:]], dtype=np.int64)
        logits = model(idx).data[0, -1]            # (V,)
        for b in banned:
            logits[b] = -np.inf
        # Discourage verbatim repetition of tokens already in the body.
        body = seq[prompt_len:]
        if repetition_penalty != 1.0 and body:
            for t in set(body):
                if logits[t] > 0:
                    logits[t] /= repetition_penalty
                else:
                    logits[t] *= repetition_penalty
        # Forbid recreating a bigram that already occurred (kills loops).
        if no_repeat_bigram and body:
            prev = seq[-1]
            for a, b in zip(body[:-1], body[1:]):
                if a == prev:
                    logits[b] = -np.inf
        if temperature > 0:
            logits = logits / temperature
        logits = _filter_logits(logits, top_k, top_p)
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        nxt = int(rng.choice(len(probs), p=probs))
        if nxt == EOS_ID:
            break
        seq.append(nxt)

    return tk.decode(seq[prompt_len:])
