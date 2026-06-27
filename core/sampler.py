"""Sampling Engine for SSLE-1 (documentation 4.8).

Converts logits into a probability distribution and samples the next token,
supporting temperature, top-k, top-p (nucleus) and repeat penalty.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(temperature, 1e-6)
    z = logits / t
    z = z - np.max(z)
    e = np.exp(z)
    s = e.sum()
    if s == 0.0:
        return np.full_like(e, 1.0 / e.shape[0])
    return e / s


def apply_repeat_penalty(logits: np.ndarray, seen: Dict[int, int],
                         repeat_decay: float) -> np.ndarray:
    """Multiplicatively penalize tokens already produced.

    repeat_decay == 1.0 means no penalty. Lower values penalize more.
    Penalty compounds with how many times a token was seen.
    """
    if repeat_decay >= 1.0 or not seen:
        return logits
    out = logits.copy()
    for tok, cnt in seen.items():
        if tok >= out.shape[0]:
            continue
        factor = repeat_decay ** cnt
        # For positive logits, shrink; for negative, push further down.
        if out[tok] > 0:
            out[tok] *= factor
        else:
            out[tok] -= (1.0 - factor) * abs(out[tok])
    return out


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= probs.shape[0]:
        return probs
    out = np.zeros_like(probs)
    idx = np.argpartition(probs, -k)[-k:]
    out[idx] = probs[idx]
    s = out.sum()
    return out / s if s > 0 else probs


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    if p >= 1.0:
        return probs
    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]
    cum = np.cumsum(sorted_probs)
    # Keep the smallest set whose cumulative prob >= p (always >= 1 token).
    cutoff = np.searchsorted(cum, p) + 1
    keep = order[:cutoff]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    s = out.sum()
    return out / s if s > 0 else probs


class Sampler:
    def __init__(self, top_k: int = 10, top_p: float = 0.9, temperature: float = 0.8,
                 repeat_decay: float = 0.7, seed: int | None = None):
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repeat_decay = repeat_decay
        self.rng = np.random.default_rng(seed)

    def sample(self, logits: np.ndarray, seen: Dict[int, int] | None = None,
               greedy: bool = False) -> int:
        seen = seen or {}
        logits = apply_repeat_penalty(logits, seen, self.repeat_decay)
        if greedy or self.top_k == 1:
            return int(np.argmax(logits))
        probs = softmax(logits, self.temperature)
        probs = top_k_filter(probs, self.top_k)
        probs = top_p_filter(probs, self.top_p)
        s = probs.sum()
        if s <= 0:
            return int(np.argmax(logits))
        probs = probs / s
        return int(self.rng.choice(len(probs), p=probs))
