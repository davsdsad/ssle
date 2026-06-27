"""Embedding layer (learned) + Context Encoder for SSLE-1.

See documentation sections 4.2 and 4.3.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class Embeddings:
    """Learned token embeddings, Xavier-initialized, updated by SGD."""

    def __init__(self, vocab_size: int, dim: int = 128, lr: float = 0.001, seed: int = 42):
        self.vocab_size = vocab_size
        self.dim = dim
        self.lr = lr
        scale = math.sqrt(2.0 / (vocab_size + dim))
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, scale, size=(vocab_size, dim)).astype(np.float32)

    def vec(self, token_id: int) -> np.ndarray:
        return self.weights[token_id]

    def update(self, token_id: int, gradient: np.ndarray) -> None:
        """SGD step: embedding[i] = embedding[i] - lr * gradient[i]."""
        self.weights[token_id] -= self.lr * gradient

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        # Store as a compact list-of-lists (rounded to keep file small).
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "weights": np.round(self.weights, 5).tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict, lr: float = 0.001) -> "Embeddings":
        emb = cls(int(data["vocab_size"]), int(data["dim"]), lr=lr)
        emb.weights = np.asarray(data["weights"], dtype=np.float32)
        return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ContextEncoder:
    """Turns a list of context tokens into a single representative vector.

    Uses a weighted average of embeddings, and produces a per-token context
    bias from cosine similarity (documentation 4.3).
    """

    def __init__(self, embeddings: Embeddings, strength: float = 1.0):
        self.embeddings = embeddings
        self.strength = strength

    def encode(self, token_ids: Sequence[int],
               weights: Sequence[float] | None = None) -> np.ndarray:
        if len(token_ids) == 0:
            return np.zeros(self.embeddings.dim, dtype=np.float32)
        if weights is None:
            weights = [1.0] * len(token_ids)
        total = float(sum(weights)) or 1.0
        acc = np.zeros(self.embeddings.dim, dtype=np.float32)
        for tid, w in zip(token_ids, weights):
            acc += (w / total) * self.embeddings.vec(tid)
        return acc

    def context_bias(self, context_vec: np.ndarray) -> np.ndarray:
        """Vector of cosine-sim-based bias for every token in the vocab."""
        if np.linalg.norm(context_vec) == 0.0:
            return np.zeros(self.embeddings.vocab_size, dtype=np.float32)
        W = self.embeddings.weights
        norms = np.linalg.norm(W, axis=1)
        cn = np.linalg.norm(context_vec)
        denom = norms * cn
        denom[denom == 0.0] = 1e-9
        sims = (W @ context_vec) / denom
        return (sims * self.strength).astype(np.float32)
