"""Long-Range Context Buffer for SSLE-1 (documentation 4.5).

Provides unlimited context that persists for an entire generation session,
operating in four simultaneous layers:
    1. Recent context (N-gram) -- handled by the matrix, exposed here as window
    2. Sliding window (last K tokens)
    3. Compressed summary (weighted average of all past embeddings)
    4. Semantic anchors (high-importance tokens that always influence sampling)
"""

from __future__ import annotations

from typing import List, Sequence, Set

import numpy as np

from .encoder import ContextEncoder, Embeddings


class LongRangeContextBuffer:
    def __init__(self, embeddings: Embeddings, encoder: ContextEncoder,
                 buffer_window: int = 50, compression_ratio: int = 50,
                 decay_factor: float = 0.995, anchor_weight: float = 0.5):
        self.embeddings = embeddings
        self.encoder = encoder
        self.buffer_window = buffer_window
        self.compression_ratio = compression_ratio
        self.decay_factor = decay_factor
        self.anchor_weight = anchor_weight

        self.full_history: List[int] = []
        self.compressed_summary = np.zeros(embeddings.dim, dtype=np.float32)
        self.window_recent: List[int] = []
        self.semantic_anchors: Set[int] = set()

    def reset(self, anchors: Sequence[int] | None = None) -> None:
        self.full_history = []
        self.compressed_summary = np.zeros(self.embeddings.dim, dtype=np.float32)
        self.window_recent = []
        self.semantic_anchors = set(anchors or [])

    def add_anchor(self, token: int) -> None:
        self.semantic_anchors.add(token)

    def append(self, token: int) -> None:
        self.full_history.append(token)
        self.window_recent.append(token)
        if len(self.window_recent) > self.buffer_window:
            self.window_recent = self.window_recent[-self.buffer_window:]
        if len(self.full_history) % self.compression_ratio == 0:
            self._compress()

    def _compress(self) -> None:
        n = len(self.full_history)
        if n == 0:
            return
        # Recent tokens get higher weight: decay_factor ** (n - i).
        idx = np.arange(n)
        weights = self.decay_factor ** (n - 1 - idx)
        weights = weights / (weights.sum() or 1.0)
        vecs = self.embeddings.weights[self.full_history]
        self.compressed_summary = (weights[:, None] * vecs).sum(axis=0).astype(np.float32)

    def get_vector(self) -> np.ndarray:
        """Representative vector of the whole history so far.

        Blends the sliding window mean with the compressed summary so callers
        always get a meaningful long-range context vector.
        """
        if not self.full_history:
            return np.zeros(self.embeddings.dim, dtype=np.float32)
        window_vec = self.embeddings.weights[self.window_recent].mean(axis=0)
        if np.linalg.norm(self.compressed_summary) == 0.0:
            return window_vec.astype(np.float32)
        return (0.5 * window_vec + 0.5 * self.compressed_summary).astype(np.float32)

    def context_bias(self, concept_graph=None) -> np.ndarray:
        """Per-token bias from the long-range context + semantic anchors."""
        bias = self.encoder.context_bias(self.get_vector())
        if concept_graph is not None and self.semantic_anchors:
            for anchor in self.semantic_anchors:
                for tok, s in concept_graph.edges.get(anchor, {}).items():
                    if tok < bias.shape[0]:
                        bias[tok] += self.anchor_weight * s
        return bias
