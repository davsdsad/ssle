"""N-gram Transition Matrix with backoff and learned logits.

See documentation section 4.6.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Sequence, Tuple

import numpy as np

Context = Tuple[int, ...]


class TransitionMatrix:
    def __init__(self, n_order: int = 3, smoothing: float = 0.1):
        self.n_order = n_order
        self.smoothing = smoothing
        # counts[context][next_token] = frequency
        self.counts: Dict[Context, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        # learned_logits[context][next_token] = adjustment learned by SGD
        self.logits: Dict[Context, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    # ------------------------------------------------------------------ #
    # Counting
    # ------------------------------------------------------------------ #
    def update_counts(self, ids: Sequence[int], weight: float = 1.0) -> None:
        """Accumulate n-gram counts for all orders from 1..n_order."""
        ids = list(ids)
        for t in range(1, len(ids)):
            nxt = ids[t]
            for order in range(1, self.n_order + 1):
                start = max(0, t - (order - 1))
                ctx = tuple(ids[start:t])
                self.counts[ctx][nxt] += weight

    # ------------------------------------------------------------------ #
    # Logit lookup with backoff
    # ------------------------------------------------------------------ #
    def _resolve_context(self, context: Sequence[int]) -> Context | None:
        """Find the longest context (up to n_order-1) that has counts."""
        context = list(context)
        for order in range(self.n_order, 0, -1):
            take = order - 1
            ctx = tuple(context[-take:]) if take > 0 else tuple()
            if ctx in self.counts and self.counts[ctx]:
                return ctx
        # Fall back to empty (unigram) context if present.
        if tuple() in self.counts and self.counts[tuple()]:
            return tuple()
        return None

    def get_logits(self, context: Sequence[int], vocab_size: int) -> np.ndarray:
        """Return a dense logit vector over the vocabulary for ``context``.

        logit(j|ctx) = log(count(j|ctx) + smoothing) + learned_logit(j|ctx)
        """
        out = np.full(vocab_size, math.log(self.smoothing), dtype=np.float32)
        ctx = self._resolve_context(context)
        if ctx is None:
            return out
        ctx_counts = self.counts[ctx]
        ctx_logits = self.logits.get(ctx, {})
        for tok, cnt in ctx_counts.items():
            if tok >= vocab_size:
                continue
            out[tok] = math.log(cnt + self.smoothing) + ctx_logits.get(tok, 0.0)
        # Apply learned logits even where there is no count (rare).
        for tok, lg in ctx_logits.items():
            if tok < vocab_size and tok not in ctx_counts:
                out[tok] += lg
        return out

    def resolved_context(self, context: Sequence[int]) -> Context:
        ctx = self._resolve_context(context)
        return ctx if ctx is not None else tuple()

    def update_logit(self, context: Context, token: int, delta: float) -> None:
        self.logits[context][token] += delta

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        def enc(d: Dict[Context, Dict[int, float]]) -> Dict[str, Dict[str, float]]:
            out: Dict[str, Dict[str, float]] = {}
            for ctx, m in d.items():
                if not m:
                    continue
                key = ",".join(str(x) for x in ctx)
                out[key] = {str(k): round(float(v), 5) for k, v in m.items()}
            return out

        return {"n_order": self.n_order, "smoothing": self.smoothing,
                "counts": enc(self.counts), "logits": enc(self.logits)}

    @classmethod
    def from_dict(cls, data: dict) -> "TransitionMatrix":
        m = cls(int(data["n_order"]), float(data["smoothing"]))

        def dec(src: Dict[str, Dict[str, float]], dst: Dict[Context, Dict[int, float]]) -> None:
            for key, mm in src.items():
                ctx = tuple(int(x) for x in key.split(",")) if key else tuple()
                for k, v in mm.items():
                    dst[ctx][int(k)] = float(v)

        dec(data.get("counts", {}), m.counts)
        dec(data.get("logits", {}), m.logits)
        return m
