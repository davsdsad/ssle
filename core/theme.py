"""Theme profiles for SSLE-1.

A theme-conditioned token prior learned directly from the dataset: for each
theme, how characteristic each token is of that theme's content. This is a
robust, data-driven theme-steering signal (unlike embedding cosine, which is
noisy early in training) and is added to the logits during generation so that
the requested theme actually controls the vocabulary used.

The prior is a smoothed pointwise-mutual-information (PMI) score:

    prior(token | theme) = log( P(token | theme) / P(token) )

so that theme-characteristic tokens get a positive boost, off-theme tokens get
a negative push, and theme-neutral / structural tokens (including EOS) stay
near zero — which lets generations terminate naturally instead of running on.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from .tokenizer import PAD_ID, UNK_ID

_SMOOTH = 0.5


class ThemeProfiles:
    def __init__(self):
        # counts[theme][token_id] = weighted frequency in that theme's content
        self.counts: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        # global counts across all themes (for the PMI denominator)
        self.global_counts: Dict[int, float] = defaultdict(float)

    def observe(self, theme: str, ids: Sequence[int], weight: float = 1.0) -> None:
        if not theme:
            return
        theme = theme.upper()
        for tid in ids:
            if tid in (PAD_ID, UNK_ID):
                continue
            self.counts[theme][tid] += weight
            self.global_counts[tid] += weight

    def prior(self, theme: str, vocab_size: int) -> np.ndarray:
        """Bounded PMI log-prior over the vocabulary for ``theme``."""
        out = np.zeros(vocab_size, dtype=np.float32)
        if not theme:
            return out
        theme = theme.upper()
        m = self.counts.get(theme)
        if not m:
            return out
        total_theme = sum(m.values())
        total_global = sum(self.global_counts.values()) or 1.0
        v = max(vocab_size, 1)
        denom_theme = total_theme + _SMOOTH * v
        denom_global = total_global + _SMOOTH * v
        for tid, c in m.items():
            if tid >= vocab_size:
                continue
            p_theme = (c + _SMOOTH) / denom_theme
            p_global = (self.global_counts.get(tid, 0.0) + _SMOOTH) / denom_global
            out[tid] = math.log(p_theme / p_global)
        return out

    def themes(self) -> List[str]:
        return list(self.counts.keys())

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "counts": {theme: {str(t): round(float(c), 4) for t, c in m.items()}
                       for theme, m in self.counts.items() if m},
            "global": {str(t): round(float(c), 4) for t, c in self.global_counts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ThemeProfiles":
        tp = cls()
        if not data:
            return tp
        # Support both the new {counts, global} layout and the legacy flat one.
        counts = data.get("counts", data)
        for theme, m in counts.items():
            if theme in ("counts", "global"):
                continue
            for t, c in m.items():
                tid = int(t)
                tp.counts[theme.upper()][tid] = float(c)
        glob = data.get("global")
        if glob:
            for t, c in glob.items():
                tp.global_counts[int(t)] = float(c)
        else:
            # Rebuild global counts from per-theme counts (legacy models).
            for m in tp.counts.values():
                for tid, c in m.items():
                    tp.global_counts[tid] += c
        return tp
