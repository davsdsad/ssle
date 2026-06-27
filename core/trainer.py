"""Training loop, loss and gradient updates for SSLE-1 (documentation 5)."""

from __future__ import annotations

import math
import random
import time
from typing import Callable, List, Sequence

import numpy as np

from .dataset import Sample
from .engine import SSLEEngine
from .sampler import softmax
from .tokenizer import normalize


def nll_loss(prob_target: float) -> float:
    return -math.log(max(prob_target, 1e-12))


class Trainer:
    """Implements the per-epoch algorithm from documentation 5.1.

    Phases per sample:
        1. update n-gram counts
        2. update concept graph (co-occurrence)
        3. token-by-token gradient update (matrix learned logits + embeddings)
    """

    def __init__(self, engine: SSLEEngine, lr: float | None = None,
                 neg_samples: int = 4, seed: int = 42):
        self.engine = engine
        self.lr = lr if lr is not None else engine.config.lr
        self.neg_samples = neg_samples
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    # ------------------------------------------------------------------ #
    def _gradient_step(self, ids: Sequence[int], t: int, sample_bias: np.ndarray,
                       seen: set) -> float:
        eng = self.engine
        vocab_size = eng.tokenizer.vocab_size
        recent = list(ids[max(0, t - (eng.config.n_order - 1)):t])
        target = ids[t]

        logits = eng.matrix.get_logits(recent, vocab_size)
        if sample_bias is not None:
            logits = logits + sample_bias
        probs = softmax(logits, temperature=1.0)
        p_t = float(probs[target]) if target < vocab_size else 1e-12
        loss = nll_loss(p_t)

        # --- Matrix learned-logit gradient (sparse) ---
        ctx = eng.matrix.resolved_context(recent)
        # Increase target, decrease the strongest competitors.
        eng.matrix.update_logit(ctx, target, self.lr * (1.0 - p_t))
        top = np.argpartition(probs, -6)[-6:]
        for j in top:
            j = int(j)
            if j == target:
                continue
            eng.matrix.update_logit(ctx, j, -self.lr * float(probs[j]))

        # --- Embedding contrastive update ---
        # Pull target embedding toward recent-context centroid; push negatives.
        if recent:
            c = eng.embeddings.weights[recent].mean(axis=0)
            cn = np.linalg.norm(c)
            if cn > 0:
                c_unit = c / cn
                tv = eng.embeddings.vec(target)
                eng.embeddings.weights[target] += self.lr * (c_unit - 0.0 * tv)
                for _ in range(self.neg_samples):
                    neg = int(self.rng.integers(5, vocab_size))
                    if neg == target:
                        continue
                    eng.embeddings.weights[neg] -= self.lr * 0.5 * c_unit
        return loss

    # ------------------------------------------------------------------ #
    def train(self, samples: List[Sample], epochs: int = 10,
              log: Callable[[str], None] = print) -> dict:
        eng = self.engine
        history: List[float] = []
        t_start = time.time()

        # Phases 1 & 2 (counts + concept graph) only need one pass over data,
        # but we follow the documented loop and let counts accumulate per epoch
        # would double-count; so we do counts/graph once up front.
        log("Phase 1+2: building n-gram counts and concept graph...")
        for s in samples:
            ids = eng.tokenizer.encode(s.target)
            eng.matrix.update_counts(ids, weight=s.weight)
            eng.semantic.update_graph(ids)
            eng.memory.observe(normalize(s.target).split())
            # Theme-conditioned token prior (also include context tokens).
            ctx_ids = [eng.tokenizer.token_id(c) for c in s.context]
            eng.theme_profiles.observe(s.theme, ids + ctx_ids, weight=s.weight)
        eng.semantic.finalize_graph()

        log(f"Phase 3: gradient training for {epochs} epoch(s)...")
        for epoch in range(epochs):
            order = list(range(len(samples)))
            random.shuffle(order)
            total_loss = 0.0
            n_tokens = 0
            for si in order:
                s = samples[si]
                ids = eng.tokenizer.encode(s.target)
                # Sample-level context bias (computed once, documentation 5.1).
                ctx_ids = [eng.tokenizer.token_id(t)
                           for t in (normalize(s.theme).split()
                                     + [c for c in s.context])]
                ctx_ids = [c for c in ctx_ids if c not in (0, 1)]
                if ctx_ids:
                    ctx_vec = eng.encoder.encode(ctx_ids)
                    sample_bias = eng.encoder.context_bias(ctx_vec)
                else:
                    sample_bias = None
                seen: set = set()
                for t in range(1, len(ids)):
                    loss = self._gradient_step(ids, t, sample_bias, seen)
                    total_loss += loss
                    n_tokens += 1
                    seen.add(ids[t])
            avg_loss = total_loss / max(n_tokens, 1)
            history.append(avg_loss)
            log(f"  Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.4f}")

        elapsed = time.time() - t_start
        stats = {
            "epochs": epochs,
            "final_loss": round(history[-1], 4) if history else None,
            "loss_history": [round(x, 4) for x in history],
            "dataset_size": len(samples),
            "train_seconds": round(elapsed, 2),
            "vocab_size": eng.tokenizer.vocab_size,
        }
        eng.training_stats = stats
        return stats
