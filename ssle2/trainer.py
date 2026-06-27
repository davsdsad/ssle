"""Training loop for NexusLM (teacher forcing + masked cross-entropy)."""

from __future__ import annotations

import gc
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from nn.autograd import Tensor
from nn.optim import Adam

from .model import NexusLM


def masked_cross_entropy(logits: Tensor, targets: np.ndarray,
                         weights: np.ndarray) -> Tensor:
    """Weighted mean NLL. logits (N,V), targets (N,), weights (N,)."""
    n, v = logits.shape
    probs = logits.softmax(axis=-1)
    onehot = np.zeros((n, v), dtype=np.float32)
    onehot[np.arange(n), targets] = 1.0
    sel = (probs * Tensor(onehot, requires_grad=False)).sum(axis=-1)  # (N,)
    nll = sel.log() * -1.0
    w = Tensor(weights, requires_grad=False)
    denom = float(weights.sum()) + 1e-8
    return (nll * w).sum() * (1.0 / denom)


def evaluate(model: NexusLM, batches: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """Average masked cross-entropy over a list of (idx, mask) batches."""
    total = 0.0
    weight = 0.0
    for idx, mask in batches:
        inp = idx[:, :-1]
        tgt = idx[:, 1:]
        wmask = mask[:, 1:]
        B, T = inp.shape
        logits = model(inp)
        V = logits.shape[-1]
        loss = masked_cross_entropy(
            logits.reshape(B * T, V),
            tgt.reshape(B * T).astype(np.int64),
            wmask.reshape(B * T),
        )
        w = float(wmask.sum())
        total += float(loss.data) * w
        weight += w
        loss.free_graph()
        del logits, loss
    gc.collect()
    return total / max(weight, 1e-8)


def train(model: NexusLM, batches_fn: Callable[[], List[Tuple[np.ndarray, np.ndarray]]],
          epochs: int, lr: float = 2e-3, weight_decay: float = 1e-5,
          log_every: int = 50, on_epoch: Optional[Callable[[int, float], None]] = None
          ) -> List[float]:
    opt = Adam(list(model.parameters()), lr=lr, weight_decay=weight_decay)
    history: List[float] = []
    for epoch in range(epochs):
        total = 0.0
        count = 0
        t0 = time.time()
        for step, (idx, mask) in enumerate(batches_fn()):
            inp = idx[:, :-1]
            tgt = idx[:, 1:]
            wmask = mask[:, 1:]
            B, T = inp.shape
            logits = model(inp)                      # (B,T,V)
            V = logits.shape[-1]
            loss = masked_cross_entropy(
                logits.reshape(B * T, V),
                tgt.reshape(B * T).astype(np.int64),
                wmask.reshape(B * T),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            lv = float(loss.data)
            total += lv
            count += 1
            if log_every and step % log_every == 0:
                print(f"  epoch {epoch+1} step {step} loss {lv:.4f}", flush=True)
            loss.free_graph()
            del logits, loss
            if step % 200 == 0:
                gc.collect()
        avg = total / max(count, 1)
        history.append(avg)
        dt = time.time() - t0
        print(f"epoch {epoch+1}/{epochs} avg_loss {avg:.4f} ({dt:.1f}s)", flush=True)
        if on_epoch:
            on_epoch(epoch, avg)
    return history
