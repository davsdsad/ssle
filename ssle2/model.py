"""SSLE-2 "Nexus" — a from-scratch attention + reasoning language model.

Deliberately NOT a transformer, NOT an RNN, NOT n-gram. Distinctive choices:

* **Resonance attention** — causal multi-head content attention, but wrapped in
  a GLU gate so each position decides how much attended context to admit.
* **Concept memory** — a bank of learnable key/value slots every position can
  read from (global, sequence-independent "knowledge"), separate from
  self-attention.
* **Weight-tied reasoning loop** — instead of stacking N independent layers, the
  SAME block is applied R times as an iterative refinement ("thinking") of a
  latent state. Depth = reasoning steps, not parameter count.

Everything runs on the hand-written NumPy autograd in `nn/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from nn.autograd import Tensor
from nn.module import Embedding, LayerNorm, Linear, Module, Parameter, xavier


@dataclass
class NexusConfig:
    vocab_size: int = 4000
    dim: int = 128
    heads: int = 4
    reasoning_steps: int = 3
    memory_slots: int = 64
    ffn_mult: int = 3
    max_len: int = 48
    dropout: float = 0.0  # reserved; inference is deterministic

    def to_dict(self) -> Dict:
        return {
            "vocab_size": self.vocab_size, "dim": self.dim, "heads": self.heads,
            "reasoning_steps": self.reasoning_steps, "memory_slots": self.memory_slots,
            "ffn_mult": self.ffn_mult, "max_len": self.max_len, "dropout": self.dropout,
        }

    @staticmethod
    def from_dict(d: Dict) -> "NexusConfig":
        return NexusConfig(**{k: d[k] for k in d if k in NexusConfig.__annotations__})


class ResonanceAttention(Module):
    """Causal multi-head attention gated by a GLU on the query stream."""

    def __init__(self, dim: int, heads: int, rng: np.random.Generator):
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.scale = 1.0 / np.sqrt(self.dh)
        self.Wq = Linear(dim, dim, rng, bias=False)
        self.Wk = Linear(dim, dim, rng, bias=False)
        self.Wv = Linear(dim, dim, rng, bias=False)
        self.Wo = Linear(dim, dim, rng, bias=False)
        self.Wg = Linear(dim, dim, rng)  # gate

    def _split(self, x: Tensor, B: int, T: int) -> Tensor:
        return x.reshape(B, T, self.heads, self.dh).swapaxes(1, 2)  # (B,H,T,dh)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, _ = x.shape
        q = self._split(self.Wq(x), B, T)
        k = self._split(self.Wk(x), B, T)
        v = self._split(self.Wv(x), B, T)
        scores = (q @ k.swapaxes(-1, -2)) * self.scale + mask  # (B,H,T,T)
        attn = scores.softmax(axis=-1)
        ctx = attn @ v                                          # (B,H,T,dh)
        ctx = ctx.swapaxes(1, 2).reshape(B, T, self.dim)        # (B,T,D)
        gate = self.Wg(x).sigmoid()
        return self.Wo(ctx * gate)


class ConceptMemory(Module):
    """Learnable key/value memory slots read by content-based attention."""

    def __init__(self, dim: int, slots: int, rng: np.random.Generator):
        self.dim = dim
        self.scale = 1.0 / np.sqrt(dim)
        self.keys = Parameter(xavier((slots, dim), rng))
        self.vals = Parameter(xavier((slots, dim), rng))
        self.Wq = Linear(dim, dim, rng, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.Wq(x).reshape(B * T, D)            # (B*T, D)
        scores = (q @ self.keys.transpose()) * self.scale   # (B*T, slots)
        attn = scores.softmax(axis=-1)
        out = attn @ self.vals                      # (B*T, D)
        return out.reshape(B, T, D)


class GatedFFN(Module):
    def __init__(self, dim: int, mult: int, rng: np.random.Generator):
        hidden = dim * mult
        self.W1 = Linear(dim, hidden, rng)
        self.Wg = Linear(dim, hidden, rng)
        self.W2 = Linear(hidden, dim, rng)

    def forward(self, x: Tensor) -> Tensor:
        return self.W2(self.W1(x).gelu() * self.Wg(x).sigmoid())


class ReasoningBlock(Module):
    """One refinement step: attention -> memory -> gated FFN, all residual."""

    def __init__(self, cfg: NexusConfig, rng: np.random.Generator):
        self.ln1 = LayerNorm(cfg.dim)
        self.attn = ResonanceAttention(cfg.dim, cfg.heads, rng)
        self.ln2 = LayerNorm(cfg.dim)
        self.mem = ConceptMemory(cfg.dim, cfg.memory_slots, rng)
        self.ln3 = LayerNorm(cfg.dim)
        self.ffn = GatedFFN(cfg.dim, cfg.ffn_mult, rng)

    def forward(self, h: Tensor, mask: Tensor) -> Tensor:
        h = h + self.attn(self.ln1(h), mask)
        h = h + self.mem(self.ln2(h))
        h = h + self.ffn(self.ln3(h))
        return h


class NexusLM(Module):
    def __init__(self, cfg: NexusConfig, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.tok = Embedding(cfg.vocab_size, cfg.dim, rng)
        self.pos = Embedding(cfg.max_len, cfg.dim, rng)
        self.block = ReasoningBlock(cfg, rng)   # weight-tied across reasoning steps
        self.ln_f = LayerNorm(cfg.dim)
        self.head = Linear(cfg.dim, cfg.vocab_size, rng, bias=True)
        self._mask_cache: Dict[int, Tensor] = {}

    def _causal_mask(self, T: int) -> Tensor:
        if T not in self._mask_cache:
            m = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
            self._mask_cache[T] = Tensor(m.reshape(1, 1, T, T), requires_grad=False)
        return self._mask_cache[T]

    def forward(self, idx: np.ndarray) -> Tensor:
        B, T = idx.shape
        positions = np.arange(T)
        h = self.tok(idx) + self.pos(positions)
        mask = self._causal_mask(T)
        for _ in range(self.cfg.reasoning_steps):
            h = self.block(h, mask)
        h = self.ln_f(h)
        return self.head(h)   # (B, T, vocab)
