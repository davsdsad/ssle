"""Optimizers — Adam implemented from scratch."""

from __future__ import annotations

from typing import List

import numpy as np

from .module import Parameter


class Adam:
    def __init__(self, params: List[Parameter], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, clip: float = 5.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip = clip
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        if self.clip > 0:
            total = 0.0
            for p in self.params:
                if p.grad is not None:
                    total += float(np.sum(p.grad * p.grad))
            norm = np.sqrt(total)
            scale = self.clip / (norm + 1e-6) if norm > self.clip else 1.0
        else:
            scale = 1.0
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad * scale
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data = p.data - self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
