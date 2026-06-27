"""Module / Parameter system and core layers — built on the autograd Tensor."""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import numpy as np

from .autograd import Tensor


class Parameter(Tensor):
    """A Tensor that is a learnable parameter (tracked by optimizers)."""


def xavier(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    fan_in = shape[0]
    fan_out = shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


class Module:
    """Base class. Subclasses register Parameters / sub-Modules as attributes."""

    def parameters(self) -> Iterator[Parameter]:
        seen: set[int] = set()
        for _, v in self.__dict__.items():
            if isinstance(v, Parameter):
                if id(v) not in seen:
                    seen.add(id(v))
                    yield v
            elif isinstance(v, Module):
                for p in v.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        yield p
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        for p in item.parameters():
                            if id(p) not in seen:
                                seen.add(id(p))
                                yield p

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        for name, v in self.__dict__.items():
            full = f"{prefix}{name}"
            if isinstance(v, Parameter):
                yield full, v
            elif isinstance(v, Module):
                yield from v.named_parameters(prefix=full + ".")
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if isinstance(item, Module):
                        yield from item.named_parameters(prefix=f"{full}.{i}.")

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> Dict[str, list]:
        return {name: p.data.tolist() for name, p in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, list]) -> None:
        params = dict(self.named_parameters())
        for name, arr in state.items():
            if name in params:
                params[name].data = np.asarray(arr, dtype=np.float32)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator, bias: bool = True):
        self.W = Parameter(xavier((in_dim, out_dim), rng))
        self.b = Parameter(np.zeros((out_dim,), dtype=np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out


class Embedding(Module):
    def __init__(self, vocab: int, dim: int, rng: np.random.Generator):
        self.weight = Parameter(rng.normal(0.0, 0.02, size=(vocab, dim)).astype(np.float32))

    def forward(self, idx: np.ndarray) -> Tensor:
        return self.weight.gather_rows(idx)


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.g = Parameter(np.ones((dim,), dtype=np.float32))
        self.b = Parameter(np.zeros((dim,), dtype=np.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        inv = (var + self.eps) ** -0.5
        return xc * inv * self.g + self.b


def parameter_count(module: Module) -> int:
    return int(sum(p.data.size for p in module.parameters()))
