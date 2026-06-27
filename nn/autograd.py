"""Mini reverse-mode autodiff over NumPy — built from scratch (no PyTorch).

A single `Tensor` type wraps an ndarray, records the operations that produced
it, and `backward()` walks the graph in reverse topological order accumulating
gradients. Broadcasting is handled by `_unbroadcast`, which sums gradients back
to the original shape. This is the foundation every layer in `ssle2` is built
on.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Set, Tuple, Union

import numpy as np

ArrayLike = Union["Tensor", np.ndarray, float, int]


def _as_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.data
    arr = np.asarray(x)
    # Preserve float64 (useful for gradient checking); otherwise default float32.
    if arr.dtype == np.float64:
        return arr
    return arr.astype(np.float32)


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum `grad` so it matches `shape` (reverse of NumPy broadcasting)."""
    if grad.shape == shape:
        return grad
    # Reduce extra leading dims.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Reduce broadcasted (size-1) dims.
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An ndarray that remembers how it was computed."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev")

    def __init__(self, data: ArrayLike, requires_grad: bool = True,
                 _children: Iterable["Tensor"] = ()):
        self.data: np.ndarray = _as_array(data)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple[Tensor, ...] = tuple(_children)

    # --- helpers ---------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def _accumulate(self, g: np.ndarray) -> None:
        if not self.requires_grad:
            return
        self.grad = g if self.grad is None else self.grad + g

    def zero_grad(self) -> None:
        self.grad = None

    # --- elementwise -----------------------------------------------------
    def __add__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, _children=(self, other))

        def _backward() -> None:
            self._accumulate(_unbroadcast(out.grad, self.shape))
            other._accumulate(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, _children=(self, other))

        def _backward() -> None:
            self._accumulate(_unbroadcast(out.grad * other.data, self.shape))
            other._accumulate(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __pow__(self, p: float) -> "Tensor":
        out = Tensor(self.data ** p, _children=(self,))

        def _backward() -> None:
            self._accumulate(out.grad * (p * self.data ** (p - 1)))

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: ArrayLike) -> "Tensor":
        o = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-o)

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** -1.0)

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return (-self) + other

    # --- matmul ----------------------------------------------------------
    def matmul(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, _children=(self, other))

        def _backward() -> None:
            g = out.grad
            a, b = self.data, other.data
            # Move last two dims for matmul transpose, supporting batched inputs.
            ga = g @ np.swapaxes(b, -1, -2)
            gb = np.swapaxes(a, -1, -2) @ g
            self._accumulate(_unbroadcast(ga, self.shape))
            other._accumulate(_unbroadcast(gb, other.shape))

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    # --- reductions / shape ---------------------------------------------
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
            keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _children=(self,))

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis if isinstance(axis, int) else tuple(axis))
            self._accumulate(np.broadcast_to(g, self.shape).copy())

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> "Tensor":
        n = self.data.size if axis is None else np.prod(
            [self.data.shape[a] for a in ([axis] if isinstance(axis, int) else axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), _children=(self,))

        def _backward() -> None:
            self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> "Tensor":
        out = Tensor(np.transpose(self.data, axes), _children=(self,))

        def _backward() -> None:
            if axes is None:
                self._accumulate(np.transpose(out.grad))
            else:
                inv = np.argsort(axes)
                self._accumulate(np.transpose(out.grad, inv))

        out._backward = _backward
        return out

    def swapaxes(self, a: int, b: int) -> "Tensor":
        out = Tensor(np.swapaxes(self.data, a, b), _children=(self,))

        def _backward() -> None:
            self._accumulate(np.swapaxes(out.grad, a, b))

        out._backward = _backward
        return out

    # --- activations -----------------------------------------------------
    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(e, _children=(self,))

        def _backward() -> None:
            self._accumulate(out.grad * e)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data + 1e-12), _children=(self,))

        def _backward() -> None:
            self._accumulate(out.grad / (self.data + 1e-12))

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,))

        def _backward() -> None:
            self._accumulate(out.grad * (1.0 - t * t))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, _children=(self,))

        def _backward() -> None:
            self._accumulate(out.grad * s * (1.0 - s))

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), _children=(self,))

        def _backward() -> None:
            self._accumulate(out.grad * (self.data > 0.0))

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        # tanh approximation of GELU.
        c = np.sqrt(2.0 / np.pi)
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = Tensor(0.5 * x * (1.0 + t), _children=(self,))

        def _backward() -> None:
            dt = (1.0 - t * t) * c * (1.0 + 3 * 0.044715 * x ** 2)
            grad = 0.5 * (1.0 + t) + 0.5 * x * dt
            self._accumulate(out.grad * grad)

        out._backward = _backward
        return out

    # --- indexing / gather ----------------------------------------------
    def gather_rows(self, idx: np.ndarray) -> "Tensor":
        """Embedding lookup: rows of a (V, D) table indexed by integer array."""
        out = Tensor(self.data[idx], _children=(self,))

        def _backward() -> None:
            grad = np.zeros_like(self.data)
            np.add.at(grad, idx, out.grad)
            self._accumulate(grad)

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        m = np.max(self.data, axis=axis, keepdims=True)
        e = np.exp(self.data - m)
        s = e / np.sum(e, axis=axis, keepdims=True)
        out = Tensor(s, _children=(self,))

        def _backward() -> None:
            g = out.grad
            dot = np.sum(g * s, axis=axis, keepdims=True)
            self._accumulate(s * (g - dot))

        out._backward = _backward
        return out

    # --- backward --------------------------------------------------------
    def backward(self) -> None:
        topo: list[Tensor] = []
        visited: Set[int] = set()

        def build(t: "Tensor") -> None:
            if id(t) in visited:
                return
            visited.add(id(t))
            for child in t._prev:
                build(child)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()

    def free_graph(self) -> None:
        """Break the autograd graph so reference cycles (each ``_backward``
        closure captures its own tensor) are reclaimed by refcounting instead
        of waiting on the cyclic garbage collector. Call after the optimizer
        step, once grads are consumed."""
        visited: Set[int] = set()
        stack: list[Tensor] = [self]
        while stack:
            t = stack.pop()
            if id(t) in visited:
                continue
            visited.add(id(t))
            stack.extend(t._prev)
            t._prev = ()
            t._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape})"


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy of integer `targets` against (N, V) `logits`."""
    probs = logits.softmax(axis=-1)
    n = targets.shape[0]
    picked = Tensor(np.zeros((n, 1), dtype=np.float32), _children=(probs,))
    rows = np.arange(n)
    picked.data[:, 0] = probs.data[rows, targets]

    def _backward() -> None:
        grad = np.zeros_like(probs.data)
        grad[rows, targets] = picked.grad[:, 0]
        probs._accumulate(grad)

    picked._backward = _backward
    return (picked.log() * -1.0).mean()
