"""Numerical gradient checks for the from-scratch autograd engine."""

from __future__ import annotations

import numpy as np

from nn.autograd import Tensor, cross_entropy


def numeric_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def _check(f_tensor, f_numpy, x0, tol=1e-4):
    x0 = x0.astype(np.float64)
    t = Tensor(x0.copy())
    out = f_tensor(t)
    out.backward()
    ng = numeric_grad(f_numpy, x0.copy())
    assert np.allclose(t.grad, ng, atol=tol), f"max diff {np.abs(t.grad - ng).max()}"


def test_grad_mul_sum():
    x = np.random.randn(3, 4).astype(np.float32)
    _check(lambda t: (t * t).sum(), lambda a: float((a * a).sum()), x)


def test_grad_matmul():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4)).astype(np.float32)
    W = Tensor(rng.standard_normal((4, 5)).astype(np.float32))

    def f_tensor(t):
        return (t @ W).sum()

    def f_numpy(a):
        return float((a @ W.data).sum())

    _check(f_tensor, f_numpy, x)


def test_grad_tanh_layernormish():
    x = np.random.randn(2, 6).astype(np.float32)

    def f_tensor(t):
        mu = t.mean(axis=-1, keepdims=True)
        xc = t - mu
        return (xc.tanh() * xc.tanh()).sum()

    def f_numpy(a):
        mu = a.mean(axis=-1, keepdims=True)
        xc = a - mu
        return float((np.tanh(xc) ** 2).sum())

    _check(f_tensor, f_numpy, x, tol=2e-3)


def test_grad_softmax():
    x = np.random.randn(4, 7).astype(np.float32)
    w = np.random.randn(4, 7).astype(np.float32)

    def f_tensor(t):
        return (t.softmax(axis=-1) * Tensor(w)).sum()

    def f_numpy(a):
        m = a.max(axis=-1, keepdims=True)
        e = np.exp(a - m)
        s = e / e.sum(axis=-1, keepdims=True)
        return float((s * w).sum())

    _check(f_tensor, f_numpy, x, tol=2e-3)


def test_grad_cross_entropy():
    x = np.random.randn(5, 8).astype(np.float32)
    tgt = np.array([0, 3, 7, 1, 5])

    def f_tensor(t):
        return cross_entropy(t, tgt)

    def f_numpy(a):
        m = a.max(axis=-1, keepdims=True)
        e = np.exp(a - m)
        s = e / e.sum(axis=-1, keepdims=True)
        return float(-np.log(s[np.arange(5), tgt] + 1e-12).mean())

    _check(f_tensor, f_numpy, x, tol=2e-3)


def test_grad_gelu():
    x = np.random.randn(3, 5).astype(np.float32)

    def f_tensor(t):
        return t.gelu().sum()

    def f_numpy(a):
        c = np.sqrt(2.0 / np.pi)
        return float((0.5 * a * (1.0 + np.tanh(c * (a + 0.044715 * a ** 3)))).sum())

    _check(f_tensor, f_numpy, x, tol=2e-3)
