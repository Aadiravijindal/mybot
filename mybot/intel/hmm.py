"""Minimal Gaussian HMM (numpy) for offline regime benchmarking.

IMPORTANT: live/backtest use must rely on *filtered* probabilities
(forward pass only, P(state_t | x_1..t)). The textbook *smoothed* probabilities
(forward-backward) use future observations and are look-ahead bias.
"""
from __future__ import annotations

import numpy as np


class GaussianHMM:
    def __init__(self, k: int = 2, iters: int = 50, seed: int = 0):
        self.k, self.iters = k, iters
        self.rng = np.random.default_rng(seed)

    def _emis(self, x: np.ndarray) -> np.ndarray:
        var = self.var[None, :]
        return np.exp(-0.5 * (x[:, None] - self.mu[None, :]) ** 2 / var) / np.sqrt(2 * np.pi * var) + 1e-300

    def fit(self, x: np.ndarray) -> "GaussianHMM":
        x = np.asarray(x, float)
        k = self.k
        qs = np.quantile(np.abs(x), np.linspace(0.2, 0.8, k))
        self.mu = np.zeros(k)
        self.var = (qs + x.std() * 0.1) ** 2
        self.A = np.full((k, k), 0.05 / (k - 1)) + np.eye(k) * (0.95 - 0.05 / (k - 1))
        self.pi = np.full(k, 1.0 / k)
        for _ in range(self.iters):
            B = self._emis(x)
            T = len(x)
            a = np.zeros((T, k)); c = np.zeros(T)
            a[0] = self.pi * B[0]; c[0] = a[0].sum(); a[0] /= c[0]
            for t in range(1, T):
                a[t] = (a[t - 1] @ self.A) * B[t]; c[t] = a[t].sum(); a[t] /= c[t]
            b = np.ones((T, k))
            for t in range(T - 2, -1, -1):
                b[t] = (self.A @ (B[t + 1] * b[t + 1])) / c[t + 1]
            g = a * b
            g /= g.sum(1, keepdims=True)
            xi = np.zeros((k, k))
            for t in range(T - 1):
                m = a[t][:, None] * self.A * (B[t + 1] * b[t + 1])[None, :] / c[t + 1]
                xi += m
            self.A = xi / xi.sum(1, keepdims=True)
            self.pi = g[0]
            w = g.sum(0) + 1e-12
            self.mu = (g * x[:, None]).sum(0) / w
            self.var = np.maximum((g * (x[:, None] - self.mu) ** 2).sum(0) / w, 1e-12)
        order = np.argsort(self.var)  # state 0 = calm, k-1 = volatile
        self.mu, self.var = self.mu[order], self.var[order]
        self.A, self.pi = self.A[np.ix_(order, order)], self.pi[order]
        return self

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Causal state probabilities P(s_t | x_1..t)."""
        x = np.asarray(x, float)
        B = self._emis(x)
        out = np.zeros((len(x), self.k))
        p = self.pi * B[0]; out[0] = p / p.sum()
        for t in range(1, len(x)):
            p = (out[t - 1] @ self.A) * B[t]
            out[t] = p / p.sum()
        return out
