"""Incremental (O(1) per update) indicators.

All indicators are causal: they only consume closed bars, in order. Swing points
are *confirmed* with a lag of ``k`` bars and are only reported once confirmed --
a classic source of look-ahead in naive backtests.
"""
from __future__ import annotations

import math
from collections import deque


class EMA:
    def __init__(self, n: int):
        self.n, self.a, self.value, self.count = n, 2.0 / (n + 1), None, 0

    def update(self, x: float) -> float:
        self.count += 1
        self.value = x if self.value is None else self.value + self.a * (x - self.value)
        return self.value

    @property
    def ready(self) -> bool:
        return self.count >= self.n


class Rolling:
    """Rolling window with running sum / sum of squares."""

    def __init__(self, n: int):
        self.n, self.buf = n, deque(maxlen=n)
        self.s = self.ss = 0.0

    def update(self, x: float) -> None:
        if len(self.buf) == self.n:
            old = self.buf[0]
            self.s -= old
            self.ss -= old * old
        self.buf.append(x)
        self.s += x
        self.ss += x * x

    @property
    def ready(self) -> bool:
        return len(self.buf) == self.n

    @property
    def mean(self) -> float:
        return self.s / len(self.buf) if self.buf else float("nan")

    @property
    def std(self) -> float:
        k = len(self.buf)
        if k < 2:
            return float("nan")
        var = (self.ss - self.s * self.s / k) / (k - 1)
        return math.sqrt(max(var, 0.0))

    def percentile_rank(self, x: float) -> float:
        if not self.buf:
            return float("nan")
        return sum(1 for v in self.buf if v <= x) / len(self.buf)


class Wilder:
    def __init__(self, n: int):
        self.n, self.value, self.count, self._seed = n, None, 0, 0.0

    def update(self, x: float) -> float | None:
        self.count += 1
        if self.count <= self.n:
            self._seed += x
            if self.count == self.n:
                self.value = self._seed / self.n
            return self.value
        self.value = (self.value * (self.n - 1) + x) / self.n
        return self.value


class RSI:
    def __init__(self, n: int = 14):
        self.up, self.dn, self.prev = Wilder(n), Wilder(n), None
        self.value = None

    def update(self, close: float) -> float | None:
        if self.prev is not None:
            ch = close - self.prev
            u, d = self.up.update(max(ch, 0.0)), self.dn.update(max(-ch, 0.0))
            if u is not None and d is not None:
                self.value = 100.0 if d == 0 else 100.0 - 100.0 / (1.0 + u / d)
        self.prev = close
        return self.value


class ATR:
    def __init__(self, n: int = 14):
        self.w, self.prev_close, self.value = Wilder(n), None, None

    def update(self, h: float, l: float, c: float) -> float | None:
        tr = h - l if self.prev_close is None else max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        self.prev_close = c
        self.value = self.w.update(tr)
        return self.value


class ADX:
    def __init__(self, n: int = 14):
        self.n = n
        self.tr, self.pdm, self.ndm, self.dx = Wilder(n), Wilder(n), Wilder(n), Wilder(n)
        self.prev = None
        self.value = self.pdi = self.ndi = None

    def update(self, h: float, l: float, c: float) -> float | None:
        if self.prev is not None:
            ph, pl, pc = self.prev
            up, dn = h - ph, pl - l
            pdm = up if up > dn and up > 0 else 0.0
            ndm = dn if dn > up and dn > 0 else 0.0
            tr = max(h - l, abs(h - pc), abs(l - pc))
            t, p, n = self.tr.update(tr), self.pdm.update(pdm), self.ndm.update(ndm)
            if t:
                self.pdi, self.ndi = 100 * p / t, 100 * n / t
                s = self.pdi + self.ndi
                self.value = self.dx.update(100 * abs(self.pdi - self.ndi) / s if s else 0.0)
        self.prev = (h, l, c)
        return self.value


class MACD:
    def __init__(self, fast: int = 12, slow: int = 26, sig: int = 9):
        self.f, self.s, self.sig = EMA(fast), EMA(slow), EMA(sig)
        self.macd = self.signal = self.hist = None

    def update(self, x: float):
        self.macd = self.f.update(x) - self.s.update(x)
        self.signal = self.sig.update(self.macd)
        self.hist = self.macd - self.signal
        return self.hist


class SessionVWAP:
    """VWAP reset each session, with volume-weighted std bands."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pv = self.v = self.pv2 = 0.0

    def update(self, price: float, vol: float) -> float | None:
        self.pv += price * vol
        self.pv2 += price * price * vol
        self.v += vol
        return self.value

    @property
    def value(self) -> float | None:
        return self.pv / self.v if self.v > 0 else None

    @property
    def std(self) -> float | None:
        if self.v <= 0:
            return None
        m = self.pv / self.v
        return math.sqrt(max(self.pv2 / self.v - m * m, 0.0))


class EfficiencyRatio:
    """Kaufman efficiency ratio: |net move| / path length. ~1 trending, ~0 choppy."""

    def __init__(self, n: int = 20):
        self.buf = deque(maxlen=n + 1)

    def update(self, x: float) -> float | None:
        self.buf.append(x)
        if len(self.buf) < self.buf.maxlen:
            return None
        b = list(self.buf)
        path = sum(abs(b[i] - b[i - 1]) for i in range(1, len(b)))
        return abs(b[-1] - b[0]) / path if path > 0 else 0.0


class SwingDetector:
    """Fractal pivots confirmed after ``k`` bars on each side (so reported k bars late)."""

    def __init__(self, k: int = 3):
        self.k, self.buf = k, deque(maxlen=2 * k + 1)
        self.last_high: float | None = None
        self.last_low: float | None = None

    def update(self, h: float, l: float) -> None:
        self.buf.append((h, l))
        if len(self.buf) < self.buf.maxlen:
            return
        mid = self.k
        hs = [x[0] for x in self.buf]
        ls = [x[1] for x in self.buf]
        if hs[mid] == max(hs) and hs.count(hs[mid]) == 1:
            self.last_high = hs[mid]
        if ls[mid] == min(ls) and ls.count(ls[mid]) == 1:
            self.last_low = ls[mid]


class RobustZ:
    """Median/MAD z-score over a rolling window -- resistant to the outliers we want to detect."""

    def __init__(self, n: int = 100):
        self.buf = deque(maxlen=n)

    def score(self, x: float) -> float:
        if len(self.buf) < 20:
            return 0.0
        s = sorted(self.buf)
        med = s[len(s) // 2]
        mad = sorted(abs(v - med) for v in s)[len(s) // 2] * 1.4826
        return (x - med) / mad if mad > 1e-12 else 0.0

    def update(self, x: float) -> float:
        z = self.score(x)
        self.buf.append(x)
        return z
