"""Minimal vectorised interval arithmetic for sound Jacobian / vector-field enclosures.

An Interval holds numpy arrays lo, hi of equal shape. System models are written with
the generic helpers below (sin, cos, ...) so the same code evaluates on floats and on
intervals. Rounding is not directed; we add a small relative outward pad on transcendental
functions, which is adequate for the double-precision margins used here.
"""
from __future__ import annotations

import numpy as np

_PAD = 1e-12


class Interval:
    __array_priority__ = 1000
    __array_ufunc__ = None  # make numpy binops defer to our reflected operators

    def __init__(self, lo, hi=None):
        lo = np.asarray(lo, dtype=float)
        hi = lo.copy() if hi is None else np.asarray(hi, dtype=float)
        self.lo, self.hi = np.minimum(lo, hi), np.maximum(lo, hi)

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def of(x):
        return x if isinstance(x, Interval) else Interval(x, x)

    @property
    def mid(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def rad(self):
        return 0.5 * (self.hi - self.lo)

    @property
    def mag(self):
        return np.maximum(np.abs(self.lo), np.abs(self.hi))

    @property
    def shape(self):
        return self.lo.shape

    def __getitem__(self, k):
        return Interval(self.lo[k], self.hi[k])

    def __len__(self):
        return len(self.lo)

    def __repr__(self):
        return f"Interval(lo={self.lo}, hi={self.hi})"

    # ---- arithmetic ------------------------------------------------------------
    def __add__(self, o):
        if hasattr(o, 'dsin'):
            return NotImplemented
        o = Interval.of(o)
        return Interval(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, o):
        if hasattr(o, 'dsin'):
            return NotImplemented
        return self + (-Interval.of(o))

    def __rsub__(self, o):
        if hasattr(o, 'dsin'):
            return NotImplemented
        return Interval.of(o) - self

    def __mul__(self, o):
        if hasattr(o, 'dsin'):
            return NotImplemented
        o = Interval.of(o)
        c = np.stack([self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi])
        return Interval(c.min(0), c.max(0))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if hasattr(o, 'dsin'):
            return NotImplemented
        if isinstance(o, Interval):
            if np.any((o.lo <= 0) & (o.hi >= 0)):
                raise ZeroDivisionError("interval division by interval containing 0")
            return self * Interval(1.0 / o.hi, 1.0 / o.lo)
        return self * (1.0 / np.asarray(o, dtype=float))

    def __rtruediv__(self, o):
        if hasattr(o, 'dsin'):
            return NotImplemented
        if np.any((self.lo <= 0) & (self.hi >= 0)):
            raise ZeroDivisionError("interval division by interval containing 0")
        return Interval(1.0 / self.hi, 1.0 / self.lo) * o

    def sq(self):
        lo2, hi2 = self.lo ** 2, self.hi ** 2
        lo = np.where((self.lo <= 0) & (self.hi >= 0), 0.0, np.minimum(lo2, hi2))
        return Interval(lo, np.maximum(lo2, hi2))

    def __pow__(self, p):
        if p == 2:
            return self.sq()
        if p == 3:  # monotone
            return Interval(self.lo ** 3, self.hi ** 3)
        raise NotImplementedError(p)

    def hull(self, o):
        o = Interval.of(o)
        return Interval(np.minimum(self.lo, o.lo), np.maximum(self.hi, o.hi))


# ---- generic elementary functions (float or Interval) ---------------------------

def _sin_iv(x: Interval) -> Interval:
    lo, hi = x.lo, x.hi
    out_lo, out_hi = np.sin(lo), np.sin(hi)
    out_lo, out_hi = np.minimum(out_lo, out_hi), np.maximum(out_lo, out_hi)
    # interior extrema at pi/2 + 2k pi (max) and -pi/2 + 2k pi (min)
    kmax = np.ceil((lo - np.pi / 2) / (2 * np.pi))
    has_max = np.pi / 2 + 2 * np.pi * kmax <= hi
    kmin = np.ceil((lo + np.pi / 2) / (2 * np.pi))
    has_min = -np.pi / 2 + 2 * np.pi * kmin <= hi
    out_hi = np.where(has_max, 1.0, out_hi)
    out_lo = np.where(has_min, -1.0, out_lo)
    wide = (hi - lo) >= 2 * np.pi
    out_lo = np.where(wide, -1.0, out_lo)
    out_hi = np.where(wide, 1.0, out_hi)
    return Interval(out_lo - _PAD, out_hi + _PAD)


def sin(x):
    if hasattr(x, "dsin"):
        return x.dsin()
    return _sin_iv(x) if isinstance(x, Interval) else np.sin(x)


def cos(x):
    if hasattr(x, "dcos"):
        return x.dcos()
    return _sin_iv(x + np.pi / 2) if isinstance(x, Interval) else np.cos(x)


def stack(items):
    """Stack a list of floats/Intervals into an Interval (or ndarray if all float)."""
    if any(isinstance(i, Interval) for i in items):
        iv = [Interval.of(i) for i in items]
        return Interval(np.stack([i.lo for i in iv]), np.stack([i.hi for i in iv]))
    return np.stack([np.asarray(i, dtype=float) for i in items])


def matvec(A, x):
    """Interval-aware matrix-vector product A @ x, A ndarray or Interval (n,m), x (m,)."""
    if not isinstance(A, Interval) and not isinstance(x, Interval):
        return A @ x
    A, x = Interval.of(A), Interval.of(x)
    prod = A * Interval(x.lo[None, :], x.hi[None, :])
    return Interval(prod.lo.sum(1), prod.hi.sum(1))
