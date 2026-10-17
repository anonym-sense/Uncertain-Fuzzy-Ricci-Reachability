"""Forward-mode automatic differentiation over floats / Intervals.

D(v, d): value v (float, ndarray (C,), or Interval (C,)) and derivative d with leading
axis m (number of independent variables), or None for a constant. Running a model's
vector field on D inputs yields sound interval Jacobian enclosures when v are Intervals.
"""
from __future__ import annotations

import numpy as np

from .interval import Interval


def _v(x):
    return x.v if isinstance(x, D) else x


def _d(x):
    return x.d if isinstance(x, D) else None


def _plus(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(b, Interval) and not isinstance(a, Interval):
        return b + a
    return a + b


def _times(a, b):
    if a is None or b is None:
        return None
    if isinstance(b, Interval) and not isinstance(a, Interval):
        return b * a
    return a * b


def _sq(v):
    return v.sq() if isinstance(v, Interval) else v * v


class D:
    __slots__ = ("v", "d")
    __array_priority__ = 2000
    __array_ufunc__ = None

    def __init__(self, v, d=None):
        self.v, self.d = v, d

    def __add__(self, o):
        return D(_plus(self.v, _v(o)), _plus(self.d, _d(o)))

    __radd__ = __add__

    def __neg__(self):
        return D(-self.v, None if self.d is None else -self.d)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        ov, od = _v(o), _d(o)
        return D(_times(self.v, ov), _plus(_times(self.d, ov), _times(od, self.v)))

    __rmul__ = __mul__

    def recip(self):
        v = self.v
        if isinstance(v, Interval):
            if np.any((v.lo <= 0) & (v.hi >= 0)):
                raise ZeroDivisionError("reciprocal of an interval containing 0")
            r = Interval(1.0 / v.hi, 1.0 / v.lo)
        else:
            r = 1.0 / v
        return D(r, _times(self.d, -_sq(r)))

    def __truediv__(self, o):
        if isinstance(o, D):
            return self * o.recip()
        return self * D(o).recip()

    def __rtruediv__(self, o):
        return self.recip() * o

    def __pow__(self, p):
        v = self.v
        if p == 2:
            return D(_sq(v), _times(self.d, _times(v, 2.0)))
        if p == 3:
            return D(v ** 3, _times(self.d, _times(_sq(v), 3.0)))
        raise NotImplementedError(p)

    # used by interval.sin / interval.cos dispatch
    def dsin(self):
        from .interval import cos, sin
        return D(sin(self.v), _times(self.d, cos(self.v)))

    def dcos(self):
        from .interval import cos, sin
        return D(cos(self.v), _times(self.d, -sin(self.v)))


def _lohi(x):
    if isinstance(x, Interval):
        return x.lo, x.hi
    a = np.asarray(x, float)
    return a, a


def jacobian(fn, xs, ths):
    """Run fn(X, TH) on dual inputs. xs: n values, ths: p values (float/ndarray/Interval).

    Returns (vals, Jx, Jth); J* have shape batch + (n_out, n|p) and are Interval if any
    input is an Interval, ndarray otherwise.
    """
    n, p = len(xs), len(ths)
    m = n + p

    def seed(i, v):
        d = np.zeros((m,) + np.shape(_lohi(v)[0]))
        d[i] = 1.0
        return D(v, d)

    out = fn([seed(i, v) for i, v in enumerate(xs)], [seed(n + j, v) for j, v in enumerate(ths)])
    vals = [_v(o) for o in out]
    ders = [_d(o) for o in out]
    shapes = [np.shape(_lohi(d)[0])[1:] for d in ders if d is not None]
    shapes += [np.shape(_lohi(v)[0]) for v in vals]
    b = np.broadcast_shapes(*shapes)
    lo = np.zeros(b + (len(out), m))
    hi = np.zeros(b + (len(out), m))
    is_iv = any(isinstance(d, Interval) for d in ders)
    for j, d in enumerate(ders):
        if d is None:
            continue
        dl, dh = _lohi(d)
        lo[..., j, :] = np.moveaxis(np.broadcast_to(dl, (m,) + b), 0, -1)
        hi[..., j, :] = np.moveaxis(np.broadcast_to(dh, (m,) + b), 0, -1)
    if is_iv:
        return vals, Interval(lo[..., :n], hi[..., :n]), Interval(lo[..., n:], hi[..., n:])
    return vals, lo[..., :n], lo[..., n:]
