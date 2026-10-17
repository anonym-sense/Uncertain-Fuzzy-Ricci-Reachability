"""Second-order forward-mode AD over floats / Intervals (value, gradient, Hessian).

T2(v, g, H): g has leading axis m, H leading axes (m, m); None means zero. Running a
model's vector field on T2 inputs yields sound interval Hessian enclosures when the
values are Intervals (used for second-order Lagrange remainders).
"""
from __future__ import annotations

import numpy as np

from .interval import Interval


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


def _outer(g1, g2):
    """g1 (m, ...), g2 (m, ...) -> (m, m, ...) outer product (Interval aware)."""
    if g1 is None or g2 is None:
        return None
    if isinstance(g1, Interval) or isinstance(g2, Interval):
        a, b = Interval.of(g1), Interval.of(g2)
        A = Interval(a.lo[:, None], a.hi[:, None])
        B = Interval(b.lo[None, :], b.hi[None, :])
        return A * B
    return g1[:, None] * g2[None, :]


def _neg(x):
    return None if x is None else -x


def _sq(v):
    return v.sq() if isinstance(v, Interval) else v * v


def _v(x):
    return x.v if isinstance(x, T2) else x


class T2:
    __slots__ = ("v", "g", "H")
    __array_priority__ = 3000
    __array_ufunc__ = None

    def __init__(self, v, g=None, H=None):
        self.v, self.g, self.H = v, g, H

    @staticmethod
    def lift(x):
        return x if isinstance(x, T2) else T2(x)

    def __add__(self, o):
        o = T2.lift(o)
        return T2(_plus(self.v, o.v), _plus(self.g, o.g), _plus(self.H, o.H))

    __radd__ = __add__

    def __neg__(self):
        return T2(-self.v, _neg(self.g), _neg(self.H))

    def __sub__(self, o):
        return self + (-T2.lift(o))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        o = T2.lift(o)
        v = _times(self.v, o.v)
        g = _plus(_times(self.g, o.v), _times(o.g, self.v))
        H = _plus(_plus(_times(self.H, o.v), _times(o.H, self.v)),
                  _plus(_outer(self.g, o.g), _outer(o.g, self.g)))
        return T2(v, g, H)

    __rmul__ = __mul__

    def _unary(self, f0, f1, f2):
        """Chain rule for a scalar function with values f0, f1 = f', f2 = f''."""
        g = _times(self.g, f1)
        H = _plus(_times(self.H, f1), _times(_outer(self.g, self.g), f2))
        return T2(f0, g, H)

    def recip(self):
        v = self.v
        if isinstance(v, Interval):
            if np.any((v.lo <= 0) & (v.hi >= 0)):
                raise ZeroDivisionError("reciprocal of an interval containing 0")
            r = Interval(1.0 / v.hi, 1.0 / v.lo)
            r2 = r.sq()
            r3 = Interval(r.lo ** 3, r.hi ** 3)
            return self._unary(r, -r2, r3 * 2.0)
        r = 1.0 / v
        return self._unary(r, -r * r, 2.0 * r ** 3)

    def __truediv__(self, o):
        return self * T2.lift(o).recip()

    def __rtruediv__(self, o):
        return self.recip() * o

    def __pow__(self, p):
        v = self.v
        if p == 2:
            return self._unary(_sq(v), _times(v, 2.0), 2.0)
        if p == 3:
            return self._unary(v ** 3, _times(_sq(v), 3.0), _times(v, 6.0))
        raise NotImplementedError(p)

    def dsin(self):
        from .interval import cos, sin
        s, c = sin(self.v), cos(self.v)
        return self._unary(s, c, -s)

    def dcos(self):
        from .interval import cos, sin
        s, c = sin(self.v), cos(self.v)
        return self._unary(c, -s, -c)


def _lohi(x):
    if isinstance(x, Interval):
        return x.lo, x.hi
    a = np.asarray(x, float)
    return a, a


def hessian(fn, zs):
    """Run fn(Z) on second-order duals; zs: list of m values (float/ndarray/Interval).

    Returns (vals, J, H): J batch+(n_out, m), H batch+(n_out, m, m); Interval if any
    input is an Interval.
    """
    m = len(zs)

    def seed(i, v):
        sh = np.shape(_lohi(v)[0])
        g = np.zeros((m,) + sh)
        g[i] = 1.0
        return T2(v, g, None)

    out = fn([seed(i, v) for i, v in enumerate(zs)])
    out = [T2.lift(o) for o in out]
    shapes = [np.shape(_lohi(o.v)[0]) for o in out]
    shapes += [np.shape(_lohi(o.g)[0])[1:] for o in out if o.g is not None]
    shapes += [np.shape(_lohi(o.H)[0])[2:] for o in out if o.H is not None]
    b = np.broadcast_shapes(*shapes)
    n = len(out)
    Jl, Jh = np.zeros(b + (n, m)), np.zeros(b + (n, m))
    Hl, Hh = np.zeros(b + (n, m, m)), np.zeros(b + (n, m, m))
    is_iv = False
    for j, o in enumerate(out):
        if o.g is not None:
            lo, hi = _lohi(o.g)
            is_iv |= isinstance(o.g, Interval)
            Jl[..., j, :] = np.moveaxis(np.broadcast_to(lo, (m,) + b), 0, -1)
            Jh[..., j, :] = np.moveaxis(np.broadcast_to(hi, (m,) + b), 0, -1)
        if o.H is not None:
            lo, hi = _lohi(o.H)
            is_iv |= isinstance(o.H, Interval)
            Hl[..., j, :, :] = np.moveaxis(np.broadcast_to(lo, (m, m) + b), (0, 1), (-2, -1))
            Hh[..., j, :, :] = np.moveaxis(np.broadcast_to(hi, (m, m) + b), (0, 1), (-2, -1))
    vals = [o.v for o in out]
    if is_iv:
        return vals, Interval(Jl, Jh), Interval(Hl, Hh)
    return vals, Jl, Hl
