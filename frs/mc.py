"""Monte Carlo + falsification: an *inner* (under-approximate) estimate of alpha*.

A sample is a vector z in [-1, 1]^D that scales every uncertain input inside its alpha
band (theta, piecewise-constant disturbance segments, initial state). Any sample that
hits the unsafe set certifies unsafety at that alpha, so alpha*_MC >= true alpha*.
A cross-entropy falsifier searches z to minimise the min-over-time unsafe sdf.
"""
from __future__ import annotations

import numpy as np

from .systems import System

ALPHA_MAX = 0.999  # beyond this, bands leave the physically meaningful parameter range


def _rk4(sys: System, X, th, w, dt):
    k1 = sys.F(X, th, w)
    k2 = sys.F(X + 0.5 * dt * k1, th, w)
    k3 = sys.F(X + 0.5 * dt * k2, th, w)
    k4 = sys.F(X + dt * k3, th, w)
    return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Sampler:
    def __init__(self, sys: System, n_seg=8, sub=1):
        self.sys, self.n_seg, self.sub = sys, n_seg, sub
        self.p, self.q, self.n = len(sys.th), len(sys.w), sys.n
        self.D = self.p + self.q * n_seg + self.n

    def decode(self, Z, alpha):
        s = self.sys
        th_lo, th_hi = (s.bands(alpha)[0].lo, s.bands(alpha)[0].hi)
        w_iv = s.bands(alpha)[1]
        th = (0.5 * (th_lo + th_hi))[:, None] + (0.5 * (th_hi - th_lo))[:, None] * Z[:, : self.p].T
        W = Z[:, self.p: self.p + self.q * self.n_seg].reshape(len(Z), self.q, self.n_seg)
        W = w_iv.mid[None, :, None] + w_iv.rad[None, :, None] * W
        z0 = Z[:, self.p + self.q * self.n_seg:].T
        x0 = 0.5 * (s.x0.lo + s.x0.hi)[:, None] + 0.5 * (s.x0.hi - s.x0.lo)[:, None] * z0
        return th, W, x0

    def simulate(self, Z, alpha, record=False):
        """Return min-over-time unsafe sdf per sample (and optionally trajectories)."""
        s = self.sys
        th, W, X = self.decode(Z, alpha)
        steps = int(round(s.T / s.dt))
        dt = s.dt / self.sub
        dmin = s.sdf(X)
        traj = [X.copy()] if record else None
        for k in range(steps):
            seg = min(int(k * self.n_seg / steps), self.n_seg - 1)
            w = W[:, :, seg].T
            for _ in range(self.sub):
                X = _rk4(s, X, th, w, dt)
            dmin = np.minimum(dmin, s.sdf(X))
            if record:
                traj.append(X.copy())
        return (dmin, np.stack(traj)) if record else dmin

    def random(self, B, rng, extreme=0.5):
        Z = rng.uniform(-1, 1, (B, self.D))
        mask = rng.random((B, self.D)) < extreme
        Z[mask] = np.sign(Z[mask])
        return Z


def falsify(sys: System, alpha, B=512, iters=12, seed=0, n_seg=8):
    """Cross-entropy falsification. Returns (best sdf, best z)."""
    rng = np.random.default_rng(seed)
    S = Sampler(sys, n_seg=n_seg)
    Z = S.random(B, rng)
    d = S.simulate(Z, alpha)
    best = int(np.argmin(d))
    best_d, best_z = d[best], Z[best].copy()
    mu, sd = Z[np.argsort(d)[: B // 8]].mean(0), np.full(S.D, 0.6)
    for _ in range(iters):
        if best_d <= 0:
            break
        Z = np.clip(mu + sd * rng.standard_normal((B, S.D)), -1, 1)
        Z[: B // 8] = np.sign(Z[: B // 8]) * (np.abs(Z[: B // 8]) > 0.3)  # vertex probes
        d = S.simulate(Z, alpha)
        elite = Z[np.argsort(d)[: B // 8]]
        mu, sd = 0.7 * mu + 0.3 * elite.mean(0), 0.7 * sd + 0.3 * (elite.std(0) + 0.05)
        i = int(np.argmin(d))
        if d[i] < best_d:
            best_d, best_z = d[i], Z[i].copy()
    return best_d, best_z


def alpha_star_mc(sys: System, tol=2e-3, **kw):
    """Inner estimate of alpha* by bisection on falsification success."""
    lo, hi = 1e-4, ALPHA_MAX
    if falsify(sys, lo, **kw)[0] <= 0:
        return 0.0
    if falsify(sys, hi, **kw)[0] > 0:
        return 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if falsify(sys, mid, **kw)[0] <= 0:
            hi = mid
        else:
            lo = mid
    return hi
