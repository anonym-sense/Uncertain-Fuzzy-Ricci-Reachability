"""Liu uncertainty theory primitives and the linguistic (fuzzy-partition) readout.

Normal uncertain variable N(e, s): Phi(x) = (1 + exp(pi (e - x) / (sqrt(3) s)))^-1,
inverse Phi^-1(a) = e + (sqrt(3) s / pi) ln(a / (1 - a)).
The central alpha band is the interval with belief alpha centred on e:
    [Phi^-1((1-alpha)/2), Phi^-1((1+alpha)/2)].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SQ3_PI = np.sqrt(3.0) / np.pi


@dataclass(frozen=True)
class LiuNormal:
    e: float = 0.0
    s: float = 1.0

    def cdf(self, x):
        return 1.0 / (1.0 + np.exp((self.e - np.asarray(x)) / (SQ3_PI * self.s)))

    def inv(self, a):
        a = np.asarray(a, dtype=float)
        return self.e + SQ3_PI * self.s * np.log(a / (1.0 - a))

    def half_width(self, alpha):
        """Half width of the central alpha band (alpha in [0,1))."""
        alpha = np.clip(alpha, 0.0, 1.0 - 1e-15)
        return SQ3_PI * self.s * np.log((1.0 + alpha) / (1.0 - alpha))

    def band(self, alpha):
        h = self.half_width(alpha)
        return self.e - h, self.e + h


def bands(vars_, alpha):
    """Stacked (lo, hi) arrays for a list of LiuNormal at belief level alpha."""
    if not vars_:
        return np.zeros(0), np.zeros(0)
    lo, hi = zip(*(v.band(alpha) for v in vars_))
    return np.array(lo), np.array(hi)


# ---------------- linguistic readout (fuzzy partition, Eq. 2 of the 2020 paper) ------
LABELS = ("VERY LOW", "LOW", "MEDIUM", "HIGH", "VERY HIGH")
_PEAKS = np.linspace(0.0, 1.0, len(LABELS))


def memberships(m):
    """Triangular Ruspini partition on [0,1]: memberships sum to 1 (Eq. 2)."""
    m = float(np.clip(m, 0.0, 1.0))
    w = _PEAKS[1] - _PEAKS[0]
    mu = np.maximum(0.0, 1.0 - np.abs(m - _PEAKS) / w)
    return mu / mu.sum()


def linguistic(m):
    """Label for an upper bound m on the uncertain measure of reaching the unsafe set."""
    mu = memberships(m)
    k = int(np.argmax(mu))
    return LABELS[k], float(mu[k])
