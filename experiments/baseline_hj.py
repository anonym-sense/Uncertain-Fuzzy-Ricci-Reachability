"""Hamilton-Jacobi reachability baseline (hj_reachability) for Van der Pol.

Augmented grid (x1, x2, mu, w) with mu' = w' = 0 (alpha-path semantics). One backward
reachable tube of the unsafe set {x2 >= 2.75} over [0, T] covers every alpha: the system
is safe at belief alpha iff the BRT value is > 0 on the alpha-initial set.
HJ is grid-based (not formally sound: interpolation and numerical dissipation); it
serves as the classical low-dimensional reference.
Writes experiments/results/van_der_pol__hj__curvature__path.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import jax.numpy as jnp  # noqa: E402
import hj_reachability as hj  # noqa: E402

from frs.systems import van_der_pol  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"


class VdP(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self):
        z = hj.sets.Box(jnp.zeros(1), jnp.zeros(1))
        super().__init__("max", "min", z, z)

    def open_loop_dynamics(self, s, t):
        x1, x2, mu, w = s
        return jnp.array([x2, mu * (1 - x1 ** 2) * x2 - x1 + w, 0.0, 0.0])

    def control_jacobian(self, s, t):
        return jnp.zeros((4, 1))

    def disturbance_jacobian(self, s, t):
        return jnp.zeros((4, 1))


def main(nx=101, npar=11):
    sys_ = van_der_pol()
    mu_v, w_v = sys_.th[0], sys_.w[0]
    a_max = 0.999
    mu_h, w_h = mu_v.half_width(a_max), w_v.half_width(a_max)
    lo = np.array([-3.0, -3.5, mu_v.e - mu_h, w_v.e - w_h])
    hi = np.array([3.0, 3.5, mu_v.e + mu_h, w_v.e + w_h])
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array(lo), jnp.array(hi)), (nx, nx, npar, npar))
    target = 2.75 - grid.states[..., 1]           # <= 0 inside the unsafe set
    settings = hj.SolverSettings.with_accuracy("high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    t0 = time.perf_counter()
    V = hj.step(settings, VdP(), grid, 0.0, target, -sys_.T)
    V = np.asarray(V)
    t_solve = time.perf_counter() - t0

    xs = [np.asarray(c) for c in grid.coordinate_vectors]

    def safe_at(alpha):
        m_lo, m_hi = mu_v.band(alpha)
        w_lo, w_hi = w_v.band(alpha)
        sel = [(xs[0] >= sys_.x0.lo[0]) & (xs[0] <= sys_.x0.hi[0]),
               (xs[1] >= sys_.x0.lo[1]) & (xs[1] <= sys_.x0.hi[1]),
               (xs[2] >= m_lo - 1e-12) & (xs[2] <= m_hi + 1e-12),
               (xs[3] >= w_lo - 1e-12) & (xs[3] <= w_hi + 1e-12)]
        # always include the nearest grid layer so a thin band is not empty
        for d, s in enumerate(sel):
            if not s.any():
                c = 0.5 * ((m_lo + m_hi) if d == 2 else (w_lo + w_hi)) if d >= 2 else 0
                s[np.argmin(np.abs(xs[d] - c))] = True
        sub = V[np.ix_(*sel)]
        return float(sub.min()) > 0

    lo_a, hi_a = 1e-4, a_max
    if not safe_at(lo_a):
        a_star = 0.0
    elif safe_at(hi_a):
        a_star = 1.0
    else:
        while hi_a - lo_a > 1e-3:
            mid = 0.5 * (lo_a + hi_a)
            lo_a, hi_a = (mid, hi_a) if safe_at(mid) else (lo_a, mid)
        a_star = lo_a
    res = {"alpha_star": a_star, "engine": "HJ (hj_reachability)", "benchmark": "van_der_pol",
           "grid": [nx, nx, npar, npar], "wall_time": t_solve, "n": 2, "p": 2, "sound": False}
    (OUT / "van_der_pol__hj__curvature__path.json").write_text(json.dumps(res, indent=2))
    print(f"HJ van_der_pol alpha*={a_star:.4f} solve {t_solve:.0f}s grid {nx}x{nx}x{npar}x{npar}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
