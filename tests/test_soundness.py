"""Containment tests: sampled true trajectories must lie inside every engine's enclosure."""
import sys

import numpy as np
import pytest
from scipy.optimize import linprog

sys.path.insert(0, ".")
from frs.engine import Cells, FlowEngine, TubeEngine, ZonoEngine, rk4_batch  # noqa: E402
from frs.systems import chain, pendulum, s_curve, van_der_pol  # noqa: E402


def _short(sys_, T):
    s = sys_.path_semantics()
    s.T = T
    return s


CASES = [_short(pendulum(), 1.0), _short(van_der_pol(), 1.5), _short(s_curve(), 2.0), _short(chain(2), 1.5)]


def _samples(s, cells, alpha, B=120, seed=0):
    rng = np.random.default_rng(seed)
    TH = cells.theta(s, alpha)
    u = rng.uniform(-1, 1, (B, s.n + s.p))
    u[: B // 3] = np.sign(u[: B // 3])  # vertices
    x0 = 0.5 * (cells.xlo[0] + cells.xhi[0]) + 0.5 * (cells.xhi[0] - cells.xlo[0]) * u[:, : s.n]
    th = TH.mid[:, 0] + TH.rad[:, 0] * u[:, s.n:]
    W = np.zeros((0, B))
    return rk4_batch(s, x0.T, th.T, W, s.dt, int(round(s.T / s.dt)), sub=8), th


def _cell(s):
    c = Cells.root(s)
    # a moderately sized sub-cell so that the engines do not hit the unsafe set early
    mid = 0.5 * (c.xlo + c.xhi)
    q = 0.25 * (c.xhi - c.xlo)
    return Cells(mid - q, mid + q, -0.5 * np.ones_like(c.ulo), 0.5 * np.ones_like(c.uhi))


@pytest.mark.parametrize("s", CASES, ids=lambda s: s.name)
def test_zonotope_contains_samples(s):
    a = 0.4
    cells = _cell(s)
    s.unsafe = []  # containment only
    o = ZonoEngine(s).run(a, cells, record=True)
    xs, th = _samples(s, cells, a)
    for k in np.linspace(0, len(o["G"]) - 1, 12).astype(int):
        G, c = o["G"][k][0], o["c"][k][0]
        ref = o["xs"][k][:, 0]
        for b in range(0, xs.shape[2], 6):
            z = np.r_[xs[k, :, b], th[b]] - ref - c
            m = G.shape[1]
            res = linprog(np.zeros(m), A_eq=G, b_eq=z, bounds=[(-1 - 1e-7, 1 + 1e-7)] * m, method="highs")
            assert res.status == 0, f"sample {b} escapes zonotope at step {k}"


@pytest.mark.parametrize("s", CASES, ids=lambda s: s.name)
def test_flow_contains_samples(s):
    a = 0.4
    cells = _cell(s)
    s.unsafe = []
    o = FlowEngine(s).run(a, cells, record=True)
    xs, th = _samples(s, cells, a)
    for k in np.linspace(0, len(o["Q"]) - 1, 12).astype(int):
        Q, c, ref = o["Q"][k][0], o["c"][k][0], o["xs"][k][:, 0]
        Qi = np.linalg.inv(Q)
        for b in range(xs.shape[2]):
            z = np.r_[xs[k, :, b], th[b]] - ref - c
            assert z @ Qi @ z <= 1 + 1e-6, f"sample {b} escapes ellipsoid at step {k}"


@pytest.mark.parametrize("s", CASES[:2] + CASES[3:], ids=lambda s: s.name)
def test_ricci_tube_contains_samples(s):
    a = 0.4
    cells = _cell(s)
    s.unsafe = []
    o = TubeEngine(s).run(a, cells, record=True)
    xs, _ = _samples(s, cells, a)
    P = o["P"]
    for k in np.linspace(0, len(o["rho"]) - 1, 12).astype(int):
        rho, ref = o["rho"][k][0], o["xs"][k][:, 0]
        if o["hit"][0] and k >= o["hit_step"][0]:  # flagged (diverged): no claim made
            continue
        d = np.linalg.norm(P @ (xs[k] - ref[:, None]), axis=0)
        assert np.all(d <= rho * (1 + 1e-6)), f"escape at step {k}: {d.max()} > {rho}"
