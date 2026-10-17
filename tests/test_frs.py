import sys, numpy as np, pytest
sys.path.insert(0, '.')
from frs.systems import BENCHMARKS, chain
from frs.interval import Interval
from frs.uncertain import LiuNormal, memberships

SYSTEMS = [f() for f in BENCHMARKS.values()] + [chain(3)]


@pytest.mark.parametrize("s", SYSTEMS, ids=lambda s: s.name)
def test_ad_matches_finite_difference(s):
    rng = np.random.default_rng(0)
    x = 0.5 * (s.x0.lo + s.x0.hi) + 0.05 * rng.standard_normal(s.n)
    th, w = s.th_hat, s.w_hat + 0.1
    Jx, Jt = s.jacs(x, th, w)
    eps = 1e-6
    for i in range(s.n):
        d = np.zeros(s.n); d[i] = eps
        fd = (s.F(x + d, th, w) - s.F(x - d, th, w)) / (2 * eps)
        assert np.allclose(Jx[:, i], fd, atol=1e-5)
    for j in range(s.p):
        d = np.zeros(s.p); d[j] = eps
        fd = (s.F(x, th + d, w) - s.F(x, th - d, w)) / (2 * eps)
        assert np.allclose(Jt[:, j], fd, atol=1e-5)


@pytest.mark.parametrize("s", SYSTEMS, ids=lambda s: s.name)
def test_interval_jacobian_encloses_samples(s):
    rng = np.random.default_rng(1)
    c = 0.5 * (s.x0.lo + s.x0.hi)
    h = 0.5 * (s.x0.hi - s.x0.lo) + 0.1
    TH, W = s.bands(0.7)
    X = Interval((c - h)[:, None], (c + h)[:, None])
    Jx, Jt = s.jacs(X, Interval(TH.lo[:, None], TH.hi[:, None]), W)
    Fi = s.F(X, Interval(TH.lo[:, None], TH.hi[:, None]), W)
    for _ in range(200):
        x = c + h * rng.uniform(-1, 1, s.n)
        th = TH.lo + (TH.hi - TH.lo) * rng.random(s.p)
        w = W.lo + (W.hi - W.lo) * rng.random(len(s.w))
        jx, jt = s.jacs(x, th, w)
        assert np.all(jx >= Jx.lo[0] - 1e-9) and np.all(jx <= Jx.hi[0] + 1e-9)
        assert np.all(jt >= Jt.lo[0] - 1e-9) and np.all(jt <= Jt.hi[0] + 1e-9)
        fx = s.F(x, th, w)
        assert np.all(fx >= Fi.lo[:, 0] - 1e-9) and np.all(fx <= Fi.hi[:, 0] + 1e-9)


def test_liu_bands_nested_and_partition():
    v = LiuNormal(1.0, 0.3)
    prev = (1.0, 1.0)
    for a in np.linspace(0.01, 0.99, 30):
        lo, hi = v.band(a)
        assert lo <= prev[0] and hi >= prev[1]
        assert abs(v.cdf(hi) - v.cdf(lo) - a) < 1e-9
        prev = (lo, hi)
    for m in np.linspace(0, 1, 21):
        assert abs(memberships(m).sum() - 1) < 1e-12
