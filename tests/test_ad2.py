import sys, numpy as np, pytest
sys.path.insert(0, '.')
from frs.ad2 import hessian
from frs.interval import Interval
from frs.systems import BENCHMARKS, chain

SYSTEMS = [f().path_semantics() for f in BENCHMARKS.values()] + [chain(3).path_semantics()]


def fz(s):
    n = s.n
    return lambda Z: list(s.f(Z[:n], Z[n:], []))


@pytest.mark.parametrize("s", SYSTEMS, ids=lambda s: s.name)
def test_hessian_matches_fd(s):
    rng = np.random.default_rng(0)
    z = np.r_[0.5 * (s.x0.lo + s.x0.hi), s.th_hat] + 0.03 * rng.standard_normal(s.n + s.p)
    _, J, H = hessian(fz(s), list(z))
    eps = 1e-5
    m = len(z)
    for i in range(m):
        d = np.zeros(m); d[i] = eps
        _, Jp, _ = hessian(fz(s), list(z + d))
        _, Jm, _ = hessian(fz(s), list(z - d))
        assert np.allclose(H[:, :, i], (Jp - Jm) / (2 * eps), atol=1e-4)


@pytest.mark.parametrize("s", SYSTEMS, ids=lambda s: s.name)
def test_interval_hessian_encloses(s):
    rng = np.random.default_rng(1)
    c = np.r_[0.5 * (s.x0.lo + s.x0.hi), s.th_hat]
    h = np.r_[0.5 * (s.x0.hi - s.x0.lo) + 0.05, 0.02 * np.abs(s.th_hat) + 0.01]
    Z = [Interval(c[i] - h[i], c[i] + h[i]) for i in range(len(c))]
    _, Ji, Hi = hessian(fz(s), Z)
    for _ in range(100):
        z = c + h * rng.uniform(-1, 1, len(c))
        _, J, H = hessian(fz(s), list(z))
        assert np.all(H >= Hi.lo - 1e-9) and np.all(H <= Hi.hi + 1e-9)
        assert np.all(J >= Ji.lo - 1e-9) and np.all(J <= Ji.hi + 1e-9)
