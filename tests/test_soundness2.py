import sys
import numpy as np
import pytest
from scipy.optimize import linprog
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from frs.engine2 import Zono2Engine  # noqa: E402
from test_soundness import CASES, _cell, _samples  # noqa: E402


@pytest.mark.parametrize("s", CASES, ids=lambda s: s.name)
def test_zono2_contains_samples(s):
    a = 0.4
    cells = _cell(s)
    s.unsafe = []
    o = Zono2Engine(s).run(a, cells, record=True)
    xs, th = _samples(s, cells, a)
    for k in np.linspace(0, len(o["G"]) - 1, 12).astype(int):
        G, c, ref = o["G"][k][0], o["c"][k][0], o["xs"][k][:, 0]
        for b in range(0, xs.shape[2], 6):
            z = np.r_[xs[k, :, b], th[b]] - ref - c
            res = linprog(np.zeros(G.shape[1]), A_eq=G, b_eq=z, bounds=[(-1 - 1e-7, 1 + 1e-7)] * G.shape[1],
                          method="highs")
            assert res.status == 0, f"sample {b} escapes at step {k}"
