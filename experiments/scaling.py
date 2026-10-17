"""Scaling with state dimension: chain of N Duffing masses (n = 2N), alpha-path semantics.

For each N: wall time of one sound reachability run (root cell) per engine, whether it
certifies safety at alpha = 0.5, and an HJ grid solve (hj_reachability) where feasible.
Writes experiments/results/scaling.json
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from frs.engine import BoxEngine, Cells, FlowEngine, TubeEngine, ZonoEngine  # noqa: E402
from frs.mc import falsify  # noqa: E402
from frs.systems import chain  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"


def hj_time(s, alpha, pts_per_dim):
    """One HJ backward-reachable-tube solve on the augmented (x, theta) grid."""
    import jax.numpy as jnp
    import hj_reachability as hj

    n, p = s.n, s.p
    TH = s.bands(alpha)[0]
    lo = np.r_[np.full(n, -1.5), TH.lo]
    hi = np.r_[np.full(n, 1.5), TH.hi]
    N = n + p

    class Dyn(hj.ControlAndDisturbanceAffineDynamics):
        def __init__(self):
            super().__init__(hj.dynamics.ControlMode.MAX if False else "max", "min",
                             hj.sets.Box(jnp.zeros(1), jnp.zeros(1)),
                             hj.sets.Box(jnp.zeros(1), jnp.zeros(1)))

        def open_loop_dynamics(self, z, time):
            x, th = [z[i] for i in range(n)], [z[n + j] for j in range(p)]
            fx = s.f(x, th, [])
            return jnp.concatenate([jnp.stack(fx), jnp.zeros(p)])

        def control_jacobian(self, z, time):
            return jnp.zeros((N, 1))

        def disturbance_jacobian(self, z, time):
            return jnp.zeros((N, 1))

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array(lo), jnp.array(hi)), tuple([pts_per_dim] * N))
    l0 = grid.states[..., 0] - 0.0  # placeholder target; cost is what we measure
    settings = hj.SolverSettings.with_accuracy("low")
    t0 = time.perf_counter()
    hj.solve(settings, Dyn(), grid, np.array([0.0, -0.5]), l0).block_until_ready()
    return time.perf_counter() - t0, int(np.prod(grid.shape))


def main(Ns=(1, 2, 3, 4, 6, 8, 10), alpha=0.5):
    rows = []
    for N in Ns:
        s = chain(N).path_semantics()
        root = Cells.root(s)
        row = {"N": N, "n": s.n, "aug": s.n + s.p}
        for key, E in (("zono", ZonoEngine(s)), ("flow", FlowEngine(s)), ("ricci", TubeEngine(s)),
                       ("box", BoxEngine(s))):
            t0 = time.perf_counter()
            o = E.run(alpha, root)
            row[key] = {"time": time.perf_counter() - t0, "safe": bool(not o["hit"][0])}
        t0 = time.perf_counter()
        d, _ = falsify(s, alpha, B=512, iters=8)
        row["mc"] = {"time": time.perf_counter() - t0, "min_sdf": float(d)}
        if s.n + s.p <= 4:
            try:
                tt, pts = hj_time(s, alpha, 41 if s.n + s.p <= 4 else 15)
                row["hj"] = {"time": tt, "grid_points": pts}
            except Exception as ex:  # noqa: BLE001
                row["hj"] = {"error": repr(ex)}
        print(row, flush=True)
        rows.append(row)
    (OUT / "scaling.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
