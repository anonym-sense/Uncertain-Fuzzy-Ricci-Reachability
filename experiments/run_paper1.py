"""Paper 1 main experiment: belief-graded verification alpha* per benchmark and engine.

usage:  python run_paper1.py <benchmark> <engine|mc> [budget] [split] [semantics]
  engine in {zono, flow, ricci, box}; 'mc' runs the falsification inner bound.
  semantics in {path (default, Yao-Chen alpha-paths), adv (time-varying adversary)}.
Writes experiments/results/<benchmark>__<engine>__<split>__<semantics>.json
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from frs.engine import BoxEngine, FlowEngine, TubeEngine, ZonoEngine, alpha_star  # noqa: E402
from frs.engine2 import Zono2Engine  # noqa: E402
from frs.mc import alpha_star_mc  # noqa: E402
from frs.systems import BENCHMARKS, chain  # noqa: E402
from frs.uncertain import linguistic  # noqa: E402

ENGINES = {"zono": ZonoEngine, "zono2": Zono2Engine, "flow": FlowEngine, "ricci": TubeEngine, "box": BoxEngine}
OUT = pathlib.Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def make(name):
    if name in BENCHMARKS:
        return BENCHMARKS[name]()
    if name.startswith("chain"):
        return chain(int(name[5:]))
    raise KeyError(name)


def main():
    name, eng = sys.argv[1], sys.argv[2]
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    split = sys.argv[4] if len(sys.argv) > 4 else "curvature"
    sem = sys.argv[5] if len(sys.argv) > 5 else "path"
    s = make(name)
    if sem == "path":
        s = s.path_semantics()
    tag = f"{name}__{eng}__{split}__{sem}"
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.perf_counter()
    if eng == "mc":
        a = alpha_star_mc(s, tol=2e-3, B=512, iters=10)
        res = {"alpha_star": a}
    else:
        E = ENGINES[eng](s)
        a, stats = alpha_star(E, tol=1e-2, max_cells=budget, log=log, split=split)
        res = {"alpha_star": a, "engine": E.name, "budget": budget,
               "stats": [dict(alpha=x[0], safe=x[1], cells=x[2], time=x[3]) for x in stats]}
        if eng == "ricci":
            res["lmi_gamma"] = float(E.gamma)
    lab, mu = linguistic(1 - res["alpha_star"])
    res.update(benchmark=name, n=s.n, p=s.p, split=split, semantics=sem,
               wall_time=time.perf_counter() - t0, label=lab, label_membership=mu,
               measure_bound=1 - res["alpha_star"])
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2))
    log(f"DONE {tag}: alpha*={res['alpha_star']:.4f}  M<= {1 - res['alpha_star']:.4f} ({lab})  "
        f"{res['wall_time']:.0f}s")


if __name__ == "__main__":
    main()
