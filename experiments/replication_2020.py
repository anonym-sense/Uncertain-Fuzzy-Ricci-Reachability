"""Replicate the 2020 Sensors-Letters experiment and contrast it with the sound relaxation.

1. Certified tube T of the S-curve scenario at belief level alpha (zonotope engine with
   curvature-guided partitions), rasterised in the (x1, x2) plane.
2. H0 = convex hull of T (the 2020 'uncertain hull').
3. 2020 flow: unconstrained curvature flow of H0 for 10/100/1000/2000 iterations.
   We measure covered MC states (soundness) and whether the set meets the unsafe box.
4. Sound relaxation (Theorem 2): constrained flow with constant and linguistic lambda;
   check containment and nesting in lambda; margin to the unsafe set vs lambda.
Outputs: experiments/results/replication_2020.json and figs/replication_*.png
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from frs.engine import Cells, ZonoEngine, verify  # noqa: E402
from frs.mc import Sampler  # noqa: E402
from frs.relax import Grid, convex_hull_mask, relax, sdf, zonotope_polygon  # noqa: E402
from frs.systems import s_curve  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"
FIG = pathlib.Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def certified_cells(engine, alpha, budget):
    """Refine at alpha; return the leaf cells (safe or not) for rasterisation."""
    from frs.engine import sensitivity
    sens = sensitivity(engine.sys)
    todo, leaves = Cells.root(engine.sys), []
    while len(todo) and len(leaves) + len(todo) <= budget:
        o = engine.run(alpha, todo)
        hits = o["hit"]
        leaves.append(todo.take(np.where(~hits)[0]))
        bad = np.where(hits)[0]
        if not len(bad):
            todo = todo.take(bad)
            break
        sc = o["score"][bad]
        todo = todo.split_dims(bad, np.argmax(sc, 1))
    leaves.append(todo)
    return Cells.cat(leaves), len(todo) == 0


def rasterise(engine, alpha, cells, grid, i=0, j=1):
    o = engine.run(alpha, cells, record=True)
    xs = o["xs"]
    for k, (G, c, D) in enumerate(zip(o["G"], o["c"], o["D"])):
        xa, xb = xs[k].T, xs[k + 1].T
        for q in range(len(cells)):
            dl = xb[q] - xa[q]
            Gq = np.hstack([G[q], np.diag(D[q]), 0.5 * dl[:, None]])[[i, j]]
            ctr = (xa[q] + c[q] + 0.5 * dl)[[i, j]]
            grid.fill_polygon(zonotope_polygon(ctr, Gq))
    return grid.mask.copy()


def main(alpha=0.3, budget=256):
    s = s_curve().path_semantics()
    U = s.unsafe[0]
    eng = ZonoEngine(s)
    t0 = time.perf_counter()
    cells, safe = certified_cells(eng, alpha, budget)
    print(f"alpha={alpha}: {len(cells)} cells, certified safe={safe}, {time.perf_counter() - t0:.0f}s")

    grid = Grid((-1.2, 7.0), (-2.6, 2.6), 328, 208)
    masks = {}
    T = rasterise(eng, alpha, cells, grid)
    H0 = convex_hull_mask(grid, T)
    Umask = (grid.X >= U.lo[0]) & (grid.X <= U.hi[0]) & (grid.Y >= U.lo[1]) & (grid.Y <= U.hi[1])

    # MC states at this alpha (all times)
    S = Sampler(s)
    _, tr = S.simulate(S.random(800, np.random.default_rng(0)), alpha, record=True)
    P = tr[:, :2, :].transpose(0, 2, 1).reshape(-1, 2)[::7]
    ii = np.clip(np.rint((P[:, 1] - grid.y[0]) / grid.h[1]).astype(int), 0, len(grid.y) - 1)
    jj = np.clip(np.rint((P[:, 0] - grid.x[0]) / grid.h[0]).astype(int), 0, len(grid.x) - 1)

    def cover(mask):
        return float(mask[ii, jj].mean())

    def meets(mask):
        return bool((mask & Umask).any())

    def margin(mask):
        d = sdf(mask, grid.h)
        return float(d[Umask].min())  # distance of unsafe set to the set (<=0 if meeting)

    res = {"alpha": alpha, "cells": len(cells), "certified_safe": safe,
           "tube": {"area": grid.area(T), "cover": cover(T), "meets_unsafe": meets(T), "margin": margin(T)},
           "hull": {"area": grid.area(H0), "cover": cover(H0), "meets_unsafe": meets(H0), "margin": margin(H0)}}
    print("tube", res["tube"], "\nhull", res["hull"])

    # 2020 flow: unconstrained curvature motion with linguistic speed weights
    near = np.clip(1 - sdf(Umask, grid.h) / 2.0, 0, 1)  # membership 'near unsafe'
    w2020 = 0.02 * (0.25 + 0.75 * (1 - near))  # 2020: high importance (near) morphs slowly
    # explicit curvature flow moves ~sqrt(0.4 h^2 iters); collapse of the hull needs ~R^2/(0.4h^2)
    # iterations.  Snapshots at 1/200, 1/20, 1/2 and 1x that count mirror the 2020 Table 1
    # schedule (10, 100, 1000, 2000 iterations of a 2000-iteration collapse).
    R = 0.5 * (grid.y[-1] - grid.y[0])
    n_coll = int(R ** 2 / (0.4 * min(grid.h) ** 2) * 0.35)
    sched = {10: n_coll // 200, 100: n_coll // 20, 1000: n_coll // 2, 2000: n_coll}
    _, snaps_raw = relax(grid, T, H0, None, iters=n_coll, constrained=False, w=w2020,
                         log_at=tuple(sched.values()))
    snaps = {k: snaps_raw[v] for k, v in sched.items()}
    res["flow2020"] = {str(k): {"iters": sched[k], "area": grid.area(m), "cover": cover(m),
                                "meets_unsafe": meets(m), "margin": margin(m)} for k, m in snaps.items()}
    print("2020 flow", res["flow2020"])

    # sound relaxation (area + lambda perimeter, obstacle T), then linguistic lambda
    res["sound"] = {}
    prev = None
    for lam in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):  # increasing lam -> nested, growing sets
        t1 = time.perf_counter()
        it = int(3.0 / (0.2 * min(grid.h) ** 2 / lam))
        Hf, _ = relax(grid, T, H0, lam, iters=it)
        res["sound"][str(lam)] = {"area": grid.area(Hf), "cover": cover(Hf), "meets_unsafe": meets(Hf),
                                  "margin": margin(Hf), "contains_tube": bool((T <= Hf).all()),
                                  "nested_in_next": None, "iters": it, "time_s": time.perf_counter() - t1}
        if prev is not None:
            res["sound"][prev[0]]["nested_in_next"] = float((prev[1] & ~Hf).sum() / max(prev[1].sum(), 1))
        prev = (str(lam), Hf)
        masks[f"H{lam}"] = Hf
        print("lambda", lam, res["sound"][str(lam)], flush=True)
    lam_field = 0.1 + 1.4 * (1 - near)  # tight near the unsafe set, loose elsewhere
    Hl, _ = relax(grid, T, H0, lam_field, iters=int(3.0 / (0.2 * min(grid.h) ** 2 / lam_field.max())))
    res["sound"]["linguistic"] = {"area": grid.area(Hl), "cover": cover(Hl), "meets_unsafe": meets(Hl),
                                  "margin": margin(Hl), "contains_tube": bool((T <= Hl).all())}
    print("linguistic", res["sound"]["linguistic"])

    np.savez_compressed(OUT / "replication_2020_masks.npz", T=T, H0=H0, U=Umask, Hl=Hl, x=grid.x, y=grid.y,
                        P=P, **{f"f{k}": m for k, m in snaps.items()}, **masks)
    (OUT / "replication_2020.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    main(alpha=a)
