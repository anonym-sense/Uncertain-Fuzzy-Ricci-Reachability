"""Safety landscape (alpha, lambda) -> margin of H*_lambda(alpha) to the unsafe set (S-curve).

For each belief level alpha: refine (UTR-Z, curvature splits), rasterise the tube T,
relax with the constrained flow for each lambda, and record the signed margin
dist(H*_lambda, U) (one-pixel tolerance; exact verdicts come from the engine).
Writes experiments/results/landscape.json
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
from scipy.ndimage import binary_erosion  # noqa: E402

from frs.engine import ZonoEngine  # noqa: E402
from frs.relax import Grid, closing, convex_hull_mask, relax, sdf  # noqa: E402
from frs.systems import s_curve, s_curve_pocket  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from replication_2020 import certified_cells, rasterise  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"
LAMS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def main(alphas=(0.05, 0.25, 0.45, 0.65, 0.8, 0.9, 0.95, 0.98), pocket=True, out=None, method="flow"):
    s = (s_curve_pocket() if pocket else s_curve()).path_semantics()
    U = s.unsafe[0]
    eng = ZonoEngine(s)
    rows = []
    for a in alphas:
        t0 = time.perf_counter()
        cells, safe = certified_cells(eng, a, 512)
        grid = Grid((-1.2, 7.0), (-2.6, 2.6), 392, 248)
        T = rasterise(eng, a, cells, grid)
        H0 = convex_hull_mask(grid, T)
        Um = (grid.X >= U.lo[0]) & (grid.X <= U.hi[0]) & (grid.Y >= U.lo[1]) & (grid.Y <= U.hi[1])
        Uin = binary_erosion(Um)  # one-pixel tolerance for raster inflation
        row = {"alpha": a, "certified_safe": safe, "cells": len(cells), "lam": {}}
        for lam in LAMS:
            if method == "closing":  # exact, nested in lam by construction
                H = closing(T, lam, grid.h) & H0
            else:  # constrained curvature flow from the hull
                it = int(4.0 / (0.2 * min(grid.h) ** 2 / lam))
                H, _ = relax(grid, T, H0, lam, iters=it)
            d = sdf(H, grid.h)
            row["lam"][str(lam)] = {"margin": float(d[Um].min()), "meets": bool((H & Uin).any()),
                                    "area": grid.area(H)}
        row["hull_meets"] = bool((H0 & Uin).any())
        row["time_s"] = time.perf_counter() - t0
        print(json.dumps(row), flush=True)
        rows.append(row)
    out = out or OUT / ("landscape_pocket.json" if pocket else "landscape.json")
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:  # single alpha -> results/landscape_pocket_<alpha>.json
        a = float(sys.argv[1])
        meth = sys.argv[2] if len(sys.argv) > 2 else "flow"
        tag = "" if meth == "flow" else "closing_"
        main(alphas=(a,), out=OUT / f"landscape_pocket_{tag}{a}.json", method=meth)
    else:
        main()
