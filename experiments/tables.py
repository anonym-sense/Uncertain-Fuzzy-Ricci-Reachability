"""Collect results into paper1/table_main.tex, paper1/table_split.tex and results/summary.md;
also draw the scaling and landscape figures."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from figures import C1, C2, C3, C4, GRID, INK, INK2, UNSAFE, _save  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
P1 = HERE.parent / "paper1"

BENCH = [("acc", "ACC", 3), ("pendulum", "Pendulum", 2), ("s_curve", "S-curve~\\cite{prior2020}", 2),
         ("s_curve_pocket", "S-curve pocket", 2), ("van_der_pol", "Van der Pol", 2),
         ("planar_quad", "Planar quadrotor", 6)]
ENG = [("zono", "UTR-Z"), ("flow", "UTR-flow"), ("ricci", "UTR-Ricci"), ("box", "$\\alpha$-cut int.")]


def load(b, e, split="curvature", sem="path"):
    f = RES / f"{b}__{e}__{split}__{sem}.json"
    return json.loads(f.read_text()) if f.exists() else None


def cell(r):
    if r is None:
        return "--"
    return f"{r['alpha_star']:.3f}"


def main_table():
    lines = [r"\begin{table}[t]\centering\caption{Certified belief level $\underline{\alpha}$ "
             r"(higher is better; sound) vs.\ falsification inner bound $\alpha^\star_{\mathrm{MC}}$. "
             r"Label: linguistic reading of $\M\{\text{unsafe}\}\le1-\underline{\alpha}$ for UTR-Z. "
             r"Time: total UTR-Z wall time of the $\alpha$ bisection (s).}\label{tab:main}",
             r"\setlength{\tabcolsep}{3pt}\footnotesize",
             r"\begin{tabular}{lc" + "c" * len(ENG) + r"ccr}\toprule",
             "System & $n{+}p$ & " + " & ".join(n for _, n in ENG) +
             r" & $\alpha^\star_{\mathrm{MC}}$ & label & time\\\midrule"]
    md = ["| system | n+p | " + " | ".join(n for _, n in ENG) + " | MC | gap closed (UTR-Z) | label | time s |",
          "|" + "---|" * (len(ENG) + 6)]
    for b, name, n in BENCH:
        rs = [load(b, e) for e, _ in ENG]
        mc = load(b, "mc")
        z = rs[0]
        p = (z or mc or {}).get("p", 0)
        lab = z["label"].lower() if z else "--"
        tm = f"{z['wall_time']:.0f}" if z else "--"
        best = max([r["alpha_star"] for r in rs if r] + [0])
        vals = [(f"\\textbf{{{cell(r)}}}" if r and r["alpha_star"] == best and best > 0 else cell(r)) for r in rs]
        lines.append(f"{name} & {n}+{p} & " + " & ".join(vals) +
                     f" & {cell(mc)} & {lab} & {tm}\\\\")
        frac = (z["alpha_star"] / mc["alpha_star"]) if (z and mc and mc["alpha_star"] > 0) else float("nan")
        md.append(f"| {b} | {n}+{p} | " + " | ".join(cell(r) for r in rs) +
                  f" | {cell(mc)} | {100 * frac:.0f}% | {lab} | {tm} |")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (P1 / "table_main.tex").write_text("\n".join(lines))
    return md


SOTA = [("jr_none", "JuliaReach"), ("jr_uniform", "JR+uniform"), ("jr_utr", "UTR-TM (ours)"),
        ("zono", "UTR-Z (ours)"), ("zono2", "UTR-Z2 (ours)")]


def sota_table():
    """Certified alpha and wall time vs JuliaReach (TMJets), same budgets and bisection."""
    lines = [r"\begin{table*}[t]\centering\caption{Comparison with JuliaReach (TMJets, order 6/2) under "
             r"identical semantics, budgets and bisection. Cells: certified $\underline\alpha$ / wall time [s]. "
             r"UTR-TM = TMJets inside our framework (curvature-guided refinement).}\label{tab:sota}",
             r"\footnotesize\begin{tabular}{l" + "c" * len(SOTA) + r"c}\toprule",
             "System & " + " & ".join(n for _, n in SOTA) + r" & $\alpha^\star_{\mathrm{MC}}$\\\midrule"]
    md = ["| system | " + " | ".join(n for _, n in SOTA) + " | MC |", "|" + "---|" * (len(SOTA) + 2)]
    rows = BENCH + [(f"chain{N}", f"Duffing chain $N{{=}}{N}$", 2 * N) for N in (1, 2, 4, 8, 10)]
    for b, name, _ in rows:
        rs = [load(b, e) for e, _ in SOTA]
        mc = load(b, "mc")
        best = max([r["alpha_star"] for r in rs if r] + [0])

        def f(r):
            if r is None:
                return "--"
            s = f"{r['alpha_star']:.3f}/{r['wall_time']:.0f}"
            return f"\\textbf{{{s}}}" if r["alpha_star"] == best and best > 0 else s
        lines.append(f"{name} & " + " & ".join(f(r) for r in rs) + f" & {cell(mc)}\\\\")
        md.append(f"| {b} | " + " | ".join(
            "--" if r is None else f"{r['alpha_star']:.3f} / {r['wall_time']:.0f}s" for r in rs) + f" | {cell(mc)} |")
    lines += [r"\bottomrule\end{tabular}\end{table*}"]
    (P1 / "table_sota.tex").write_text("\n".join(lines))
    return md


def split_table():
    md = ["| system | curvature | sensitivity | width |", "|---|---|---|---|"]
    lines = [r"\begin{table}[t]\centering\caption{Partition strategy ablation (UTR-Z): certified "
             r"$\underline{\alpha}$ and cells used at that level.}\label{tab:split}\footnotesize",
             r"\begin{tabular}{lccc}\toprule System & curvature-guided & sensitivity & width\\\midrule"]
    for b, name, _ in BENCH:
        rs = [load(b, "zono", sp) for sp in ("curvature", "sensitivity", "width")]
        if not any(rs[1:]):
            continue

        def fmt(r):
            if r is None:
                return "--"
            ok = [s for s in r["stats"] if s["safe"] and abs(s["alpha"] - r["alpha_star"]) < 1e-9]
            c = ok[0]["cells"] if ok else 0
            return f"{r['alpha_star']:.3f} ({c})"
        lines.append(f"{name} & " + " & ".join(fmt(r) for r in rs) + r"\\")
        md.append(f"| {b} | " + " | ".join(fmt(r) for r in rs) + " |")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (P1 / "table_split.tex").write_text("\n".join(lines))
    return md


def fig_scaling():
    Ns = (1, 2, 3, 4, 6, 8, 10)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.0))
    for e, lab, col in (("zono", "UTR-Z", C1), ("zono2", "UTR-Z2", C2), ("jr_utr", "UTR-TM", C4),
                        ("jr_none", "JuliaReach", "#6250d6"), ("ricci", "UTR-Ricci", C3),
                        ("mc", "falsification", INK2)):
        xs, a, t = [], [], []
        for N in Ns:
            r = load(f"chain{N}", e)
            if r:
                xs.append(2 * N + r["p"])
                a.append(r["alpha_star"])
                t.append(r["wall_time"])
        if not xs:
            continue
        ls = "--" if e == "mc" else "-"
        axs[0].plot(xs, a, ls, color=col, lw=2, marker="o", ms=4, label=lab)
        axs[1].plot(xs, t, ls, color=col, lw=2, marker="o", ms=4, label=lab)
    # HJ: measured 4D point and grid-size extrapolation (41 points per dimension)
    sc = RES / "scaling.json"
    if sc.exists():
        rows = json.loads(sc.read_text())
        hj = [r for r in rows if "hj" in r and "time" in r["hj"]]
        if hj:
            t4 = hj[0]["hj"]["time"] / 0.5 * 4.0  # per simulated second -> horizon 4 s
            dims = np.arange(4, 9)
            axs[1].plot(dims, t4 * 41.0 ** (dims - 4), ":", color=C2, lw=2, label="HJ grid (extrap.)")
            axs[1].plot([4], [t4], "o", color=C2, ms=4)
    axs[0].set_xlabel("augmented dimension $n+p$")
    axs[1].set_xlabel("augmented dimension $n+p$")
    axs[0].set_ylabel("certified $\\underline{\\alpha}$")
    axs[1].set_ylabel("wall time [s]")
    axs[1].set_yscale("log")
    for ax in axs:
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.legend(fontsize=6)
    _save(fig, "fig_scaling")


def fig_landscape():
    files = sorted(RES.glob("landscape_pocket_closing_*.json"))
    if not files:
        return
    rows = sorted((json.loads(f.read_text())[0] for f in files), key=lambda r: r["alpha"])
    lams = sorted(rows[0]["lam"].keys(), key=float)
    # colour sign follows the (one-pixel tolerant) verdict, magnitude follows the margin
    A = np.array([[(-1 if r["lam"][l]["meets"] else 1) * max(abs(r["lam"][l]["margin"]), 0.01)
                   for l in lams] for r in rows])
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    v = np.nanmax(np.abs(A))
    im = ax.imshow(A.T, origin="lower", aspect="auto", cmap="RdBu", vmin=-v, vmax=v)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['alpha']:.2f}" for r in rows], fontsize=6)
    ax.set_yticks(range(len(lams)))
    ax.set_yticklabels(lams, fontsize=6)
    ax.set_xlabel("belief level $\\alpha$")
    ax.set_ylabel("relaxation radius $\\rho$")
    for i, r in enumerate(rows):
        for j, l in enumerate(lams):
            ax.text(i, j, "safe" if not r["lam"][l]["meets"] else "hit", ha="center", va="center", fontsize=5,
                    color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("margin to $\\mathcal{U}$", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    _save(fig, "fig_landscape")


if __name__ == "__main__":
    md = ["# Paper 1 results summary", "", "## Main table (certified alpha*)", *main_table(), "",
          "## Split ablation (alpha*, cells)", *split_table(), "",
          "## SOTA comparison (alpha* / wall time)", *sota_table()]
    (RES / "summary.md").write_text("\n".join(md))
    fig_scaling()
    fig_landscape()
    print("\n".join(md))
