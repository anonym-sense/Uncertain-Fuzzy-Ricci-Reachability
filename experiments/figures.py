"""Paper 1 figures (static, print-ready PDF + PNG), reference categorical palette.

fig_replication: (a) certified tube, uncertain hull, MC states, unsafe set
                 (b) 2020 unconstrained flow at the Table-1 schedule, lost MC states
                 (c) sound relaxation family H*_lambda
fig_alpha:       certified alpha* per benchmark and engine vs falsification bound
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figs"
FIG.mkdir(exist_ok=True)

# reference palette (light): categorical slots in fixed order; red reserved for unsafe
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
UNSAFE = "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "axes.grid": False, "legend.frameon": False,
    "font.family": "serif", "pdf.fonttype": 42,
})


def _save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_replication():
    d = np.load(RES / "replication_2020_masks.npz")
    js = json.loads((RES / "replication_2020.json").read_text())
    x, y, T, H0, U, P = d["x"], d["y"], d["T"], d["H0"], d["U"], d["P"]
    ext = [x[0], x[-1], y[0], y[-1]]

    def contour(ax, m, color, lw=1.0, ls="-", label=None):
        ax.contour(x, y, m.astype(float), levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)
        if label:
            ax.plot([], [], color=color, lw=lw, ls=ls, label=label)

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True)
    for ax in axs:
        ax.imshow(np.where(U, 1.0, np.nan), extent=ext, origin="lower", cmap="Reds", vmin=0, vmax=1.6,
                  aspect="auto")
        ax.set_xlabel("$x_1$")
    ax = axs[0]
    ax.scatter(P[::6, 0], P[::6, 1], s=0.3, color=INK2, alpha=0.25, lw=0, label="MC states")
    ax.imshow(np.where(T, 1.0, np.nan), extent=ext, origin="lower", cmap="Blues", vmin=0, vmax=1.4,
              aspect="auto")
    contour(ax, H0, INK2, 0.8, "--", "uncertain hull $H_0$")
    ax.plot([], [], color=C1, lw=4, alpha=0.5, label="certified tube $T$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(a) certified tube, $\\alpha=%.1f$" % js["alpha"], fontsize=8, loc="left")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.5)

    ax = axs[1]
    ii = np.clip(np.rint((P[:, 1] - y[0]) / (y[1] - y[0])).astype(int), 0, len(y) - 1)
    jj = np.clip(np.rint((P[:, 0] - x[0]) / (x[1] - x[0])).astype(int), 0, len(x) - 1)
    shades = [C1, C3, C4, C2]
    for (k, col) in zip(("10", "100", "1000", "2000"), shades):
        m = d[f"f{k}"]
        cov = js["flow2020"][k]["cover"]
        contour(ax, m, col, 0.9, label=f"{k} it.: {100 * cov:.1f}% states")
    lost = ~d["f2000"][ii, jj]
    ax.scatter(P[lost][::3, 0], P[lost][::3, 1], s=0.6, color=C2, lw=0)
    ax.set_title("(b) 2020 flow (unconstrained)", fontsize=8, loc="left")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.2)

    ax = axs[2]
    ax.imshow(np.where(T, 1.0, np.nan), extent=ext, origin="lower", cmap="Blues", vmin=0, vmax=1.4,
              aspect="auto")
    for lam, col, ls in (("0.25", C3, "-"), ("1.0", C1, "-"), ("2.0", C4, "-")):
        if f"H{lam}" in d:
            cov = js["sound"][lam]["cover"]
            contour(ax, d[f"H{lam}"], col, 0.9, ls, f"$\\lambda$={lam}: {100 * cov:.0f}% states")
    contour(ax, H0, INK2, 0.8, "--")
    ax.set_title("(c) sound relaxation $H^*_\\lambda$", fontsize=8, loc="left")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.2)
    for ax in axs:
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    _save(fig, "fig_replication")


ENG = [("zono", "UTR-Z (zonotope)", C1), ("flow", "UTR-flow (ellipsoid)", C2),
       ("ricci", "UTR-Ricci (contraction)", C3), ("box", "$\\alpha$-cut interval", C4)]
BENCH = [("acc", "ACC\n3D"), ("pendulum", "Pendulum\n2D"), ("s_curve", "S-curve\n[2020]"),
         ("van_der_pol", "Van der\nPol"), ("planar_quad", "Quadrotor\n6D")]


def load(b, e, split="curvature", sem="path"):
    f = RES / f"{b}__{e}__{split}__{sem}.json"
    return json.loads(f.read_text()) if f.exists() else None


def fig_alpha():
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    w = 0.18
    xs = np.arange(len(BENCH))
    for i, (e, lab, col) in enumerate(ENG):
        vals = [(load(b, e) or {}).get("alpha_star", np.nan) for b, _ in BENCH]
        ax.bar(xs + (i - 1.5) * w, vals, w * 0.9, color=col, label=lab, zorder=2)
    mc = [(load(b, "mc") or {}).get("alpha_star", np.nan) for b, _ in BENCH]
    ax.scatter(xs, mc, marker="_", s=260, color=INK, lw=1.4, zorder=3, label="falsification (inner)")
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in BENCH], rotation=0, fontsize=6.5)
    ax.set_ylabel("certified $\\underline{\\alpha}$")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.legend(fontsize=5.8, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.32))
    _save(fig, "fig_alpha")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["replication", "alpha"]
    if "replication" in which:
        fig_replication()
    if "alpha" in which:
        fig_alpha()
    print("figures written to", FIG)
