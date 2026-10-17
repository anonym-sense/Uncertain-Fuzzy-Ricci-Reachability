"""Attractive visuals (README + papers): animations and a scoreboard.

  python visuals.py tube      -> figs/hero_belief_tube.gif   (belief dial on the S-curve)
  python visuals.py flow      -> figs/flow_vs_sound.gif      (2020 flow vs sound relaxation)
  python visuals.py score     -> figs/scoreboard.png/.pdf    (certificate vs compute)
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
RES, FIG = HERE / "results", HERE / "figs"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, INK, INK2, BG = "#e34948", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c3c2b7",
                     "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2})


def _label(m):
    from frs.uncertain import linguistic
    return linguistic(m)[0].lower()


# ------------------------------------------------------------------ belief dial
def tube_frames(alphas, budget=256):
    from frs.engine import ZonoEngine
    from frs.mc import Sampler
    from frs.relax import Grid
    from frs.systems import s_curve
    from replication_2020 import certified_cells, rasterise
    s = s_curve().path_semantics()
    eng = ZonoEngine(s)
    frames = []
    for a in alphas:
        cells, safe = certified_cells(eng, a, budget)
        g = Grid((-1.0, 6.6), (-2.5, 2.3), 300, 190)
        T = rasterise(eng, a, cells, g)
        S = Sampler(s)
        _, tr = S.simulate(S.random(120, np.random.default_rng(1)), a, record=True)
        frames.append(dict(alpha=a, safe=bool(safe), T=T, tr=tr[:, :2, :]))
        print(f"  frame alpha={a:.2f} safe={safe} cells={len(cells)}", flush=True)
    np.savez_compressed(RES / "hero_frames.npz", alphas=np.array(alphas), safe=np.array([f["safe"] for f in frames]),
                        T=np.array([f["T"] for f in frames]), x=g.x, y=g.y,
                        tr=np.array([f["tr"] for f in frames]))
    return frames, g, s.unsafe[0]


def hero_gif():
    alphas = [0.05, 0.15, 0.25, 0.35, 0.45, 0.52, 0.58, 0.62, 0.66, 0.72, 0.82, 0.92]
    frames, g, U = tube_frames(alphas)
    ext = [g.x[0], g.x[-1], g.y[0], g.y[-1]]
    fig = plt.figure(figsize=(8, 4.2), facecolor=BG)
    ax = fig.add_axes([0.06, 0.12, 0.62, 0.74], facecolor=BG)
    dial = fig.add_axes([0.72, 0.18, 0.26, 0.6], facecolor=BG)

    def draw(i):
        f = frames[i]
        ax.clear(); dial.clear()
        ax.set_facecolor(BG)
        for k in range(0, f["tr"].shape[2], 3):
            ax.plot(f["tr"][:, 0, k], f["tr"][:, 1, k], color=INK2, lw=0.4, alpha=0.25)
        col = BLUE if f["safe"] else RED
        ax.imshow(np.where(f["T"], 1.0, np.nan), extent=ext, origin="lower", aspect="auto",
                  cmap=matplotlib.colors.ListedColormap([col]), alpha=0.55)
        ax.add_patch(plt.Rectangle((U.lo[0], U.lo[1]), U.hi[0] - U.lo[0], U.hi[1] - U.lo[1],
                                   color=RED, alpha=0.85, lw=0))
        ax.text(U.lo[0] - 0.05, U.lo[1] - 0.18, "unsafe", color=RED, fontsize=9, ha="right")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("certified reach tube of every behaviour the system can have",
                     fontsize=10, color=INK2, loc="left")
        # dial
        th = np.linspace(np.pi, 0, 200)
        dial.plot(np.cos(th), np.sin(th), color="#e4e3df", lw=14, solid_capstyle="round")
        frac = f["alpha"]
        th2 = np.linspace(np.pi, np.pi * (1 - frac), 100)
        dial.plot(np.cos(th2), np.sin(th2), color=col, lw=14, solid_capstyle="round")
        dial.text(0, 0.18, f"belief α = {frac:.2f}", ha="center", fontsize=12, color=INK)
        verdict = "PROVEN SAFE" if f["safe"] else "NOT PROVABLE"
        dial.text(0, -0.25, verdict, ha="center", fontsize=15, weight="bold", color=col)
        sub = (f"risk ≤ {1 - frac:.2f}  ({_label(1 - frac)})" if f["safe"]
               else "beliefs too wide to guarantee")
        dial.text(0, -0.52, sub, ha="center", fontsize=9, color=INK2)
        dial.set_xlim(-1.25, 1.25); dial.set_ylim(-0.7, 1.15); dial.axis("off")
        fig.suptitle("How sure must you be about the uncertain parameters to prove safety?",
                     fontsize=12, color=INK, x=0.06, ha="left", y=0.97)

    seq = list(range(len(frames))) + [len(frames) - 1] * 3 + list(range(len(frames) - 1, -1, -1)) + [0] * 2
    anim = FuncAnimation(fig, lambda k: draw(seq[k]), frames=len(seq), interval=450)
    anim.save(FIG / "hero_belief_tube.gif", writer=PillowWriter(fps=2.2), dpi=90)
    draw(alphas.index(0.58)); fig.savefig(FIG / "hero_belief_tube.png", dpi=160, facecolor=BG)
    print("wrote hero_belief_tube.gif")


# ------------------------------------------------------------------ flow vs sound
def flow_gif(nframes=24):
    from frs.relax import Grid, relax
    d = np.load(RES / "replication_2020_masks.npz")
    x, y, T, H0, U, P = d["x"], d["y"], d["T"], d["H0"], d["U"], d["P"]
    g = Grid((x[0], x[-1]), (y[0], y[-1]), len(x), len(y))
    from frs.relax import sdf
    near = np.clip(1 - sdf(U, g.h) / 2.0, 0, 1)
    w = 0.02 * (0.25 + 0.75 * (1 - near))
    R = 0.5 * (y[-1] - y[0])
    n_coll = int(R ** 2 / (0.4 * min(g.h) ** 2) * 0.35)
    ticks = np.unique(np.linspace(1, n_coll, nframes).astype(int))
    cache = RES / "flow_frames.npz"
    if cache.exists():
        c = np.load(cache)
        ticks = c["ticks"]
        old = {int(k): m for k, m in zip(ticks, c["old"])}
        new = list(c["new"])
    else:
        _, old = relax(g, T, H0, None, iters=n_coll, constrained=False, w=w, log_at=tuple(ticks))
        lams = np.geomspace(3.0, 0.05, len(ticks))
        new = []
        for lam in lams:  # sound relaxation path: hull -> tube
            it = int(3.0 / (0.2 * min(g.h) ** 2 / lam))
            new.append(relax(g, T, H0, lam, iters=it)[0])
        np.savez_compressed(cache, ticks=ticks, old=np.array([old[int(k)] for k in ticks]), new=np.array(new))
    ii = np.clip(np.rint((P[:, 1] - y[0]) / g.h[1]).astype(int), 0, len(y) - 1)
    jj = np.clip(np.rint((P[:, 0] - x[0]) / g.h[0]).astype(int), 0, len(x) - 1)
    ext = [x[0], x[-1], y[0], y[-1]]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.9), facecolor=BG)
    fig.subplots_adjust(top=0.74, bottom=0.03, left=0.02, right=0.98, wspace=0.08)

    def panel(ax, m, title, col):
        ax.clear(); ax.set_facecolor(BG)
        inside = m[ii, jj]
        ax.scatter(P[inside][::5, 0], P[inside][::5, 1], s=0.5, color=INK2, alpha=0.3, lw=0)
        ax.scatter(P[~inside][::2, 0], P[~inside][::2, 1], s=3, color=RED, lw=0)
        ax.contour(x, y, m.astype(float), levels=[0.5], colors=[col], linewidths=1.6)
        ax.contour(x, y, H0.astype(float), levels=[0.5], colors=["#c3c2b7"], linewidths=0.8, linestyles="--")
        ax.set_xticks([]); ax.set_yticks([])
        lost = 100 * (1 - inside.mean())
        ax.set_title(f"{title}\nreal states lost: {lost:.1f}%", fontsize=10, color=RED if lost > 0.05 else AQUA)

    def draw(k):
        panel(axs[0], old[ticks[k]], "original Ricci flow (2020)", ORANGE)
        panel(axs[1], new[k], "sound topography relaxation (ours)", BLUE)
        fig.suptitle("Simplifying the reachable set: one method silently drops real behaviours",
                     fontsize=11, color=INK, x=0.02, y=0.98, ha="left")

    seq = list(range(len(ticks))) + [len(ticks) - 1] * 5
    FuncAnimation(fig, lambda f: draw(seq[f]), frames=len(seq), interval=300).save(
        FIG / "flow_vs_sound.gif", writer=PillowWriter(fps=3.3), dpi=85)
    draw(len(ticks) - 1); fig.savefig(FIG / "flow_vs_sound.png", dpi=160, facecolor=BG)
    print("wrote flow_vs_sound.gif")


# ------------------------------------------------------------------ scoreboard
def scoreboard():
    def ld(b, e):
        f = RES / f"{b}__{e}__curvature__path.json"
        return json.loads(f.read_text()) if f.exists() else None
    benches = [("s_curve", "S-curve"), ("van_der_pol", "Van der Pol"), ("pendulum", "Pendulum"),
               ("acc", "ACC"), ("planar_quad", "Quadrotor 6D")]
    methods = [("zono2", "ours: UTR-Z2", BLUE, "o"), ("zono", "ours: UTR-Z", AQUA, "o"),
               ("jr_uniform", "JuliaReach + partitions", ORANGE, "s"), ("jr_none", "JuliaReach", YELLOW, "s")]
    fig, axs = plt.subplots(1, len(benches), figsize=(12, 3.0), facecolor=BG, sharey=True)
    for ax, (b, name) in zip(axs, benches):
        ax.set_facecolor(BG)
        mc = ld(b, "mc")
        if mc:
            ax.axhline(mc["alpha_star"], color=INK2, lw=1, ls=":")
            ax.text(0.03, mc["alpha_star"] + 0.015, "true limit (falsification)", fontsize=6.5, color=INK2,
                    transform=ax.get_yaxis_transform(), clip_on=True)
        for e, lab, col, mk in methods:
            r = ld(b, e)
            if r:
                ax.scatter(max(r["wall_time"], 1), r["alpha_star"], s=70, color=col, marker=mk,
                           edgecolor="white", lw=1.2, zorder=3, label=lab)
        if b == "van_der_pol":
            hj = RES / "van_der_pol__hj__curvature__path.json"
            if hj.exists():
                h = json.loads(hj.read_text())
                ax.scatter(h["wall_time"], h["alpha_star"], s=80, color=RED, marker="X", zorder=3,
                           label="HJ grid (claims too much)")
        ax.set_xscale("log"); ax.set_title(name, fontsize=10, color=INK)
        ax.set_xlabel("compute time [s]", fontsize=8)
        ax.grid(True, color="#ecebe7", lw=0.6)
    axs[0].set_ylabel("certified safe up to belief α*")
    h, l = [], []
    for ax in axs:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, -0.06, "higher is better (more uncertainty tolerated), left is faster; never above the dotted line = sound",
             ha="center", fontsize=8.5, color=INK2)
    fig.savefig(FIG / "scoreboard.png", dpi=170, bbox_inches="tight", facecolor=BG)
    fig.savefig(FIG / "scoreboard.pdf", bbox_inches="tight", facecolor=BG)
    print("wrote scoreboard")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["score", "flow", "tube"]:
        {"tube": hero_gif, "flow": flow_gif, "score": scoreboard}[w]()
