"""Batched sound reachability engines + partition refinement + alpha* search.

A *cell* is a sub-box of the initial set X0 together with a sub-box of the normalised
parameter cube [-1,1]^p (scaled to the alpha band at run time). Disturbances w(t) are
never split (they are time varying inside their alpha band).

Engines (all sound enclosures, same reference trajectories, same cells):
  FlowEngine        UTR-flow: ellipsoid = unit ball of a metric g_k transported by the
                    linearisation plus conformal/curvature correction (augmented state)
  TubeEngine        UTR-Ricci: scalar radius in a fixed Ricci-optimal (LMI contraction)
                    metric, r' = -kappa r + d  (Theorem 1); parameters kept as bands
  ZonoEngine        first-order zonotope propagation with interval remainder (augmented)
  BoxEngine         plain alpha-cut interval propagation, classic fuzzy reachability

verify(): fuzzy-partition refinement. Cells whose enclosure meets the unsafe set (or
diverges) are split along the input with largest sensitivity-weighted width, until all
cells are certified SAFE or the cell budget is exhausted (UNSAFE-POSSIBLE).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog, lsq_linear

from .interval import Interval, stack
from .systems import System

ALPHA_MAX = 0.999


# ----------------------------------------------------------------------------- cells
@dataclass
class Cells:
    xlo: np.ndarray  # (C, n)
    xhi: np.ndarray
    ulo: np.ndarray  # (C, p) normalised parameter sub-box in [-1, 1]
    uhi: np.ndarray

    def __len__(self):
        return len(self.xlo)

    @staticmethod
    def root(sys: System):
        return Cells(sys.x0.lo[None].copy(), sys.x0.hi[None].copy(),
                     -np.ones((1, sys.p)), np.ones((1, sys.p)))

    def take(self, idx):
        return Cells(self.xlo[idx], self.xhi[idx], self.ulo[idx], self.uhi[idx])

    @staticmethod
    def cat(cs):
        cs = [c for c in cs if len(c)]
        return Cells(*(np.concatenate([getattr(c, a) for c in cs]) for a in ("xlo", "xhi", "ulo", "uhi")))

    def theta(self, sys: System, alpha):
        """Interval (p, C) of parameters for each cell at belief level alpha."""
        TH, _ = sys.bands(alpha)
        mid, rad = TH.mid[:, None], TH.rad[:, None]
        return Interval(mid + rad * self.ulo.T, mid + rad * self.uhi.T)

    def split(self, idx, sens):
        """Split selected cells in two along argmax(sens * width)."""
        lo = np.hstack([self.xlo, self.ulo])[idx]
        hi = np.hstack([self.xhi, self.uhi])[idx]
        return self.split_dims(idx, np.argmax((hi - lo) * sens[None, :], 1))

    def split_dims(self, idx, dim):
        """Split selected cells in two along the given dimensions."""
        n = self.xlo.shape[1]
        lo = np.hstack([self.xlo, self.ulo])[idx]
        hi = np.hstack([self.xhi, self.uhi])[idx]
        r = np.arange(len(idx))
        mid = 0.5 * (lo[r, dim] + hi[r, dim])
        lo2, hi1 = lo.copy(), hi.copy()
        hi1[r, dim] = mid
        lo2[r, dim] = mid
        L, H = np.vstack([lo, lo2]), np.vstack([hi1, hi])
        return Cells(L[:, :n], H[:, :n], L[:, n:], H[:, n:])


# ----------------------------------------------------------------- reference paths
def rk4_batch(sys: System, X, TH, W, dt, steps, sub=4):
    xs = [X.copy()]
    h = dt / sub
    for _ in range(steps):
        for _ in range(sub):
            k1 = sys.F(X, TH, W)
            k2 = sys.F(X + 0.5 * h * k1, TH, W)
            k3 = sys.F(X + 0.5 * h * k2, TH, W)
            k4 = sys.F(X + h * k3, TH, W)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(X.copy())
    return np.stack(xs)  # (K+1, n, C)


def sensitivity(sys: System, alpha=0.5):
    """Finite-difference trajectory sensitivity to each splittable input (x0 dims, theta)."""
    n, p = sys.n, sys.p
    c = 0.5 * (sys.x0.lo + sys.x0.hi)
    TH = sys.bands(alpha)[0]
    steps = int(round(sys.T / sys.dt))
    X, th = [], []
    for i in range(n + p):
        dx, du = np.zeros(n), np.zeros(p)
        if i < n:
            dx[i] = 1e-4 * (sys.x0.hi[i] - sys.x0.lo[i])
        else:
            du[i - n] = 1e-4
        X += [c + dx, c - dx]
        th += [TH.mid + TH.rad * du, TH.mid - TH.rad * du]
    X, th = np.array(X).T, np.array(th).T
    W = np.zeros((len(sys.w), X.shape[1])) + sys.w_hat[:, None]
    xs = rk4_batch(sys, X, th, W, sys.dt, steps, sub=1)
    scale = np.maximum(np.abs(xs).max((0, 2)), 1e-3)
    diff = (xs[:, :, 0::2] - xs[:, :, 1::2]) / scale[None, :, None]
    s = np.abs(diff).max((0, 1)) / 2e-4
    return s / max(s.max(), 1e-12)


class Model:
    """Engine view of a system at belief level alpha for a batch of cells.

    augment=True treats parameters as constant states (theta' = 0), which preserves
    their correlation over time; augment=False keeps them as interval bands.
    """

    def __init__(self, sys: System, alpha, cells: Cells, augment=True):
        self.sys, self.augment = sys, augment
        n, p = sys.n, sys.p
        self.TH = cells.theta(sys, alpha)
        self.W = sys.bands(alpha)[1]
        steps = int(round(sys.T / sys.dt))
        Xc = 0.5 * (cells.xlo + cells.xhi).T
        THc = self.TH.mid
        Wc = np.broadcast_to(sys.w_hat[:, None], (len(sys.w), len(cells))).copy()
        xs = rk4_batch(sys, Xc, THc, Wc, sys.dt, steps)
        if augment and p:
            self.N = n + p
            self.xs = np.concatenate([xs, np.broadcast_to(THc[None], (len(xs),) + THc.shape)], 1)
            self.x0lo = np.hstack([cells.xlo, self.TH.lo.T])
            self.x0hi = np.hstack([cells.xhi, self.TH.hi.T])
            self.unsafe = [U.extend(p) for U in sys.unsafe]
        else:
            self.augment = False
            self.N, self.xs = n, xs
            self.x0lo, self.x0hi = cells.xlo, cells.xhi
            self.unsafe = sys.unsafe

    def F(self, X: Interval):
        s, n = self.sys, self.sys.n
        if not self.augment:
            return s.F(X, self.TH, self.W)
        fx = s.F(X[:n], X[n:], self.W)
        z = np.zeros((s.p,) + fx.lo.shape[1:])
        return Interval(np.concatenate([fx.lo, z]), np.concatenate([fx.hi, z]))

    def J(self, X: Interval):
        s, n, p = self.sys, self.sys.n, self.sys.p
        if not self.augment:
            return s.J(X, self.TH, self.W)
        Jx, Jt = s.jacs(X[:n], X[n:], self.W)
        C = Jx.lo.shape[0]
        lo = np.zeros((C, n + p, n + p))
        hi = np.zeros((C, n + p, n + p))
        lo[:, :n, :n], hi[:, :n, :n] = Jx.lo, Jx.hi
        lo[:, :n, n:], hi[:, :n, n:] = Jt.lo, Jt.hi
        return Interval(lo, hi)


# ------------------------------------------------------------------- LMI metric
def lmi_metric(Js, cond=1e2, iters=18):
    """M = P^T P maximising uniform contraction: J^T M + M J <= 2 gamma M (all J)."""
    n = Js[0].shape[0]
    M = cp.Variable((n, n), symmetric=True)

    def feasible(g):
        cons = [M >> np.eye(n), M << cond * np.eye(n)]
        cons += [J.T @ M + M @ J - 2 * g * M << 0 for J in Js]
        prob = cp.Problem(cp.Minimize(0), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return None
        return M.value if prob.status in ("optimal", "optimal_inaccurate") else None

    lo = -max(np.abs(J).sum(1).max() for J in Js) - 1
    hi = max(np.max(np.linalg.eigvalsh(0.5 * (J + J.T))) for J in Js) + 1e-3
    best = feasible(hi)
    if best is None:
        return np.eye(n), hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Mv = feasible(mid)
        if Mv is None:
            lo = mid
        else:
            hi, best = mid, Mv
    best = 0.5 * (best + best.T)
    P = np.linalg.cholesky(best + 1e-12 * np.eye(n)).T
    return P / np.linalg.norm(P, 2), hi


def _box_overlap(lo, hi, U):
    return ~(np.any(hi < U.lo[None], 1) | np.any(lo > U.hi[None], 1))


def _ellseg_meets_box(P, xa, xb, rho, U):
    """Exact test: exists s in [0,1], x in U with ||P (x - xa - s (xb - xa))|| <= rho.

    Unbounded box coordinates are eliminated analytically: the projection of the
    ellipsoid {e : e^T M e <= rho^2} onto bounded coords B has matrix ((M^-1)_BB)^-1.
    """
    B = np.where(np.isfinite(U.lo) | np.isfinite(U.hi))[0]
    Minv = np.linalg.inv(P.T @ P)
    S = Minv[np.ix_(B, B)]
    PB = np.linalg.cholesky(np.linalg.inv(0.5 * (S + S.T))).T
    a, dlt = xa[B], (xb - xa)[B]
    A = np.hstack([PB, -(PB @ dlt)[:, None]])
    lb = np.r_[np.where(np.isfinite(U.lo[B]), U.lo[B], -1e6), 0.0]
    ub = np.r_[np.where(np.isfinite(U.hi[B]), U.hi[B], 1e6), 1.0]
    res = lsq_linear(A, PB @ a, bounds=(lb, ub), method="bvls", tol=1e-13)
    return np.linalg.norm(A @ res.x - PB @ a) <= rho * (1 + 1e-9)


def _apriori(m: Model, elo, ehi, seg, dv, dt):
    """Box E with  box(e_k) + [0,dt]([J](seg+E) E + dv)  subset of E (Picard enclosure).

    Returns ((lo, hi), Jiv, (inc_lo, inc_hi)) or (None, None, None) if no enclosure.
    """
    tiny = 1e-12 + 1e-9 * (np.abs(elo) + np.abs(ehi))
    lo, hi = elo - tiny, ehi + tiny
    for _ in range(40):
        try:
            Jiv = m.J(Interval(seg.lo + lo.T, seg.hi + hi.T))
        except ZeroDivisionError:
            break
        E = Interval(lo[:, None, :], hi[:, None, :])
        JE = Jiv * E
        JE = Interval(JE.lo.sum(2), JE.hi.sum(2))
        inc_lo = np.minimum(0, dt * (JE.lo - dv))
        inc_hi = np.maximum(0, dt * (JE.hi + dv))
        nlo, nhi = elo + inc_lo, ehi + inc_hi
        if np.all(nlo >= lo) and np.all(nhi <= hi):
            return (lo, hi), Jiv, (inc_lo, inc_hi)
        # inflate only the increment over e_k (inflating the whole width feeds back)
        lo = elo + 1.5 * np.minimum(lo - elo, inc_lo) - tiny
        hi = ehi + 1.5 * np.maximum(hi - ehi, inc_hi) + tiny
        if not np.all(np.isfinite(hi - lo)) or np.max(hi - lo) > 1e4:
            break
    return None, None, None


@dataclass
class Step:
    """Per-step reference data and the defect integral eta(s) = int_0^s delta (batched).

    delta(u) = f(xhat(u), theta, w(u)) - slope on the piecewise-linear reference.
    Mean-value form about the segment midpoint xm (J~ in [J](seg) = Jm +- Rs):
        eta(s)  in s C0 + Jm D (s^2/(2dt) - s/2) +- Rs |D| dt/4,  C0 = f(xm, W) - slope
        eta(dt) in dt C0 +- Rs |D| dt/4            (O(dt^3) besides the disturbance)
    """
    xa: np.ndarray
    xb: np.ndarray
    slo: np.ndarray
    shi: np.ndarray
    seg: Interval
    dv: np.ndarray        # crude |delta| bound, used only inside one step (a priori box)
    eta_mid: np.ndarray   # centre of eta(dt)
    eta_rad: np.ndarray   # radius of eta(dt)
    eta_max: np.ndarray   # sup_s |eta(s)|


def _step_data(m: Model, k, dt):
    xa, xb = m.xs[k].T, m.xs[k + 1].T  # (C, N)
    slo, shi = np.minimum(xa, xb), np.maximum(xa, xb)
    seg = Interval(slo.T, shi.T)
    slope = (xb - xa) / dt
    dv = (m.F(seg) - slope.T).mag.T
    xm = 0.5 * (xa + xb)
    C0 = m.F(Interval(xm.T, xm.T)) - slope.T
    c0m, c0r = C0.mid.T, C0.rad.T
    Js = m.J(seg)
    Dl = xb - xa
    JmD = np.abs(np.einsum("cij,cj->ci", Js.mid, Dl))
    RD = np.einsum("cij,cj->ci", Js.rad, np.abs(Dl))
    eta_mid = dt * c0m
    eta_rad = dt * c0r + RD * dt / 4
    eta_max = dt * (np.abs(c0m) + c0r) + JmD * dt / 8 + RD * dt / 4
    return Step(xa, xb, slo, shi, seg, dv, eta_mid, eta_rad, eta_max)


# ------------------------------------------------------------------- engines
class Engine:
    name = "engine"
    augment = True

    def __init__(self, sys: System):
        self.sys = sys

    def model(self, alpha, cells):
        return Model(self.sys, alpha, cells, augment=self.augment)

    def run(self, alpha, cells: Cells, record=False):
        raise NotImplementedError


def _perturbation(Jiv, Erad, st: Step, dt):
    """Box beta enclosing int_0^dt Phi(dt-s) v(s) ds, v = (J - Jc) e~ + J eta."""
    Jc, R = Jiv.mid, Jiv.rad
    et = Erad + st.eta_max  # |e~| = |e - eta|
    b = np.einsum("cij,cj->ci", R, et) + np.einsum("cij,cj->ci", Jiv.mag, st.eta_max)
    return dt * np.einsum("cij,cj->ci", expm(np.abs(Jc) * dt), b)


def _split_score(Pacc, h0, B):
    """Directional stretching of each initial input toward the unsafe coordinates B:
    score_i = || Pacc[B, i] || h0_i  (local finite-time Lyapunov / negative curvature)."""
    return np.linalg.norm(Pacc[B], axis=0) * h0


def _outer_sum(Q, S):
    """Trace-optimal outer ellipsoid of E(Q) (+) E(S), batched. Sound for any p > 0."""
    tq = np.trace(Q, axis1=1, axis2=2)
    ts = np.trace(S, axis1=1, axis2=2)
    p = np.sqrt(np.maximum(ts, 1e-300) / np.maximum(tq, 1e-300))[:, None, None]
    out = (1 + p) * Q + (1 + 1 / p) * S
    return np.where((ts > 0)[:, None, None], out, Q)


def _box_ellipsoid(beta):
    """Shape matrix of an ellipsoid containing the box [-beta, beta] (batched)."""
    m = np.maximum((beta > 0).sum(1), 1)[:, None]
    C, n = beta.shape
    S = np.zeros((C, n, n))
    i = np.arange(n)
    S[:, i, i] = m * beta ** 2
    return S


class FlowEngine(Engine):
    """UTR-flow: ellipsoidal metric-flow tube.

    The set is c + unit ball of the evolving metric g_k = Q_k^-1. Per step
    (Phi = expm(Jc dt)):  c <- Phi c + eta_mid,
    Q <- outer_sum(Phi Q Phi^T, box(beta + eta_rad)):  Lie transport of the metric
    followed by a conformal / curvature correction. Every step is an enclosure.
    """

    name = "UTR-flow"

    def run(self, alpha, cells: Cells, record=False):
        m, dt = self.model(alpha, cells), self.sys.dt
        C, N = len(cells), m.N
        scale = np.maximum(np.abs(m.xs).max((0, 2)), 1e-3)
        h = np.maximum(0.5 * (m.x0hi - m.x0lo), 1e-7 * scale[None, :])
        idx = np.arange(N)
        Q = np.zeros((C, N, N))
        Q[:, idx, idx] = N * h ** 2
        ce = np.zeros((C, N))
        hit = np.zeros(C, bool)
        hit_step = np.full(C, -1)
        alive = np.ones(C, bool)
        rec = {"Q": [], "c": []} if record else None
        Pacc = np.broadcast_to(np.eye(N), (C, N, N)).copy()
        h0 = 0.5 * (m.x0hi - m.x0lo)
        score = np.zeros((C, N))
        for k in range(len(m.xs) - 1):
            st = _step_data(m, k, dt)
            hE = np.sqrt(np.maximum(np.diagonal(Q, axis1=1, axis2=2), 0))
            E, Jiv, inc = _apriori(m, ce - hE, ce + hE, st.seg, st.dv, dt)
            if E is None:
                hit[alive], hit_step[alive] = True, k
                break
            Erad = np.maximum(np.abs(E[0]), np.abs(E[1]))
            Drad = np.maximum(np.abs(inc[0]), np.abs(inc[1]))
            Qsw = _outer_sum(Q, _box_ellipsoid(Drad))  # swept set over the step
            for U in m.unsafe:
                half = np.sqrt(np.diagonal(Qsw, axis1=1, axis2=2))
                lo, hi = st.slo + ce - half, st.shi + ce + half
                cand = np.where(alive & _box_overlap(lo, hi, U))[0]
                for c in cand:
                    Pc = np.linalg.cholesky(np.linalg.inv(Qsw[c])).T
                    if _ellseg_meets_box(Pc, st.xa[c] + ce[c], st.xb[c] + ce[c], 1.0, U):
                        hit[c], hit_step[c], alive[c] = True, k, False
                        B = np.where(np.isfinite(U.lo) | np.isfinite(U.hi))[0]
                        score[c] = _split_score(Pacc[c], h0[c], B)
            if record:
                rec["Q"].append(Qsw.copy())
                rec["c"].append(ce.copy())
            if not alive.any() and not record:
                break
            Phi = expm(Jiv.mid * dt)
            Pacc = Phi @ Pacc
            beta = _perturbation(Jiv, Erad, st, dt) + st.eta_rad
            ce = np.einsum("cij,cj->ci", Phi, ce) + st.eta_mid
            Q = _outer_sum(Phi @ Q @ np.swapaxes(Phi, 1, 2), _box_ellipsoid(beta))
            Q = 0.5 * (Q + np.swapaxes(Q, 1, 2))
        out = {"hit": hit, "hit_step": hit_step, "score": score}
        if record:
            out.update(xs=m.xs, Q=np.array(rec["Q"]), c=np.array(rec["c"]))
        return out


class TubeEngine(Engine):
    """UTR-Ricci: radius in a fixed contraction (Ricci lower bound) metric P.

    With e = e~ + eta:  ||P e~||' <= mu ||P e~|| + || |P| |J| eta_max ||,
    mu = lam_max(sym(P Jc P^-1)) + min(|| |P| R |P^-1| ||, sum R_ij |P_:i| |P^-1_j:|)
    (kappa = -mu is the local Ricci / contraction bound), and r_{k+1} = r~(dt) + ||P eta(dt)||.
    Parameters stay as interval bands (a contraction argument needs no augmentation).
    """

    augment = False

    def __init__(self, sys: System, cond=1e2):
        super().__init__(sys)
        self.name = "UTR-Ricci"
        xs = rk4_batch(sys, (0.5 * (sys.x0.lo + sys.x0.hi))[:, None], sys.th_hat[:, None],
                       sys.w_hat[:, None], sys.dt, int(round(sys.T / sys.dt)))[:, :, 0]
        idx = np.linspace(0, len(xs) - 1, 25).astype(int)
        Js = [sys.J(xs[i], sys.th_hat, sys.w_hat) for i in idx]
        self.P0, self.gamma = lmi_metric(Js, cond=cond)

    def _r0(self, P, lo, hi):
        h = 0.5 * (hi - lo)
        n = h.shape[1]
        if n <= 12:
            V = np.array(list(itertools.product([-1, 1], repeat=n)))
            return np.linalg.norm((V[None] * h[:, None, :]) @ P.T, axis=2).max(1)
        return np.linalg.norm(P, 2) * np.linalg.norm(h, axis=1)

    def run(self, alpha, cells: Cells, record=False, r_cap=1e3):
        m, dt = self.model(alpha, cells), self.sys.dt
        C, n = len(cells), m.N
        P = self.P0
        Pinv = np.linalg.inv(P)
        prow, pcol = np.linalg.norm(Pinv, axis=1), np.linalg.norm(P, axis=0)
        aP, aPi = np.abs(P), np.abs(Pinv)
        r = self._r0(P, m.x0lo, m.x0hi)
        hit = np.zeros(C, bool)
        hit_step = np.full(C, -1)
        alive = np.ones(C, bool)
        rec = {"rho": [], "mu": []} if record else None
        for k in range(len(m.xs) - 1):
            st = _step_data(m, k, dt)
            pe_max = np.linalg.norm(st.eta_max @ aP.T, axis=1)
            pe_end = np.linalg.norm((np.abs(st.eta_mid) + st.eta_rad) @ aP.T, axis=1)
            rbar = r * 1.02 + 1e-12
            done = ~alive.copy()
            mu_o, r_o, rmax_o = np.zeros(C), r.copy(), r.copy()
            for _ in range(40):
                half = (rbar + pe_max)[:, None] * prow[None]
                Jiv = m.J(Interval((st.slo - half).T, (st.shi + half).T))
                Jc, R = Jiv.mid, Jiv.rad
                PJP = P[None] @ Jc @ Pinv[None]
                prp1 = np.linalg.norm(aP[None] @ R @ aPi[None], 2, axis=(1, 2))
                prp2 = np.einsum("i,cij,j->c", pcol, R, prow)
                mu = np.linalg.eigvalsh(0.5 * (PJP + np.swapaxes(PJP, 1, 2)))[:, -1] + np.minimum(prp1, prp2)
                d = np.linalg.norm(np.einsum("cij,cj->ci", Jiv.mag, st.eta_max) @ aP.T, axis=1)
                z = np.minimum(mu * dt, 50)
                phi = np.where(np.abs(z) < 1e-12, dt, np.expm1(z) / np.where(mu == 0, 1, mu))
                rt = r * np.exp(z) + d * phi
                rmax = np.maximum(r, rt)
                ok = (rmax <= rbar) & ~done
                mu_o[ok], r_o[ok], rmax_o[ok] = mu[ok], rt[ok] + pe_end[ok], rmax[ok]
                done |= ok
                if done.all():
                    break
                nd = ~done
                rbar[nd] = np.maximum(rbar[nd] * 1.25, rmax[nd] * 1.05)
                over = nd & (rbar > r_cap)
                if over.any():
                    done |= over
                    hit[over & alive], hit_step[over & alive] = True, k
                    alive &= ~over
            rho = rmax_o + pe_max
            for U in m.unsafe:
                half = rho[:, None] * prow[None]
                cand = np.where(alive & _box_overlap(st.slo - half, st.shi + half, U))[0]
                for c in cand:
                    if _ellseg_meets_box(P, st.xa[c], st.xb[c], rho[c], U):
                        hit[c], hit_step[c], alive[c] = True, k, False
            if record:
                rec["rho"].append(rho.copy())
                rec["mu"].append(mu_o.copy())
            if not alive.any() and not record:
                break
            r = np.where(alive, r_o, 0.0)
        out = {"hit": hit, "hit_step": hit_step}
        if record:
            out.update(xs=m.xs, rho=np.array(rec["rho"]), mu=np.array(rec["mu"]), P=P)
        return out


class BoxEngine(Engine):
    """Classic alpha-cut interval reachability (interval Euler with a priori enclosure)."""

    name = "alpha-cut interval"

    def run(self, alpha, cells: Cells, record=False):
        m, dt = self.model(alpha, cells), self.sys.dt
        C = len(cells)
        c0 = m.xs[0].T
        elo, ehi = m.x0lo - c0, m.x0hi - c0
        hit = np.zeros(C, bool)
        hit_step = np.full(C, -1)
        alive = np.ones(C, bool)
        for k in range(len(m.xs) - 1):
            st = _step_data(m, k, dt)
            E, Jiv, inc = _apriori(m, elo, ehi, st.seg, st.dv, dt)
            if E is None:
                hit[alive], hit_step[alive] = True, k
                break
            for U in m.unsafe:
                mk = alive & _box_overlap(st.slo + E[0], st.shi + E[1], U)
                hit[mk], hit_step[mk] = True, k
                alive &= ~mk
            if not alive.any():
                break
            # e_{k+1} in e_k + dt [J] (E + eta) + eta(dt)
            Et = Interval((E[0] - st.eta_max)[:, None, :], (E[1] + st.eta_max)[:, None, :])
            JE = Jiv * Et
            elo = elo + dt * JE.lo.sum(2) + st.eta_mid - st.eta_rad
            ehi = ehi + dt * JE.hi.sum(2) + st.eta_mid + st.eta_rad
        return {"hit": hit, "hit_step": hit_step}


class ZonoEngine(Engine):
    """First-order zonotope reachability, exact linear part, interval remainder."""

    name = "zonotope"

    def __init__(self, sys: System, order=8):
        super().__init__(sys)
        self.order = order

    def _reduce(self, G, max_gen):
        C, n, mg = G.shape
        if mg <= max_gen:
            return G
        score = np.abs(G).sum(1) - np.abs(G).max(1)
        order = np.argsort(score, 1)
        nb = mg - max_gen + n
        small = np.take_along_axis(G, order[:, None, :nb], 2)
        keep = np.take_along_axis(G, order[:, None, nb:], 2)
        boxg = np.zeros((C, n, n))
        i = np.arange(n)
        boxg[:, i, i] = np.abs(small).sum(2)
        return np.concatenate([keep, boxg], 2)

    def run(self, alpha, cells: Cells, record=False):
        m, dt = self.model(alpha, cells), self.sys.dt
        C, n = len(cells), m.N
        idx = np.arange(n)
        G = np.zeros((C, n, n))
        G[:, idx, idx] = 0.5 * (m.x0hi - m.x0lo)
        cz = np.zeros((C, n))
        hit = np.zeros(C, bool)
        hit_step = np.full(C, -1)
        alive = np.ones(C, bool)
        rec = {"G": [], "c": [], "D": []} if record else None
        Pacc = np.broadcast_to(np.eye(n), (C, n, n)).copy()
        h0 = 0.5 * (m.x0hi - m.x0lo)
        score = np.zeros((C, n))
        for k in range(len(m.xs) - 1):
            st = _step_data(m, k, dt)
            gb = np.abs(G).sum(2)
            E, Jiv, inc = _apriori(m, cz - gb, cz + gb, st.seg, st.dv, dt)
            if E is None:
                hit[alive], hit_step[alive] = True, k
                break
            Erad = np.maximum(np.abs(E[0]), np.abs(E[1]))
            Drad = np.maximum(np.abs(inc[0]), np.abs(inc[1]))
            if record:
                rec["G"].append(G.copy()); rec["c"].append(cz.copy()); rec["D"].append(Drad.copy())
            for U in m.unsafe:
                gbox = gb + Drad
                cand = np.where(alive & _box_overlap(st.slo + cz - gbox, st.shi + cz + gbox, U))[0]
                B = np.where(np.isfinite(U.lo) | np.isfinite(U.hi))[0]
                for c in cand:
                    dl = st.xb[c] - st.xa[c]
                    Gc = np.hstack([G[c], np.diag(Drad[c]), 0.5 * dl[:, None]])[B]
                    ctr = (st.xa[c] + cz[c] + 0.5 * dl)[B]
                    A_ub, b_ub = [], []
                    for j, bj in enumerate(B):
                        if np.isfinite(U.hi[bj]):
                            A_ub.append(Gc[j]); b_ub.append(U.hi[bj] - ctr[j])
                        if np.isfinite(U.lo[bj]):
                            A_ub.append(-Gc[j]); b_ub.append(ctr[j] - U.lo[bj])
                    res = linprog(np.zeros(Gc.shape[1]), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                                  bounds=[(-1, 1)] * Gc.shape[1], method="highs")
                    if res.status == 0:
                        hit[c], hit_step[c], alive[c] = True, k, False
                        score[c] = _split_score(Pacc[c], h0[c], B)
            if not alive.any() and not record:
                break
            Phi = expm(Jiv.mid * dt)
            Pacc = Phi @ Pacc
            cz = np.einsum("cij,cj->ci", Phi, cz) + st.eta_mid
            G = Phi @ G
            extra = _perturbation(Jiv, Erad, st, dt) + st.eta_rad
            boxg = np.zeros((C, n, n))
            boxg[:, idx, idx] = extra
            G = self._reduce(np.concatenate([G, boxg], 2), self.order * n)
        out = {"hit": hit, "hit_step": hit_step, "score": score}
        if record:
            out.update(xs=m.xs, G=rec["G"], c=np.array(rec["c"]), D=np.array(rec["D"]))
        return out


# ------------------------------------------------------------ refinement / alpha*
def verify(engine: Engine, alpha, max_cells=256, sens=None, batch=256, split="curvature", time_cap=1200.0):
    """Partition refinement. Returns (safe, cells_used, wall_time).

    split: 'curvature'   per-cell directional stretching toward the unsafe set (engines
                         that report it; falls back to 'sensitivity'),
           'sensitivity' global finite-difference sensitivity x width,
           'width'       largest relative width (naive bisection).
    """
    t0 = time.perf_counter()
    sys = engine.sys
    sens = sensitivity(sys) if sens is None else sens
    root = Cells.root(sys)
    rel = 1.0 / np.maximum(np.r_[root.xhi[0] - root.xlo[0], root.uhi[0] - root.ulo[0]], 1e-12)
    todo = root
    certified = 0
    while len(todo):
        if certified + len(todo) > max_cells or time.perf_counter() - t0 > time_cap:
            return False, certified + len(todo), time.perf_counter() - t0
        outs = [engine.run(alpha, todo.take(np.arange(i, min(i + batch, len(todo)))))
                for i in range(0, len(todo), batch)]
        hits = np.concatenate([o["hit"] for o in outs])
        certified += int((~hits).sum())
        bad = np.where(hits)[0]
        if not len(bad):
            break
        if split == "curvature" and "score" in outs[0]:
            sc = np.concatenate([o["score"] for o in outs])[bad]
            good = sc.sum(1) > 0
            dims = np.where(good, np.argmax(sc, 1),
                            np.argmax(np.hstack([todo.xhi - todo.xlo, todo.uhi - todo.ulo])[bad] * sens, 1))
            todo = todo.split_dims(bad, dims)
        elif split == "width":
            todo = todo.split(bad, rel)
        else:
            todo = todo.split(bad, sens)
    return True, certified, time.perf_counter() - t0


def alpha_star(engine: Engine, tol=5e-3, max_cells=256, log=None, split="curvature"):
    """Largest alpha certified SAFE: a sound lower bound on the true alpha*."""
    sens = sensitivity(engine.sys)
    stats = []

    def safe(a):
        ok, used, t = verify(engine, a, max_cells=max_cells, sens=sens, split=split)
        stats.append((a, ok, used, t))
        if log:
            log(f"    {engine.name} alpha={a:.4f} safe={ok} cells={used} {t:.1f}s")
        return ok

    lo, hi = 1e-4, ALPHA_MAX
    if not safe(lo):
        return 0.0, stats
    if not engine.run(hi, Cells.root(engine.sys))["hit"][0]:  # cheap probe: single cell
        return 1.0, stats
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if safe(mid):
            lo = mid
        else:
            hi = mid
    return lo, stats
