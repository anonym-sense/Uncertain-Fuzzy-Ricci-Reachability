"""UTR-Z2: second-order zonotope engine (Hessian Lagrange remainder on the zonotope).

Per step, with xhat(s) = xa + (s/dt) D on [0, dt], midpoint xm, Jm = J(xm) (point),
Phi = expm(Jm dt), Psi = int_0^dt expm(Jm u) du, e = x - xhat, e~ = e - eta:

  e~' = Jm e~ + Jm eta + B(s) e + r2(s),
        B(s) = J(xhat(s)) - Jm = H(zeta)[D] (s/dt - 1/2)        (zero-mean along the step)
        r2(s) = 1/2 e^T H(xi) e                                   (Lagrange remainder)

  e_{k+1} in Phi Z_k + Psi q_mid + eta_mid
            (+) box( |T1| + |T2| + exp(|Jm|dt) dt (q_rad + rho2) + eta_rad )

  T1 = int Phi Jm eta          |T1| <= exp(|Jm|dt) |Jm| eta_max dt
  T2 = int Phi B e             with B = Bc + Br (Hessian centre / radius over the segment):
       Bc: the e_k part integrates to 0; |rest| <= |Mc||D_e| dt/2 + (exp(|Jm|dt)-I)|Mc||E| dt/2
       Br: <= exp(|Jm|dt) Mr |E| dt/4
  q    = quadratic map of the zonotope Z_k with the centre Hessian over the a-priori box,
  rho2 = Hessian-radius part + cross terms with the within-step increment D_e.
All bounds are enclosures; see tests/test_soundness2.py.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog

from .ad2 import hessian
from .engine import Cells, Engine, Model, _apriori, _box_overlap, _split_score, _step_data
from .interval import Interval


def _hess(m: Model, Z: Interval):
    """Interval (J, H) of the augmented field over the box Z (Interval (N, C))."""
    s, n, p = m.sys, m.sys.n, m.sys.p

    def fn(zl):
        out = list(s.f(zl[:n], zl[n:], m.W))
        return out + [0.0 * zl[0]] * p

    _, J, H = hessian(fn, [Z[i] for i in range(n + p)])
    return J, H


def _jac_point(m: Model, X):
    """Point Jacobian of the augmented field at X (N, C) floats -> (C, N, N)."""
    s, n, p = m.sys, m.sys.n, m.sys.p
    W = m.W.mid if len(m.W.lo) else m.W

    def fn(zl):
        return list(s.f(zl[:n], zl[n:], W)) + [0.0 * zl[0]] * p

    _, J, _ = hessian(fn, [X[i] for i in range(n + p)])
    return J


def _psi(Jm, dt):
    """Psi = int_0^dt expm(Jm u) du via the augmented-matrix exponential (batched)."""
    C, N, _ = Jm.shape
    A = np.zeros((C, 2 * N, 2 * N))
    A[:, :N, :N] = Jm * dt
    A[:, :N, N:] = np.eye(N) * dt
    E = expm(A)
    return E[:, :N, :N], E[:, :N, N:]


def _quadmap(cz, G, Hc):
    """Range of 1/2 z^T Hc_i z over z = c + G beta, beta in [-1,1]^m (per output i)."""
    A = np.einsum("cjm,cijk,ckl->ciml", G, Hc, G)            # (C, N, m, m)
    b = np.einsum("cjm,cijk,ck->cim", G, Hc, cz)             # (C, N, m)
    c0 = np.einsum("cj,cijk,ck->ci", cz, Hc, cz)             # (C, N)
    diag = np.diagonal(A, axis1=2, axis2=3)                  # (C, N, m)
    off = np.abs(A).sum((2, 3)) - np.abs(diag).sum(2)
    lo = c0 + np.minimum(diag, 0).sum(2) - off - 2 * np.abs(b).sum(2)
    hi = c0 + np.maximum(diag, 0).sum(2) + off + 2 * np.abs(b).sum(2)
    return 0.5 * lo, 0.5 * hi


class Zono2Engine(Engine):
    name = "UTR-Z2"

    def __init__(self, sys, order=8):
        super().__init__(sys)
        self.order = order

    _reduce = staticmethod(lambda G, mg: Zono2Engine._red(G, mg))

    @staticmethod
    def _red(G, max_gen):
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
        I = np.eye(n)
        G = np.zeros((C, n, n))
        G[:, idx, idx] = 0.5 * (m.x0hi - m.x0lo)
        cz = np.zeros((C, n))
        hit = np.zeros(C, bool)
        hit_step = np.full(C, -1)
        alive = np.ones(C, bool)
        Pacc = np.broadcast_to(I, (C, n, n)).copy()
        h0 = 0.5 * (m.x0hi - m.x0lo)
        score = np.zeros((C, n))
        rec = {"G": [], "c": [], "D": []} if record else None
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

            Dl = st.xb - st.xa
            xm = 0.5 * (st.xa + st.xb)
            Jm = _jac_point(m, xm.T)                                     # (C, n, n)
            Phi, Psi = _psi(Jm, dt)
            eJ = expm(np.abs(Jm) * dt)
            # Hessian over the segment (for B(s)) and over the a-priori box (for r2)
            _, Hs = _hess(m, st.seg)
            _, Hb = _hess(m, Interval(st.seg.lo + E[0].T, st.seg.hi + E[1].T))
            Mc = np.abs(np.einsum("cijk,ck->cij", Hs.mid, Dl))
            Mr = np.einsum("cijk,ck->cij", Hs.rad, np.abs(Dl))
            mv = lambda A, x: np.einsum("cij,cj->ci", A, x)  # noqa: E731
            T1 = dt * mv(eJ, mv(np.abs(Jm), st.eta_max))
            T2 = 0.5 * dt * mv(Mc, Drad) + 0.5 * dt * mv(eJ - I, mv(Mc, Erad)) + 0.25 * dt * mv(eJ, mv(Mr, Erad))
            # quadratic remainder: centre Hessian on the zonotope, radius + cross terms on boxes
            ql, qh = _quadmap(cz, G, Hb.mid)
            ebox = gb + np.abs(cz)
            Hmag = Hb.mag
            rho2 = 0.5 * np.einsum("cj,cijk,ck->ci", ebox, Hb.rad, ebox) \
                + np.einsum("cj,cijk,ck->ci", ebox, Hmag, Drad) \
                + 0.5 * np.einsum("cj,cijk,ck->ci", Drad, Hmag, Drad)
            qm, qr = 0.5 * (ql + qh), 0.5 * (qh - ql)
            box = T1 + T2 + dt * mv(eJ, qr + rho2) + st.eta_rad
            Pacc = Phi @ Pacc
            cz = mv(Phi, cz) + mv(Psi, qm) + st.eta_mid
            G = Phi @ G
            boxg = np.zeros((C, n, n))
            boxg[:, idx, idx] = box
            G = self._red(np.concatenate([G, boxg], 2), self.order * n)
        out = {"hit": hit, "hit_step": hit_step, "score": score}
        if record:
            out.update(xs=m.xs, G=rec["G"], c=np.array(rec["c"]), D=np.array(rec["D"]))
        return out
