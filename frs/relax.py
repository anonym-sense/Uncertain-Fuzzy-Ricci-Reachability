"""Topography relaxation (Theorem 2): sound, curvature-driven relaxation of a hull.

Everything here operates on a 2D raster of a state projection.

H_0 = convex hull of the certified tube T (what the 2020 paper bounds and morphs).
Constrained flow (level set phi < 0 inside):
    phi_tau = (1 - lam(x) kappa) |grad phi|          inward speed 1 - lam kappa
    phi <- min(phi, s_T)   (T stays inside: soundness)
    phi <- max(phi, s_H0)  (never grows beyond the hull)
This is the L2 gradient flow of the energy  |H| + int_{dH} lam ds  with obstacle T
(and ceiling H0). Free boundary arcs at equilibrium are convex with curvature 1/lam;
minimisers are nested in lam, equal T as lam -> 0 and H0 (convex hull) as lam -> inf.
The *2020 flow* is the unconstrained curvature flow phi_tau = w kappa |grad phi|,
which shrinks the hull to a point and drops reachable states (unsound).
closing() is kept as a fast morphological alternative (not the flow's limit).
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import ConvexHull


class Grid:
    def __init__(self, xlim, ylim, nx=400, ny=400):
        self.x = np.linspace(*xlim, nx)
        self.y = np.linspace(*ylim, ny)
        self.h = (self.x[1] - self.x[0], self.y[1] - self.y[0])
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.pts = np.c_[self.X.ravel(), self.Y.ravel()]
        self.mask = np.zeros(self.X.shape, bool)

    def ij(self, x, y):
        return (int(np.clip(np.rint((y - self.y[0]) / self.h[1]), 0, len(self.y) - 1)),
                int(np.clip(np.rint((x - self.x[0]) / self.h[0]), 0, len(self.x) - 1)))

    def fill_polygon(self, V, mask=None):
        """Mark all pixels whose centre lies in polygon V (k,2), plus touched pixels."""
        mask = self.mask if mask is None else mask
        lo, hi = V.min(0), V.max(0)
        i0, j0 = self.ij(lo[0] - self.h[0], lo[1] - self.h[1])
        i1, j1 = self.ij(hi[0] + self.h[0], hi[1] + self.h[1])
        sub = (slice(i0, i1 + 1), slice(j0, j1 + 1))
        P = np.c_[self.X[sub].ravel(), self.Y[sub].ravel()]
        # inflate by half a pixel diagonal so that the raster over-approximates the set
        r = 0.5 * np.hypot(*self.h)
        inside = Path(V).contains_points(P, radius=-r) | Path(V).contains_points(P, radius=r)
        mask[sub] |= inside.reshape(self.X[sub].shape)
        for v in V:  # thin sets: make sure vertices are marked
            mask[self.ij(*v)] = True

    def area(self, mask):
        return mask.sum() * self.h[0] * self.h[1]


def zonotope_polygon(c, G):
    """Vertices of the 2D zonotope c + G [-1,1]^m (counter-clockwise)."""
    G = G[:, np.linalg.norm(G, axis=0) > 1e-14]
    if G.shape[1] == 0:
        return c[None]
    G = G * np.where(G[1] < 0, -1, 1) + 0.0
    G = G * np.where((G[1] == 0) & (G[0] < 0), -1, 1)
    ang = np.arctan2(G[1], G[0])
    G = G[:, np.argsort(ang)]
    start = c - G.sum(1)
    up = start + np.cumsum(2 * G, axis=1).T
    down = up[-1] - np.cumsum(2 * G, axis=1).T
    return np.vstack([start, up, down[:-1]])


def ellipse_polygon(c, S, k=48):
    """Polygon circumscribing the ellipse {c + y : y^T S^-1 y <= 1}."""
    L = np.linalg.cholesky(S + 1e-15 * np.eye(2))
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    circ = np.c_[np.cos(t), np.sin(t)] / np.cos(np.pi / k)  # circumscribed polygon
    return c + circ @ L.T


def sdf(mask, h):
    """Signed distance (negative inside) of a raster set."""
    out = distance_transform_edt(~mask, sampling=(h[1], h[0]))
    inn = distance_transform_edt(mask, sampling=(h[1], h[0]))
    return out - inn


def convex_hull_mask(grid: Grid, mask):
    pts = grid.pts[mask.ravel()]
    if len(pts) < 3:
        return mask.copy()
    hull = ConvexHull(pts)
    m = np.zeros_like(mask)
    grid.fill_polygon(pts[hull.vertices], m)
    return m | mask


def closing(mask, rho, h):
    """Morphological closing by a disc of radius rho (exact via distance transforms)."""
    if rho <= 0:
        return mask.copy()
    d = distance_transform_edt(~mask, sampling=(h[1], h[0]))
    dil = d <= rho
    d2 = distance_transform_edt(dil, sampling=(h[1], h[0]))
    return (d2 > rho) | mask


def _curvature(phi, h):
    """Mean-curvature term with a compact stencil (damps odd-even modes).

    Returns (kappa, |grad phi|) with kappa |grad phi| =
    (phi_xx phi_y^2 - 2 phi_x phi_y phi_xy + phi_yy phi_x^2) / |grad phi|^2.
    """
    hx, hy = h
    pe, pw = np.roll(phi, -1, 1), np.roll(phi, 1, 1)
    pn, ps = np.roll(phi, -1, 0), np.roll(phi, 1, 0)
    px = (pe - pw) / (2 * hx)
    py = (pn - ps) / (2 * hy)
    pxx = (pe - 2 * phi + pw) / hx ** 2
    pyy = (pn - 2 * phi + ps) / hy ** 2
    pxy = (np.roll(pn, -1, 1) - np.roll(pn, 1, 1) - np.roll(ps, -1, 1) + np.roll(ps, 1, 1)) / (4 * hx * hy)
    g2 = px ** 2 + py ** 2 + 1e-12
    nrm = np.sqrt(g2)
    kap = (pxx * py ** 2 - 2 * px * py * pxy + pyy * px ** 2) / (g2 * nrm)
    kmax = 1.0 / min(hx, hy)  # curvature is unresolvable beyond the grid scale
    return np.clip(kap, -kmax, kmax), nrm


def _grad_upwind(phi, h, speed):
    """Godunov |grad phi| for phi_t = speed |grad phi| (speed > 0 moves boundary inward)."""
    dxm = (phi - np.roll(phi, 1, 1)) / h[0]
    dxp = (np.roll(phi, -1, 1) - phi) / h[0]
    dym = (phi - np.roll(phi, 1, 0)) / h[1]
    dyp = (np.roll(phi, -1, 0) - phi) / h[1]
    gp = np.sqrt(np.maximum(dxm, 0) ** 2 + np.minimum(dxp, 0) ** 2 +
                 np.maximum(dym, 0) ** 2 + np.minimum(dyp, 0) ** 2)
    gm = np.sqrt(np.minimum(dxm, 0) ** 2 + np.maximum(dxp, 0) ** 2 +
                 np.minimum(dym, 0) ** 2 + np.maximum(dyp, 0) ** 2)
    return np.where(speed > 0, gm, gp)


def relax(grid: Grid, tube, hull, lam, iters=2000, reinit=50, constrained=True, w=None, log_at=(), init=None):
    """Run the (constrained) topography relaxation. Returns final mask and snapshots.

    lam: scalar or field (curvature radius). constrained=False with lam=None gives the
    2020-paper flow phi_t = w kappa |grad phi| (pure curvature flow, weights w).
    """
    h = grid.h
    sT, sH = sdf(tube, h), sdf(hull, h)
    phi = sH.copy() if init is None else np.maximum(np.minimum(sdf(init | tube, h), sT), sH)
    hmin = min(h)
    lam_f = np.broadcast_to(np.asarray(lam if lam is not None else 0.0, float), phi.shape)
    wf = np.broadcast_to(np.asarray(1.0 if w is None else w, float), phi.shape)
    lmax = max(float(lam_f.max()), float(wf.max()) if lam is None else 0.0, 1e-9)
    dt = 0.2 * hmin ** 2 / lmax if lmax > 0 else 0.5 * hmin
    dt = min(dt, 0.5 * hmin)
    snaps = {}
    for it in range(1, iters + 1):
        kap, nrm = _curvature(phi, h)
        if lam is None:  # 2020 flow: pure (weighted) curvature motion
            phi = phi + dt * wf * kap * nrm
        else:
            g = _grad_upwind(phi, h, np.ones_like(phi))
            phi = phi + dt * (g - lam_f * kap * nrm)
        if constrained:
            phi = np.maximum(np.minimum(phi, sT), sH)
        if it % reinit == 0:
            inside = phi < 0
            if constrained:  # components not touching T only add energy: drop them
                lab, _ = label(inside)
                keep = np.unique(lab[tube & inside])
                inside = np.isin(lab, keep[keep > 0])
            phi = sdf(inside, h)
            if constrained:
                phi = np.maximum(np.minimum(phi, sT), sH)
        if it in log_at:
            snaps[it] = phi < 0
    return phi < 0, snaps
