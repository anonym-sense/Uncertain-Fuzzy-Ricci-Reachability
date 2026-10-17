"""Uncertain closed-loop benchmark systems.

Each model writes its vector field f(x, th, w) with generic operations, so the same code
runs on batched floats (x shape (n, B)) for Monte Carlo, on Interval vectors for sound
enclosures, and on dual numbers (frs.ad) for sound interval Jacobians.

th: uncertain constant parameters (LiuNormal each); w: uncertain disturbance channels
(Liu process alpha-path drift, allowed to be time varying inside its alpha band).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.linalg import solve_continuous_are

from .ad import jacobian
from .interval import Interval, cos, sin, stack
from .uncertain import LiuNormal, bands


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        self.lo = np.asarray(self.lo, float)
        self.hi = np.asarray(self.hi, float)

    def sdf(self, X):
        """Inf-norm signed distance (negative inside) for points X shape (n, B)."""
        lo, hi = self.lo[:, None], self.hi[:, None]
        d = np.maximum(lo - X, X - hi)
        d = np.where(np.isfinite(d), d, -np.inf)
        return d.max(0)

    def extend(self, p):
        """Same box in a state space augmented by p unconstrained coordinates."""
        return Box(np.r_[self.lo, np.full(p, -np.inf)], np.r_[self.hi, np.full(p, np.inf)])


def _split(x, n):
    if isinstance(x, Interval):
        return [x[i] for i in range(n)]
    return [x[i] for i in range(n)]


@dataclass
class System:
    name: str
    n: int
    f: Callable
    th: list
    w: list
    x0: Box
    unsafe: list
    T: float
    dt: float
    labels: tuple = ()
    proj: tuple = (0, 1)  # state pair used for 2D figures
    meta: dict = field(default_factory=dict)

    @property
    def p(self):
        return len(self.th)

    @property
    def th_hat(self):
        return np.array([v.e for v in self.th])

    @property
    def w_hat(self):
        return np.array([v.e for v in self.w])

    def bands(self, alpha):
        return Interval(*bands(self.th, alpha)), Interval(*bands(self.w, alpha))

    def F(self, x, th, w):
        out = self.f(x, th, w)
        return out if isinstance(out, (np.ndarray, Interval)) else stack(out)

    def jacs(self, x, th, w):
        """(Jx, Jth) via interval AD; batch shape (C,) -> (C, n, n), (C, n, p)."""
        xs, ths = _split(x, self.n), _split(th, self.p)
        _, Jx, Jt = jacobian(lambda X, TH: self.f(X, TH, w), xs, ths)
        return Jx, Jt

    def J(self, x, th, w):
        return self.jacs(x, th, w)[0]

    def sdf(self, X):
        return np.min(np.stack([u.sdf(X) for u in self.unsafe]), 0)

    def path_semantics(self):
        """Yao-Chen alpha-path semantics of the Liu-process disturbance.

        For dX = f dt + sigma dC_t the inverse uncertainty distribution of X_t is given
        by alpha-paths, in which the Liu-process term is the *constant* drift
        sigma Phi^-1(beta). The disturbances therefore become constant uncertain
        parameters (augmentable, splittable). The default System keeps the stronger
        'adversarial' semantics (w(t) arbitrary inside its band).
        """
        if not self.w:
            return self
        p, f = len(self.th), self.f

        def f2(x, th, w):
            return f(x, th[:p], th[p:])

        return System(self.name, self.n, f2, list(self.th) + list(self.w), [], self.x0,
                      self.unsafe, self.T, self.dt, self.labels, self.proj,
                      dict(self.meta, semantics="alpha-path"))


def _lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


# --------------------------------------------------------------------------------------
# B1  Reconstruction of the 2020 Sensors-Letters S-curve scenario (2D)
#     x1' = c,  x2' = -a (x2 - A sin(om x1 + ph)) + b w
def s_curve():
    om, ph, A, b = 2.24, -1.12, 1.75, 1.0

    def f(x, th, w):
        c, a = th[0], th[1]
        return [c + 0.0 * x[0], -a * (x[1] - A * sin(om * x[0] + ph)) + b * w[0]]

    return System(
        "s_curve", 2, f,
        th=[LiuNormal(0.6, 0.006), LiuNormal(3.0, 0.15)],
        w=[LiuNormal(0.0, 0.4)],
        x0=Box([-0.5, -0.6], [0.0, 0.0]),
        unsafe=[Box([5.3, -2.3], [6.0, -1.72])],
        T=10.0, dt=0.01, labels=("x1", "x2"),
    )


# --------------------------------------------------------------------------------------
# B2  Van der Pol, ARCH-COMP style initial set, spec x2 <= 2.75, uncertain mu + disturbance
def van_der_pol():
    def f(x, th, w):
        mu = th[0]
        return [x[1] + 0.0 * x[0], mu * (1 - x[0] ** 2) * x[1] - x[0] + w[0]]

    return System(
        "van_der_pol", 2, f,
        th=[LiuNormal(1.0, 0.02)],
        w=[LiuNormal(0.0, 0.05)],
        x0=Box([1.25, 2.25], [1.55, 2.35]),
        unsafe=[Box([-np.inf, 2.75], [np.inf, np.inf])],
        T=7.0, dt=0.005, labels=("x1", "x2"),
    )


# --------------------------------------------------------------------------------------
# B3  Inverted pendulum, LQR, uncertain mass and friction; unsafe: overshoot past -0.095
def pendulum():
    g, l = 9.81, 1.0
    m0, b0 = 1.0, 0.1
    A = np.array([[0, 1], [g / l, -b0 / (m0 * l * l)]])
    B = np.array([[0], [1 / (m0 * l * l)]])
    K = _lqr(A, B, np.diag([1.0, 0.05]), np.array([[0.02]]))[0]

    def f(x, th, w):
        m, b = th[0], th[1]
        u = -(K[0] * x[0] + K[1] * x[1])
        ml2 = m * (l * l)
        return [x[1] + 0.0 * x[0], (g / l) * sin(x[0]) - (b * x[1] - u - w[0]) / ml2]

    return System(
        "pendulum", 2, f,
        th=[LiuNormal(1.0, 0.08), LiuNormal(0.1, 0.03)],
        w=[LiuNormal(0.0, 0.4)],
        x0=Box([0.3, -0.1], [0.4, 0.1]),
        unsafe=[Box([-np.inf, -np.inf], [-0.095, np.inf])],
        T=3.0, dt=0.005, labels=("theta", "omega"), meta={"K": K},
    )


# --------------------------------------------------------------------------------------
# B4  Adaptive cruise control (3D): gap D, ego speed v, lead speed vl; quadratic drag.
def acc():
    mass, f0, f1 = 1650.0, 0.1, 5.0
    h, kD, kv = 1.4, 0.35, 0.9

    def f(x, th, w):
        f2, al = th[0], th[1]
        D, v, vl = x[0], x[1], x[2]
        u = mass * (kD * (D - h * v) + kv * (vl - v))
        drag = f0 + f1 * v + f2 * v ** 2
        return [vl - v, (u - drag) / mass, al + w[0] + 0.0 * vl]

    return System(
        "acc", 3, f,
        th=[LiuNormal(0.25, 0.03), LiuNormal(-1.2, 0.15)],
        w=[LiuNormal(0.0, 0.25)],
        x0=Box([38.0, 26.0, 21.5], [42.0, 27.0, 22.5]),
        unsafe=[Box([-np.inf, -np.inf, -np.inf], [12.0, np.inf, np.inf])],
        T=8.0, dt=0.01, labels=("gap", "v_ego", "v_lead"),
    )


# --------------------------------------------------------------------------------------
# B5  Planar quadrotor (6D), LQR to hover at origin, uncertain mass, wind disturbance.
def planar_quad():
    g, m0, Jz, arm = 9.81, 0.5, 0.01, 0.15
    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1
    A[3, 2] = -g
    B = np.zeros((6, 2))
    B[4, :] = 1 / m0
    B[5, 0], B[5, 1] = -arm / Jz, arm / Jz
    K = _lqr(A, B, np.diag([4, 4, 1, 1, 1, 0.5]), np.eye(2) * 2.0)
    ue = m0 * g / 2

    def f(x, th, w):
        m = th[0]
        dx = [x[i] for i in range(6)]
        u1 = ue - sum(float(K[0, j]) * dx[j] for j in range(6))
        u2 = ue - sum(float(K[1, j]) * dx[j] for j in range(6))
        T_ = u1 + u2
        return [x[3], x[4], x[5],
                -(T_ / m) * sin(x[2]) + w[0],
                (T_ / m) * cos(x[2]) - g + w[1],
                arm * (u2 - u1) / Jz]

    return System(
        "planar_quad", 6, f,
        th=[LiuNormal(0.5, 0.02)],
        w=[LiuNormal(0.0, 0.3), LiuNormal(0.0, 0.3)],
        x0=Box([-1.55, -0.05, -0.02, -0.05, -0.05, -0.02], [-1.45, 0.05, 0.02, 0.05, 0.05, 0.02]),
        unsafe=[Box([-0.9, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                    [-0.5, -0.19, np.inf, np.inf, np.inf, np.inf])],
        T=4.0, dt=0.005, labels=("x", "z", "phi", "vx", "vz", "omega"), meta={"K": K},
    )


# --------------------------------------------------------------------------------------
# B6  Scalable chain of N masses with cubic (Duffing) springs; dimension 2N.
def chain(N=4):
    k, c = 2.0, 0.8

    def f(x, th, w):
        k3 = th[0]
        pos = [x[i] for i in range(N)]
        vel = [x[N + i] for i in range(N)]
        acc_ = []
        for i in range(N):
            left = pos[i] - pos[i - 1] if i > 0 else pos[i]
            a = -k * left - k3 * left ** 3 - c * vel[i]
            if i < N - 1:
                right = pos[i] - pos[i + 1]
                a = a - k * right - k3 * right ** 3
            if i == N - 1:
                a = a + w[0]
            acc_.append(a)
        return vel + acc_

    n = 2 * N
    lo = np.zeros(n)
    hi = np.zeros(n)
    lo[N - 1], hi[N - 1] = 0.8, 1.0  # last mass displaced
    lo[N:], hi[N:] = -0.05, 0.05
    # unsafe: displaced (last) mass dropping 0.12 below its nominal minimum
    X = (0.5 * (lo + hi))[:, None]
    th0, xmin, h = np.array([[0.5], [0.0]]), 0.0, 0.0025
    for _ in range(int(4.0 / h)):
        k1 = np.stack(f(X, th0, [0.0]))
        k2 = np.stack(f(X + 0.5 * h * k1, th0, [0.0]))
        k3 = np.stack(f(X + 0.5 * h * k2, th0, [0.0]))
        k4 = np.stack(f(X + h * k3, th0, [0.0]))
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xmin = min(xmin, float(X[N - 1, 0]))
    ulo = np.full(n, -np.inf)
    uhi = np.full(n, np.inf)
    uhi[N - 1] = xmin - 0.12
    return System(
        f"chain{N}", n, f,
        th=[LiuNormal(0.5, 0.05)],
        w=[LiuNormal(0.0, 0.1)],
        x0=Box(lo, hi),
        unsafe=[Box(ulo, uhi)],
        T=4.0, dt=0.01, labels=tuple(f"p{i}" for i in range(N)) + tuple(f"v{i}" for i in range(N)),
        proj=(0, N),
    )


def s_curve_pocket():
    """S-curve with the unsafe box inside a concave pocket of the reach tube: the
    convex 'uncertain hull' of the 2020 paper covers it, the true tube does not."""
    s = s_curve()
    s.name = "s_curve_pocket"
    s.unsafe = [Box([2.3, 0.4], [3.2, 1.2])]
    return s


BENCHMARKS = {
    "s_curve": s_curve,
    "van_der_pol": van_der_pol,
    "pendulum": pendulum,
    "acc": acc,
    "planar_quad": planar_quad,
    "s_curve_pocket": s_curve_pocket,
}
