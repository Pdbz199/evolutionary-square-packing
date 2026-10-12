"""Geometry kernels for packing unit squares that are free to rotate.

A *configuration* is an ``(N, 3)`` float64 array whose rows are ``(x, y, theta)``:
the centre of a unit square and its rotation in radians. A unit square looks
the same after a quarter turn, so ``theta`` only matters modulo pi/2.

Overlap between two squares is measured with the separating axis theorem (SAT).
For two squares the four candidate axes are the edge normals of each square.
Projected onto any of those axes, the two squares have the *same* combined
half-width ``R = 0.5 * (1 + |cos dθ| + |sin dθ|)``, so the penetration depth is
simply ``R - max_k |Δ·a_k|`` where Δ is the centre offset. The energy that the
local search minimises is the sum of squared depths plus a wall term, and its
gradient with respect to ``(x, y, theta)`` is computed analytically here.
"""

import math

import numpy as np
from numba import njit

H = 0.5  # half the side of a unit square
QUARTER = 0.5 * math.pi
EIGHTH = 0.25 * math.pi
# Two unit squares cannot touch once their centres are further apart than the
# sum of their circumradii (2 * sqrt(2)/2 = sqrt(2)); compare squared distances.
CUTOFF2 = 2.0


@njit(cache=True, inline="always")
def _sign(v):
    if v > 0.0:
        return 1.0
    if v < 0.0:
        return -1.0
    return 0.0


@njit(cache=True, inline="always")
def extent(t):
    """Half-width of a rotated unit square projected onto the x (or y) axis."""
    return H * (abs(math.cos(t)) + abs(math.sin(t)))


@njit(cache=True, inline="always")
def extent_deriv(t):
    """d extent / d t (a subgradient at the kinks, 0 when axis-aligned)."""
    c = math.cos(t)
    s = math.sin(t)
    return H * (-_sign(c) * s + _sign(s) * c)


@njit(cache=True, inline="always")
def wrap_angle(t):
    """Map an angle onto the canonical range [-pi/4, pi/4)."""
    return t - QUARTER * math.floor((t + EIGHTH) / QUARTER)


@njit(cache=True)
def wrap_angles(cfg):
    for i in range(cfg.shape[0]):
        cfg[i, 2] = wrap_angle(cfg[i, 2])


@njit(cache=True)
def bbox(cfg):
    """Axis-aligned bounding box (xmin, xmax, ymin, ymax) of all squares."""
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for i in range(cfg.shape[0]):
        r = extent(cfg[i, 2])
        xmin = min(xmin, cfg[i, 0] - r)
        xmax = max(xmax, cfg[i, 0] + r)
        ymin = min(ymin, cfg[i, 1] - r)
        ymax = max(ymax, cfg[i, 1] + r)
    return xmin, xmax, ymin, ymax


@njit(cache=True)
def side(cfg):
    """Side length of the smallest axis-aligned square containing the packing."""
    xmin, xmax, ymin, ymax = bbox(cfg)
    return max(xmax - xmin, ymax - ymin)


@njit(cache=True)
def center(cfg):
    """Translate the configuration so its bounding box is centred on the origin."""
    xmin, xmax, ymin, ymax = bbox(cfg)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    for i in range(cfg.shape[0]):
        cfg[i, 0] -= cx
        cfg[i, 1] -= cy


@njit(cache=True, inline="always")
def _max_projection(dx, dy, ti, tj):
    """Largest |Δ·a| over the four SAT axes, plus the signed projection and axis data."""
    best = -1.0
    best_d = 0.0
    best_c = 1.0
    best_s = 0.0
    best_k = 0
    for k in range(4):
        a = (ti if k < 2 else tj) + (k % 2) * QUARTER
        ca = math.cos(a)
        sa = math.sin(a)
        d = dx * ca + dy * sa
        if abs(d) > best:
            best = abs(d)
            best_d = d
            best_c = ca
            best_s = sa
            best_k = k
    return best, best_d, best_c, best_s, best_k


@njit(cache=True)
def pair_depth(xi, yi, ti, xj, yj, tj):
    """SAT penetration depth of two unit squares (0 if they do not overlap)."""
    dx = xj - xi
    dy = yj - yi
    if dx * dx + dy * dy >= CUTOFF2:
        return 0.0
    dth = ti - tj
    R = H * (1.0 + abs(math.cos(dth)) + abs(math.sin(dth)))
    best, _, _, _, _ = _max_projection(dx, dy, ti, tj)
    return max(R - best, 0.0)


@njit(cache=True)
def pair_energy_grad(cfg, i, j, grad):
    """Add d(depth^2) for the pair (i, j) to ``grad``; return depth^2."""
    xi, yi, ti = cfg[i, 0], cfg[i, 1], cfg[i, 2]
    xj, yj, tj = cfg[j, 0], cfg[j, 1], cfg[j, 2]
    dx = xj - xi
    dy = yj - yi
    if dx * dx + dy * dy >= CUTOFF2:
        return 0.0
    dth = ti - tj
    R = H * (1.0 + abs(math.cos(dth)) + abs(math.sin(dth)))
    best, d, ca, sa, k = _max_projection(dx, dy, ti, tj)
    depth = R - best
    if depth <= 0.0:
        return 0.0
    s = _sign(d)
    nx = s * ca  # unit normal of the minimum-translation axis, pointing i -> j
    ny = s * sa
    # dR/dti, and the change of |Δ·a| as the axis a turns with its owner.
    gR = H * (-_sign(math.cos(dth)) * math.sin(dth) + _sign(math.sin(dth)) * math.cos(dth))
    dperp = -s * (-dx * sa + dy * ca)
    dti = gR + (dperp if k < 2 else 0.0)
    dtj = -gR + (dperp if k >= 2 else 0.0)
    c = 2.0 * depth
    grad[i, 0] += c * nx
    grad[i, 1] += c * ny
    grad[i, 2] += c * dti
    grad[j, 0] -= c * nx
    grad[j, 1] -= c * ny
    grad[j, 2] += c * dtj
    return depth * depth


@njit(cache=True)
def energy_grad(cfg, half, w_wall, grad):
    """Overlap energy of ``cfg`` in the container [-half, half]^2 and its gradient.

    E = sum_pairs depth^2 + w_wall * sum_walls violation^2. ``grad`` is overwritten.
    """
    n = cfg.shape[0]
    grad[:, :] = 0.0
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            e += pair_energy_grad(cfg, i, j, grad)
    if w_wall > 0.0:
        for i in range(n):
            r = extent(cfg[i, 2])
            rp = extent_deriv(cfg[i, 2])
            for dim in range(2):
                c = cfg[i, dim]
                v = c + r - half
                if v > 0.0:
                    e += w_wall * v * v
                    grad[i, dim] += 2.0 * w_wall * v
                    grad[i, 2] += 2.0 * w_wall * v * rp
                v = r - half - c
                if v > 0.0:
                    e += w_wall * v * v
                    grad[i, dim] -= 2.0 * w_wall * v
                    grad[i, 2] += 2.0 * w_wall * v * rp
    return e


@njit(cache=True)
def max_depth(cfg):
    """Largest pairwise penetration depth in the configuration."""
    n = cfg.shape[0]
    m = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = pair_depth(cfg[i, 0], cfg[i, 1], cfg[i, 2], cfg[j, 0], cfg[j, 1], cfg[j, 2])
            if d > m:
                m = d
    return m


@njit(cache=True)
def repair_scale(cfg):
    """Smallest factor k >= 1 such that scaling every centre by k removes all overlaps.

    Scaling centres (not squares) multiplies every projection |Δ·a| by k while the
    combined half-width R stays fixed, so pair (i, j) separates exactly when
    k * max_a |Δ·a| >= R. Returns inf if two centres coincide.
    """
    n = cfg.shape[0]
    k = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = cfg[j, 0] - cfg[i, 0]
            dy = cfg[j, 1] - cfg[i, 1]
            if dx * dx + dy * dy >= CUTOFF2:
                continue
            dth = cfg[i, 2] - cfg[j, 2]
            R = H * (1.0 + abs(math.cos(dth)) + abs(math.sin(dth)))
            best, _, _, _, _ = _max_projection(dx, dy, cfg[i, 2], cfg[j, 2])
            if best < R:
                if best < 1e-9:
                    return np.inf
                k = max(k, R / best)
    if k > 1.0:
        k *= 1.0 + 1e-12
    return k


@njit(cache=True)
def make_feasible(cfg):
    """Centre the packing and push centres apart until nothing overlaps.

    Returns the resulting side length (inf if the packing could not be repaired).
    """
    center(cfg)
    k = repair_scale(cfg)
    if not np.isfinite(k) or k > 50.0:
        return np.inf
    if k > 1.0:
        for i in range(cfg.shape[0]):
            cfg[i, 0] *= k
            cfg[i, 1] *= k
        center(cfg)
    return side(cfg)
