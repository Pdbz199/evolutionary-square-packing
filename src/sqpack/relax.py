"""Local search: physical relaxation and Lamarckian compression of packings.

Every individual the GA produces is "developed" before it is scored:

1. ``fire_relax`` treats the squares as soft bodies in a square container and
   minimises the overlap energy with FIRE (the Fast Inertial Relaxation Engine,
   Bitzek et al. 2006), moving and rotating squares to resolve overlaps.
2. ``make_feasible`` scales the centres apart just enough to remove whatever
   overlap is left, so every scored individual is a *valid* packing and its
   fitness is simply the side of its bounding square.
3. ``compress`` repeatedly tries a slightly smaller container (affine squeeze,
   relax, repair) and keeps the result when it is genuinely smaller. The squeeze
   factor ``eps`` grows on success and shrinks on failure, and is inherited by
   offspring as a self-adapted strategy parameter.

The improved configuration is written back into the genome (Lamarckian
evolution), so the population always consists of valid packings.
"""

import math

import numpy as np
from numba import njit, prange

from .geometry import center, energy_grad, make_feasible, side, wrap_angles

EPS_MIN = 1e-5
EPS_MAX = 0.08
INV_MASS = (1.0, 1.0, 6.0)  # unit mass; moment of inertia of a unit square is 1/6


@njit(cache=True)
def fire_relax(cfg, half, w_wall, max_steps, e_tol=1e-20):
    """Minimise the overlap energy of ``cfg`` (in place) inside [-half, half]^2.

    Returns the final energy. Stops early once the energy is below ``e_tol``
    (overlaps resolved) or the forces vanish (a jammed local minimum).
    """
    n = cfg.shape[0]
    v = np.zeros((n, 3))
    g = np.zeros((n, 3))
    dt = 0.05
    dt_max = 0.25
    dt_min = 1e-3
    alpha = 0.1
    n_pos = 0
    e = energy_grad(cfg, half, w_wall, g)
    for _ in range(max_steps):
        if e < e_tol:
            break
        power = 0.0
        fnorm2 = 0.0
        for i in range(n):
            for k in range(3):
                power -= g[i, k] * v[i, k]
                fnorm2 += g[i, k] * g[i, k]
        if fnorm2 < 1e-24:
            break
        if power > 0.0:
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            n_pos = 0
            dt = max(dt * 0.5, dt_min)
            alpha = 0.1
            for i in range(n):
                for k in range(3):
                    cfg[i, k] -= 0.5 * dt * v[i, k]
                    v[i, k] = 0.0
        # semi-implicit Euler step followed by FIRE velocity mixing
        vnorm2 = 0.0
        for i in range(n):
            for k in range(3):
                v[i, k] -= dt * g[i, k] * INV_MASS[k]
                vnorm2 += v[i, k] * v[i, k]
        mix = alpha * math.sqrt(vnorm2 / fnorm2)
        for i in range(n):
            for k in range(3):
                v[i, k] = (1.0 - alpha) * v[i, k] - mix * g[i, k]
                cfg[i, k] += dt * v[i, k]
        e = energy_grad(cfg, half, w_wall, g)
    return e


@njit(cache=True)
def compress(cfg, eps, attempts, steps, w_wall):
    """Try to shrink a valid packing ``attempts`` times; keep only real improvements.

    Returns (side, eps): the side of the (possibly improved) packing written back
    into ``cfg`` and the adapted squeeze factor.
    """
    center(cfg)
    s_cur = side(cfg)
    trial = np.empty_like(cfg)
    for _ in range(attempts):
        f = 1.0 - eps
        for i in range(cfg.shape[0]):
            trial[i, 0] = cfg[i, 0] * f
            trial[i, 1] = cfg[i, 1] * f
            trial[i, 2] = cfg[i, 2]
        fire_relax(trial, 0.5 * s_cur * f, w_wall, steps)
        s_new = make_feasible(trial)
        if s_new < s_cur - 1e-12:
            cfg[:, :] = trial
            s_cur = s_new
            eps = min(eps * 1.5, EPS_MAX)
        else:
            eps = max(eps * 0.5, EPS_MIN)
    wrap_angles(cfg)
    return s_cur, eps


@njit(cache=True)
def develop(cfg, eps, target_side, attempts, steps, w_wall):
    """Turn an arbitrary (possibly overlapping) genome into a scored, valid packing.

    Relaxes inside a container of side ``target_side``, repairs, then compresses.
    Returns (side, eps); ``side`` is inf if the genome could not be repaired.
    """
    center(cfg)
    fire_relax(cfg, 0.5 * target_side, w_wall, 2 * steps)
    s = make_feasible(cfg)
    if not np.isfinite(s):
        return np.inf, eps
    return compress(cfg, eps, attempts, steps, w_wall)


@njit(cache=True)
def squeeze_resistance(cfg, delta, steps, w_wall):
    """RMS overlap left after squeezing a valid packing by ``delta`` and relaxing it.

    0 means the packing could shrink by ``delta``; larger values mean it is more
    firmly jammed. Used to break ties between packings of (nearly) equal side,
    e.g. the many grid-like packings of 19 squares that all have side exactly 5.
    """
    s = side(cfg)
    trial = cfg.copy()
    f = 1.0 - delta
    for i in range(cfg.shape[0]):
        trial[i, 0] *= f
        trial[i, 1] *= f
    e = fire_relax(trial, 0.5 * s * f, w_wall, steps)
    return math.sqrt(max(e, 0.0))


@njit(cache=True, parallel=True)
def develop_batch(pop, eps, targets, attempts, steps, w_wall, out_side, out_eps, probe, out_res):
    """``develop`` every individual of ``pop`` in parallel (deterministic per individual).

    With ``probe > 0`` also measures each result's squeeze resistance into ``out_res``
    (otherwise ``out_res`` is zeroed).
    """
    for p in prange(pop.shape[0]):
        s, e = develop(pop[p], eps[p], targets[p], attempts, steps, w_wall)
        out_side[p] = s
        out_eps[p] = e
        out_res[p] = 0.0
        if probe > 0.0 and np.isfinite(s):
            out_res[p] = squeeze_resistance(pop[p], probe, steps, w_wall)


@njit(cache=True)
def polish(cfg, eps, rounds, attempts, steps, w_wall):
    """Long final compression of the best packing with a slowly cooled squeeze factor."""
    s = side(cfg)
    for _ in range(rounds):
        s, eps = compress(cfg, max(eps, 1e-4), attempts, steps, w_wall)
    return s, eps
