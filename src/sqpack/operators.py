"""Variation operators: selection, crossover and mutation (Numba, serial RNG).

All squares are identical, so swapping two squares changes nothing and index-wise
crossover is meaningless. Crossover is therefore *geometric*: a random line cuts
the container, and the child takes one parent's squares on one side and the
other parent's squares on the other side (after rotating/reflecting the second
parent to best match the first). The child usually overlaps along the seam; the
local search in ``relax.develop`` heals it.

Every child is recorded with an operator code: bit 0 says whether crossover was
used and ``code >> 1`` is the mutation that followed (``MUTATIONS[code >> 1]``).
"""

import math

import numpy as np
from numba import njit

from .geometry import extent, pair_depth
from .relax import EPS_MAX, EPS_MIN

MUTATIONS = ("none", "jitter", "reinsert", "rotate", "cluster-turn", "row-shift")
N_MUT = len(MUTATIONS)


@njit(cache=True)
def seed_rng(seed):
    """Seed Numba's internal generator (separate from NumPy's)."""
    np.random.seed(seed)


@njit(cache=True)
def tournament(fit, k):
    best = np.random.randint(fit.shape[0])
    for _ in range(k - 1):
        c = np.random.randint(fit.shape[0])
        if fit[c] < fit[best]:
            best = c
    return best


# --------------------------------------------------------------------------- D4


@njit(cache=True)
def apply_d4(cfg, t, out):
    """Apply symmetry ``t`` (0-7) of the square to ``cfg``: optional mirror, then t%4 quarter turns."""
    for i in range(cfg.shape[0]):
        x = cfg[i, 0]
        y = cfg[i, 1]
        th = cfg[i, 2]
        if t >= 4:
            y = -y
            th = -th
        r = t % 4
        if r == 1:
            x, y = -y, x
        elif r == 2:
            x, y = -x, -y
        elif r == 3:
            x, y = y, -x
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = th  # quarter turns leave a square's orientation unchanged


@njit(cache=True)
def align_d4(ref, cfg, out):
    """Write into ``out`` the symmetry of ``cfg`` whose centres best match ``ref``'s."""
    n = ref.shape[0]
    tmp = np.empty_like(cfg)
    best_cost = np.inf
    best_t = 0
    for t in range(8):
        apply_d4(cfg, t, tmp)
        cost = 0.0
        for i in range(n):
            m = np.inf
            for j in range(n):
                dx = ref[i, 0] - tmp[j, 0]
                dy = ref[i, 1] - tmp[j, 1]
                m = min(m, dx * dx + dy * dy)
            cost += m
        if cost < best_cost:
            best_cost = cost
            best_t = t
    apply_d4(cfg, best_t, out)
    return best_t


# ------------------------------------------------------------------- crossover


@njit(cache=True)
def crossover_cut(a, b, child):
    """Geometric cut-and-splice crossover of two packings (both centred)."""
    n = a.shape[0]
    bb = np.empty_like(b)
    align_d4(a, b, bb)
    phi = np.random.random() * math.pi
    nx = math.cos(phi)
    ny = math.sin(phi)
    spread = 0.0
    for i in range(n):
        spread = max(spread, abs(a[i, 0] * nx + a[i, 1] * ny))
    off = (np.random.random() - 0.5) * spread
    # signed distance of every candidate square to the cut line; A on the negative
    # side, B on the positive side. Candidates 0..n-1 are A's, n..2n-1 are B's.
    dist = np.empty(2 * n)
    take = np.zeros(2 * n, dtype=np.bool_)
    count = 0
    for i in range(n):
        dist[i] = a[i, 0] * nx + a[i, 1] * ny - off
        dist[n + i] = bb[i, 0] * nx + bb[i, 1] * ny - off
        if dist[i] < 0.0:
            take[i] = True
            count += 1
        if dist[n + i] >= 0.0:
            take[n + i] = True
            count += 1
    order = np.argsort(np.abs(dist))
    # too many squares: drop the selected ones closest to the seam;
    # too few: add unselected ones closest to the seam.
    for idx in order:
        if count == n:
            break
        if count > n and take[idx]:
            take[idx] = False
            count -= 1
        elif count < n and not take[idx]:
            take[idx] = True
            count += 1
    k = 0
    for idx in range(2 * n):
        if take[idx]:
            src = a[idx] if idx < n else bb[idx - n]
            child[k, 0] = src[0]
            child[k, 1] = src[1]
            child[k, 2] = src[2]
            k += 1


# -------------------------------------------------------------------- mutations


@njit(cache=True)
def _random_angle():
    """Axis-aligned half the time, otherwise a uniformly random tilt."""
    if np.random.random() < 0.5:
        return 0.0
    return (np.random.random() - 0.5) * 0.5 * math.pi


@njit(cache=True)
def mutate_jitter(cfg):
    sigma = 0.01 * math.exp(np.random.random() * math.log(15.0))  # log-uniform in [0.01, 0.15]
    for i in range(cfg.shape[0]):
        cfg[i, 0] += np.random.normal() * sigma
        cfg[i, 1] += np.random.normal() * sigma
        cfg[i, 2] += np.random.normal() * sigma


@njit(cache=True)
def _placement_cost(cfg, skip, x, y, t, half):
    e = 0.0
    for j in range(cfg.shape[0]):
        if j == skip:
            continue
        d = pair_depth(x, y, t, cfg[j, 0], cfg[j, 1], cfg[j, 2])
        e += d * d
    r = extent(t)
    for c in (x, y):
        v = abs(c) + r - half
        if v > 0.0:
            e += v * v
    return e


@njit(cache=True)
def mutate_reinsert(cfg, half, samples=96):
    """Pick up 1-2 squares and drop each into the emptiest of ``samples`` random spots."""
    n = cfg.shape[0]
    moves = 2 if np.random.random() < 0.35 else 1
    for _ in range(moves):
        i = np.random.randint(n)
        best = np.inf
        bx = cfg[i, 0]
        by = cfg[i, 1]
        bt = cfg[i, 2]
        lim = max(half - 0.5, 0.0)
        for _s in range(samples):
            x = (2.0 * np.random.random() - 1.0) * lim
            y = (2.0 * np.random.random() - 1.0) * lim
            t = _random_angle()
            c = _placement_cost(cfg, i, x, y, t, half)
            if c < best:
                best = c
                bx = x
                by = y
                bt = t
        cfg[i, 0] = bx
        cfg[i, 1] = by
        cfg[i, 2] = bt


@njit(cache=True)
def mutate_rotate(cfg):
    """Re-orient one square: axis-aligned, 45 degrees, or random."""
    i = np.random.randint(cfg.shape[0])
    u = np.random.random()
    if u < 0.3:
        cfg[i, 2] = 0.0
    elif u < 0.6:
        cfg[i, 2] = 0.25 * math.pi
    else:
        cfg[i, 2] = (np.random.random() - 0.5) * 0.5 * math.pi


@njit(cache=True)
def mutate_cluster_turn(cfg):
    """Rigidly turn the squares near a random pivot by up to +-45 degrees."""
    n = cfg.shape[0]
    p = np.random.randint(n)
    px = cfg[p, 0]
    py = cfg[p, 1]
    rad2 = (0.8 + np.random.random()) ** 2
    phi = (np.random.random() - 0.5) * 0.5 * math.pi
    c = math.cos(phi)
    s = math.sin(phi)
    for i in range(n):
        dx = cfg[i, 0] - px
        dy = cfg[i, 1] - py
        if dx * dx + dy * dy <= rad2:
            cfg[i, 0] = px + c * dx - s * dy
            cfg[i, 1] = py + s * dx + c * dy
            cfg[i, 2] += phi


@njit(cache=True)
def mutate_row_shift(cfg):
    """Slide the row (or column) through a random square sideways by up to half a unit."""
    n = cfg.shape[0]
    p = np.random.randint(n)
    dim = np.random.randint(2)
    other = 1 - dim
    shift = np.random.random() - 0.5
    for i in range(n):
        if abs(cfg[i, other] - cfg[p, other]) < 0.5:
            cfg[i, dim] += shift


@njit(cache=True)
def mutate(cfg, op, half):
    if op == 1:
        mutate_jitter(cfg)
    elif op == 2:
        mutate_reinsert(cfg, half)
    elif op == 3:
        mutate_rotate(cfg)
    elif op == 4:
        mutate_cluster_turn(cfg)
    elif op == 5:
        mutate_row_shift(cfg)


@njit(cache=True)
def _pick(cum):
    u = np.random.random() * cum[-1]
    for k in range(cum.shape[0]):
        if u < cum[k]:
            return k
    return cum.shape[0] - 1


@njit(cache=True)
def vary(pop, fit, sel, eps, p_cross, p_mut, op_weights, tour_k, children, ch_eps, targets, parents, ops):
    """Breed ``children.shape[0]`` offspring from ``pop`` (sorted or not).

    Tournaments compare the selection key ``sel`` (the side, possibly plus a
    tie-break); container targets use the true side ``fit``.
    ``op_weights`` has one weight per entry of ``MUTATIONS`` (index 0 is unused).
    Writes genomes, inherited squeeze factors, container targets, parent indices
    and operator codes into the output arrays.
    """
    n_pop = pop.shape[0]
    cum = np.cumsum(op_weights[1:])
    for c in range(children.shape[0]):
        a = tournament(sel, tour_k)
        code = 0
        if n_pop > 1 and np.random.random() < p_cross:
            b = tournament(sel, tour_k)
            while b == a:
                b = np.random.randint(n_pop)
            crossover_cut(pop[a], pop[b], children[c])
            parents[c, 0] = a
            parents[c, 1] = b
            targets[c] = min(fit[a], fit[b])
            e = math.sqrt(eps[a] * eps[b])
            code = 1
        else:
            children[c, :, :] = pop[a]
            parents[c, 0] = a
            parents[c, 1] = -1
            targets[c] = fit[a]
            e = eps[a]
        if code == 0 or np.random.random() < p_mut:
            op = 1 + _pick(cum)
            mutate(children[c], op, 0.5 * targets[c])
            code |= op << 1
        ch_eps[c] = min(max(e * math.exp(0.3 * np.random.normal()), EPS_MIN), EPS_MAX)
        ops[c] = code


@njit(cache=True)
def random_population(pop, box):
    """Scatter squares uniformly in a box of side ``box`` with random tilts."""
    for p in range(pop.shape[0]):
        for i in range(pop.shape[1]):
            pop[p, i, 0] = (np.random.random() - 0.5) * box
            pop[p, i, 1] = (np.random.random() - 0.5) * box
            pop[p, i, 2] = (np.random.random() - 0.5) * 0.5 * math.pi


@njit(cache=True)
def lattice_population(pop, jitter):
    """Jittered square lattice with random vacancies (a strong, boring start)."""
    n = pop.shape[1]
    k = int(math.ceil(math.sqrt(n)))
    cells = np.arange(k * k)
    for p in range(pop.shape[0]):
        np.random.shuffle(cells)
        for i in range(n):
            cx = cells[i] % k
            cy = cells[i] // k
            pop[p, i, 0] = cx - 0.5 * (k - 1) + np.random.normal() * jitter
            pop[p, i, 1] = cy - 0.5 * (k - 1) + np.random.normal() * jitter
            pop[p, i, 2] = np.random.normal() * jitter

