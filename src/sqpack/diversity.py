"""Population diversity: symmetry-aware raster fingerprints of packings.

Because the squares are interchangeable, two genomes are "the same packing" when
they cover the same region, regardless of square order or of which of the 8
symmetries of the container they are drawn in. Each packing is rasterised onto a
``res x res`` grid spanning its own bounding square; the distance between two
packings is the smallest fraction of differing cells over the 8 symmetries.
"""

import math

import numpy as np
from numba import njit

from .geometry import H, bbox


@njit(cache=True)
def raster(cfg, res):
    """Occupancy grid (flattened, uint8) of the packing in its bounding square."""
    out = np.zeros(res * res, dtype=np.uint8)
    xmin, xmax, ymin, ymax = bbox(cfg)
    s = max(xmax - xmin, ymax - ymin)
    x0 = 0.5 * (xmin + xmax) - 0.5 * s
    y0 = 0.5 * (ymin + ymax) - 0.5 * s
    n = cfg.shape[0]
    cs = np.cos(cfg[:, 2])
    sn = np.sin(cfg[:, 2])
    cell = s / res
    for iy in range(res):
        py = y0 + (iy + 0.5) * cell
        for ix in range(res):
            px = x0 + (ix + 0.5) * cell
            for i in range(n):
                dx = px - cfg[i, 0]
                dy = py - cfg[i, 1]
                if dx * dx + dy * dy > 0.5:
                    continue
                u = dx * cs[i] + dy * sn[i]
                v = -dx * sn[i] + dy * cs[i]
                if abs(u) <= H and abs(v) <= H:
                    out[iy * res + ix] = 1
                    break
    return out


def d4_perms(res):
    """Index maps so that ``r[perm[t]]`` is raster ``r`` under symmetry ``t`` (0-7)."""
    perms = np.empty((8, res * res), dtype=np.int64)
    for t in range(8):
        for iy in range(res):
            for ix in range(res):
                x, y = ix, iy
                if t >= 4:
                    y = res - 1 - y
                for _ in range(t % 4):
                    x, y = res - 1 - y, x
                perms[t, y * res + x] = iy * res + ix
    return perms


@njit(cache=True)
def raster_dist(a, b, perms):
    """Fraction of differing cells, minimised over the 8 symmetries; also returns the best t."""
    m = a.shape[0]
    best = m + 1
    best_t = 0
    for t in range(perms.shape[0]):
        d = 0
        p = perms[t]
        for k in range(m):
            if a[k] != b[p[k]]:
                d += 1
                if d >= best:
                    break
        if d < best:
            best = d
            best_t = t
    return best / m, best_t


@njit(cache=True)
def rasters_of(pop, res):
    out = np.empty((pop.shape[0], res * res), dtype=np.uint8)
    for p in range(pop.shape[0]):
        out[p] = raster(pop[p], res)
    return out


@njit(cache=True)
def mean_pairwise_distance(rasters, perms):
    n = rasters.shape[0]
    if n < 2:
        return 0.0
    tot = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d, _ = raster_dist(rasters[i], rasters[j], perms)
            tot += d
    return tot / (n * (n - 1) / 2)


@njit(cache=True)
def select_survivors(fit, rasters, perms, n_keep, min_dist):
    """Truncation selection with clearing.

    Walk candidates from best to worst and keep each one unless it is within
    ``min_dist`` of an already kept packing (a near-duplicate). If too few
    distinct candidates exist, the best duplicates fill the remaining slots.
    """
    order = np.argsort(fit)
    kept = np.empty(n_keep, dtype=np.int64)
    spare = np.empty(fit.shape[0], dtype=np.int64)
    nk = 0
    ns = 0
    for idx in order:
        if nk == n_keep:
            break
        dup = not math.isfinite(fit[idx])
        if not dup:
            for c in range(nk):
                d, _ = raster_dist(rasters[idx], rasters[kept[c]], perms)
                if d < min_dist:
                    dup = True
                    break
        if dup:
            spare[ns] = idx
            ns += 1
        else:
            kept[nk] = idx
            nk += 1
    k = 0
    while nk < n_keep:
        kept[nk] = spare[k]
        nk += 1
        k += 1
    # keep the final population sorted best-first
    return kept[np.argsort(fit[kept], kind="mergesort")]
