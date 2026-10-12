"""Independent validation of a final packing.

Deliberately does not reuse the Numba kernels: corners are computed with plain
NumPy and every pair is tested with a generic polygon SAT test, so a bug in the
optimiser's geometry cannot certify its own output.
"""

import math
from dataclasses import dataclass

import numpy as np


def square_corners(x, y, t):
    c, s = math.cos(t) * 0.5, math.sin(t) * 0.5
    return np.array([[x - c + s, y - s - c], [x + c + s, y + s - c], [x + c - s, y + s + c], [x - c - s, y - s + c]])


def _polygon_penetration(p, q):
    """Penetration depth of two convex polygons via SAT over all edge normals (<=0: disjoint)."""
    depth = math.inf
    for poly in (p, q):
        for k in range(len(poly)):
            e = poly[(k + 1) % len(poly)] - poly[k]
            axis = np.array([-e[1], e[0]]) / math.hypot(e[0], e[1])
            pp = p @ axis
            qq = q @ axis
            overlap = min(pp.max(), qq.max()) - max(pp.min(), qq.min())
            depth = min(depth, overlap)
    return depth


@dataclass
class Validation:
    valid: bool
    side: float
    max_overlap: float
    worst_pair: tuple | None
    n: int

    def as_dict(self):
        return {
            "valid": self.valid,
            "side": self.side,
            "max_overlap": self.max_overlap,
            "worst_pair": list(self.worst_pair) if self.worst_pair else None,
            "n": self.n,
        }


def validate(cfg, tol=1e-9):
    """Check that no two squares overlap by more than ``tol`` and report the container side.

    The container is the axis-aligned bounding square of all corners, so
    containment holds by construction; the side is computed from the corners.
    """
    cfg = np.asarray(cfg, dtype=float)
    polys = [square_corners(*row) for row in cfg]
    pts = np.concatenate(polys)
    side_len = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    worst = 0.0
    worst_pair = None
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if np.sum((cfg[i, :2] - cfg[j, :2]) ** 2) >= 2.0:
                continue
            d = _polygon_penetration(polys[i], polys[j])
            if d > worst:
                worst = d
                worst_pair = (i, j)
    return Validation(valid=worst <= tol, side=side_len, max_overlap=worst, worst_pair=worst_pair, n=len(cfg))


def to_container_frame(cfg):
    """Translate so the bounding square is [0, s] x [0, s] (the packing is centred in it)."""
    cfg = np.array(cfg, dtype=float)
    pts = np.concatenate([square_corners(*row) for row in cfg])
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    s = max(hi - lo)
    offset = lo - 0.5 * (s - (hi - lo))
    cfg[:, :2] -= offset
    return cfg, float(s)
