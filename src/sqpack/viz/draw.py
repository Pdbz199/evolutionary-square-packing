"""Drawing packings onto Matplotlib axes."""

import math

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from ..geometry import max_depth
from ..operators import align_d4
from . import style


def square_polys(cfg):
    """(N, 4, 2) corner array for a configuration (plain NumPy)."""
    cfg = np.asarray(cfg, dtype=float)
    c = np.cos(cfg[:, 2])[:, None] * 0.5
    s = np.sin(cfg[:, 2])[:, None] * 0.5
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    x = cfg[:, 0:1] + local[:, 0] * c - local[:, 1] * s
    y = cfg[:, 1:2] + local[:, 0] * s + local[:, 1] * c
    return np.stack([x, y], axis=-1)


def bounding_square(cfg):
    """(x0, y0, side) of the axis-aligned bounding square, centred on the packing."""
    p = square_polys(cfg).reshape(-1, 2)
    lo, hi = p.min(axis=0), p.max(axis=0)
    s = float(max(hi - lo))
    mid = 0.5 * (lo + hi)
    return mid[0] - s / 2, mid[1] - s / 2, s


def centred(cfg):
    """Copy of ``cfg`` translated so its bounding square is centred on the origin."""
    cfg = np.array(cfg, dtype=float)
    x0, y0, s = bounding_square(cfg)
    cfg[:, 0] -= x0 + s / 2
    cfg[:, 1] -= y0 + s / 2
    return cfg


def aligned(cfg, ref):
    """``cfg`` turned/mirrored (one of the 8 symmetries) to look most like ``ref``."""
    cfg = centred(cfg)
    if ref is None or len(ref) != len(cfg):
        return cfg
    out = np.empty_like(cfg)
    align_d4(np.ascontiguousarray(ref, dtype=float), np.ascontiguousarray(cfg), out)
    return out


def draw_packing(ax, cfg, half_view=None, container=True, edge_width=0.8, flag_overlaps=True, alpha=1.0):
    """Draw a packing centred on the origin. Returns the PolyCollection (for animation)."""
    cfg = np.asarray(cfg, dtype=float)
    polys = square_polys(cfg)
    coll = PolyCollection(polys, facecolors=style.tilt_colors(cfg[:, 2]), edgecolors=style.SURFACE, linewidths=edge_width, alpha=alpha, zorder=2)
    ax.add_collection(coll)
    x0, y0, s = bounding_square(cfg)
    if container:
        ax.add_patch(Rectangle((x0, y0), s, s, fill=False, edgecolor=style.INK_2, linewidth=0.9, zorder=3))
    if flag_overlaps and len(cfg) > 1 and max_depth(np.ascontiguousarray(cfg)) > 1e-6:
        ax.add_patch(Rectangle((x0, y0), s, s, fill=False, edgecolor=style.OVERLAP, linewidth=1.5, zorder=4))
    h = half_view if half_view is not None else 0.5 * s * 1.04
    ax.set_xlim(-h, h)
    ax.set_ylim(-h, h)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return coll


def tilt_colorbar(fig, cax, label="tilt of each square"):
    import matplotlib as mpl

    norm = mpl.colors.Normalize(0, 45)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=style.TILT_CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 15, 30, 45])
    cb.set_ticklabels(["0°", "15°", "30°", "45°"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=style.INK_2)
    cb.set_label(label, color=style.INK_2, fontsize=8)
    return cb


def sample_generations(total, k, log_bias=0.6):
    """``k`` distinct generations from 0..total, denser early (where most change happens)."""
    if total + 1 <= k:
        return list(range(total + 1))
    lin = np.linspace(0, total, k)
    log = np.geomspace(1, total + 1, k) - 1
    g = np.round(log_bias * log + (1 - log_bias) * lin).astype(int)
    g[0], g[-1] = 0, total
    out = sorted(set(int(v) for v in g))
    # fill up any collisions with evenly spaced extras
    extra = (int(v) for v in np.linspace(0, total, 4 * k).round())
    while len(out) < k:
        v = next(extra, None)
        if v is None:
            break
        if v not in out:
            out.append(v)
            out.sort()
    return out


def fmt_side(s):
    return "∞" if not math.isfinite(s) else f"{s:.4f}"
