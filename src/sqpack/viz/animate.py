"""Animated GIFs of a run (Matplotlib frames assembled with Pillow)."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from ..history import RunData  # noqa: E402
from . import style  # noqa: E402
from .draw import aligned, bounding_square, sample_generations, square_polys  # noqa: E402


def _frame_gens(run, max_frames):
    """Stored generations to animate: all of them if few, else denser early on."""
    stored = [int(g) for g in run.pop_gens]
    if len(stored) <= max_frames:
        return stored
    wanted = sample_generations(run.generations, max_frames, log_bias=0.5)
    out = sorted({stored[int(np.argmin(np.abs(np.array(stored) - g)))] for g in wanted})
    return out


class _PackingArtist:
    """A packing drawn as one PolyCollection plus its container, updatable in place."""

    def __init__(self, ax, n, half, lw):
        self.ax = ax
        self.coll = PolyCollection(np.zeros((n, 4, 2)), edgecolors=style.SURFACE, linewidths=lw, zorder=2)
        ax.add_collection(self.coll)
        self.box = Rectangle((0, 0), 1, 1, fill=False, edgecolor=style.INK_2, linewidth=0.8, zorder=3)
        ax.add_patch(self.box)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_axis_off()

    def set_half(self, half):
        self.ax.set_xlim(-half, half)
        self.ax.set_ylim(-half, half)

    def update(self, cfg):
        self.coll.set_verts(square_polys(cfg))
        self.coll.set_facecolor(style.tilt_colors(cfg[:, 2]))
        x0, y0, s = bounding_square(cfg)
        self.box.set_bounds(x0, y0, s, s)


def _to_image(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buf[..., :3].copy())


def _save_gif(frames, path, durations, colors=128):
    """Quantise all frames against one shared palette (stable colours, smaller file)."""
    picks = sorted({0, len(frames) // 4, len(frames) // 2, len(frames) - 1})
    w, h = frames[0].size
    sheet = Image.new("RGB", (w, h * len(picks) + 40))
    for k, i in enumerate(picks):
        sheet.paste(frames[i], (0, k * h))
    # make sure thin chart lines keep their exact series colours
    sw = w // len(style.SERIES)
    for k, c in enumerate(style.SERIES):
        sheet.paste(c, (k * sw, h * len(picks), (k + 1) * sw, h * len(picks) + 40))
    palette = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    pal = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=1)


def evolution_gif(run: RunData, path, max_frames=120, max_thumbs=48, frame_ms=110, hold_ms=2500):
    """Dashboard animation: best packing, the whole population, and the fitness curve."""
    style.apply()
    gens = _frame_gens(run, max_frames)
    ref = aligned(np.array(run.final["squares"]), None) if run.final.get("squares") else aligned(run.best_cfg[-1], None)
    P = min(run.pop_size, max_thumbs)

    # precompute aligned packings; thumbnails share one scale, the big panel another
    pops = {}
    max_best = 0.0
    for g in gens:
        _, pop, fits = run.population(g)
        al = [aligned(pop[i], ref) for i in range(P)]
        frame_half = 0.5 * max(bounding_square(c)[2] for c in al) * 1.03
        pops[g] = (al, frame_half)
        max_best = max(max_best, bounding_square(al[0])[2])
    half_best = 0.5 * max_best * 1.03

    W, H = 12.0, 6.6
    fig = plt.figure(figsize=(W, H), dpi=90)
    fig.patch.set_facecolor(style.PAGE)
    title = fig.text(0.2 / W, 1 - 0.2 / H, "", fontsize=15, fontweight="semibold", va="top")
    subtitle = fig.text(0.2 / W, 1 - 0.62 / H, "", fontsize=10, color=style.INK_2, va="top")
    big = 3.75
    ax_best = fig.add_axes([0.75 / W, 2.05 / H, big / W, big / H])
    best_art = _PackingArtist(ax_best, run.n, half_best, 1.2)
    fig.text((0.75 + big / 2) / W, 1.98 / H, "best packing this generation", fontsize=9.5, color=style.INK_2, ha="center", va="top")

    # population thumbnails: pick the grid shape that makes the cells largest
    tx0, tx1, ty0, ty1 = 5.3, W - 0.15, 0.2, H - 1.15
    tw_in, th_in = tx1 - tx0, ty1 - ty0
    tcols = max(range(1, P + 1), key=lambda c: min(tw_in / c, th_in / math.ceil(P / c)))
    cell = min(tw_in / tcols, th_in / math.ceil(P / tcols))
    thumbs = []
    for k in range(P):
        r, c = divmod(k, tcols)
        ax = fig.add_axes([(tx0 + c * cell) / W, (ty1 - (r + 1) * cell) / H, cell * 0.93 / W, cell * 0.93 / H])
        thumbs.append(_PackingArtist(ax, run.n, half_best, 0.3))
    if P == run.pop_size:
        label = f"whole population ({P} individuals, best to worst, drawn at a common scale)"
    else:
        label = f"best {P} of {run.pop_size} individuals (best to worst, drawn at a common scale)"
    fig.text(tx0 / W, (ty1 + 0.08) / H, label, fontsize=9.5, color=style.INK_2, va="bottom")

    # fitness curve
    axf = fig.add_axes([0.75 / W, 0.4 / H, 4.2 / W, 1.3 / H])
    g_all = np.arange(run.generations + 1)
    axf.plot(g_all, run.median, color=style.MEDIAN, linewidth=1.6, label="median")
    axf.plot(g_all, run.best, color=style.BEST, linewidth=1.8, label="best")
    rec = run.final.get("best_known")
    if rec is not None:
        axf.axhline(rec, color=style.MUTED, linewidth=0.8)
        axf.annotate(f"best known {rec:.4f}", (0, rec), xytext=(3, 2), textcoords="offset points", fontsize=7.5, color=style.INK_2)
    lo = min(run.best.min(), rec if rec is not None else np.inf)
    hi = max(run.median[0], run.best[0])
    axf.set_ylim(lo - 0.05 * (hi - lo), hi + 0.03 * (hi - lo))
    if run.generations > 40:
        axf.set_xscale("symlog", linthresh=10, linscale=0.6)
        ticks = [t for t in (0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if t <= run.generations]
        axf.set_xticks(ticks, [str(t) for t in ticks])
        axf.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axf.set_xlim(0, max(1, run.generations))
    axf.tick_params(labelsize=8)
    axf.set_ylabel("container side", fontsize=8.5)
    axf.legend(loc="upper right", ncols=2, fontsize=8)
    cursor = axf.axvline(0, color=style.INK_2, linewidth=1.0)
    (dot,) = axf.plot([0], [run.best[0]], "o", color=style.BEST, markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=2, zorder=5)

    frames = []
    for g in gens:
        al, frame_half = pops[g]
        best_art.update(al[0])
        for art, cfg in zip(thumbs, al):
            art.update(cfg)
            art.set_half(frame_half)
        cursor.set_xdata([g, g])
        dot.set_data([g], [run.best[g]])
        title.set_text(f"N = {run.n} unit squares · generation {g} of {run.generations}")
        gap = f" · {100 * (run.best[g] - rec) / rec:+.2f}% vs best known" if rec is not None else ""
        subtitle.set_text(f"best side {run.best[g]:.4f}{gap} · median {run.median[g]:.4f} · diversity {run.diversity[g]:.3f}")
        frames.append(_to_image(fig))
    plt.close(fig)
    durations = [frame_ms] * len(frames)
    durations[0] = 900
    durations[-1] = hold_ms
    _save_gif(frames, path, durations)
    return path


def best_gif(run: RunData, path, max_frames=90, size_px=420, frame_ms=110, hold_ms=2500):
    """Small square animation of the best packing only (handy for READMEs)."""
    style.apply()
    best = run.best
    # one frame per improvement (plus spacing frames), capped
    events = [0] + [int(i) for i in np.nonzero(np.diff(best) < -1e-9)[0] + 1] + [run.generations]
    events = sorted(set(events))
    if len(events) > max_frames:
        idx = np.unique(np.linspace(0, len(events) - 1, max_frames).round().astype(int))
        events = [events[i] for i in idx]
    ref = aligned(np.array(run.final["squares"]), None) if run.final.get("squares") else aligned(run.best_cfg[-1], None)
    cfgs = [aligned(run.best_cfg[g], ref) for g in events]
    half = 0.5 * max(bounding_square(c)[2] for c in cfgs) * 1.04
    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, (size_px + 46) / dpi), dpi=dpi)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94 * size_px / (size_px + 46)])
    art = _PackingArtist(ax, run.n, half, 1.0)
    label = fig.text(0.04, 0.975, "", fontsize=11, fontweight="semibold", va="top")
    frames = []
    for g, cfg in zip(events, cfgs):
        art.update(cfg)
        label.set_text(f"N = {run.n} · gen {g} · side {best[g]:.4f}")
        frames.append(_to_image(fig))
    plt.close(fig)
    durations = [frame_ms * 2] * len(frames)
    durations[-1] = hold_ms
    _save_gif(frames, path, durations)
    return path


def render_gifs(run_dir, out_dir=None, verbose=True, max_frames=120):
    run = RunData(run_dir)
    out = Path(out_dir or Path(run_dir) / "figures")
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn, kw in (("evolution.gif", evolution_gif, {"max_frames": max_frames}), ("best.gif", best_gif, {})):
        p = out / name
        fn(run, p, **kw)
        written.append(p)
        if verbose:
            print(f"  wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")
    return written
