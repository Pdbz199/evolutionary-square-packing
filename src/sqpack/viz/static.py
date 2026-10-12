"""Static PNG figures of a finished run."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from .. import records  # noqa: E402
from ..history import RunData  # noqa: E402
from . import style  # noqa: E402
from .draw import aligned, bounding_square, draw_packing, fmt_side, sample_generations, square_polys, tilt_colorbar  # noqa: E402


def _reference(run: RunData):
    """Final best packing, centred: every figure orients packings to match it."""
    return aligned(np.array(run.final["squares"]) if run.final.get("squares") else run.best_cfg[-1], None)


def _stored_gens(run, k):
    """Up to ``k`` stored generations, denser early."""
    wanted = sample_generations(run.generations, k)
    stored = run.pop_gens
    out = []
    for g in wanted:
        j = int(np.argmin(np.abs(stored - g)))
        if int(stored[j]) not in out:
            out.append(int(stored[j]))
    return sorted(out)


def _record_text(run):
    rec = run.final.get("best_known")
    return f"best known {rec:.4f}" if rec is not None else "no tabulated record"


# ---------------------------------------------------------------------- figures


def population_grid(run: RunData, path, rows=8, cols=10):
    """Rows = generations, columns = individuals from best (left) to worst (right)."""
    gens = _stored_gens(run, rows)
    ref = _reference(run)
    ranks = np.unique(np.linspace(0, run.pop_size - 1, cols).round().astype(int))
    cols = len(ranks)
    cells = {}
    max_side = 0.0
    for g in gens:
        _, pop, fits = run.population(g)
        for c, r in enumerate(ranks):
            cfg = aligned(pop[r], ref)
            cells[g, c] = (cfg, fits[r])
            max_side = max(max_side, bounding_square(cfg)[2])
    half = 0.5 * max_side * 1.02
    cell = 1.05
    left = 2.1
    fig = plt.figure(figsize=(left + cols * cell, 1.25 + len(gens) * (cell + 0.18) + 0.5))
    W, Hh = fig.get_size_inches()
    top = 1.15
    for ri, g in enumerate(gens):
        y = Hh - top - (ri + 1) * (cell + 0.18)
        for c in range(cols):
            ax = fig.add_axes([(left + c * cell) / W, y / Hh, (cell * 0.94) / W, (cell * 0.94) / Hh])
            cfg, s = cells[g, c]
            draw_packing(ax, cfg, half_view=half, edge_width=0.4)
            ax.text(0.5, -0.04, fmt_side(s), transform=ax.transAxes, ha="center", va="top", fontsize=7, color=style.MUTED)
            if ri == 0:
                label = "best" if ranks[c] == 0 else ("worst" if ranks[c] == run.pop_size - 1 else f"rank {ranks[c] + 1}")
                ax.text(0.5, 1.06, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=style.INK_2)
        fig.text(0.15 / W, (y + cell * 0.62) / Hh, f"generation {g}", ha="left", va="center", fontsize=10, fontweight="semibold")
        fig.text(
            0.15 / W,
            (y + cell * 0.28) / Hh,
            f"best {run.best[g]:.4f}\ndiversity {run.diversity[g]:.3f}",
            ha="left",
            va="center",
            fontsize=8,
            color=style.INK_2,
            linespacing=1.3,
        )
    fig.text(0.15 / W, (Hh - 0.25) / Hh, f"How the population converges · N = {run.n} unit squares", fontsize=14, fontweight="semibold", va="top")
    fig.text(
        0.15 / W,
        (Hh - 0.62) / Hh,
        "Each row is one generation. Columns sample the population from best to worst (number = container side, lower is better).\n"
        "Every packing is drawn at the same scale and turned to match the final winner, so rows that look alike mean the population agrees.",
        fontsize=9,
        color=style.INK_2,
        va="top",
        linespacing=1.4,
    )
    cax = fig.add_axes([(W - 3.2) / W, 0.25 / Hh, 2.6 / W, 0.12 / Hh])
    tilt_colorbar(fig, cax)
    fig.savefig(path)
    plt.close(fig)


def fitness_curves(run: RunData, path):
    """Best / median / worst container side per generation, plus population diversity."""
    g = np.arange(run.generations + 1)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.25], "hspace": 0.25})
    ax.fill_between(g, run.best, run.worst, color=style.BEST, alpha=0.08, linewidth=0, zorder=1)
    ax.plot(g, run.worst, color=style.WORST, label="worst in population", zorder=2)
    ax.plot(g, run.median, color=style.MEDIAN, label="median", zorder=3)
    ax.plot(g, run.best, color=style.BEST, label="best", zorder=4)
    rec = run.final.get("best_known")
    lb = records.lower_bound(run.n)
    trivial = records.trivial_upper_bound(run.n)
    core_lo = min(run.best.min(), rec if rec is not None else np.inf)
    hi = max(run.median[0], run.best[0])
    refs = []
    if rec is not None:
        refs.append((rec, f"best known {rec:.4f}"))
    if trivial < hi * 1.01 and (rec is None or abs(trivial - rec) > 1e-9):
        refs.append((trivial, f"plain grid {trivial:.0f}"))
    if lb >= core_lo - 0.35 * (hi - core_lo):
        refs.append((lb, f"area bound √N = {lb:.4f}"))
    lo = min([core_lo] + [v for v, _ in refs])
    span = max(hi - lo, 1e-3)
    for v, label in refs:
        ax.axhline(v, color=style.MUTED, linewidth=0.9, zorder=1.5)
        ax.annotate(label, (0, v), xytext=(4, 2), textcoords="offset points", va="bottom", ha="left", fontsize=8, color=style.INK_2)
    ax.set_ylim(lo - 0.06 * span, hi + 0.04 * span)
    ax.plot([g[-1]], [run.best[-1]], "o", color=style.BEST, markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=2, zorder=5, clip_on=False)
    ax.annotate(f"{run.best[-1]:.5f}", (g[-1], run.best[-1]), xytext=(-6, 8), textcoords="offset points", ha="right", va="bottom", fontsize=9, color=style.INK)
    ax.set_ylabel("container side (lower is better)")
    final = run.final.get("side")
    sub = f" · final after polish {final:.5f}" if final is not None else ""
    ax.set_title(f"Fitness over generations · N = {run.n}{sub}", pad=10)
    ax.legend(loc="upper right", ncols=3)
    if run.generations > 40:
        ax.set_xscale("symlog", linthresh=10, linscale=0.6)
        ticks = [t for t in (0, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000) if t <= run.generations]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0, max(1, g[-1]))

    ax2.plot(g, run.diversity, color=style.BEST)
    ax2.set_ylim(0, max(0.05, run.diversity.max() * 1.15))
    ax2.set_ylabel("diversity")
    ax2.set_xlabel("generation" + (" (log scale after 10)" if run.generations > 40 else ""))
    ax2.set_title("Population diversity: share of the container where two packings disagree (mean over pairs)", fontsize=9.5, fontweight="normal", color=style.INK_2, pad=6)
    fig.savefig(path)
    plt.close(fig)


def best_progression(run: RunData, path, max_panels=12):
    """Small multiples of the best packing each time it improved."""
    best = run.best
    events = [0] + [int(i) for i in np.nonzero(np.diff(best) < -1e-9)[0] + 1]
    if events[-1] != run.generations:
        events.append(run.generations)
    if len(events) > max_panels:
        idx = np.unique(np.round(np.geomspace(1, len(events), max_panels)).astype(int) - 1)
        idx = sorted(set(idx.tolist()) | {0, len(events) - 1})
        events = [events[i] for i in idx]
    ref = _reference(run)
    cfgs = [aligned(run.best_cfg[g], ref) for g in events]
    final_side = run.final.get("side")
    polished = final_side is not None and final_side < run.best[-1] - 1e-9
    if polished:
        cfgs.append(ref)
        events.append(None)
    half = 0.5 * max(bounding_square(c)[2] for c in cfgs) * 1.02
    k = len(cfgs)
    ncol = min(k, 6)
    nrow = math.ceil(k / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.35 * nrow + 0.9), squeeze=False)
    prev = None
    for ax, g, cfg in zip(axes.flat, events, cfgs):
        draw_packing(ax, cfg, half_view=half, edge_width=0.6)
        s = final_side if g is None else best[g]
        head = "after final polish" if g is None else f"generation {g}"
        delta = "" if prev is None else (f"  ({s - prev:+.4f})" if abs(s - prev) >= 5e-5 else "  (−<0.0001)")
        ax.set_title(f"{head}\nside {s:.4f}{delta}", fontsize=8.5, fontweight="normal", color=style.INK_2, loc="center")
        prev = s
    for ax in list(axes.flat)[k:]:
        ax.set_axis_off()
    fig.suptitle(f"The best packing each time it improved · N = {run.n} ({_record_text(run)})", x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def _occupancy(cfgs, half, res):
    """Mean occupancy of a set of (aligned) packings on a fixed res x res grid."""
    xs = (np.arange(res) + 0.5) / res * 2 * half - half
    px, py = np.meshgrid(xs, xs)
    pts = np.stack([px.ravel(), py.ravel()], axis=1)
    acc = np.zeros(len(pts))
    for cfg in cfgs:
        d = pts[:, None, :] - cfg[None, :, :2]
        c = np.cos(cfg[:, 2])[None, :]
        s = np.sin(cfg[:, 2])[None, :]
        u = d[..., 0] * c + d[..., 1] * s
        v = -d[..., 0] * s + d[..., 1] * c
        acc += ((np.abs(u) <= 0.5) & (np.abs(v) <= 0.5)).any(axis=1)
    return (acc / len(cfgs)).reshape(res, res)


def consensus_maps(run: RunData, path, k=8, res=120):
    """Population-average occupancy: blurry = disagreement, crisp = convergence."""
    gens = _stored_gens(run, k)
    ref = _reference(run)
    pops = {}
    max_side = 0.0
    for g in gens:
        _, pop, _ = run.population(g)
        al = [aligned(c, ref) for c in pop]
        pops[g] = al
        max_side = max(max_side, max(bounding_square(c)[2] for c in al))
    half = 0.5 * max_side * 1.02
    ncol = min(len(gens), 4)
    nrow = math.ceil(len(gens) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 3.5 * nrow + 1.0), squeeze=False, gridspec_kw={"hspace": 0.35})
    im = None
    for ax, g in zip(axes.flat, gens):
        occ = _occupancy(pops[g], half, res)
        im = ax.imshow(occ, origin="lower", extent=(-half, half, -half, half), cmap=style.SEQ_CMAP, vmin=0, vmax=1, interpolation="bilinear")
        ax.set_axis_off()
        ax.set_title(f"generation {g}\ndiversity {run.diversity[g]:.3f}", fontsize=9, fontweight="normal", color=style.INK_2, loc="center")
    for ax in list(axes.flat)[len(gens):]:
        ax.set_axis_off()
    fig.suptitle(f"Population consensus · N = {run.n}: how often each point is covered across the whole population", x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.1, hspace=0.3, wspace=0.05)
    cax = fig.add_axes([0.35, 0.04, 0.3, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["never covered", "half the population", "always covered"])
    cb.ax.tick_params(labelsize=8, colors=style.INK_2)
    fig.savefig(path)
    plt.close(fig)


def tilt_heatmap(run: RunData, path, bins=15):
    """Distribution of square tilts across the population over generations."""
    gens = run.pop_gens
    mat = np.zeros((bins, len(gens)))
    for j, _ in enumerate(gens):
        t = style.tilt_fraction(run.pops[j][:, :, 2].ravel()) * 45
        h, _ = np.histogram(t, bins=bins, range=(0, 45))
        mat[:, j] = h / max(1, h.sum())
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x_edges = np.append(gens, gens[-1] + max(1, gens[-1] - gens[-2] if len(gens) > 1 else 1))
    norm = matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=max(0.05, mat.max()))
    im = ax.pcolormesh(x_edges, np.linspace(0, 45, bins + 1), mat, cmap=style.SEQ_CMAP, norm=norm, shading="flat")
    ax.grid(False)
    ax.set_yticks([0, 15, 30, 45], ["0°", "15°", "30°", "45°"])
    ax.set_ylabel("tilt")
    ax.set_xlabel("generation")
    ax.set_title(f"Which tilts survive · share of all squares in the population at each tilt (N = {run.n})", pad=10)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=style.INK_2)
    cb.set_label("share of squares (square-root scale)", color=style.INK_2, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _op_names(run):
    return ["crossover only" if m == "none" else m for m in run.mutations]


def operator_stats(run: RunData, path, window=None):
    """Which variation operators produce survivors, and which set new records."""
    made = run.op_made[1:].sum(axis=1)  # (G, N_MUT) summed over crossover flag
    kept = run.op_kept[1:].sum(axis=1)
    recs = run.op_record[1:].sum(axis=1)
    names = _op_names(run)
    G = made.shape[0]
    if G == 0:
        return
    window = window or max(1, G // 40)
    kernel = np.ones(window) / window
    smooth = np.array([np.convolve(kept[:, k], kernel, mode="same") for k in range(kept.shape[1])])
    total = smooth.sum(axis=0)
    share = np.divide(smooth, total, out=np.zeros_like(smooth), where=total > 0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.35})
    x = np.arange(1, G + 1)
    ax.stackplot(x, share, colors=style.SERIES[: len(names)], labels=names, edgecolor=style.SURFACE, linewidth=0.6, alpha=0.9)
    ax.set_xlim(1, max(2, G))
    ax.set_ylim(0, 1)
    ax.set_xlabel("generation")
    ax.set_ylabel("share of surviving children")
    ax.set_title("Where the survivors come from", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=len(names), fontsize=8, handlelength=1.2, columnspacing=1.2)
    ax.grid(False)

    tot_made = made.sum(axis=0)
    tot_kept = kept.sum(axis=0)
    tot_rec = recs.sum(axis=0)
    rate = np.divide(tot_kept, tot_made, out=np.zeros(len(names)), where=tot_made > 0)
    order = np.argsort(rate)
    y = np.arange(len(names))
    ax2.barh(y, rate[order], height=0.55, color=style.BEST)
    ax2.set_yticks(y, [names[i] for i in order])
    for yi, i in zip(y, order):
        extra = f"  {rate[i]:.0%} of {tot_made[i]:,}"
        if tot_rec[i]:
            extra += f" · {tot_rec[i]} new best"
        ax2.text(rate[i], yi, extra, va="center", fontsize=8, color=style.INK_2)
    ax2.set_xlim(0, max(0.05, rate.max()) * 1.9)
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_title("Survival rate by operator", pad=10)
    ax2.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def lineage(run: RunData, path, max_panels=10):
    """The winner's ancestry: a chain of primary parents back to generation 0."""
    chain = run.lineage()
    items = []
    for gen, ident, op, side in chain:
        cfg = run.config_of(ident, gen)
        if cfg is not None:
            items.append((gen, op, side, cfg))
    if len(items) < 2:
        return False
    # keep the ancestors where the lineage got better, then thin evenly
    keep = [items[0]]
    for it in items[1:]:
        if it[2] < keep[-1][2] * (1 - 2e-5):
            keep.append(it)
    if keep[-1] is not items[-1]:
        keep.append(items[-1])
    if len(keep) > max_panels:
        idx = np.unique(np.linspace(0, len(keep) - 1, max_panels).round().astype(int))
        keep = [keep[i] for i in idx]
    ref = _reference(run)
    cfgs = [aligned(it[3], ref) for it in keep]
    half = 0.5 * max(bounding_square(c)[2] for c in cfgs) * 1.02
    names = _op_names(run)
    k = len(keep)
    ncol = min(k, 5)
    nrow = math.ceil(k / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.9 * nrow + 0.8), squeeze=False, gridspec_kw={"hspace": 0.45})
    for ax, (gen, op, side, _), cfg in zip(axes.flat, keep, cfgs):
        draw_packing(ax, cfg, half_view=half, edge_width=0.6)
        if op < 0:
            how = "random start"
        else:
            how = ("crossover + " if op & 1 else "") + names[op >> 1]
            how = how.replace("crossover + crossover only", "crossover")
        ax.set_title(f"gen {gen} · {how}\nside {side:.4f}", fontsize=8.5, fontweight="normal", color=style.INK_2, loc="center")
    for ax in list(axes.flat)[k:]:
        ax.set_axis_off()
    fig.suptitle(
        f"Ancestry of the winner · N = {run.n}: each packing is an ancestor of the next ({len(chain)} generations of descent)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84 if nrow == 1 else 0.9, bottom=0.02, wspace=0.08)
    fig.savefig(path)
    plt.close(fig)
    return True


def islands_chart(run: RunData, path):
    """Best side on each island over time, with restarts marked (island runs only)."""
    if not hasattr(run, "island") or run.island.size == 0 or int(run.island.max()) < 1:
        return False
    k = int(run.island.max()) + 1
    G = run.generations
    best = np.full((k, G + 1), np.nan)
    for g in range(G + 1):
        for i in range(k):
            sel = run.fits[g][run.island[g] == i]
            if sel.size:
                best[i, g] = sel.min()
    restarts = np.zeros(G + 1)
    gens_born = run.birth_gen[run.birth_op < 0]
    np.add.at(restarts, gens_born[gens_born > 0], 1)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(G + 1)
    if k <= 4:
        ends = best[:, -1]
        spread_ok = np.min(np.diff(np.sort(ends))) > 0.04 * (np.nanmax(best) - np.nanmin(best) + 1e-9) if k > 1 else True
        for i in range(k):
            ax.plot(x, best[i], color=style.SERIES[i], linewidth=1.8, label=f"island {i + 1}")
            if spread_ok:  # direct labels only when the line ends do not collide
                ax.annotate(f"island {i + 1}", (x[-1], best[i, -1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=style.INK_2)
    else:
        for i in range(k):
            ax.plot(x, best[i], color=style.AXIS, linewidth=1.2, label="one island" if i == 0 else None)
        ax.plot(x, np.nanmin(best, axis=0), color=style.BEST, linewidth=2, label="best of all islands")
    rec = run.final.get("best_known")
    if rec is not None:
        ax.axhline(rec, color=style.MUTED, linewidth=0.9)
        ax.annotate(f"best known {rec:.4f}", (0, rec), xytext=(4, 2), textcoords="offset points", fontsize=8, color=style.INK_2)
    lo = np.nanmin(best)
    lo = min(lo, rec) if rec is not None else lo
    hi = np.nanquantile(best[:, : max(2, G // 20)], 0.5)
    ax.set_ylim(lo - 0.05 * (hi - lo + 1e-3), hi + 0.02 * (hi - lo + 1e-3))
    ymin = ax.get_ylim()[0]
    rg = np.nonzero(restarts)[0]
    if len(rg):
        ax.plot(rg, np.full(len(rg), ymin), "|", color=style.INK_2, markersize=9, markeredgewidth=1.2, clip_on=False, label="restart (fresh half-island)")
    mig = run.config.get("migrate_every") or 0
    sub = f" · ring migration every {mig} generations" if mig else ""
    ax.set_title(f"Each island's best packing over time · N = {run.n}, {k} islands{sub}", pad=10)
    ax.set_xlabel("generation")
    ax.set_ylabel("container side")
    ax.set_xlim(0, G)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return True


def final_packing(run: RunData, path):
    """The final packing at full size, in container coordinates."""
    cfg = np.array(run.final["squares"])
    s = run.final["side"]
    fig = plt.figure(figsize=(7.2, 8.0))
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.72])
    polys = square_polys(cfg)
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    ax.add_collection(PolyCollection(polys, facecolors=style.tilt_colors(cfg[:, 2]), edgecolors=style.SURFACE, linewidths=1.2, zorder=2))
    ax.add_patch(Rectangle((0, 0), s, s, fill=False, edgecolor=style.INK, linewidth=1.2, zorder=3))
    pad = 0.04 * s
    ax.set_xlim(-pad, s + pad)
    ax.set_ylim(-pad, s + pad)
    ax.set_aspect("equal")
    ticks = np.arange(0, math.floor(s) + 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.spines[["left", "bottom"]].set_visible(False)
    rec = run.final.get("best_known")
    gap = run.final.get("gap_percent")
    fig.text(0.08, 0.965, f"{run.n} unit squares in a square of side {s:.6f}", fontsize=14, fontweight="semibold", va="top")
    lines = [f"Best known: {rec:.6f} ({run.final.get('best_known_note', '')})" if rec is not None else "No tabulated record for this N"]
    if gap is not None:
        lines[0] += f" · gap {gap:+.3f}%"
    lines.append(
        f"Area bound √N = {run.final['lower_bound']:.4f} · plain grid {run.final['trivial_side']:.0f} · "
        f"{'valid: no overlaps' if run.final['valid'] else 'INVALID: max overlap %.2e' % run.final['max_overlap']}"
    )
    lines.append(f"{run.final['generations_run']} generations · {run.final['evaluations']:,} packings evaluated · {run.final['seconds']:.1f} s")
    fig.text(0.08, 0.925, "\n".join(lines), fontsize=9, color=style.INK_2, va="top", linespacing=1.5)
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.015])
    tilt_colorbar(fig, cax)
    fig.savefig(path)
    plt.close(fig)


def render_static(run_dir, out_dir=None, verbose=True):
    """Render every static figure for a run; returns the list of written paths."""
    style.apply()
    run = RunData(run_dir)
    out = Path(out_dir or Path(run_dir) / "figures")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("01_population_grid.png", population_grid),
        ("02_fitness.png", fitness_curves),
        ("03_best_progression.png", best_progression),
        ("04_consensus.png", consensus_maps),
        ("05_tilts.png", tilt_heatmap),
        ("06_operators.png", operator_stats),
        ("07_lineage.png", lineage),
        ("08_final_packing.png", final_packing),
        ("09_islands.png", islands_chart),
    ]
    written = []
    for name, fn in jobs:
        p = out / name
        res = fn(run, p)
        if res is False or not p.exists():
            continue
        written.append(p)
        if verbose:
            print(f"  wrote {p}")
    return written
