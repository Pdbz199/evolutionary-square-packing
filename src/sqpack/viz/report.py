"""Self-contained interactive HTML report of a run."""

import base64
import json
from pathlib import Path

import numpy as np

from ..history import RunData
from .draw import aligned, bounding_square, sample_generations

TEMPLATE = Path(__file__).with_name("report_template.html")
SCALE_XY = 1000.0  # positions are stored as int16 thousandths of a unit
SCALE_T = 10000.0  # angles as int16 ten-thousandths of a radian


def _wrap(t):
    q = np.pi / 2
    return t - q * np.floor((t + q / 2) / q)


def _encode(pop):
    """Base64 int16 encoding of an aligned (P, N, 3) population."""
    q = np.empty(pop.shape, dtype=np.int16)
    q[..., 0] = np.clip(np.round(pop[..., 0] * SCALE_XY), -32767, 32767)
    q[..., 1] = np.clip(np.round(pop[..., 1] * SCALE_XY), -32767, 32767)
    q[..., 2] = np.round(_wrap(pop[..., 2]) * SCALE_T)
    return base64.b64encode(q.astype("<i2").tobytes()).decode("ascii")


def _rounded(a, d):
    return [None if not np.isfinite(v) else round(float(v), d) for v in a]


def build_data(run: RunData, max_frames=300, figures=()):
    ref = aligned(np.array(run.final["squares"]), None) if run.final.get("squares") else aligned(run.best_cfg[-1], None)
    stored = [int(g) for g in run.pop_gens]
    if len(stored) > max_frames:
        wanted = sample_generations(run.generations, max_frames, log_bias=0.5)
        stored_arr = np.array(stored)
        stored = sorted({stored[int(np.argmin(np.abs(stored_arr - g)))] for g in wanted})
    frames = []
    best_max = 0.0
    for g in stored:
        _, pop, fits = run.population(g)
        al = np.stack([aligned(c, ref) for c in pop])
        best_max = max(best_max, bounding_square(al[0])[2])
        ids = run.ids[g]
        frames.append(
            {
                "gen": g,
                "fit": _rounded(fits, 6),
                "eps": _rounded(run.eps[g], 5),
                "ids": [int(i) for i in ids],
                "born": [int(run.birth_gen[i]) for i in ids],
                "op": [int(run.birth_op[i]) for i in ids],
                "island": [int(v) for v in run.island[g]] if getattr(run, "island", np.zeros(0)).size else [0] * len(ids),
                "cfg": _encode(al),
            }
        )
    lineage = []
    chain = run.lineage()
    last_side = np.inf
    for k, (gen, ident, op, side) in enumerate(chain):
        if side >= last_side * (1 - 2e-5) and k != len(chain) - 1:
            continue
        cfg = run.config_of(ident, gen)
        if cfg is None:
            continue
        last_side = side
        lineage.append({"gen": gen, "id": ident, "op": op, "side": round(side, 6), "cfg": np.round(aligned(cfg, ref), 4).ravel().tolist()})
    final = {k: v for k, v in run.final.items() if k != "squares"}
    return {
        "n": run.n,
        "pop": run.pop_size,
        "islands": int(run.config.get("islands") or 1),
        "scale_xy": SCALE_XY,
        "scale_t": SCALE_T,
        "mutations": list(run.mutations),
        "final": final,
        "config": run.config,
        "stats": {k: _rounded(getattr(run, k), 6) for k in ("best", "median", "worst", "diversity")},
        "frames": frames,
        "best_max_side": best_max,
        "lineage": lineage,
        "figures": list(figures),
    }


def render_report(run_dir, out_path=None, max_frames=300, verbose=True):
    run = RunData(run_dir)
    out = Path(out_path or Path(run_dir) / "figures" / "report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    figs = sorted(p.name for p in out.parent.glob("*.png")) + sorted(p.name for p in out.parent.glob("*.gif"))
    data = build_data(run, max_frames=max_frames, figures=figs)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__DATA__", payload)
    out.write_text(html)
    if verbose:
        print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
