"""Command line interface: ``sqpack run | viz | bench | records``."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from . import records
from .ga import DEFAULT_OP_WEIGHTS, GAConfig, run


def _weights(text):
    w = dict(DEFAULT_OP_WEIGHTS)
    if not text:
        return w
    for part in text.split(","):
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in w:
            raise argparse.ArgumentTypeError(f"unknown mutation {name!r}; choose from {', '.join(w)}")
        w[name] = float(val)
    return w


def _add_ga_args(p):
    d = GAConfig()
    g = p.add_argument_group("evolution")
    g.add_argument("--pop", type=int, default=d.pop_size, help="population size (default %(default)s)")
    g.add_argument("--offspring", type=int, default=None, help="children per generation (default: population size)")
    g.add_argument("--generations", "-g", type=int, default=d.generations, help="maximum generations (default %(default)s)")
    g.add_argument("--seed", type=int, default=d.seed, help="random seed (default %(default)s)")
    g.add_argument("--crossover", type=float, default=d.p_crossover, help="crossover probability (default %(default)s)")
    g.add_argument("--mutation", type=float, default=d.p_mutation, help="chance a crossover child is also mutated (default %(default)s)")
    g.add_argument("--tournament", type=int, default=d.tournament, help="tournament size (default %(default)s)")
    g.add_argument("--weights", type=_weights, default=None, help="mutation weights, e.g. 'jitter=2,reinsert=3,rotate=1.5,cluster-turn=1.5,row-shift=1'")
    g.add_argument("--init", choices=["random", "lattice"], default=d.init, help="initial population (default %(default)s)")
    g.add_argument("--clearing", type=float, default=d.clearing, help="near-duplicate threshold for survival (default %(default)s)")
    g.add_argument("--islands", type=int, default=d.islands, help="independent sub-populations exchanging migrants (default %(default)s)")
    g.add_argument("--migrate-every", type=int, default=d.migrate_every, help="generations between ring migrations (default %(default)s)")
    g.add_argument("--restart-after", type=int, default=d.restart_after, help="reseed an island's worse half after this many stagnant generations; 0 = never (default %(default)s)")
    g.add_argument("--tiebreak", type=float, default=d.tiebreak, help="experimental: weight of squeeze resistance as a selection tie-break; 0 = off (default %(default)s)")
    s = p.add_argument_group("local search")
    s.add_argument("--relax-steps", type=int, default=d.relax_steps, help="FIRE steps per relaxation (default %(default)s)")
    s.add_argument("--attempts", type=int, default=d.compress_attempts, help="compression attempts per child (default %(default)s)")
    s.add_argument("--polish-rounds", type=int, default=d.polish_rounds, help="final polishing rounds (default %(default)s)")
    t = p.add_argument_group("stopping")
    t.add_argument("--patience", type=int, default=None, help="stop after this many generations without improvement")
    t.add_argument("--target-gap", type=float, default=None, help="stop once within this relative gap of the best known side (e.g. 1e-5)")
    t.add_argument("--time-limit", type=float, default=None, help="stop after this many seconds")


def _config_from(args, n, seed=None):
    return GAConfig(
        n=n,
        pop_size=args.pop,
        offspring=args.offspring,
        generations=args.generations,
        p_crossover=args.crossover,
        p_mutation=args.mutation,
        tournament=args.tournament,
        op_weights=args.weights or dict(DEFAULT_OP_WEIGHTS),
        relax_steps=args.relax_steps,
        compress_attempts=args.attempts,
        init=args.init,
        clearing=args.clearing,
        islands=args.islands,
        migrate_every=args.migrate_every,
        restart_after=args.restart_after,
        tiebreak=args.tiebreak,
        seed=args.seed if seed is None else seed,
        patience=args.patience,
        target_gap=args.target_gap,
        time_limit=args.time_limit,
        polish_rounds=args.polish_rounds,
    )


def _default_out(n, seed):
    base = Path("runs") / f"n{n}_seed{seed}"
    if not base.exists():
        return base
    return Path("runs") / f"n{n}_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"


def cmd_run(args):
    cfg = _config_from(args, args.n)
    out = Path(args.out) if args.out else _default_out(args.n, cfg.seed)
    result = run(cfg, verbose=not args.quiet, print_every=args.print_every)
    result.history.save(out)
    print(f"saved run to {out}/ (history.npz, run.json, best.json)")
    if not args.no_viz:
        from .viz import render_all

        print("rendering visuals ...")
        render_all(out, gif=not args.no_gif, html=not args.no_html, max_frames=args.frames)
        print(f"open {out / 'figures' / 'report.html'} in a browser for the interactive report")
    return 0 if result.validation["valid"] else 1


def cmd_viz(args):
    from .viz import render_all

    render_all(args.run_dir, out_dir=args.out, gif=not args.no_gif, html=not args.no_html, max_frames=args.frames)
    return 0


def cmd_records(args):
    ns = args.n or list(range(1, records.MAX_TABULATED + 1))
    print(f"{'n':>4}  {'best known':>12}  {'sqrt(n)':>8}  {'grid':>5}  note")
    for n in ns:
        rec = records.best_known(n)
        rec_s = f"{rec:.6f}" if rec is not None else "-"
        print(f"{n:>4}  {rec_s:>12}  {math.sqrt(n):8.4f}  {records.trivial_upper_bound(n):5.0f}  {records.describe(n)}")
    return 0


def cmd_bench(args):
    rows = []
    for n in args.n:
        for seed in range(args.seeds):
            cfg = _config_from(args, n, seed=seed)
            t0 = time.time()
            res = run(cfg, verbose=False)
            rec = records.best_known(n)
            gap = None if rec is None else 100 * (res.side - rec) / rec
            rows.append({"n": n, "seed": seed, "side": res.side, "best_known": rec, "gap_percent": gap, "valid": res.validation["valid"], "generations": res.history.final["generations_run"], "seconds": time.time() - t0})
            g = "" if gap is None else f"{gap:+.3f}%"
            print(f"n={n:3d} seed={seed}  side {res.side:.6f}  {g:>9}  {'valid' if rows[-1]['valid'] else 'INVALID'}  {rows[-1]['seconds']:.1f}s", flush=True)
    print()
    print("| n | best known | best of seeds | median of seeds | gap (best) | hits (≤0.01%) | mean time |")
    print("|---|---|---|---|---|---|---|")
    for n in args.n:
        rs = [r for r in rows if r["n"] == n]
        sides = sorted(r["side"] for r in rs)
        rec = rs[0]["best_known"]
        hits = sum(1 for r in rs if r["gap_percent"] is not None and r["gap_percent"] <= 0.01)
        gap = "-" if rec is None else f"{100 * (sides[0] - rec) / rec:+.3f}%"
        med = sides[len(sides) // 2]
        print(f"| {n} | {rec if rec is None else f'{rec:.4f}'} | {sides[0]:.4f} | {med:.4f} | {gap} | {hits}/{len(rs)} | {sum(r['seconds'] for r in rs) / len(rs):.1f}s |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="sqpack", description="Evolve packings of N unit squares (rotation allowed) into the smallest square, and visualise the evolution.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run the GA for one N and render all visuals")
    p.add_argument("n", type=int, help="number of unit squares")
    p.add_argument("--out", "-o", help="run directory (default runs/n<N>_seed<SEED>)")
    _add_ga_args(p)
    v = p.add_argument_group("output")
    v.add_argument("--no-viz", action="store_true", help="skip all visuals")
    v.add_argument("--no-gif", action="store_true", help="skip the animated GIFs")
    v.add_argument("--no-html", action="store_true", help="skip the HTML report")
    v.add_argument("--frames", type=int, default=120, help="maximum GIF frames (default %(default)s)")
    v.add_argument("--print-every", type=int, default=10, help="progress line every K generations")
    v.add_argument("--quiet", "-q", action="store_true", help="no progress output")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("viz", help="(re)render visuals for a saved run directory")
    p.add_argument("run_dir")
    p.add_argument("--out", help="output directory (default <run_dir>/figures)")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--frames", type=int, default=120)
    p.set_defaults(func=cmd_viz)

    p = sub.add_parser("bench", help="run several N and seeds without visuals and summarise")
    p.add_argument("n", type=int, nargs="+")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--json", help="also write per-run results to this JSON file")
    _add_ga_args(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("records", help="print the table of best known sides")
    p.add_argument("n", type=int, nargs="*")
    p.set_defaults(func=cmd_records)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
