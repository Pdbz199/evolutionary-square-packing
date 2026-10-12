"""The memetic genetic algorithm.

Each generation:

1. **Selection** - parents are chosen by tournament on fitness (container side).
2. **Variation** - geometric cut crossover and/or one of several mutations
   (``operators.vary``).
3. **Development** - every child is relaxed, repaired into a valid packing and
   compressed (``relax.develop_batch``, run in parallel). Its fitness is the side
   of the smallest square that contains it, so lower is better.
4. **Survival** - parents and children compete together ((mu + lambda)
   selection); near-duplicate packings are cleared so the population does not
   collapse onto a single configuration too early.

Optionally the population is split into islands that evolve separately, pass
their best individual around a ring every few generations, and reseed their
worse half with fresh random packings when they stagnate.
"""

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import records
from .diversity import d4_perms, mean_pairwise_distance, rasters_of, select_survivors
from .history import History
from .operators import MUTATIONS, N_MUT, lattice_population, random_population, seed_rng, vary
from .relax import develop_batch, polish
from .validate import to_container_frame, validate

DEFAULT_OP_WEIGHTS = {"jitter": 2.0, "reinsert": 3.0, "rotate": 1.5, "cluster-turn": 1.5, "row-shift": 1.0}


@dataclass
class GAConfig:
    n: int = 10
    pop_size: int = 96
    offspring: int | None = None  # children per generation; defaults to pop_size
    generations: int = 600
    p_crossover: float = 0.5
    p_mutation: float = 0.8  # chance a crossover child is also mutated (clones always are)
    tournament: int = 3
    op_weights: dict = field(default_factory=lambda: dict(DEFAULT_OP_WEIGHTS))
    relax_steps: int = 300
    compress_attempts: int = 3
    wall_weight: float = 1.0
    eps0: float = 0.02
    init: str = "random"  # "random" or "lattice"
    init_box: float | None = None
    clearing: float = 0.06  # near-duplicate radius: share of raster cells that must differ
    islands: int = 4  # independent sub-populations (at least 2 individuals each)
    migrate_every: int = 25  # generations between ring migrations of each island's best
    restart_after: int = 60  # reseed an island's worse half after this many stagnant generations (0 = never)
    improve_tol: float = 1e-4  # relative gain that counts as progress for patience and restarts
    tiebreak: float = 0.0  # weight of squeeze resistance in the selection key (0 = off)
    probe: float = 0.01  # relative squeeze used to measure squeeze resistance
    raster_res: int = 24
    seed: int = 0
    patience: int | None = None  # stop after this many generations without improvement
    target_gap: float | None = None  # stop once within this relative gap of the record
    time_limit: float | None = None  # seconds
    store_every: int | None = None  # population snapshot stride (None = automatic)
    polish_rounds: int = 30

    def weights_array(self):
        w = np.zeros(N_MUT)
        for k, name in enumerate(MUTATIONS[1:], start=1):
            w[k] = float(self.op_weights.get(name, 0.0))
        if w.sum() <= 0:
            raise ValueError("at least one mutation must have a positive weight")
        return w


@dataclass
class RunResult:
    config: GAConfig
    best_cfg: np.ndarray
    side: float
    history: History
    validation: dict

    @property
    def best_known(self):
        return records.best_known(self.config.n)


@dataclass
class Pool:
    """Aligned per-individual arrays of a (sub)population."""

    pop: np.ndarray  # (P, n, 3) genomes
    fit: np.ndarray  # (P,) container side, the fitness (lower is better)
    eps: np.ndarray  # (P,) self-adapted squeeze factor
    ids: np.ndarray  # (P,) unique individual ids
    res: np.ndarray  # (P,) squeeze resistance, the optional tie-break
    rasters: np.ndarray  # (P, res*res) fingerprints for clearing / diversity

    FIELDS = ("pop", "fit", "eps", "ids", "res", "rasters")

    def take(self, idx):
        return Pool(*(getattr(self, f)[idx] for f in self.FIELDS))

    def put(self, idx, other):
        for f in self.FIELDS:
            getattr(self, f)[idx] = getattr(other, f)

    @classmethod
    def concat(cls, a, b):
        return cls(*(np.concatenate([getattr(a, f), getattr(b, f)]) for f in cls.FIELDS))


def _auto_store_every(cfg):
    # keep stored populations under ~6M floats (~24 MB as float32)
    per_gen = cfg.pop_size * cfg.n * 3
    budget = 6_000_000
    return max(1, math.ceil(per_gen * (cfg.generations + 1) / budget))


def _fmt_gap(side, rec):
    if rec is None:
        return ""
    return f" (record {rec:.4f}, gap {100 * (side - rec) / rec:+.2f}%)"


def run(cfg: GAConfig, verbose=True, print_every=10, callback=None) -> RunResult:
    """Run the GA described by ``cfg`` and return the polished, validated best packing."""
    n, P = cfg.n, cfg.pop_size
    lam = cfg.offspring or P
    if n < 1 or P < 2:
        raise ValueError("need n >= 1 and pop_size >= 2")
    islands = max(1, min(cfg.islands, P // 2))
    rec = records.best_known(n)
    store_every = cfg.store_every or _auto_store_every(cfg)
    hist = History(n=n, pop_size=P, config=asdict(cfg), store_every=store_every)
    weights = cfg.weights_array()
    perms = d4_perms(cfg.raster_res)
    t_start = time.time()

    lam = max(lam, islands)

    def _split(total):
        bounds = np.cumsum([0] + [len(a) for a in np.array_split(np.arange(total), islands)])
        return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]

    slots = _split(P)  # island k owns pool[slots[k]] and children[cslots[k]]
    cslots = _split(lam)
    island_of = np.concatenate([np.full(sl.stop - sl.start, k) for k, sl in enumerate(slots)])
    init_side = max(cfg.init_box or math.sqrt(n) * 1.15, math.ceil(math.sqrt(n)))
    probe = cfg.probe if cfg.tiebreak > 0 else 0.0
    next_id = 0
    evaluations = 0

    def key(pool):
        """Selection key: the side, plus an optional squeeze-resistance tie-break."""
        return pool.fit + cfg.tiebreak * pool.res

    def develop(genomes, eps, targets, ids):
        nonlocal evaluations
        k = len(genomes)
        evaluations += k
        fit, eps_out, res = np.empty(k), np.empty(k), np.empty(k)
        develop_batch(genomes, eps, targets, cfg.compress_attempts, cfg.relax_steps, cfg.wall_weight, fit, eps_out, probe, res)
        return Pool(genomes, fit, eps_out, ids, res, rasters_of(genomes, cfg.raster_res))

    def fresh(count, gen):
        """``count`` new random individuals, developed into valid packings and registered."""
        nonlocal next_id
        genomes = np.empty((count, n, 3))
        if cfg.init == "lattice":
            lattice_population(genomes, 0.08)
        elif cfg.init == "random":
            random_population(genomes, cfg.init_box or math.sqrt(n) * 1.15)
        else:
            raise ValueError(f"unknown init {cfg.init!r}")
        ids = np.arange(next_id, next_id + count, dtype=np.int64)
        next_id += count
        pool = develop(genomes, np.full(count, cfg.eps0), np.full(count, init_side), ids)
        hist.register_births(gen, [(-1, -1)] * count, [-1] * count, pool.fit)
        return pool

    def sort_island(pool, sl):
        o = np.argsort(key(pool)[sl], kind="stable") + sl.start
        pool.put(sl, pool.take(o))

    def record(gen, pool, *stats, force=False):
        o = np.argsort(pool.fit, kind="stable")
        div = mean_pairwise_distance(pool.rasters, perms)
        hist.record(gen, pool.pop[o], pool.fit[o], pool.eps[o], pool.ids[o], div, time.time() - t_start, *stats, force_store=force, island=island_of[o])
        return div

    # ---------------------------------------------------------------- generation 0
    seed_rng(cfg.seed)
    pool = fresh(P, 0)
    for sl in slots:
        sort_island(pool, sl)
    record(0, pool, force=True)
    if verbose:
        isl = f" islands={islands}" if islands > 1 else ""
        print(f"[sqpack] n={n} pop={P}{isl} offspring={lam} generations={cfg.generations} seed={cfg.seed}")
        print(f"gen    0 | best {pool.fit.min():.5f}{_fmt_gap(pool.fit.min(), rec)} | median {np.median(pool.fit):.4f}")

    # ---------------------------------------------------------------- main loop
    children = np.empty((lam, n, 3))
    ch_eps = np.empty(lam)
    targets = np.empty(lam)
    parents = np.empty((lam, 2), dtype=np.int64)
    ops = np.empty(lam, dtype=np.int64)
    best_seen = pool.fit.min()
    last_improve = 0
    island_best = np.array([pool.fit[sl].min() for sl in slots])
    island_last = np.zeros(islands, dtype=int)
    stop_reason = "generations"
    gen = 0
    for gen in range(1, cfg.generations + 1):
        t0 = time.time()
        sel = key(pool)
        for k in range(islands):
            sl, cl = slots[k], cslots[k]
            vary(pool.pop[sl], pool.fit[sl], sel[sl], pool.eps[sl], cfg.p_crossover, cfg.p_mutation, weights, cfg.tournament, children[cl], ch_eps[cl], targets[cl], parents[cl], ops[cl])
            parents[cl] = np.where(parents[cl] >= 0, parents[cl] + sl.start, -1)
        kids = develop(children.copy(), ch_eps, targets, np.arange(next_id, next_id + lam, dtype=np.int64))
        next_id += lam
        parent_ids = np.where(parents >= 0, pool.ids[np.clip(parents, 0, None)], -1)
        hist.register_births(gen, parent_ids, ops, kids.fit)

        # (mu + lambda) survival with clearing, separately on every island;
        # the packing with the smallest true side always survives
        child_kept = np.zeros(lam, dtype=bool)
        for k in range(islands):
            sl, cl = slots[k], cslots[k]
            cand = Pool.concat(pool.take(sl), kids.take(cl))
            ck = key(cand)
            m = sl.stop - sl.start
            keep = select_survivors(ck, cand.rasters, perms, m, cfg.clearing)
            b = int(np.argmin(cand.fit))
            if b not in keep:
                keep[-1] = b
                keep = keep[np.argsort(ck[keep], kind="stable")]
            child_kept[cl.start + keep[keep >= m] - m] = True
            pool.put(sl, cand.take(keep))

        # operator bookkeeping: [crossover?, mutation id]
        op_made = np.zeros((2, N_MUT), dtype=np.int32)
        op_kept = np.zeros((2, N_MUT), dtype=np.int32)
        op_rec = np.zeros((2, N_MUT), dtype=np.int32)
        for c in range(lam):
            xo, mut = ops[c] & 1, ops[c] >> 1
            op_made[xo, mut] += 1
            op_kept[xo, mut] += child_kept[c]
            op_rec[xo, mut] += kids.fit[c] < best_seen - 1e-12

        if islands > 1:
            # ring migration: each island's best replaces the next island's worst
            if cfg.migrate_every and gen % cfg.migrate_every == 0:
                migrants = [pool.take(slice(sl.start, sl.start + 1)) for sl in slots]
                for k in range(islands):
                    dst = slots[(k + 1) % islands].stop - 1
                    pool.put(slice(dst, dst + 1), migrants[k])
                for sl in slots:  # re-rank so a restart below cannot overwrite a fresh migrant
                    sort_island(pool, sl)
            # restarts: an island that stopped improving gets a fresh worse half
            for k in range(islands):
                sl = slots[k]
                ib = pool.fit[sl].min()
                if ib < island_best[k] * (1 - cfg.improve_tol):
                    island_best[k] = ib
                    island_last[k] = gen
                elif cfg.restart_after and gen - island_last[k] >= cfg.restart_after:
                    half = (sl.stop - sl.start) // 2
                    if half:
                        pool.put(slice(sl.stop - half, sl.stop), fresh(half, gen))
                    island_last[k] = gen
                island_best[k] = min(island_best[k], ib)
            for sl in slots:
                sort_island(pool, sl)

        cur_best = pool.fit.min()
        if cur_best < best_seen * (1 - cfg.improve_tol):
            last_improve = gen
        best_seen = min(best_seen, cur_best)
        elapsed = time.time() - t_start
        done = gen == cfg.generations
        if cfg.patience is not None and gen - last_improve >= cfg.patience:
            stop_reason, done = "patience", True
        if cfg.target_gap is not None and rec is not None and cur_best <= rec * (1 + cfg.target_gap):
            stop_reason, done = "target", True
        if cfg.time_limit is not None and elapsed >= cfg.time_limit:
            stop_reason, done = "time limit", True
        div = record(gen, pool, op_made, op_kept, op_rec, force=done)
        if callback is not None:
            callback(gen, pool.pop, pool.fit)
        if verbose and (gen % print_every == 0 or done):
            print(f"gen {gen:4d} | best {cur_best:.5f}{_fmt_gap(cur_best, rec)} | median {np.median(pool.fit):.4f} | diversity {div:.3f} | {time.time() - t0:.2f}s/gen")
        if done:
            break

    order = np.argsort(pool.fit, kind="stable")
    pop, fit, eps = pool.pop[order], pool.fit[order], pool.eps[order]

    # ---------------------------------------------------------------- polish + validate
    best = pop[0].copy()
    side_before = float(fit[0])
    side_after, _ = polish(best, max(float(eps[0]), 1e-3), cfg.polish_rounds, 10, cfg.relax_steps * 4, cfg.wall_weight)
    if side_after > side_before:  # polish never makes things worse, but be safe
        best, side_after = pop[0].copy(), side_before
    v = validate(best)
    frame, _ = to_container_frame(best)
    hist.final = {
        "n": n,
        "side": v.side,
        "side_before_polish": side_before,
        "best_known": rec,
        "best_known_note": records.describe(n),
        "lower_bound": records.lower_bound(n),
        "trivial_side": records.trivial_upper_bound(n),
        "gap_percent": None if rec is None else 100 * (v.side - rec) / rec,
        "valid": v.valid,
        "max_overlap": v.max_overlap,
        "generations_run": gen,
        "stop_reason": stop_reason,
        "seconds": time.time() - t_start,
        "evaluations": int(evaluations),
        "squares": frame.tolist(),
    }
    if verbose:
        status = "valid" if v.valid else f"INVALID (max overlap {v.max_overlap:.2e})"
        print(f"final  | side {v.side:.6f}{_fmt_gap(v.side, rec)} | {status} | stop: {stop_reason} | {hist.final['seconds']:.1f}s")
    return RunResult(config=cfg, best_cfg=best, side=v.side, history=hist, validation=v.as_dict())
