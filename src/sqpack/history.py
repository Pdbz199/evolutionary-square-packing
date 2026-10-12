"""Recording and persisting everything a run did, for the visualisations.

A run directory contains:

* ``history.npz`` - per-generation arrays (populations, fitness, ids, operator
  statistics, best-of-generation packings, ...);
* ``run.json``    - the configuration, final result and validation report;
* ``best.json``   - the final packing in container coordinates, easy to reuse.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .operators import MUTATIONS


@dataclass
class History:
    n: int
    pop_size: int
    config: dict
    store_every: int = 1
    # per generation (index = generation)
    best: list = field(default_factory=list)
    median: list = field(default_factory=list)
    worst: list = field(default_factory=list)
    mean: list = field(default_factory=list)
    diversity: list = field(default_factory=list)
    seconds: list = field(default_factory=list)
    fits: list = field(default_factory=list)  # (P,) sorted best-first
    eps: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    best_cfg: list = field(default_factory=list)
    op_made: list = field(default_factory=list)  # (2, N_MUT): [crossover?, mutation]
    op_kept: list = field(default_factory=list)
    op_record: list = field(default_factory=list)
    island: list = field(default_factory=list)  # (P,) island of each individual
    # stored populations (every ``store_every`` generations, and the last one)
    pop_gens: list = field(default_factory=list)
    pops: list = field(default_factory=list)
    # birth register, indexed by individual id
    birth_gen: list = field(default_factory=list)
    birth_parents: list = field(default_factory=list)
    birth_op: list = field(default_factory=list)
    birth_side: list = field(default_factory=list)
    final: dict = field(default_factory=dict)

    @property
    def generations(self):
        return len(self.best) - 1

    def register_births(self, gen, parent_ids, ops, sides):
        for (a, b), op, s in zip(parent_ids, ops, sides):
            self.birth_gen.append(gen)
            self.birth_parents.append((int(a), int(b)))
            self.birth_op.append(int(op))
            self.birth_side.append(float(s))

    def record(self, gen, pop, fit, eps, ids, diversity, seconds, op_made=None, op_kept=None, op_record=None, force_store=False, island=None):
        finite = fit[np.isfinite(fit)]
        self.best.append(float(fit[0]))
        self.median.append(float(np.median(finite)))
        self.worst.append(float(finite.max()))
        self.mean.append(float(finite.mean()))
        self.diversity.append(float(diversity))
        self.seconds.append(float(seconds))
        self.fits.append(np.asarray(fit, dtype=np.float64).copy())
        self.eps.append(np.asarray(eps, dtype=np.float64).copy())
        self.ids.append(np.asarray(ids, dtype=np.int64).copy())
        self.best_cfg.append(np.asarray(pop[0], dtype=np.float64).copy())
        self.island.append(np.zeros(len(fit), dtype=np.int16) if island is None else np.asarray(island, dtype=np.int16))
        zeros = np.zeros((2, len(MUTATIONS)), dtype=np.int32)
        self.op_made.append(zeros if op_made is None else op_made)
        self.op_kept.append(zeros if op_kept is None else op_kept)
        self.op_record.append(zeros if op_record is None else op_record)
        if force_store or gen % self.store_every == 0:
            self.pop_gens.append(gen)
            self.pops.append(np.asarray(pop, dtype=np.float32).copy())

    # ------------------------------------------------------------------ io

    def arrays(self):
        return {
            "best": np.array(self.best),
            "median": np.array(self.median),
            "worst": np.array(self.worst),
            "mean": np.array(self.mean),
            "diversity": np.array(self.diversity),
            "seconds": np.array(self.seconds),
            "fits": np.array(self.fits),
            "eps": np.array(self.eps),
            "ids": np.array(self.ids),
            "best_cfg": np.array(self.best_cfg),
            "op_made": np.array(self.op_made),
            "op_kept": np.array(self.op_kept),
            "op_record": np.array(self.op_record),
            "island": np.array(self.island),
            "pop_gens": np.array(self.pop_gens, dtype=np.int64),
            "pops": np.array(self.pops, dtype=np.float32),
            "birth_gen": np.array(self.birth_gen, dtype=np.int32),
            "birth_parents": np.array(self.birth_parents, dtype=np.int64).reshape(-1, 2),
            "birth_op": np.array(self.birth_op, dtype=np.int16),
            "birth_side": np.array(self.birth_side),
        }

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "history.npz", **self.arrays())
        meta = {"n": self.n, "pop_size": self.pop_size, "store_every": self.store_every, "config": self.config, "final": self.final, "mutations": list(MUTATIONS)}
        (out / "run.json").write_text(json.dumps(meta, indent=2))
        if self.final.get("squares") is not None:
            best = {k: self.final[k] for k in ("n", "side", "best_known", "gap_percent", "valid", "squares")}
            (out / "best.json").write_text(json.dumps(best, indent=2))
        return out


class RunData:
    """Read-only view of a saved run (what the visualisations consume)."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        meta = json.loads((self.dir / "run.json").read_text())
        self.n = meta["n"]
        self.pop_size = meta["pop_size"]
        self.config = meta["config"]
        self.final = meta["final"]
        self.mutations = meta.get("mutations", list(MUTATIONS))
        with np.load(self.dir / "history.npz") as z:
            for k in z.files:
                setattr(self, k, z[k])
        self.generations = len(self.best) - 1

    def population(self, gen):
        """Stored population closest to (and not after) ``gen``; returns (gen, pop, fits)."""
        k = int(np.searchsorted(self.pop_gens, gen, side="right") - 1)
        k = max(k, 0)
        g = int(self.pop_gens[k])
        return g, self.pops[k].astype(np.float64), self.fits[g]

    def lineage(self):
        """Ancestry of the final best individual, following each child's primary parent.

        Returns a list of (birth_gen, id, op_code, side) from the initial
        population up to the final best.
        """
        cur = int(self.ids[-1][0])
        chain = []
        while cur >= 0:
            chain.append((int(self.birth_gen[cur]), cur, int(self.birth_op[cur]), float(self.birth_side[cur])))
            cur = int(self.birth_parents[cur][0])
        return chain[::-1]

    def config_of(self, ident, gen):
        """Configuration of individual ``ident`` in stored generation ``gen`` (or None)."""
        k = np.searchsorted(self.pop_gens, gen)
        if k >= len(self.pop_gens) or self.pop_gens[k] != gen:
            return None
        where = np.nonzero(self.ids[gen] == ident)[0]
        if len(where) == 0:
            return None
        return self.pops[k][where[0]].astype(np.float64)
