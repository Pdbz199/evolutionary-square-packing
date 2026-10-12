import math

import pytest

from sqpack import GAConfig, run
from sqpack.validate import validate


def test_four_squares_reach_two():
    res = run(GAConfig(n=4, pop_size=16, generations=25, seed=0), verbose=False)
    assert res.validation["valid"]
    assert res.side == pytest.approx(2.0, abs=2e-3)


def test_five_squares_find_the_tilted_optimum():
    res = run(GAConfig(n=5, pop_size=32, generations=80, seed=1), verbose=False)
    assert validate(res.best_cfg).valid
    assert res.side < 2.0 + math.sqrt(2) / 2 + 2e-3  # 2.7071..., needs the 45-degree square


def test_history_is_complete(tmp_path):
    res = run(GAConfig(n=6, pop_size=12, generations=10, seed=2), verbose=False)
    h = res.history
    assert h.generations == 10
    assert len(h.pops) == len(h.pop_gens) == 11
    assert all(b <= m <= w for b, m, w in zip(h.best, h.median, h.worst))
    assert all(h.best[i + 1] <= h.best[i] + 1e-12 for i in range(10))  # elitist
    out = h.save(tmp_path / "run")
    assert (out / "history.npz").exists() and (out / "best.json").exists()


def test_stops_on_target():
    res = run(GAConfig(n=4, pop_size=16, generations=200, target_gap=1e-3, seed=0), verbose=False)
    assert res.history.final["stop_reason"] == "target"
    assert res.history.generations < 200


def test_islands_migration_and_restarts():
    cfg = GAConfig(n=6, pop_size=16, islands=4, migrate_every=3, restart_after=4, generations=20, seed=5)
    res = run(cfg, verbose=False)
    h = res.history
    assert res.validation["valid"]
    assert h.final["evaluations"] == len(h.birth_gen)  # every evaluated packing is registered
    restarted = sum(1 for g, op in zip(h.birth_gen, h.birth_op) if g > 0 and op == -1)
    assert restarted > 0  # stagnating islands were reseeded
    islands = h.island[-1]
    assert sorted(set(islands.tolist())) == [0, 1, 2, 3]
    assert all(h.best[i + 1] <= h.best[i] + 1e-12 for i in range(h.generations))  # best never lost


def test_tiebreak_keeps_true_best():
    res = run(GAConfig(n=5, pop_size=12, generations=15, tiebreak=0.5, seed=3), verbose=False)
    h = res.history
    assert all(h.best[i + 1] <= h.best[i] + 1e-12 for i in range(h.generations))


def test_uneven_islands():
    res = run(GAConfig(n=5, pop_size=10, islands=4, generations=6, seed=1), verbose=False)
    assert res.validation["valid"]
    assert sorted(set(res.history.island[-1].tolist())) == [0, 1, 2, 3]
    assert len(res.history.fits[-1]) == 10
