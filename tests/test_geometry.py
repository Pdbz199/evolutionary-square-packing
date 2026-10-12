import math

import numpy as np
import pytest

from sqpack.geometry import energy_grad, make_feasible, max_depth, pair_depth, repair_scale, side
from sqpack.validate import validate


def test_axis_aligned_overlap_depth():
    assert pair_depth(0, 0, 0, 0.9, 0, 0) == pytest.approx(0.1)
    assert pair_depth(0, 0, 0, 1.0, 0, 0) == pytest.approx(0.0, abs=1e-12)
    assert pair_depth(0, 0, 0, 1.2, 0.3, 0) == 0.0


def test_diamond_touching_square_has_no_overlap():
    # a 45-degree square whose corner just touches the right edge of an axis-aligned one
    d = 0.5 + math.sqrt(2) / 2
    assert pair_depth(0, 0, 0, d, 0, math.pi / 4) == pytest.approx(0.0, abs=1e-12)
    assert pair_depth(0, 0, 0, d - 0.05, 0, math.pi / 4) == pytest.approx(0.05)


def test_quarter_turn_symmetry():
    a = pair_depth(0.1, -0.2, 0.3, 0.8, 0.4, -0.2)
    b = pair_depth(0.1, -0.2, 0.3 + math.pi / 2, 0.8, 0.4, -0.2 - math.pi)
    assert a == pytest.approx(b)


def test_energy_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(60):
        n = 5
        cfg = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(-0.8, 0.8, n)])
        g = np.zeros_like(cfg)
        if energy_grad(cfg, 1.2, 1.0, g) == 0:
            continue
        h = 1e-7
        num = np.zeros_like(cfg)
        for i in range(n):
            for k in range(3):
                a, b = cfg.copy(), cfg.copy()
                a[i, k] += h
                b[i, k] -= h
                num[i, k] = (energy_grad(a, 1.2, 1.0, np.zeros_like(a)) - energy_grad(b, 1.2, 1.0, np.zeros_like(b))) / (2 * h)
        assert np.allclose(num, g, atol=1e-6 * max(1, np.abs(num).max()))
        checked += 1
    assert checked > 20


def test_side_of_rotated_square():
    cfg = np.array([[0.0, 0.0, math.pi / 4]])
    assert side(cfg) == pytest.approx(math.sqrt(2))


def test_repair_scale_removes_all_overlaps():
    rng = np.random.default_rng(3)
    for _ in range(20):
        cfg = np.column_stack([rng.uniform(-1.5, 1.5, 8), rng.uniform(-1.5, 1.5, 8), rng.uniform(-0.7, 0.7, 8)])
        k = repair_scale(cfg)
        assert k >= 1
        s = make_feasible(cfg)
        assert math.isfinite(s)
        assert max_depth(cfg) < 1e-9
        v = validate(cfg)
        assert v.valid
        assert v.side == pytest.approx(s)


def test_repair_of_coincident_centres_is_infinite():
    cfg = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]])
    assert repair_scale(cfg) == math.inf
