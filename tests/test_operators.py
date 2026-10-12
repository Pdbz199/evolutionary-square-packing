import numpy as np

from sqpack.diversity import d4_perms, raster, raster_dist
from sqpack.operators import N_MUT, apply_d4, crossover_cut, seed_rng, vary


def _rand(n, seed):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.uniform(-0.7, 0.7, n)])


def test_crossover_keeps_square_count():
    seed_rng(0)
    for s in range(30):
        a, b = _rand(11, s), _rand(11, s + 100)
        child = np.empty_like(a)
        crossover_cut(a, b, child)
        assert child.shape == a.shape
        assert np.isfinite(child).all()


def test_symmetries_preserve_rasters():
    perms = d4_perms(16)
    cfg = _rand(6, 3)
    r0 = raster(cfg, 16)
    out = np.empty_like(cfg)
    for t in range(8):
        apply_d4(cfg, t, out)
        d, _ = raster_dist(r0, raster(out, 16), perms)
        assert d < 0.02, (t, d)


def test_vary_writes_valid_operator_codes():
    seed_rng(1)
    pop = np.stack([_rand(7, s) for s in range(10)])
    fit = np.linspace(3, 4, 10)
    eps = np.full(10, 0.02)
    lam = 20
    children = np.empty((lam, 7, 3))
    ch_eps, targets = np.empty(lam), np.empty(lam)
    parents = np.empty((lam, 2), dtype=np.int64)
    ops = np.empty(lam, dtype=np.int64)
    w = np.ones(N_MUT)
    vary(pop, fit, fit, eps, 0.5, 0.8, w, 3, children, ch_eps, targets, parents, ops)
    assert ((ops >> 1) < N_MUT).all() and (ops >= 0).all()
    assert (parents[:, 0] >= 0).all() and (parents[:, 0] < 10).all()
    # clones are always mutated
    assert (((ops & 1) == 1) | ((ops >> 1) > 0)).all()
