from sqpack import GAConfig, run
from sqpack.history import RunData
from sqpack.viz import render_all


def test_render_all_smoke(tmp_path):
    res = run(GAConfig(n=5, pop_size=10, generations=12, seed=0), verbose=False)
    d = res.history.save(tmp_path / "run")
    written = render_all(d, max_frames=8, verbose=False)
    names = {p.name for p in written}
    for expected in ("01_population_grid.png", "02_fitness.png", "08_final_packing.png", "evolution.gif", "best.gif", "report.html"):
        assert expected in names
    html = (d / "figures" / "report.html").read_text()
    assert "__DATA__" not in html and '"frames"' in html


def test_lineage_reaches_generation_zero(tmp_path):
    res = run(GAConfig(n=6, pop_size=12, generations=15, seed=4), verbose=False)
    d = res.history.save(tmp_path / "run")
    chain = RunData(d).lineage()
    assert chain[0][0] == 0 and chain[0][2] == -1
    assert [c[0] for c in chain] == sorted(c[0] for c in chain)
