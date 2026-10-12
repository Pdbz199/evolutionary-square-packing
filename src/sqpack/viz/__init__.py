"""Visualisations of a finished run: static PNGs, animated GIFs and an HTML report."""

from pathlib import Path


def render_all(run_dir, out_dir=None, gif=True, html=True, max_frames=120, verbose=True):
    """Render every visual for ``run_dir`` into ``out_dir`` (default ``run_dir/figures``)."""
    from .animate import render_gifs
    from .report import render_report
    from .static import render_static

    out = Path(out_dir or Path(run_dir) / "figures")
    written = list(render_static(run_dir, out, verbose=verbose))
    if gif:
        written += render_gifs(run_dir, out, verbose=verbose, max_frames=max_frames)
    if html:
        written.append(render_report(run_dir, out / "report.html", verbose=verbose))
    return written
