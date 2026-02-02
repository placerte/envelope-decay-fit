"""Unified plot export for deterministic debug sidecars."""

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from mpl_plot_report import dump_report


def export_plot(
    fig: Figure,
    path: Path,
    context: dict[str, Any] | None = None,
    dpi: int | None = None,
) -> Path:
    """Export a Matplotlib figure with JSON + Markdown sidecars.

    Args:
        fig: Matplotlib Figure to export.
        path: Target PNG path (sidecars share the same stem).
        context: Optional context metadata for the plot report.
        dpi: Optional PNG DPI override.

    Returns:
        Path to the PNG file.
    """
    out_path = Path(path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    report_context: dict[str, Any] = context or {}
    stem = out_path.stem

    if dpi is not None:
        fig.set_dpi(dpi)

    dump_report(fig, out_dir=out_dir, stem=stem, context=report_context)

    return out_path
