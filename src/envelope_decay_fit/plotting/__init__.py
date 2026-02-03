"""Plotting helpers and exporters."""

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from .storyboard import (
    create_diagnostic_plots,
    create_param_traces_plot,
    create_piecewise_fit_plot,
    create_score_traces_plot,
    create_segmentation_storyboard_plot,
    plot_segmentation_storyboard,
)


def export_plot(
    fig: Figure,
    path: Path,
    context: dict[str, Any] | None = None,
    dpi: int | None = None,
) -> Path:
    """Lazily import plot exporter to avoid hard deps on import."""
    from .plot_export import export_plot as _export_plot

    return _export_plot(fig, path, context=context, dpi=dpi)


__all__ = [
    "create_diagnostic_plots",
    "create_param_traces_plot",
    "create_piecewise_fit_plot",
    "create_score_traces_plot",
    "create_segmentation_storyboard_plot",
    "export_plot",
    "plot_segmentation_storyboard",
]
