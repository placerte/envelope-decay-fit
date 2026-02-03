"""Piecewise exponential decay fitting for time-domain response envelopes."""

from .api import (
    fit_piecewise_auto,
    fit_piecewise_manual,
    launch_manual_segmentation_ui,
    plot_segmentation_storyboard,
)
from .models import FitDiagnostics, FitResult, GlobalFitMetrics, PieceFit

__version__ = "0.1.1"

__all__ = [
    "FitDiagnostics",
    "FitResult",
    "GlobalFitMetrics",
    "PieceFit",
    "fit_piecewise_auto",
    "fit_piecewise_manual",
    "launch_manual_segmentation_ui",
    "plot_segmentation_storyboard",
]
