"""Piecewise exponential decay fitting for time-domain response envelopes."""

from .api import (
    fit_piecewise_auto,
    fit_piecewise_manual,
    launch_manual_segmentation_ui,
    launch_tx_span_ui,
    plot_segmentation_storyboard,
)
from .models import (
    FitDiagnostics,
    FitResult,
    GlobalFitMetrics,
    PieceFit,
    TxSpanMeasurement,
)

__version__ = "0.2.0"

__all__ = [
    "FitDiagnostics",
    "FitResult",
    "GlobalFitMetrics",
    "PieceFit",
    "TxSpanMeasurement",
    "fit_piecewise_auto",
    "fit_piecewise_manual",
    "launch_manual_segmentation_ui",
    "launch_tx_span_ui",
    "plot_segmentation_storyboard",
]
