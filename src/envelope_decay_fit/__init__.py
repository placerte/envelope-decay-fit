"""Piecewise exponential decay fitting for time-domain response envelopes.

This package provides tools to fit exponential decay models to envelope data
and extract damping ratios with comprehensive diagnostics.
"""

__version__ = "0.1.0"

# Public API
from .api import fit_envelope_decay, fit_envelope_decay_manual
from .result import Result, PieceRecord
from .flags import FlagRecord
from .manual_segmentation import run_manual_segmentation

__all__ = [
    "fit_envelope_decay",
    "fit_envelope_decay_manual",
    "run_manual_segmentation",
    "Result",
    "PieceRecord",
    "FlagRecord",
]
