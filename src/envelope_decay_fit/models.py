"""Typed models for the public API surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .flags import FlagRecord

if TYPE_CHECKING:
    from .segmentation.auto.window_scan import WindowFitRecord


@dataclass
class GlobalFitMetrics:
    """Summary metrics for a fit run."""

    fn_hz: float
    omega_n: float
    n_pieces: int
    n_samples: int
    duration_s: float


@dataclass
class FitDiagnostics:
    """Optional diagnostic details for a fit run."""

    flags: list[FlagRecord]
    windows_trace: list["WindowFitRecord"] | None = None


@dataclass
class PieceFit:
    """Fit results for a single piece of the decay."""

    piece_id: int
    t_start_s: float
    t_end_s: float
    n_points: int
    params: dict[str, float]
    r2: float
    flags: list[str]


@dataclass
class FitResult:
    """Top-level fit result returned by the public API."""

    pieces: list[PieceFit]
    breakpoints_t: list[float]
    global_metrics: GlobalFitMetrics | None
    diagnostics: FitDiagnostics | None


@dataclass
class TxSpanMeasurement:
    """Span-based Tx measurement payload."""

    t0: float
    t1: float
    Tx_active: str
    Tx_value: float
    slope_db_per_s: float
    linearity_r2: float
    linearity_rms_db: float
    flags: list[FlagRecord]
    zeta: float | None
