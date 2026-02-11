"""Typed models for the public API surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .flags import FlagRecord

if TYPE_CHECKING:
    from .segmentation.auto.window_scan import WindowFitRecord


@dataclass
class GlobalFitMetrics:
    """Summary metrics for a fit run.

    Attributes:
        fn_hz: Natural frequency in Hz.
        omega_n: Angular frequency in rad/s.
        n_pieces: Number of extracted pieces.
        n_samples: Number of samples in the input.
        duration_s: Total duration in seconds.
    """

    fn_hz: float
    omega_n: float
    n_pieces: int
    n_samples: int
    duration_s: float


@dataclass
class FitDiagnostics:
    """Optional diagnostic details for a fit run.

    Attributes:
        flags: List of FlagRecord diagnostics.
        windows_trace: Optional list of WindowFitRecord objects.
    """

    flags: list[FlagRecord]
    windows_trace: list["WindowFitRecord"] | None = None


@dataclass
class PieceFit:
    """Fit results for a single piece of the decay.

    Attributes:
        piece_id: Piece identifier.
        t_start_s: Start time in seconds.
        t_end_s: End time in seconds.
        n_points: Number of samples in the piece.
        params: Fit parameters (alpha in 1/s, zeta dimensionless, A0 amplitude).
        r2: Coefficient of determination.
        flags: List of string flags for the piece.
    """

    piece_id: int
    t_start_s: float
    t_end_s: float
    n_points: int
    params: dict[str, float]
    r2: float
    flags: list[str]


@dataclass
class FitResult:
    """Top-level fit result returned by the public API.

    Attributes:
        pieces: PieceFit list for the piecewise decay.
        breakpoints_t: Breakpoint times in seconds.
        global_metrics: GlobalFitMetrics summary (optional).
        diagnostics: FitDiagnostics payload (optional).
    """

    pieces: list[PieceFit]
    breakpoints_t: list[float]
    global_metrics: GlobalFitMetrics | None
    diagnostics: FitDiagnostics | None


@dataclass
class TxSpanMeasurement:
    """Span-based Tx measurement payload.

    Attributes:
        t0: Span start time in seconds.
        t1: Span end time in seconds.
        Tx_active: Active Tx label (e.g., "T60").
        Tx_value: Target dB drop (e.g., 60).
        slope_db_per_s: Implied slope in dB/s.
        linearity_r2: R2 against the Tx diagonal.
        linearity_rms_db: RMS deviation from the Tx diagonal in dB.
        flags: Diagnostic FlagRecord list.
        zeta: Optional damping ratio (dimensionless).
    """

    t0: float
    t1: float
    Tx_active: str
    Tx_value: float
    slope_db_per_s: float
    linearity_r2: float
    linearity_rms_db: float
    flags: list[FlagRecord]
    zeta: float | None
