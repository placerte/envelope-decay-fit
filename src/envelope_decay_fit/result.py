"""Result dataclasses for API return values."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .flags import FlagRecord
from .fitters import FitResult

if TYPE_CHECKING:
    from .segmentation.auto.window_scan import WindowFitRecord


@dataclass
class PieceRecord:
    """Record for an extracted decay piece.

    Attributes:
        piece_id: piece identifier (0-indexed)
        label: descriptive label (e.g., 'established_free_decay', 'transient_dominated_decay')
        i_start: start index in original arrays
        i_end: end index (exclusive)
        t_start_s: start time (seconds)
        t_end_s: end time (seconds)
        dt_s: piece duration (seconds)
        n_points: number of samples

        breakpoint_index: index in windows trace where piece was detected (or None)
        breakpoint_dt_s: window duration at breakpoint (or None)

        log_fit: representative LOG fit for this piece
        lin0_fit: representative LIN0 fit
        linc_fit: representative LINC fit
    """

    piece_id: int
    label: str
    i_start: int
    i_end: int
    t_start_s: float
    t_end_s: float
    dt_s: float
    n_points: int

    breakpoint_index: int | None
    breakpoint_dt_s: float | None

    log_fit: FitResult
    lin0_fit: FitResult
    linc_fit: FitResult


@dataclass
class Result:
    """Top-level result object returned by internal pipelines.

    Attributes:
        t: input time array (seconds)
        env: input envelope array
        fn_hz: natural frequency (Hz)
        omega_n: angular frequency (rad/s)

        pieces: list of extracted PieceRecord objects
        windows_trace: list of all WindowFitRecord objects (diagnostic)
        flags: list of FlagRecord objects (diagnostics, warnings)

        artifact_paths: dict of written file paths (if out_dir was provided)
        manual_segmentation_enabled: whether manual segmentation was used
        manual_boundaries_time_s: snapped boundary times (seconds)
        manual_boundaries_index: snapped boundary indices
    """

    # Input data
    t: np.ndarray
    env: np.ndarray
    fn_hz: float
    omega_n: float

    # Results
    pieces: list[PieceRecord]
    windows_trace: list["WindowFitRecord"]
    flags: list[FlagRecord]

    # Optional output paths
    artifact_paths: dict[str, Path] = field(default_factory=dict)

    # Manual segmentation metadata (optional)
    manual_segmentation_enabled: bool = False
    manual_boundaries_time_s: list[float] = field(default_factory=list)
    manual_boundaries_index: list[int] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary of results."""
        lines = []
        lines.append(f"Envelope Decay Fit Results")
        lines.append(f"=" * 60)
        lines.append(f"Natural frequency: {self.fn_hz:.2f} Hz")
        lines.append(f"Data: {len(self.t)} samples, {self.t[-1] - self.t[0]:.4f} s")
        lines.append(f"Pieces extracted: {len(self.pieces)}")
        lines.append(f"Flags: {len(self.flags)} ({self._count_flags_by_severity()})")
        if self.manual_segmentation_enabled:
            lines.append("Manual segmentation: enabled")
            lines.append(
                f"Manual boundaries: {len(self.manual_boundaries_time_s)} points"
            )
        lines.append("")

        for piece in self.pieces:
            lines.append(f"Piece {piece.piece_id}: {piece.label}")
            lines.append(
                f"  Range: [{piece.t_start_s:.4f}, {piece.t_end_s:.4f}] s, duration: {piece.dt_s:.4f} s"
            )
            lines.append(
                f"  LOG:  ζ = {piece.log_fit.zeta:.6f}, R² = {piece.log_fit.r2:.4f}, valid = {piece.log_fit.valid}"
            )
            lines.append(
                f"  LIN0: ζ = {piece.lin0_fit.zeta:.6f}, R² = {piece.lin0_fit.r2:.4f}, valid = {piece.lin0_fit.valid}"
            )
            lines.append(
                f"  LINC: ζ = {piece.linc_fit.zeta:.6f}, R² = {piece.linc_fit.r2:.4f}, valid = {piece.linc_fit.valid}"
            )
            lines.append("")

        if self.flags:
            lines.append("Flags:")
            for flag in self.flags:
                lines.append(f"  {flag}")

        return "\n".join(lines)

    def _count_flags_by_severity(self) -> str:
        """Count flags by severity level."""
        info_count = sum(1 for f in self.flags if f.severity == "info")
        warn_count = sum(1 for f in self.flags if f.severity == "warn")
        reject_count = sum(1 for f in self.flags if f.severity == "reject")
        return f"{info_count} info, {warn_count} warn, {reject_count} reject"
