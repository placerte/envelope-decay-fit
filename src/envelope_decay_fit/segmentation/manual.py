"""Interactive manual segmentation for envelope decay fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, cast

import numpy as np
from matplotlib.backend_bases import Event
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from ..fitters import fit_log_domain, fit_lin0_domain, fit_linc_domain, FitResult
from ..flags import FlagRecord
from ..result import PieceRecord, Result


@dataclass
class ManualUIConfig:
    """Configuration for the manual segmentation UI.

    Attributes:
        min_points: Minimum samples required per segment.
        out_dir: Optional output directory for artifacts.
    """

    min_points: int = 10
    out_dir: Path | None = None


def find_nearest_index(t: np.ndarray, x_value: float) -> int:
    """Find the nearest sample index in a sorted time array.

    Args:
        t: Time array in seconds (strictly increasing).
        x_value: Target time in seconds.

    Returns:
        Index of the closest sample in t.
    """
    idx = int(np.searchsorted(t, x_value))
    if idx <= 0:
        return 0
    if idx >= len(t):
        return len(t) - 1

    left = idx - 1
    right = idx
    if abs(float(t[right]) - x_value) < abs(float(t[left]) - x_value):
        return right
    return left


def snap_boundary_times_to_indices(
    t: np.ndarray, boundaries_time_s: list[float]
) -> list[int]:
    """Snap boundary times to nearest sample indices.

    Args:
        t: Time array in seconds (strictly increasing).
        boundaries_time_s: Boundary times in seconds.

    Returns:
        Sorted list of unique boundary indices.
    """
    indices: list[int] = []
    for time_s in boundaries_time_s:
        idx = find_nearest_index(t, float(time_s))
        indices.append(idx)
    return sorted(set(indices))


def compute_manual_pieces(
    t: np.ndarray,
    env: np.ndarray,
    fn_hz: float,
    boundary_indices: list[int],
    min_points: int = 10,
) -> tuple[list[PieceRecord], list[FlagRecord]]:
    """Compute piecewise fits using literal manual boundaries.

    Args:
        t: Time array in seconds.
        env: Envelope amplitude (linear scale).
        fn_hz: Natural frequency in Hz.
        boundary_indices: Boundary indices into t/env.
        min_points: Minimum samples required per piece.

    Returns:
        Tuple of (pieces, flags).

    Assumptions:
        - Each piece is modeled as exponential decay.
        - Damping ratio uses zeta = alpha / (2*pi*fn_hz).
    """
    flags: list[FlagRecord] = []

    if len(boundary_indices) < 2:
        return [], flags

    boundary_indices = sorted(set(boundary_indices))

    # Validate segment lengths
    for i in range(len(boundary_indices) - 1):
        i_start = boundary_indices[i]
        i_end_inclusive = boundary_indices[i + 1]
        n_points = i_end_inclusive - i_start + 1
        if n_points < min_points:
            flags.append(
                FlagRecord(
                    scope="global",
                    scope_id="manual",
                    severity="warn",
                    code="MANUAL_SEGMENT_TOO_SHORT",
                    message=(
                        "Manual segmentation rejected: segment has fewer than min_points"
                    ),
                    details=(
                        f"segment={i}, n_points={n_points}, min_points={min_points}"
                    ),
                )
            )
            return [], flags

    pieces: list[PieceRecord] = []
    for piece_id in range(len(boundary_indices) - 1):
        i_start = boundary_indices[piece_id]
        i_end_inclusive = boundary_indices[piece_id + 1]
        i_end_exclusive = i_end_inclusive + 1

        t_piece = t[i_start:i_end_exclusive]
        env_piece = env[i_start:i_end_exclusive]
        t_ref = float(t_piece[0])

        log_fit = fit_log_domain(t_piece, env_piece, fn_hz, t_ref)
        lin0_fit = fit_lin0_domain(t_piece, env_piece, fn_hz, t_ref)
        linc_fit = fit_linc_domain(t_piece, env_piece, fn_hz, t_ref)

        piece = PieceRecord(
            piece_id=piece_id,
            label=f"manual_piece_{piece_id}",
            i_start=int(i_start),
            i_end=int(i_end_exclusive),
            t_start_s=float(t_piece[0]),
            t_end_s=float(t_piece[-1]),
            dt_s=float(t_piece[-1] - t_piece[0]),
            n_points=int(len(t_piece)),
            breakpoint_index=None,
            breakpoint_dt_s=None,
            log_fit=log_fit,
            lin0_fit=lin0_fit,
            linc_fit=linc_fit,
        )
        pieces.append(piece)

    return pieces, flags


def build_result_from_manual(
    t: np.ndarray,
    env: np.ndarray,
    fn_hz: float,
    boundary_indices: list[int],
    pieces: list[PieceRecord],
    flags: list[FlagRecord],
) -> Result:
    """Build a Result object from manual segmentation outputs.

    Args:
        t: Time array in seconds.
        env: Envelope amplitude (linear scale).
        fn_hz: Natural frequency in Hz.
        boundary_indices: Boundary indices into t/env.
        pieces: PieceRecord list from manual fits.
        flags: FlagRecord list describing data or fit issues.

    Returns:
        Result object populated with manual segmentation metadata.
    """
    omega_n = 2.0 * np.pi * fn_hz
    boundary_indices = sorted(set(boundary_indices))
    boundary_times = [float(t[i]) for i in boundary_indices]
    enabled = len(boundary_indices) >= 2 and len(pieces) > 0

    return Result(
        t=t,
        env=env,
        fn_hz=fn_hz,
        omega_n=omega_n,
        pieces=pieces,
        windows_trace=[],
        flags=flags,
        manual_segmentation_enabled=enabled,
        manual_boundaries_time_s=boundary_times,
        manual_boundaries_index=boundary_indices,
    )


def _fit_result_to_dict(
    fit: FitResult,
) -> dict[str, float | bool | str | dict[str, float]]:
    params = {str(k): float(v) for k, v in fit.params.items()}
    return {
        "alpha": float(fit.alpha),
        "zeta": float(fit.zeta),
        "r2": float(fit.r2),
        "rmse": float(fit.rmse),
        "valid": bool(fit.valid),
        "params": params,
        "notes": str(fit.notes),
    }


def write_manual_segmentation_json(result: Result, out_path: Path) -> Path:
    """Write manual segmentation metadata and fit results to JSON.

    Args:
        result: Result object from the manual workflow.
        out_path: Output JSON path.

    Returns:
        Path to the written JSON file.
    """
    data: dict[str, object] = {
        "manual_segmentation": {
            "enabled": bool(result.manual_segmentation_enabled),
            "boundaries_time_s": list(result.manual_boundaries_time_s),
            "boundaries_index": list(result.manual_boundaries_index),
        },
        "pieces": [],
        "flags": [],
    }

    pieces_payload: list[dict[str, object]] = []
    for piece in result.pieces:
        pieces_payload.append(
            {
                "piece_id": int(piece.piece_id),
                "label": piece.label,
                "i_start": int(piece.i_start),
                "i_end": int(piece.i_end),
                "t_start_s": float(piece.t_start_s),
                "t_end_s": float(piece.t_end_s),
                "dt_s": float(piece.dt_s),
                "n_points": int(piece.n_points),
                "log_fit": _fit_result_to_dict(piece.log_fit),
                "lin0_fit": _fit_result_to_dict(piece.lin0_fit),
                "linc_fit": _fit_result_to_dict(piece.linc_fit),
            }
        )
    data["pieces"] = pieces_payload

    flags_payload: list[dict[str, str]] = []
    for flag in result.flags:
        flags_payload.append(
            {
                "scope": flag.scope,
                "scope_id": flag.scope_id,
                "severity": flag.severity,
                "code": flag.code,
                "message": flag.message,
                "details": flag.details,
            }
        )
    data["flags"] = flags_payload

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(data, handle, indent=2)

    return out_path


@dataclass
class ManualSegmentationUI:
    """Interactive Matplotlib UI for manual segmentation."""

    t: np.ndarray
    env: np.ndarray
    fn_hz: float
    min_points: int = 10
    out_dir: Path | None = None
    initial_boundaries_time_s: list[float] = field(default_factory=list)

    boundary_order: list[int] = field(default_factory=list)
    boundary_indices: list[int] = field(default_factory=list)
    current_pieces: list[PieceRecord] = field(default_factory=list)
    current_flags: list[FlagRecord] = field(default_factory=list)
    committed: bool = False
    skip_save: bool = False
    last_mouse_x: float | None = None
    last_action: str = "Ready"
    help_visible: bool = False
    last_toolbar_mode: str = "NONE"
    env_line: Line2D | None = None
    legend: Legend | None = None

    def run(self) -> Result | None:
        """Launch the interactive session and return a Result if committed."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        if self.initial_boundaries_time_s:
            initial_indices = snap_boundary_times_to_indices(
                self.t, self.initial_boundaries_time_s
            )
            self.boundary_order = list(initial_indices)
            self._rebuild_boundaries()
            self.last_action = "Loaded initial boundaries"

        title_size = 18
        label_size = 16
        tick_size = 14
        overlay_size = 16

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.env_line = self.ax.plot(
            self.t, self.env, "k-", alpha=0.6, linewidth=1, label="Envelope"
        )[0]
        self.ax.set_xlabel("Time (s)", fontsize=label_size)
        self.ax.set_ylabel("Envelope amplitude", fontsize=label_size)
        self.ax.set_title(
            f"Manual Segmentation (keyboard-driven boundaries) - fn={self.fn_hz:.1f} Hz",
            fontsize=title_size,
        )
        self.ax.grid(True, alpha=0.3)
        self.ax.set_yscale("log", nonpositive="clip")
        self.ax.tick_params(axis="both", labelsize=tick_size)

        self.boundary_lines: list[Line2D] = []
        self.fit_lines: list[Line2D] = []
        self.info_text = self.ax.text(
            0.98,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="right",
            fontsize=overlay_size,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        self.help_text = self.ax.text(
            0.02,
            0.02,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=overlay_size,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._refresh_plot()
        plt.show()

        if self.skip_save:
            return None

        pieces, flags = compute_manual_pieces(
            self.t,
            self.env,
            self.fn_hz,
            self.boundary_indices,
            min_points=self.min_points,
        )
        flags = self.current_flags if self.current_flags else flags
        result = build_result_from_manual(
            self.t,
            self.env,
            self.fn_hz,
            self.boundary_indices,
            pieces,
            flags,
        )

        if self.out_dir is not None:
            out_dir = Path(self.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = write_manual_segmentation_json(
                result, out_dir / "manual_segmentation.json"
            )
            result.artifact_paths["manual_segmentation_json"] = json_path

        return result

    def _on_key(self, event: Event) -> None:
        key = getattr(event, "key", None)
        if key == "h":
            self.help_visible = not self.help_visible
            self._update_help_text()
            self.fig.canvas.draw_idle()
            return

        if key == "a":
            self._add_boundary_at_cursor()
            return

        if key == "x":
            self._delete_nearest_boundary()
            return

        if key == "c":
            self.boundary_order = []
            self._rebuild_boundaries()
            self.last_action = "Cleared all boundaries"
            self._refresh_plot()
            return

        if key == "l":
            self._toggle_scale()
            return

        if key == "r":
            self.last_action = "Recomputed fits"
            self._refresh_plot(recompute=True)
            return

        if key == "q":
            self.committed = True
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def _on_motion(self, event: Event) -> None:
        if getattr(event, "inaxes", None) != self.ax:
            return
        xdata = getattr(event, "xdata", None)
        if xdata is None:
            return
        self.last_mouse_x = float(xdata)
        self._update_help_text()

    def _on_close(self, event: Event) -> None:
        if self.skip_save:
            return
        self.committed = True

    def _delete_nearest_boundary(self) -> None:
        if not self.boundary_indices:
            self.last_action = "No boundaries to delete"
            self._update_info_text([float(self.t[i]) for i in self.boundary_indices])
            self.fig.canvas.draw_idle()
            return

        x_target = None
        if self.last_mouse_x is not None:
            x_target = self.last_mouse_x

        if x_target is None:
            self.last_action = "Move cursor over plot to delete"
            self._update_info_text([float(self.t[i]) for i in self.boundary_indices])
            self.fig.canvas.draw_idle()
            return

        distances = [abs(float(self.t[i]) - x_target) for i in self.boundary_indices]
        nearest_idx = int(np.argmin(distances))
        boundary_to_remove = self.boundary_indices[nearest_idx]

        self.boundary_order = [
            i for i in self.boundary_order if i != boundary_to_remove
        ]
        self._rebuild_boundaries()
        self.last_action = (
            f"Deleted boundary at t={float(self.t[boundary_to_remove]):.6f} s"
        )
        self._refresh_plot()

    def _toggle_scale(self) -> None:
        current = self.ax.get_yscale()
        if current == "linear":
            self.ax.set_yscale("log", nonpositive="clip")
            self.last_action = "Switched to log scale"
        else:
            self.ax.set_yscale("linear")
            self.last_action = "Switched to linear scale"
        self._update_info_text([float(self.t[i]) for i in self.boundary_indices])
        self.fig.canvas.draw_idle()

    def _rebuild_boundaries(self) -> None:
        self.boundary_indices = sorted(set(self.boundary_order))

    def _add_boundary_at_cursor(self) -> None:
        if self.last_mouse_x is None:
            self.last_action = "Move cursor over plot to add"
            self._update_info_text([float(self.t[i]) for i in self.boundary_indices])
            self.fig.canvas.draw_idle()
            return

        idx = find_nearest_index(self.t, float(self.last_mouse_x))
        if idx not in self.boundary_order:
            self.boundary_order.append(idx)
        self._rebuild_boundaries()
        self.last_action = f"Added boundary at t={float(self.t[idx]):.6f} s"
        self._refresh_plot()

    def _get_toolbar_mode(self) -> str:
        toolbar = None
        if hasattr(self.fig.canvas, "manager") and self.fig.canvas.manager is not None:
            toolbar = getattr(self.fig.canvas.manager, "toolbar", None)
        if toolbar is None:
            return "NONE"
        mode = getattr(toolbar, "mode", "")
        if mode is None:
            return "NONE"
        mode_text = str(mode).lower().strip()
        if "pan" in mode_text:
            return "PAN"
        if "zoom" in mode_text:
            return "ZOOM"
        return "NONE"

    def _refresh_plot(self, recompute: bool = True) -> None:
        for line in self.boundary_lines:
            line.remove()
        for line in self.fit_lines:
            line.remove()
        self.boundary_lines = []
        self.fit_lines = []

        boundary_times = [float(self.t[i]) for i in self.boundary_indices]

        if recompute:
            self.current_pieces, self.current_flags = compute_manual_pieces(
                self.t,
                self.env,
                self.fn_hz,
                self.boundary_indices,
                min_points=self.min_points,
            )

        for time_s in boundary_times:
            line = self.ax.axvline(
                time_s, color="red", linestyle="--", alpha=0.7, label="Boundary"
            )
            self.boundary_lines.append(line)

        colors = ["blue", "green", "orange", "purple", "brown"]
        for idx, piece in enumerate(self.current_pieces):
            if not piece.log_fit.valid:
                continue

            t_piece = self.t[piece.i_start : piece.i_end]
            t_ref = float(t_piece[0])
            t_shifted = t_piece - t_ref
            A = float(np.exp(piece.log_fit.params["b"]))
            env_fit = A * np.exp(-piece.log_fit.alpha * t_shifted)
            color = colors[idx % len(colors)]
            line = self.ax.plot(
                t_piece,
                env_fit,
                "--",
                color=color,
                linewidth=2,
                alpha=0.8,
                label=f"Log fit {piece.piece_id}",
            )[0]
            self.fit_lines.append(line)

        self._update_legend()
        self._update_help_text()
        self._update_info_text(boundary_times)
        self.fig.canvas.draw_idle()

    def _update_info_text(self, boundary_times: list[float]) -> None:
        k = len(boundary_times)
        n_pieces = max(k - 1, 0)
        time_text = ", ".join(f"{t_val:.6f}" for t_val in boundary_times)
        scale = self.ax.get_yscale()

        lines = [
            f"fn_hz={self.fn_hz:.1f} | scale={scale}",
            f"boundaries k={k} | pieces={n_pieces}",
            f"t_s=[{time_text}]" if time_text else "t_s=[]",
            f"last: {self.last_action}",
        ]

        if self.current_pieces:
            lines.append("fit summary:")
            for piece in self.current_pieces:
                if piece.log_fit.valid and "b" in piece.log_fit.params:
                    A0 = float(np.exp(piece.log_fit.params["b"]))
                    lines.append(
                        "Piece "
                        + f"{piece.piece_id}: "
                        + f"ζ={piece.log_fit.zeta:.5f} | "
                        + f"R²={piece.log_fit.r2:.3f} | "
                        + f"A0={A0:.4g}"
                    )
                else:
                    lines.append(f"Piece {piece.piece_id}: invalid fit")

        for flag in self.current_flags:
            if flag.code == "MANUAL_SEGMENT_TOO_SHORT":
                lines.append(f"warning: {flag.details}")

        self.info_text.set_text("\n".join(lines))
        self._place_info_overlay()

    def _update_help_text(self) -> None:
        mode = self._get_toolbar_mode()
        if mode != self.last_toolbar_mode:
            self.last_toolbar_mode = mode

        self.help_text.set_visible(True)
        if not self.help_visible:
            self.help_text.set_text("h: help")
            self._place_help_overlay()
            return

        lines = [
            "a: add boundary",
            "x: delete nearest boundary",
            "c: clear boundaries",
            "l: toggle lin/log",
            "h: toggle help",
            "q: save and quit",
        ]
        self.help_text.set_text("\n".join(lines))
        self._place_help_overlay()

    def _update_legend(self) -> None:
        handles: list[Line2D] = []
        labels: list[str] = []

        if self.env_line is not None:
            handles.append(self.env_line)
            labels.append("Envelope")

        if self.boundary_lines:
            handles.append(self.boundary_lines[0])
            labels.append("Boundary")

        for line in self.fit_lines:
            handles.append(line)
            labels.append(str(line.get_label()))

        if not handles:
            if self.legend is not None:
                self.legend.remove()
                self.legend = None
            return

        if self.legend is not None:
            self.legend.remove()

        self.legend = self.ax.legend(
            handles=handles,
            labels=labels,
            loc="upper right",
            framealpha=0.9,
        )

    def _place_help_overlay(self) -> None:
        self.help_text.set_position((0.98, 0.02))
        self.help_text.set_horizontalalignment("right")
        self.help_text.set_verticalalignment("bottom")

    def _place_info_overlay(self) -> None:
        if self.fig is None:
            return

        self.fig.canvas.draw()
        renderer = cast(Any, self.fig.canvas).get_renderer()

        legend_bbox = None
        if self.legend is not None:
            legend_bbox = self.legend.get_window_extent(renderer=renderer)

        help_bbox = None
        if self.help_text is not None:
            help_bbox = self.help_text.get_window_extent(renderer=renderer)

        padding_ax = 0.02
        candidates: list[tuple[float, float]] = [(0.98, 0.98)]

        if legend_bbox is not None:
            legend_bbox_ax = self.ax.transAxes.inverted().transform_bbox(legend_bbox)
            candidate_y = legend_bbox_ax.y0 - padding_ax
            if candidate_y > 0.02:
                candidates.append((0.98, candidate_y))

        if help_bbox is not None:
            help_bbox_ax = self.ax.transAxes.inverted().transform_bbox(help_bbox)
            candidate_y = help_bbox_ax.y0 - padding_ax
            if candidate_y > 0.02:
                candidates.append((0.98, candidate_y))

        candidates.append((0.98, 0.02))

        for x_pos, y_pos in candidates:
            self.info_text.set_position((x_pos, y_pos))
            self.info_text.set_horizontalalignment("right")
            self.info_text.set_verticalalignment("top" if y_pos > 0.5 else "bottom")
            self.fig.canvas.draw()
            info_bbox = self.info_text.get_window_extent(renderer=renderer)

            overlaps = False
            if legend_bbox is not None and info_bbox.overlaps(legend_bbox):
                overlaps = True
            if help_bbox is not None and info_bbox.overlaps(help_bbox):
                overlaps = True

            if not overlaps:
                return


def run_manual_segmentation(
    t: np.ndarray,
    env: np.ndarray,
    fn_hz: float,
    min_points: int = 10,
    out_dir: Path | str | None = None,
    initial_boundaries_time_s: list[float] | None = None,
) -> Result | None:
    """Run the interactive manual segmentation workflow.

    Args:
        t: Time array in seconds.
        env: Envelope amplitude (linear scale).
        fn_hz: Natural frequency in Hz.
        min_points: Minimum samples required per piece.
        out_dir: Optional output directory for artifacts.
        initial_boundaries_time_s: Optional initial boundaries in seconds.

    Returns:
        Result if committed, otherwise None.
    """
    out_path = Path(out_dir) if out_dir is not None else None
    ui = ManualSegmentationUI(
        t=t,
        env=env,
        fn_hz=fn_hz,
        min_points=min_points,
        out_dir=out_path,
        initial_boundaries_time_s=initial_boundaries_time_s or [],
    )
    return ui.run()
