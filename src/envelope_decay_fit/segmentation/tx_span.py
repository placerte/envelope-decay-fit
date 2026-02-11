"""Interactive span-based Tx measurement for log-amplitude decay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from matplotlib.backend_bases import Event
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector

from ..flags import FlagRecord
from ..models import TxSpanMeasurement


@dataclass
class TxSpanConfig:
    """Configuration for span-based Tx measurement.

    Attributes:
        min_points: Minimum samples required inside the span.
        min_span_s: Minimum span duration in seconds.
        low_linearity_r2: R2 threshold for flagging low linearity.
        high_rms_db: RMS threshold in dB for flagging high deviation.
        mismatch_db: Allowed mismatch between actual drop and Tx value (dB).
        noise_floor_margin_db: Margin to flag proximity to noise floor (dB).
        non_monotonic_tolerance_db: Positive tolerance for non-monotonic steps (dB).
    """

    min_points: int = 10
    min_span_s: float = 0.01
    low_linearity_r2: float = 0.95
    high_rms_db: float = 1.0
    mismatch_db: float = 3.0
    noise_floor_margin_db: float = 3.0
    non_monotonic_tolerance_db: float = 0.0


def _compute_log_envelope_db(env: np.ndarray) -> np.ndarray:
    """Convert envelope to log-amplitude dB re max(env).

    Args:
        env: Envelope amplitude (linear scale).

    Returns:
        Log-amplitude in dB, referenced to the maximum envelope value.
        Non-positive values map to NaN.

    Assumptions:
        - Reference level is max(|env|) over the provided array.
        - Uses 20*log10 for amplitude (not energy).
    """
    env_abs = np.abs(env)
    ref = float(np.max(env_abs)) if env_abs.size else 0.0
    if ref <= 0.0:
        ref = 1.0
    safe_env = np.where(env_abs > 0.0, env_abs, np.nan)
    return 20.0 * np.log10(safe_env / ref)


def _format_tx_label(tx_db: float) -> str:
    return f"T{int(tx_db)}" if float(tx_db).is_integer() else f"T{tx_db:.1f}"


def _compute_linearity_metrics(
    t: np.ndarray,
    y_db: np.ndarray,
    t0: float,
    t1: float,
    tx_db: float,
) -> tuple[float, float, float, float, int]:
    """Compute linearity metrics against the Tx reference diagonal.

    Args:
        t: Time array in seconds.
        y_db: Log-amplitude envelope in dB re max(env).
        t0: Span start time in seconds.
        t1: Span end time in seconds.
        tx_db: Target dB drop over the span (e.g., 60 for T60).

    Returns:
        Tuple of (y0_data, y1_data, r2, rms_db, n_points).

    Assumptions:
        - Reference line is defined by y(t0) and tx_db drop at t1.
        - Metrics use all finite samples within [t0, t1].
    """
    if t1 <= t0:
        return np.nan, np.nan, np.nan, np.nan, 0

    finite_mask = np.isfinite(y_db)
    span_mask = (t >= t0) & (t <= t1) & finite_mask
    t_span = t[span_mask]
    y_span = y_db[span_mask]
    if t_span.size == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    y0_data = float(np.interp(t0, t_span, y_span))
    y1_data = float(np.interp(t1, t_span, y_span))
    y1_line = y0_data - float(tx_db)
    slope = (y1_line - y0_data) / (t1 - t0)
    y_line = y0_data + slope * (t_span - t0)

    residuals = y_span - y_line
    rms = float(np.sqrt(np.mean(residuals**2))) if residuals.size else np.nan

    y_mean = float(np.mean(y_span)) if y_span.size else 0.0
    ss_res = float(np.sum((y_span - y_line) ** 2))
    ss_tot = float(np.sum((y_span - y_mean) ** 2))
    if ss_tot <= 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return y0_data, y1_data, r2, rms, int(t_span.size)


def compute_tx_span_measurement(
    t: np.ndarray,
    env: np.ndarray,
    t0: float,
    t1: float,
    tx_db: float,
    *,
    fn_hz: float | None = None,
    config: TxSpanConfig | None = None,
) -> TxSpanMeasurement:
    """Compute Tx span measurement and diagnostics.

    Args:
        t: Time array in seconds.
        env: Envelope amplitude (linear scale).
        t0: Span start time in seconds.
        t1: Span end time in seconds.
        tx_db: Target dB drop over the span (e.g., 10, 20, 30, 60).
        fn_hz: Natural frequency in Hz (optional for zeta).
        config: Threshold configuration for diagnostic flags.

    Returns:
        TxSpanMeasurement payload with Tx, slope, linearity metrics, and flags.

    Assumptions:
        - Uses log-amplitude 20*log10(|env|/max(env)).
        - Tx is defined as t1 - t0 with slope = tx_db / (t1 - t0).
        - zeta uses (slope_db_per_s * ln(10)) / (20 * 2*pi*fn_hz).
    """
    if config is None:
        config = TxSpanConfig()

    if t1 < t0:
        t0, t1 = t1, t0

    y_db = _compute_log_envelope_db(env)
    y0_data, y1_data, r2, rms_db, n_points = _compute_linearity_metrics(
        t, y_db, t0, t1, tx_db
    )

    dt_s = float(t1 - t0)
    slope_db_per_s = float(tx_db / dt_s) if dt_s > 0.0 else np.nan
    zeta = None
    if fn_hz is not None and fn_hz > 0.0 and np.isfinite(slope_db_per_s):
        zeta = float((slope_db_per_s * np.log(10.0)) / (20.0 * 2.0 * np.pi * fn_hz))

    flags: list[FlagRecord] = []
    if dt_s <= 0.0 or dt_s < config.min_span_s or n_points < config.min_points:
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="tx_span",
                severity="warn",
                code="flag_span_too_short",
                message="Span is too short for reliable diagnostics",
                details=f"dt_s={dt_s:.6f}, n_points={n_points}",
            )
        )

    if np.isfinite(r2) and r2 < config.low_linearity_r2:
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="tx_span",
                severity="warn",
                code="flag_low_linearity",
                message="Linearity is below the threshold",
                details=f"r2={r2:.4f}, threshold={config.low_linearity_r2:.2f}",
            )
        )

    if np.isfinite(rms_db) and rms_db > config.high_rms_db:
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="tx_span",
                severity="warn",
                code="flag_high_rms_db",
                message="RMS deviation is above the threshold",
                details=f"rms_db={rms_db:.3f}, threshold={config.high_rms_db:.2f}",
            )
        )

    if np.isfinite(y0_data) and np.isfinite(y1_data):
        actual_drop = float(y0_data - y1_data)
        if abs(actual_drop - tx_db) > config.mismatch_db:
            flags.append(
                FlagRecord(
                    scope="global",
                    scope_id="tx_span",
                    severity="warn",
                    code="flag_db_drop_mismatch",
                    message="Measured dB drop differs from selected Tx",
                    details=f"drop_db={actual_drop:.3f}, tx_db={tx_db:.1f}",
                )
            )

    finite_mask = np.isfinite(y_db)
    span_mask = (t >= t0) & (t <= t1) & finite_mask
    y_span = y_db[span_mask]
    if y_span.size > 1:
        diffs = np.diff(y_span)
        if np.any(diffs > config.non_monotonic_tolerance_db):
            flags.append(
                FlagRecord(
                    scope="global",
                    scope_id="tx_span",
                    severity="info",
                    code="flag_non_monotonic",
                    message="Span contains non-monotonic decay",
                )
            )

    if finite_mask.any():
        floor_db = float(np.min(y_db[finite_mask]))
        if (
            y_span.size > 0
            and np.min(y_span) <= floor_db + config.noise_floor_margin_db
        ):
            flags.append(
                FlagRecord(
                    scope="global",
                    scope_id="tx_span",
                    severity="info",
                    code="flag_hits_noise_floor",
                    message="Span approaches the noise floor",
                    details=f"floor_db={floor_db:.2f}",
                )
            )

    return TxSpanMeasurement(
        t0=float(t0),
        t1=float(t1),
        Tx_active=_format_tx_label(tx_db),
        Tx_value=float(tx_db),
        slope_db_per_s=float(slope_db_per_s),
        linearity_r2=float(r2),
        linearity_rms_db=float(rms_db),
        flags=flags,
        zeta=zeta,
    )


@dataclass
class TxSpanUI:
    """Interactive Matplotlib UI for span-based Tx measurement."""

    t: np.ndarray
    env: np.ndarray
    fn_hz: float | None = None
    tx_options_db: list[float] = field(default_factory=lambda: [10.0, 20.0, 30.0, 60.0])
    config: TxSpanConfig = field(default_factory=TxSpanConfig)

    t0: float | None = None
    t1: float | None = None
    tx_index: int = 0
    measurement: TxSpanMeasurement | None = None
    committed: bool = False
    display_mode: str = "db"
    env_line: Line2D | None = None
    span_selector: SpanSelector | None = None
    overlay_rect: Rectangle | None = None
    overlay_line: Line2D | None = None
    _y_db: np.ndarray | None = None
    _y_amp: np.ndarray | None = None

    def run(self) -> TxSpanMeasurement | None:
        """Launch the interactive session and return a measurement if committed."""
        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self._y_db = _compute_log_envelope_db(self.env)
        self._y_amp = np.abs(self.env)
        self.env_line = self.ax.plot(
            self.t,
            self._current_display_y(),
            "k-",
            alpha=0.6,
            linewidth=1,
            label=self._current_display_label(),
        )[0]

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(self._current_display_label())
        title = "Span-based Tx measurement"
        if self.fn_hz is not None:
            title = f"{title} (fn={self.fn_hz:.1f} Hz)"
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)

        self.info_text = self.ax.text(
            0.98,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="right",
            fontsize=12,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        self.span_selector = SpanSelector(
            self.ax,
            onselect=self._on_span_select,
            direction="horizontal",
            useblit=True,
            interactive=True,
            props=dict(facecolor="tab:blue", alpha=0.15),
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._set_initial_span()
        self._refresh_overlay()
        plt.show()

        return self.measurement if self.measurement is not None else None

    def _set_initial_span(self) -> None:
        if self.t.size == 0:
            return
        self.t0 = float(self.t[0])
        self.t1 = float(self.t[-1])

    def _on_span_select(self, xmin: float, xmax: float) -> None:
        self.t0 = float(min(xmin, xmax))
        self.t1 = float(max(xmin, xmax))
        self._refresh_overlay()

    def _on_key(self, event: Event) -> None:
        key = getattr(event, "key", None)
        if key == "t":
            self.tx_index = (self.tx_index + 1) % len(self.tx_options_db)
            self._refresh_overlay()
            return

        if key == "l":
            self._toggle_display_mode()
            return

        if key == "q":
            self.committed = True
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def _on_close(self, event: Event) -> None:
        self.committed = True

    def _current_tx(self) -> float:
        if not self.tx_options_db:
            return 60.0
        return float(self.tx_options_db[self.tx_index])

    def _current_display_label(self) -> str:
        if self.display_mode == "amp":
            return "Envelope amplitude"
        return "Envelope (dB re max)"

    def _current_display_y(self) -> np.ndarray:
        if self.display_mode == "amp":
            if self._y_amp is None:
                self._y_amp = np.abs(self.env)
            return self._y_amp
        if self._y_db is None:
            self._y_db = _compute_log_envelope_db(self.env)
        return self._y_db

    def _set_display_ylim(self) -> None:
        y_display = self._current_display_y()
        finite_mask = np.isfinite(y_display)
        if not finite_mask.any():
            return

        y_min = float(np.min(y_display[finite_mask]))
        y_max = float(np.max(y_display[finite_mask]))
        if self.display_mode == "amp":
            y_min = max(0.0, y_min)

        if y_max <= y_min:
            y_max = y_min + 1.0

        pad = 0.05 * (y_max - y_min)
        lower = y_min - pad
        upper = y_max + pad
        if self.display_mode == "amp":
            lower = max(0.0, lower)

        self.ax.set_ylim(lower, upper)

    def _toggle_display_mode(self) -> None:
        self.display_mode = "amp" if self.display_mode == "db" else "db"
        if self.env_line is not None:
            y_display = self._current_display_y()
            self.env_line.set_data(self.t, y_display)
            self.env_line.set_label(self._current_display_label())
        self.ax.set_ylabel(self._current_display_label())
        self.ax.set_yscale("linear")
        self._set_display_ylim()
        self._refresh_overlay()
        self.fig.canvas.draw_idle()

    def _refresh_overlay(self) -> None:
        if self.t0 is None or self.t1 is None:
            return

        tx_db = self._current_tx()
        self.measurement = compute_tx_span_measurement(
            self.t,
            self.env,
            self.t0,
            self.t1,
            tx_db,
            fn_hz=self.fn_hz,
            config=self.config,
        )

        y_display = self._current_display_y()
        finite_mask = np.isfinite(y_display)
        if not finite_mask.any():
            return

        t0 = self.measurement.t0
        t1 = self.measurement.t1
        span_mask = (self.t >= t0) & (self.t <= t1) & finite_mask
        t_span = self.t[span_mask]
        y_span = y_display[span_mask]
        if t_span.size == 0:
            return

        y0 = float(np.interp(t0, t_span, y_span))
        if self.display_mode == "amp":
            ratio = 10.0 ** (-tx_db / 20.0)
            y1 = y0 * ratio
            rect_height = y0 - y1
        else:
            y1 = y0 - tx_db
            rect_height = tx_db

        if self.overlay_rect is None:
            self.overlay_rect = Rectangle(
                (t0, y1),
                t1 - t0,
                rect_height,
                facecolor="tab:orange",
                edgecolor="tab:orange",
                alpha=0.2,
                linewidth=1.5,
            )
            self.ax.add_patch(self.overlay_rect)
        else:
            self.overlay_rect.set_x(t0)
            self.overlay_rect.set_y(y1)
            self.overlay_rect.set_width(t1 - t0)
            self.overlay_rect.set_height(rect_height)

        if self.display_mode == "amp":
            t_line = np.linspace(t0, t1, 100)
            span_dt = t1 - t0
            if span_dt <= 0.0:
                return
            slope_db_per_s = tx_db / span_dt
            y_line = y0 * 10.0 ** (-(slope_db_per_s * (t_line - t0)) / 20.0)
            if self.overlay_line is None:
                self.overlay_line = Line2D(
                    t_line,
                    y_line,
                    color="tab:orange",
                    linewidth=2,
                )
                self.ax.add_line(self.overlay_line)
            else:
                self.overlay_line.set_data(t_line, y_line)
        else:
            if self.overlay_line is None:
                self.overlay_line = Line2D(
                    [t0, t1],
                    [y0, y1],
                    color="tab:orange",
                    linewidth=2,
                )
                self.ax.add_line(self.overlay_line)
            else:
                self.overlay_line.set_data([t0, t1], [y0, y1])

        self._update_info_text(self.measurement)
        self.fig.canvas.draw_idle()

    def _update_info_text(self, measurement: TxSpanMeasurement) -> None:
        flags = [flag.code for flag in measurement.flags]
        flags_text = ", ".join(flags) if flags else "none"
        lines = [
            f"Tx: {measurement.Tx_active}",
            f"span: {measurement.t0:.4f} to {measurement.t1:.4f} s",
            f"slope: {measurement.slope_db_per_s:.3f} dB/s",
        ]
        if measurement.zeta is not None:
            lines.append(f"zeta: {measurement.zeta:.6f}")
        lines.extend(
            [
                f"R2: {measurement.linearity_r2:.4f}",
                f"RMS: {measurement.linearity_rms_db:.3f} dB",
                f"flags: {flags_text}",
                "t: cycle Tx | l: toggle display | q: commit",
            ]
        )
        self.info_text.set_text("\n".join(lines))


def run_tx_span_measurement_ui(
    t: np.ndarray,
    env: np.ndarray,
    *,
    fn_hz: float | None = None,
    tx_options_db: Iterable[float] | None = None,
    config: TxSpanConfig | None = None,
) -> TxSpanMeasurement | None:
    """Run the interactive Tx span measurement UI.

    Args:
        t: Time array in seconds.
        env: Envelope amplitude (linear scale).
        fn_hz: Natural frequency in Hz (optional for zeta).
        tx_options_db: Candidate Tx drops in dB.
        config: Threshold configuration for diagnostic flags.

    Returns:
        TxSpanMeasurement payload if committed, otherwise None.
    """
    tx_list = (
        list(tx_options_db) if tx_options_db is not None else [10.0, 20.0, 30.0, 60.0]
    )
    ui = TxSpanUI(
        t=t,
        env=env,
        fn_hz=fn_hz,
        tx_options_db=tx_list,
        config=config or TxSpanConfig(),
    )
    return ui.run()
