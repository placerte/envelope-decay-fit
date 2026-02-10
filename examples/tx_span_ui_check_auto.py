"""Tx span UI check script (headless export)."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib
import numpy as np

from envelope_decay_fit.segmentation.tx_span import TxSpanUI
from envelope_decay_fit.plotting.plot_export import export_plot


matplotlib.use("Agg")


def _make_fixture() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.4, 600)
    env = 1.1 * np.exp(-2.0 * t) + 0.02
    bump = 0.1 * np.exp(-((t - 0.2) ** 2) / 0.004)
    ripple = 0.02 * np.sin(2.0 * np.pi * 5.0 * t)
    env = env + bump + ripple
    env = np.clip(env, 1e-6, None)
    return t, env


def _measurement_to_dict(measurement) -> dict[str, object]:
    flags_payload = []
    for flag in measurement.flags:
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

    return {
        "t0": float(measurement.t0),
        "t1": float(measurement.t1),
        "Tx_active": measurement.Tx_active,
        "Tx_value": float(measurement.Tx_value),
        "slope_db_per_s": float(measurement.slope_db_per_s),
        "linearity_r2": float(measurement.linearity_r2),
        "linearity_rms_db": float(measurement.linearity_rms_db),
        "flags": flags_payload,
        "zeta": None if measurement.zeta is None else float(measurement.zeta),
    }


def main() -> None:
    t, env = _make_fixture()
    ui = TxSpanUI(
        t=t,
        env=env,
        fn_hz=120.0,
        tx_options_db=[10.0, 20.0, 30.0, 60.0],
    )

    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: None
    measurement = ui.run()

    if measurement is None or ui.fig is None:
        return

    ui.fig.canvas.draw()

    png_path = Path("examples/tx_span_ui_check.png")
    json_path = Path("examples/tx_span_measurement.json")
    context = {
        "plot_kind": "tx_span",
        "fn_hz": 120.0,
        "Tx_active": measurement.Tx_active,
        "Tx_value": measurement.Tx_value,
        "t0": measurement.t0,
        "t1": measurement.t1,
    }

    export_plot(ui.fig, png_path, context=context, dpi=150)

    payload = _measurement_to_dict(measurement)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"tx_span plot: {png_path}")
    print(f"tx_span measurement: {json_path}")


if __name__ == "__main__":
    main()
