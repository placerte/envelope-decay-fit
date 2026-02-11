"""Tx span UI smoke tests."""

import matplotlib

matplotlib.use("Agg")

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import Event

from envelope_decay_fit.segmentation.tx_span import TxSpanUI, _compute_log_envelope_db


def test_tx_span_ui_launch_sanity(monkeypatch) -> None:
    """Ensure the Tx span UI can launch headlessly."""
    t = np.linspace(0.0, 1.0, 200)
    env = np.exp(-3.0 * t)

    ui = TxSpanUI(t=t, env=env, fn_hz=120.0)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    measurement = ui.run()

    assert measurement is not None
    assert measurement.t0 == float(t[0])
    assert measurement.t1 == float(t[-1])
    assert measurement.Tx_value in ui.tx_options_db
    assert np.isfinite(measurement.slope_db_per_s)


def test_tx_span_ui_toggle_display_mode(monkeypatch) -> None:
    t = np.linspace(0.0, 1.0, 200)
    env = np.exp(-3.0 * t)

    ui = TxSpanUI(t=t, env=env, fn_hz=120.0)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    ui.run()

    class EventStub:
        def __init__(self, key: str) -> None:
            self.key = key

    assert ui.display_mode == "db"
    ui._on_key(cast(Event, EventStub("l")))

    assert ui.display_mode == "amp"
    assert ui.env_line is not None
    np.testing.assert_allclose(np.asarray(ui.env_line.get_ydata()), np.abs(env))
    assert ui.ax.get_ylabel() == "Envelope amplitude"

    ui._on_key(cast(Event, EventStub("l")))
    assert ui.display_mode == "db"
    np.testing.assert_allclose(
        np.asarray(ui.env_line.get_ydata()), _compute_log_envelope_db(env)
    )
    assert ui.ax.get_ylabel() == "Envelope (dB re max)"


def test_tx_span_ui_overlay_amp_conversion(monkeypatch) -> None:
    t = np.linspace(0.0, 1.0, 200)
    env = np.exp(-2.0 * t)

    ui = TxSpanUI(t=t, env=env, fn_hz=120.0, tx_options_db=[20.0])
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    ui.run()

    class EventStub:
        def __init__(self, key: str) -> None:
            self.key = key

    ui._on_key(cast(Event, EventStub("l")))

    y0_amp = float(np.abs(env[0]))
    y1_amp = y0_amp * 10.0 ** (-20.0 / 20.0)
    assert ui.overlay_line is not None
    line_y = np.asarray(ui.overlay_line.get_ydata())
    np.testing.assert_allclose(line_y[0], y0_amp)
    np.testing.assert_allclose(line_y[-1], y1_amp)
    assert np.all(np.diff(line_y) <= 0.0)
