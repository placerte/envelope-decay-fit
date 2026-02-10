"""Tx span UI smoke tests."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from envelope_decay_fit.segmentation.tx_span import TxSpanUI


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
