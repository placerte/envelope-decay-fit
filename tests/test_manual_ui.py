"""Manual UI smoke tests."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from envelope_decay_fit.segmentation.manual import ManualSegmentationUI


def test_manual_ui_launch_sanity(monkeypatch) -> None:
    """Ensure the manual UI can launch headlessly."""
    t = np.linspace(0.0, 1.0, 200)
    env = np.exp(-2.0 * t)

    ui = ManualSegmentationUI(t=t, env=env, fn_hz=120.0)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    result = ui.run()

    assert result is not None
    assert result.fn_hz == 120.0
