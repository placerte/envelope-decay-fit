"""Tx span UI check script for visual inspection."""

from __future__ import annotations

import numpy as np

from envelope_decay_fit.segmentation.tx_span import TxSpanUI


def _make_fixture() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.4, 600)
    env = 1.1 * np.exp(-2.0 * t) + 0.02
    bump = 0.1 * np.exp(-((t - 0.2) ** 2) / 0.004)
    ripple = 0.02 * np.sin(2.0 * np.pi * 5.0 * t)
    env = env + bump + ripple
    env = np.clip(env, 1e-6, None)
    return t, env


def main() -> None:
    t, env = _make_fixture()
    ui = TxSpanUI(
        t=t,
        env=env,
        fn_hz=120.0,
        tx_options_db=[10.0, 20.0, 30.0, 60.0],
    )
    ui.run()


if __name__ == "__main__":
    main()
