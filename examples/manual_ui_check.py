"""Manual UI check script for visual inspection."""

from __future__ import annotations

import numpy as np

from envelope_decay_fit.segmentation.manual import ManualSegmentationUI


def _make_fixture() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.2, 500)
    env = 1.2 * np.exp(-2.4 * t) + 0.02
    bump = 0.12 * np.exp(-((t - 0.15) ** 2) / 0.002)
    ripple = 0.015 * np.sin(2.0 * np.pi * 6.0 * t)
    env = env + bump + ripple
    env = np.clip(env, 1e-6, None)
    return t, env


def main() -> None:
    t, env = _make_fixture()
    ui = ManualSegmentationUI(
        t=t,
        env=env,
        fn_hz=120.0,
        min_points=20,
        initial_boundaries_time_s=[0.0, 0.6, 1.2],
    )
    ui.run()


if __name__ == "__main__":
    main()
