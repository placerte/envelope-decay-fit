"""Manual UI check script for envelope-decay-fit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np

from envelope_decay_fit.segmentation.manual import ManualSegmentationUI


matplotlib.use("Agg")


def _make_fixture() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.2, 500)
    env = 1.2 * np.exp(-2.4 * t) + 0.02
    bump = 0.12 * np.exp(-((t - 0.15) ** 2) / 0.002)
    ripple = 0.015 * np.sin(2.0 * np.pi * 6.0 * t)
    env = env + bump + ripple
    env = np.clip(env, 1e-6, None)
    return t, env


def _save_ui_png(ui: ManualSegmentationUI, out_path: Path) -> None:
    if ui.fig is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ui.fig.savefig(out_path, dpi=150)


def _check_box_overlap(ui: ManualSegmentationUI) -> list[str]:
    if ui.fig is None:
        return ["figure_not_initialized"]

    renderer = cast(Any, ui.fig.canvas).get_renderer()
    issues: list[str] = []

    legend_bbox = None
    if ui.legend is not None:
        legend_bbox = ui.legend.get_window_extent(renderer=renderer)

    help_bbox = None
    if ui.help_text is not None:
        help_bbox = ui.help_text.get_window_extent(renderer=renderer)

    info_bbox = None
    if ui.info_text is not None:
        info_bbox = ui.info_text.get_window_extent(renderer=renderer)

    if legend_bbox is not None and help_bbox is not None:
        if legend_bbox.overlaps(help_bbox):
            issues.append("legend_overlaps_help")

    if legend_bbox is not None and info_bbox is not None:
        if legend_bbox.overlaps(info_bbox):
            issues.append("legend_overlaps_info")

    if help_bbox is not None and info_bbox is not None:
        if help_bbox.overlaps(info_bbox):
            issues.append("help_overlaps_info")

    return issues


def main() -> None:
    t, env = _make_fixture()
    ui = ManualSegmentationUI(
        t=t,
        env=env,
        fn_hz=120.0,
        min_points=20,
        initial_boundaries_time_s=[0.0, 0.6, 1.2],
    )
    ui.help_visible = True

    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: None
    ui.run()

    ui._refresh_plot(recompute=False)
    ui.fig.canvas.draw()

    out_path = Path("examples/manual_ui_check.png")
    _save_ui_png(ui, out_path)

    issues = _check_box_overlap(ui)
    if issues:
        print("overlap_issues:", ", ".join(issues))
    else:
        print("overlap_issues: none")


if __name__ == "__main__":
    main()
