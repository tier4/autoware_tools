#!/usr/bin/env python3

# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive comfort plots: speed, accel, jerk, and control commands across models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from comfort_metrics import ComfortAnalysisConfig
from comfort_metrics import ComfortTraceSeries
from comfort_metrics import load_trace_comfort_series


def discover_models(output_dir: Path) -> list[str]:
    models = []
    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            if child.name in ("comparisons", "quantitative_analysis"):
                continue
            if any(child.rglob("ego_pose.csv")):
                models.append(child.name)
    return models


def discover_bag_keys(output_dir: Path, models: list[str]) -> list[str]:
    keys: set[str] = set()
    for model in models:
        model_dir = output_dir / model
        for ego_csv in model_dir.rglob("ego_pose.csv"):
            keys.add(str(ego_csv.parent.relative_to(model_dir)))
    return sorted(keys)


def shorten_model_name(name: str, max_len: int = 36) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def shade_harsh_events(ax, series: ComfortTraceSeries, color: str, alpha: float = 0.12) -> None:
    for start, end in series.harsh_spans:
        ax.axvspan(start, end, color=color, alpha=alpha)


def plot_comfort_comparison(
    output_dir: Path,
    bag_key: str,
    models: list[str],
    config: ComfortAnalysisConfig,
    save_path: Path | None,
    *,
    show_control: bool = True,
    show_plan: bool = True,
    show_lateral: bool = False,
) -> None:
    colors = plt.cm.tab10.colors
    series_list: list[ComfortTraceSeries] = []

    for model in models:
        trace_dir = output_dir / model / bag_key
        label = shorten_model_name(model)
        series = load_trace_comfort_series(trace_dir, label, config)
        if series is not None:
            series_list.append(series)

    if not series_list:
        print(f"[error] No comfort traces found for bag_key={bag_key}")
        return

    nrows = 4 if show_lateral else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 3.0 * nrows), sharex=True)
    if nrows == 3:
        ax_speed, ax_accel, ax_jerk = axes
    else:
        ax_speed, ax_accel, ax_lat, ax_jerk = axes

    for idx, series in enumerate(series_list):
        color = colors[idx % len(colors)]
        accel_note = "meas" if series.uses_measured_accel else "derived"
        ego_label = f"{series.model_label} ego ({accel_note})"

        shade_harsh_events(ax_accel, series, color)
        shade_harsh_events(ax_jerk, series, color)

        ax_speed.plot(
            series.time_sec,
            series.speed_mps,
            color=color,
            linewidth=2.0,
            label=ego_label,
        )
        ax_accel.plot(
            series.time_sec,
            series.a_long_mps2,
            color=color,
            linewidth=2.0,
            label=f"{series.model_label} ego a_long",
        )

        if show_control and series.control_time_sec:
            ax_speed.plot(
                series.control_time_sec,
                series.control_velocity_mps,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                label=f"{series.model_label} cmd v",
            )
            ax_accel.plot(
                series.control_time_sec,
                series.control_accel_mps2,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                label=f"{series.model_label} cmd a",
            )
            ax_jerk.plot(
                series.control_time_sec,
                series.control_jerk_mps3,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                label=f"{series.model_label} cmd jerk",
            )

        if show_plan and series.plan_time_sec:
            ax_accel.plot(
                series.plan_time_sec,
                series.plan_accel_mps2,
                color=color,
                linewidth=1.0,
                linestyle=":",
                alpha=0.7,
                label=f"{series.model_label} plan a",
            )

        ax_jerk.plot(
            series.jerk_time_sec,
            series.jerk_long_mps3,
            color=color,
            linewidth=2.0,
            label=f"{series.model_label} ego jerk",
        )

        if show_lateral:
            ax_lat.plot(
                series.time_sec,
                series.a_lat_mps2,
                color=color,
                linewidth=2.0,
                label=f"{series.model_label} ego a_lat",
            )

    ax_speed.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax_accel.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax_accel.axhline(
        config.harsh_decel_threshold_mps2,
        color="red",
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
        label=f"harsh decel ({config.harsh_decel_threshold_mps2} m/s²)",
    )
    ax_jerk.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    if show_lateral:
        ax_lat.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)

    ax_speed.set_ylabel("speed [m/s]")
    ax_accel.set_ylabel("long. accel [m/s²]")
    if show_lateral:
        ax_lat.set_ylabel("lat. accel [m/s²]")
    ax_jerk.set_ylabel("long. jerk [m/s³]")
    ax_jerk.set_xlabel("time since run start [s]")

    ax_speed.set_title(f"Comfort comparison — {bag_key}")
    ax_speed.grid(True, alpha=0.3)
    ax_accel.grid(True, alpha=0.3)
    ax_jerk.grid(True, alpha=0.3)
    if show_lateral:
        ax_lat.grid(True, alpha=0.3)

    ax_speed.legend(loc="upper right", fontsize=8)
    ax_accel.legend(loc="upper right", fontsize=8)
    ax_jerk.legend(loc="upper right", fontsize=8)
    if show_lateral:
        ax_lat.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[info] Saved {save_path}")
        plt.close(fig)
    else:
        print("[info] Interactive plot — close window to exit")
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot speed / acceleration / jerk and control commands across models."
    )
    parser.add_argument("-o", "--output-dir", required=True, type=Path, help="Batch eval output dir")
    parser.add_argument("--models", help="Comma-separated model names (default: all with traces)")
    parser.add_argument("--bag-key", help="Bag key relative to model dir, e.g. ID1/bag_name")
    parser.add_argument("--list-bags", action="store_true", help="List available bag keys and exit")
    parser.add_argument("--save", type=Path, help="Save PNG instead of interactive window")
    parser.add_argument(
        "--render-all",
        action="store_true",
        help="Save PNG for all bag keys (requires --models with two models)",
    )
    parser.add_argument("--no-control", action="store_true", help="Hide control command overlays")
    parser.add_argument("--no-plan", action="store_true", help="Hide planned trajectory overlays")
    parser.add_argument("--show-lateral", action="store_true", help="Add lateral acceleration subplot")
    parser.add_argument("--comfort-sample-dt", type=float, default=0.1)
    parser.add_argument("--harsh-decel-threshold-mps2", type=float, default=-2.5)
    parser.add_argument("--harsh-decel-min-duration-sec", type=float, default=0.3)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_dir():
        print(f"[error] Output directory not found: {output_dir}")
        sys.exit(1)

    models = (
        [name.strip() for name in args.models.split(",") if name.strip()]
        if args.models
        else discover_models(output_dir)
    )
    if not models:
        print("[error] No models with ego_pose.csv traces found.")
        sys.exit(1)

    bag_keys = discover_bag_keys(output_dir, models)

    if args.list_bags:
        print(f"Models: {', '.join(models)}")
        for key in bag_keys:
            print(f"  {key}")
        return

    config = ComfortAnalysisConfig(
        sample_dt=args.comfort_sample_dt,
        min_derivative_dt=0.05,
        max_derivative_dt=0.5,
        harsh_decel_threshold_mps2=args.harsh_decel_threshold_mps2,
        harsh_decel_min_duration_sec=args.harsh_decel_min_duration_sec,
    )

    if args.render_all:
        out_dir = output_dir / "comparisons" / "comfort"
        out_dir.mkdir(parents=True, exist_ok=True)
        for bag_key in bag_keys:
            safe_name = bag_key.replace("/", "__")
            plot_comfort_comparison(
                output_dir,
                bag_key,
                models,
                config,
                out_dir / f"{safe_name}.png",
                show_control=not args.no_control,
                show_plan=not args.no_plan,
                show_lateral=args.show_lateral,
            )
        print(f"[info] Wrote comfort plots to {out_dir}")
        return

    if not args.bag_key:
        print("[error] Pass --bag-key or use --list-bags / --render-all")
        sys.exit(1)

    plot_comfort_comparison(
        output_dir,
        args.bag_key,
        models,
        config,
        args.save,
        show_control=not args.no_control,
        show_plan=not args.no_plan,
        show_lateral=args.show_lateral,
    )


if __name__ == "__main__":
    main()
