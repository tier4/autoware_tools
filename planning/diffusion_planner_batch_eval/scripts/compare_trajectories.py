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

"""Compare diffusion planner trajectories across models for the same rosbag."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def discover_models(output_dir: Path) -> list[str]:
    models = []
    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            if child.name == "comparisons":
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


def load_trajectory_polylines(path: Path) -> list[tuple[float, list[tuple[float, float]]]]:
    rows = read_csv_rows(path)
    grouped: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        stamp = float(row["stamp_sec"])
        grouped[stamp].append((float(row["x"]), float(row["y"])))
    return sorted((stamp, points) for stamp, points in grouped.items())


def load_ego_path(path: Path) -> list[tuple[float, float]]:
    return [(float(row["x"]), float(row["y"])) for row in read_csv_rows(path)]


def load_objects_at_stamp(path: Path, stamp_sec: float, tolerance: float = 0.25):
    rows = read_csv_rows(path)
    if not rows:
        return []
    stamps = sorted({float(row["stamp_sec"]) for row in rows})
    target = min(stamps, key=lambda value: abs(value - stamp_sec))
    if abs(target - stamp_sec) > tolerance:
        return []
    return [row for row in rows if abs(float(row["stamp_sec"]) - target) < 1e-6]


def ego_point_at_focus(ego_path: list[tuple[float, float]], focus: str) -> tuple[float, float] | None:
    if not ego_path:
        return None
    if focus == "start":
        return ego_path[0]
    if focus == "end":
        return ego_path[-1]
    mid = len(ego_path) // 2
    return ego_path[mid]


def apply_plot_limits(
    ax,
    *,
    ego_path: list[tuple[float, float]],
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    zoom_range: float | None,
    focus: str,
    focus_xy: tuple[float, float] | None,
) -> None:
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None or ylim is not None:
        return

    if zoom_range is None:
        return

    center = focus_xy
    if center is None:
        center = ego_point_at_focus(ego_path, focus)
    if center is None:
        return

    half = zoom_range / 2.0
    cx, cy = center
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def shorten_model_name(name: str, max_len: int = 40) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def combined_ego_paths(paths: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    if not paths:
        return []
    if len(paths) == 1:
        return paths[0]
    return [point for path in paths for point in path]


def plot_comparison(
    output_dir: Path,
    bag_key: str,
    models: list[str],
    snapshot_stamp: float | None,
    save_path: Path | None,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zoom_range: float | None = None,
    focus: str = "end",
    focus_xy: tuple[float, float] | None = None,
    show_plans: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10.colors
    model_ego_paths: list[tuple[str, list[tuple[float, float]]]] = []

    for idx, model in enumerate(models):
        trace_dir = output_dir / model / bag_key
        ego_csv = trace_dir / "ego_pose.csv"
        traj_csv = trace_dir / "planned_trajectory.csv"
        color = colors[idx % len(colors)]
        short_name = shorten_model_name(model)

        ego_path = load_ego_path(ego_csv) if ego_csv.exists() else []
        if ego_path:
            model_ego_paths.append((model, ego_path))
            xs, ys = zip(*ego_path)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.5,
                linestyle="-",
                label=f"{short_name} (ego)",
                zorder=3,
            )

        if not show_plans or not traj_csv.exists():
            continue
        polylines = load_trajectory_polylines(traj_csv)
        if not polylines:
            continue

        if snapshot_stamp is None:
            sample = polylines[:: max(1, len(polylines) // 15)]
        else:
            sample = [min(polylines, key=lambda item: abs(item[0] - snapshot_stamp))]

        for stamp, points in sample:
            if len(points) < 2:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs,
                ys,
                color=color,
                alpha=0.2,
                linewidth=0.8,
                linestyle="--",
            )
        stamp, points = polylines[-1]
        if len(points) >= 2:
            xs, ys = zip(*points)
            ax.plot(
                xs,
                ys,
                color=color,
                alpha=0.55,
                linewidth=1.2,
                linestyle="--",
                label=f"{short_name} (plan)",
            )

    focus_path = combined_ego_paths([path for _, path in model_ego_paths])

    if snapshot_stamp is not None and model_ego_paths:
        objects_csv = None
        for model in models:
            candidate = output_dir / model / bag_key / "objects.csv"
            if candidate.exists():
                objects_csv = candidate
                break
        if objects_csv is not None:
            for row in load_objects_at_stamp(objects_csv, snapshot_stamp):
                x = float(row["x"])
                y = float(row["y"])
                length = float(row["length_m"])
                width = float(row["width_m"])
                yaw = float(row["yaw_rad"])
                label = row["label"]
                rect = Rectangle(
                    (x - length / 2.0, y - width / 2.0),
                    length,
                    width,
                    angle=math.degrees(yaw),
                    rotation_point="center",
                    fill=False,
                    linewidth=1.0,
                    edgecolor="gray",
                )
                ax.add_patch(rect)
                ax.text(x, y, label[:3], fontsize=7, ha="center", va="center", color="gray")

    ax.set_aspect("equal", adjustable="box")
    apply_plot_limits(
        ax,
        ego_path=focus_path,
        xlim=xlim,
        ylim=ylim,
        zoom_range=zoom_range,
        focus=focus,
        focus_xy=focus_xy,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Per-model ego paths: {bag_key}")
    ax.legend(loc="best")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[info] Saved {save_path}")
        plt.close(fig)
    else:
        enable_interactive_navigation(fig, ax)
        print(
            "[info] Interactive view — scroll=zoom, left-drag=pan, "
            "r=reset, h=fit all, q=close"
        )
        plt.show()


def enable_interactive_navigation(fig, ax) -> None:
    """Mouse scroll zoom, left-drag pan, and keyboard shortcuts."""
    initial_xlim = ax.get_xlim()
    initial_ylim = ax.get_ylim()
    pan_state = {
        "active": False,
        "x": 0.0,
        "y": 0.0,
        "xlim": initial_xlim,
        "ylim": initial_ylim,
    }

    def on_scroll(event) -> None:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.85 if event.button == "up" else 1.0 / 0.85
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        width = (x1 - x0) * scale
        height = (y1 - y0) * scale
        rel_x = (x1 - event.xdata) / (x1 - x0)
        rel_y = (y1 - event.ydata) / (y1 - y0)
        ax.set_xlim(event.xdata - width * (1.0 - rel_x), event.xdata + width * rel_x)
        ax.set_ylim(event.ydata - height * (1.0 - rel_y), event.ydata + height * rel_y)
        fig.canvas.draw_idle()

    def on_press(event) -> None:
        # Left or right mouse button starts pan (works on laptop trackpads).
        if event.inaxes is not ax or event.button not in (1, 3):
            return
        pan_state["active"] = True
        pan_state["x"] = event.x
        pan_state["y"] = event.y
        pan_state["xlim"] = ax.get_xlim()
        pan_state["ylim"] = ax.get_ylim()

    def on_release(_event) -> None:
        pan_state["active"] = False

    def on_motion(event) -> None:
        if not pan_state["active"]:
            return
        dx = event.x - pan_state["x"]
        dy = event.y - pan_state["y"]
        x0, x1 = pan_state["xlim"]
        y0, y1 = pan_state["ylim"]
        scale_x = (x1 - x0) / ax.bbox.width
        scale_y = (y1 - y0) / ax.bbox.height
        ax.set_xlim(x0 - dx * scale_x, x1 - dx * scale_x)
        ax.set_ylim(y0 - dy * scale_y, y1 - dy * scale_y)
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in ("r", "R"):
            ax.set_xlim(initial_xlim)
            ax.set_ylim(initial_ylim)
            fig.canvas.draw_idle()
        elif event.key in ("h", "H"):
            ax.autoscale()
            ax.set_aspect("equal", adjustable="box")
            fig.canvas.draw_idle()
        elif event.key in ("q", "Q"):
            plt.close(fig)

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-model executed ego paths (and optional planned trajectories) "
            "for the same rosbag."
        )
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Batch eval output directory containing per-model folders",
    )
    parser.add_argument(
        "--bag-key",
        help="Relative trace path, e.g. ID1/94eca28e-..._p0900_27. List with --list-bags.",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model names. Default: all models found in output-dir.",
    )
    parser.add_argument(
        "--snapshot-stamp",
        type=float,
        default=None,
        help="Optional time [s] to overlay NPC boxes from objects.csv",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Save PNG instead of showing interactively",
    )
    parser.add_argument(
        "--ego-only",
        action="store_true",
        help="Show only executed ego paths per model (hide planned trajectory overlays)",
    )
    parser.add_argument(
        "--list-bags",
        action="store_true",
        help="List bag keys that have trajectory logs and exit",
    )
    parser.add_argument(
        "--render-all",
        action="store_true",
        help="Render comparison PNG for every bag key into output-dir/comparisons/",
    )
    parser.add_argument(
        "--zoom-range",
        type=float,
        default=None,
        metavar="M",
        help="Zoom to a square window of M meters (e.g. 80 = 80m x 80m)",
    )
    parser.add_argument(
        "--focus",
        choices=["start", "center", "end"],
        default="end",
        help="Center --zoom-range on ego path start/center/end (default: end)",
    )
    parser.add_argument(
        "--focus-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Center --zoom-range on map x,y instead of --focus",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("XMIN", "XMAX"),
        help="Explicit x axis limits [m]",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        help="Explicit y axis limits [m]",
    )
    args = parser.parse_args()

    focus_xy = tuple(args.focus_xy) if args.focus_xy is not None else None
    xlim = tuple(args.xlim) if args.xlim is not None else None
    ylim = tuple(args.ylim) if args.ylim is not None else None
    plot_kwargs = {
        "xlim": xlim,
        "ylim": ylim,
        "zoom_range": args.zoom_range,
        "focus": args.focus,
        "focus_xy": focus_xy,
        "show_plans": not args.ego_only,
    }

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
        print(f"[error] No model result folders found in {output_dir}")
        sys.exit(1)

    bag_keys = discover_bag_keys(output_dir, models)
    if args.list_bags:
        for key in bag_keys:
            print(key)
        sys.exit(0)

    if args.render_all:
        comparison_dir = output_dir / "comparisons"
        for bag_key in bag_keys:
            save_path = comparison_dir / f"{bag_key.replace('/', '__')}.png"
            plot_comparison(
                output_dir, bag_key, models, args.snapshot_stamp, save_path, **plot_kwargs
            )
        print(f"[info] Rendered {len(bag_keys)} comparison plot(s) to {comparison_dir}")
        sys.exit(0)

    if not args.bag_key:
        print("[error] Specify --bag-key, or use --list-bags / --render-all")
        sys.exit(1)

    plot_comparison(
        output_dir, args.bag_key, models, args.snapshot_stamp, args.save, **plot_kwargs
    )


if __name__ == "__main__":
    main()
