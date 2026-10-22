#!/usr/bin/env python3
# Copyright 2026 TIER IV, Inc.
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
"""Plot ADE/FDE by location along the route.

Reads time_step_based_trajectory_frame_geometry.jsonl, time_step_based_trajectory_route.json and
time_step_based_trajectory_result.jsonl written by autoware_planning_data_analyzer with
export_frame_geometry:=true, bins the per-frame errors (ADE, FDE, average lateral / longitudinal
deviation) by route arc length, and writes one PNG per (metric, horizon):

- top: the route colored by the mean error of each bin
- bottom: mean / max error of each bin along the route arc length

Example:
    ros2 run autoware_planning_data_analyzer plot_error_map.py <analyzer output_dir>
"""

import argparse
from dataclasses import dataclass
from dataclasses import field
import json
import math
from pathlib import Path
import sys
from typing import Dict
from typing import List
from typing import Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FRAME_GEOMETRY_FILE = "time_step_based_trajectory_frame_geometry.jsonl"
ROUTE_FILE = "time_step_based_trajectory_route.json"
RESULT_FILE = "time_step_based_trajectory_result.jsonl"


@dataclass(frozen=True)
class Metric:
    name: str  # shown in the title, e.g. "FDE"
    info_key: str  # key in Frame.<horizon>.Info of time_step_based_trajectory_result.jsonl
    average_label: str  # label of the average in the title, e.g. "Average FDE"


METRICS = {
    "ade": Metric("ADE", "ADE", "Average ADE"),
    "fde": Metric("FDE", "FDE", "Average FDE"),
    "avg_lateral": Metric(
        "Lateral deviation", "average_lateral_deviation", "Average lateral deviation"
    ),
    "avg_longitudinal": Metric(
        "Longitudinal deviation", "average_longitudinal_deviation", "Average longitudinal deviation"
    ),
}
POSITIONS = {
    "inference": "position at inference time",
    "evaluated": "position at evaluated time",
}

# Colors (light theme)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
ROUTE_NO_DATA = "#c3c2b7"
INTERSECTION_BAND = "#f0efec"
SERIES_MEAN = "#2a78d6"
SERIES_MAX = "#eb6834"
# Sequential blue ramp (light -> dark = small -> large error). The lightest steps are skipped
# so that small errors stay visible on the light surface.
SEQUENTIAL_BLUE = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


@dataclass
class Route:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    length: float
    intersections: List[tuple] = field(default_factory=list)  # (s_start, s_end)


@dataclass
class FrameError:
    """Error of one frame at one horizon, with the arc length of both positions."""

    s_inference: Optional[float]
    s_evaluated: Optional[float]
    values: Dict[str, float]  # metric key -> value


@dataclass
class Run:
    """One analyzer output. A report may compare several runs in the future."""

    label: str
    route: Route
    # horizon key ("1.0s", ...) -> errors of the frames that have the horizon
    errors: Dict[str, List[FrameError]]
    num_frames: int
    num_selected_frames: int


@dataclass
class Bins:
    edges: np.ndarray  # bin edges along route_s [m]
    mean: np.ndarray  # NaN for bins without data
    max: np.ndarray
    count: np.ndarray


def load_route(path: Path) -> Route:
    with path.open() as f:
        route_json = json.load(f)
    centerline = route_json["centerline"]
    intersections = [
        (lanelet["s_start"], lanelet["s_end"])
        for lanelet in route_json.get("lanelets", [])
        if lanelet.get("is_intersection")
    ]
    return Route(
        s=np.asarray(centerline["s"], dtype=float),
        x=np.asarray(centerline["x"], dtype=float),
        y=np.asarray(centerline["y"], dtype=float),
        length=float(route_json["length_m"] or 0.0),
        intersections=intersections,
    )


def load_run(output_dir: Path, label: str, control_mode: str) -> Run:
    frame_path = output_dir / FRAME_GEOMETRY_FILE
    route_path = output_dir / ROUTE_FILE
    result_path = output_dir / RESULT_FILE
    for path in (frame_path, route_path, result_path):
        if not path.exists():
            sys.exit(
                f"{path} not found. Run autoware_planning_data_analyzer with "
                "export_frame_geometry:=true."
            )

    errors: Dict[str, List[FrameError]] = {}
    num_frames = 0
    num_selected = 0
    # The N-th lines of frame_geometry.jsonl and result.jsonl are the same frame.
    with frame_path.open() as frame_file, result_path.open() as result_file:
        for frame_line, result_line in zip(frame_file, result_file):
            frame = json.loads(frame_line)
            result = json.loads(result_line)
            if frame["timestamp_sec"] != result["Stamp"]["ROS"]:
                sys.exit(
                    f"Frame {num_frames} of {frame_path.name} and {result_path.name} do not "
                    "match. Use the files of the same analyzer run."
                )
            num_frames += 1
            if not frame["valid"]:
                continue
            if control_mode != "all" and frame["control_mode"] != control_mode:
                continue
            num_selected += 1
            ego = frame["ego"] or {}
            gt_route_s = frame["ground_truth"]["route_s"]
            for key, horizon in frame["horizons"].items():
                index = horizon["index"]
                info = result["Frame"][key]["Info"]
                errors.setdefault(key, []).append(
                    FrameError(
                        s_inference=ego.get("route_s"),
                        s_evaluated=gt_route_s[index] if index < len(gt_route_s) else None,
                        values={k: info[m.info_key] for k, m in METRICS.items()},
                    )
                )
    return Run(label, load_route(route_path), errors, num_frames, num_selected)


def compute_bins(
    errors: List[FrameError], metric: str, position: str, length: float, bin_size: float
) -> Bins:
    num_bins = max(1, int(math.ceil(length / bin_size)))
    edges = np.arange(num_bins + 1) * bin_size
    values: List[List[float]] = [[] for _ in range(num_bins)]
    for error in errors:
        s = error.s_inference if position == "inference" else error.s_evaluated
        if s is None:
            continue
        index = min(max(int(s // bin_size), 0), num_bins - 1)
        values[index].append(error.values[metric])

    mean = np.full(num_bins, np.nan)
    maximum = np.full(num_bins, np.nan)
    count = np.zeros(num_bins, dtype=int)
    for i, bin_values in enumerate(values):
        if bin_values:
            mean[i] = float(np.mean(bin_values))
            maximum[i] = float(np.max(bin_values))
            count[i] = len(bin_values)
    return Bins(edges, mean, maximum, count)


def route_segments(route: Route, max_step: float):
    """Centerline segments, skipping the gaps between joined routes."""
    points = np.column_stack([route.x, route.y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    mid_s = 0.5 * (route.s[:-1] + route.s[1:])
    keep = np.linalg.norm(points[1:] - points[:-1], axis=1) <= max_step
    return segments[keep], mid_s[keep]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(ROUTE_NO_DATA)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_report(
    run: Run,
    metric: str,
    horizon: str,
    position: str,
    bin_size: float,
    vmax: Optional[float],
    control_mode: str,
    output_path: Path,
) -> Bins:
    errors = run.errors.get(horizon, [])
    route = run.route
    bins = compute_bins(errors, metric, position, route.length, bin_size)
    finite_mean = bins.mean[np.isfinite(bins.mean)]
    if vmax is None:
        vmax = float(np.percentile(finite_mean, 99)) if finite_mean.size else 1.0
    vmax = max(vmax, 1.0e-3)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("error", SEQUENTIAL_BLUE)

    fig = plt.figure(figsize=(12, 13), facecolor=SURFACE, constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0])
    ax_map = fig.add_subplot(grid[0])
    ax_line = fig.add_subplot(grid[1])

    metric_name = METRICS[metric].name
    # Average over the frames used in this figure (filtered by control mode)
    all_values = [error.values[metric] for error in errors]
    average_text = f"{np.mean(all_values):.3f} m" if all_values else "n/a"
    fig.suptitle(
        f"{metric_name} @ {horizon}: {run.label}, route {route.length:.0f} m\n"
        f"{METRICS[metric].average_label}={average_text}, {POSITIONS[position]}",
        color=TEXT_PRIMARY,
        fontsize=16,
    )

    # --- map ---
    style_axes(ax_map)
    segments, mid_s = route_segments(route, max_step=3.0 * bin_size)
    bin_index = np.clip((mid_s // bin_size).astype(int), 0, len(bins.mean) - 1)
    segment_mean = bins.mean[bin_index]
    has_data = np.isfinite(segment_mean)
    ax_map.add_collection(
        LineCollection(segments[~has_data], colors=ROUTE_NO_DATA, linewidths=2.0, zorder=1)
    )
    colored = LineCollection(
        segments[has_data], cmap=cmap, norm=norm, linewidths=5.0, capstyle="round", zorder=2
    )
    colored.set_array(segment_mean[has_data])
    ax_map.add_collection(colored)
    ax_map.set_aspect("equal")
    ax_map.autoscale_view()
    ax_map.ticklabel_format(useOffset=False, style="plain")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    colorbar = fig.colorbar(colored, ax=ax_map, shrink=0.6, extend="max", pad=0.02)
    colorbar.set_label(f"{metric_name}, mean per bin [m]", color=TEXT_SECONDARY)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    colorbar.outline.set_visible(False)

    # Arc length ticks along the route so the map can be matched with the line chart
    tick_step = 500.0 if route.length > 2000.0 else 250.0
    for s_tick in np.arange(0.0, route.length + 1.0e-6, tick_step):
        i = int(np.searchsorted(route.s, s_tick))
        if i >= len(route.s):
            continue
        ax_map.plot(route.x[i], route.y[i], "o", ms=4, color=TEXT_SECONDARY, zorder=3)
        ax_map.annotate(
            f"{s_tick:.0f} m",
            (route.x[i], route.y[i]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_SECONDARY,
        )

    # --- line chart along the route ---
    style_axes(ax_line)
    for s_start, s_end in route.intersections:
        ax_line.axvspan(s_start, s_end, color=INTERSECTION_BAND, zorder=0, linewidth=0)
    centers = 0.5 * (bins.edges[:-1] + bins.edges[1:])
    ax_line.plot(centers, bins.max, color=SERIES_MAX, linewidth=1.0, label="max per bin")
    ax_line.plot(centers, bins.mean, color=SERIES_MEAN, linewidth=2.0, label="mean per bin")
    ax_line.set_xlim(0.0, route.length)
    ax_line.set_ylim(bottom=0.0)
    ax_line.set_xlabel("route arc length [m]  (shaded: intersection lanelets)")
    ax_line.set_ylabel(f"{metric_name} [m]")
    legend = ax_line.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.savefig(output_path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return bins


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ADE/FDE by location along the route from the frame geometry output of "
        "autoware_planning_data_analyzer."
    )
    parser.add_argument(
        "input_dir", type=Path, help="output_dir of autoware_planning_data_analyzer"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory to write the PNG files (default: <input_dir>/error_map)",
    )
    parser.add_argument("--label", help="name of the run shown in the title (default: input_dir)")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(METRICS),
        default=list(METRICS),
        help="metrics to plot (default: all). avg_lateral / avg_longitudinal are the "
        "average_lateral_deviation / average_longitudinal_deviation of the result",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["1.0s", "2.0s", "4.0s"],
        help="horizon keys of the result (default: 1.0s 2.0s 4.0s)",
    )
    parser.add_argument(
        "--position",
        choices=sorted(POSITIONS),
        default="inference",
        help="where the error of a frame is placed: ego position at the inference time "
        "(default) or the ground-truth position at the evaluated time",
    )
    parser.add_argument("--bin-size", type=float, default=5.0, help="bin size [m] (default: 5)")
    parser.add_argument(
        "--control-mode",
        default="AUTONOMOUS",
        help='frames to use by control_mode, or "all" (default: AUTONOMOUS)',
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help="upper limit of the map color scale [m] (default: 99th percentile of the bins)",
    )
    args = parser.parse_args()

    if args.bin_size <= 0.0:
        parser.error("--bin-size must be positive")
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir / "error_map").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or input_dir.name

    run = load_run(input_dir, label, args.control_mode)
    print(
        f"{run.num_selected_frames} of {run.num_frames} frames selected "
        f"(control_mode: {args.control_mode}), route {run.route.length:.0f} m, "
        f"{args.bin_size:g} m bins"
    )
    for horizon in args.horizons:
        if horizon not in run.errors:
            print(f"WARNING: horizon {horizon} not found. Available: {sorted(run.errors)}")
            continue
        for metric in args.metrics:
            output_path = output_dir / f"{metric}_{horizon}_{args.position}.png"
            bins = plot_report(
                run,
                metric,
                horizon,
                args.position,
                args.bin_size,
                args.vmax,
                args.control_mode,
                output_path,
            )
            print(
                f"{output_path}  ({int((bins.count > 0).sum())} of {len(bins.count)} bins "
                "have data)"
            )


if __name__ == "__main__":
    main()
