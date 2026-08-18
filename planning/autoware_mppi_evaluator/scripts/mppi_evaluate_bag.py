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

"""Evaluate MPPI configurations against synchronized MCAP frames."""

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

from autoware_mppi_evaluator import mppi_optimizer_py as mppi_cpp
from autoware_mppi_evaluator.dataset_io import load_dataset
from autoware_mppi_evaluator.evaluator_config import make_configuration
from autoware_mppi_evaluator.mcap_reader import McapZohSynchronizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def parse_named_path(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for each optimizer configuration")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Both NAME and PATH are required")
    return name, path


def finite_or_none(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def evaluate_frame(session, frame):
    messages = frame.messages
    return session.evaluate(
        frame.frame_id,
        frame.timestamp_ns,
        messages["reference_trajectory"],
        messages["odometry"],
        messages["tracked_objects"],
        messages.get("acceleration"),
        messages.get("steering"),
        frame.ages_ms["odometry"],
        frame.ages_ms.get("acceleration"),
        frame.ages_ms.get("steering"),
        frame.ages_ms.get("tracked_objects"),
    )


def percentile(values: Iterable[float], fraction: float):
    ordered = sorted(values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: List[Dict]) -> Dict:
    summary = {}
    names = sorted({row["config_name"] for row in rows})
    for name in names:
        selected = [row for row in rows if row["config_name"] == name and "error" not in row]
        latencies = [row["execution_time_ms"] for row in selected]
        summary[name] = {
            "evaluated_frames": len(selected),
            "failed_frames": sum(row["config_name"] == name and "error" in row for row in rows),
            "rejected_frames": sum(row["was_rejected"] for row in selected),
            "invalid_frames": sum(not row["is_valid"] for row in selected),
            "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
            "latency_ms_p50": percentile(latencies, 0.50),
            "latency_ms_p95": percentile(latencies, 0.95),
        }
    return summary


def select_visualization_frames(records: List[Dict], limit: int) -> List[Dict]:
    """Prioritize invalid/rejected frames, then the valid frames with the largest error."""
    failed_or_rejected = [
        record
        for record in records
        if not record["result"]["metrics"]["is_valid"]
        or record["result"]["metrics"]["was_rejected"]
    ]
    valid = [
        record
        for record in records
        if record["result"]["metrics"]["is_valid"]
        and not record["result"]["metrics"]["was_rejected"]
    ]

    def cross_track_error(record: Dict) -> float:
        value = finite_or_none(record["result"]["metrics"]["max_cross_track_error_m"])
        return -math.inf if value is None else float(value)

    valid.sort(key=cross_track_error, reverse=True)
    remaining = max(0, limit - len(failed_or_rejected))
    return failed_or_rejected[:limit] + valid[:remaining]


def box_outline(tracked_object: Dict) -> Tuple[List[float], List[float]]:
    x = float(tracked_object["x"])
    y = float(tracked_object["y"])
    yaw = float(tracked_object["yaw"])
    half_length = 0.5 * float(tracked_object["length"])
    half_width = 0.5 * float(tracked_object["width"])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    corners = []
    for longitudinal, lateral in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
        (half_length, half_width),
    ):
        corners.append(
            (
                x + longitudinal * cosine - lateral * sine,
                y + longitudinal * sine + lateral * cosine,
            )
        )
    return [corner[0] for corner in corners], [corner[1] for corner in corners]


def trajectory_values(trajectory: Dict, field: str) -> Tuple[List[float], List[float]]:
    points = trajectory["points"]
    times = [float(point["time_from_start_ns"]) / 1.0e9 for point in points]
    return times, [float(point[field]) for point in points]


def nominal_control_values(profile: Dict, field: str) -> Tuple[List[float], List[float]]:
    values = [float(value) for value in profile[field]]
    time_step_s = float(profile["time_step_s"])
    return [index * time_step_s for index in range(len(values))], values


COST_BREAKDOWN_COMPONENTS = (
    ("speed", "Speed"),
    ("track", "Track"),
    ("heading", "Heading"),
    ("lateral_distance", "Lateral distance"),
    ("lateral_boundary", "Lateral boundary"),
    ("lateral_yaw_error", "Lateral yaw error"),
    ("remaining_distance", "Remaining distance"),
    ("path_overshoot", "Path overshoot"),
    ("track_center", "Track center"),
    ("corner_buffer", "Corner buffer"),
    ("drivable_area", "Drivable area"),
    ("obstacle", "Obstacle"),
    ("road_border", "Road border"),
    ("acceleration_command", "Acceleration command"),
    ("steering_command", "Steering command"),
    ("lateral_acceleration", "Lateral acceleration"),
    ("lateral_jerk", "Lateral jerk"),
    ("longitudinal_jerk", "Longitudinal jerk"),
    ("steering_rate", "Steering rate"),
)


def add_bev_trajectory(figure, trajectory: Dict, name: str, color: str, dash: str) -> None:
    points = trajectory["points"]
    figure.add_trace(
        go.Scatter(
            x=[point["x"] for point in points],
            y=[point["y"] for point in points],
            mode="lines",
            name=name,
            legendgroup=name,
            line={"color": color, "width": 3, "dash": dash},
        ),
        row=1,
        col=1,
    )


def add_segments(figure, segments: Iterable, name: str, color: str, dash: str = "solid") -> None:
    for index, (x0, y0, x1, y1) in enumerate(segments):
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=index == 0,
                line={"color": color, "width": 2, "dash": dash},
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )


def make_frame_figure(record: Dict):
    result = record["result"]
    messages = record["messages"]
    metrics = result["metrics"]
    output_color = "green" if metrics["is_valid"] else "red"
    execution_time_ms = finite_or_none(float(metrics["execution_time_ms"]))
    execution_time_text = "N/A" if execution_time_ms is None else f"{execution_time_ms:.3f}"
    effective_sample_size = finite_or_none(float(metrics["effective_sample_size"]))
    effective_sample_size_text = (
        "N/A" if effective_sample_size is None else f"{effective_sample_size:.3f}"
    )
    max_importance_weight = finite_or_none(float(metrics["max_importance_weight"]))
    max_importance_weight_text = (
        "N/A" if max_importance_weight is None else f"{max_importance_weight:.6f}"
    )
    output_cost_breakdown = result["cost_breakdown"]
    nominal_cost_breakdown = result["nominal_cost_breakdown"]
    output_cost = finite_or_none(float(output_cost_breakdown["total"]))
    nominal_cost = finite_or_none(float(nominal_cost_breakdown["total"]))
    costs_available = (
        int(output_cost_breakdown.get("evaluated_timesteps", 0)) > 0
        and int(nominal_cost_breakdown.get("evaluated_timesteps", 0)) > 0
        and output_cost is not None
        and nominal_cost is not None
    )
    delta_cost = output_cost - nominal_cost if costs_available else None
    nominal_cost_text = "N/A" if not costs_available else f"{nominal_cost:.4g}"
    output_cost_text = "N/A" if not costs_available else f"{output_cost:.4g}"
    delta_cost_text = "N/A" if delta_cost is None else f"{delta_cost:+.4g}"
    invalid_index = metrics["first_invalid_index"]
    invalid_index_text = "N/A" if invalid_index is None else str(invalid_index)
    figure_title = (
        f"{result['config_name']} — {result['frame_id']}<br>"
        f"<sup>execution_time_ms: {execution_time_text} | "
        f"effective_sample_size: {effective_sample_size_text} | "
        f"max_importance_weight: {max_importance_weight_text} | "
        f"nominal_cost: {nominal_cost_text} | output_cost: {output_cost_text} | "
        f"delta_cost (output − nominal): {delta_cost_text}<br>"
        f"is_valid: {metrics['is_valid']} | was_rejected: {metrics['was_rejected']} | "
        f"invalidity_reasons: {metrics['invalidity_reason_names']} "
        f"({metrics['invalidity_reasons']}) | first_invalid_index: {invalid_index_text}</sup>"
    )
    figure = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"rowspan": 2}, {}, {"rowspan": 2}],
            [None, {}, None],
            [{"colspan": 3}, None, None],
        ],
        subplot_titles=("BEV", "Velocity", "Lateral", "Acceleration", "Cost breakdown"),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        row_heights=[0.32, 0.32, 0.36],
    )

    reference = result["reference_trajectory"]
    optimized = result["optimized_trajectory"]
    add_bev_trajectory(figure, reference, "Reference", "gray", "dash")
    add_bev_trajectory(figure, optimized, "Optimized", output_color, "solid")
    add_segments(figure, result["road_borders"], "Road borders", "firebrick")
    add_segments(figure, result["drivable_area"], "Drivable bounds", "teal", "dash")
    for index, tracked_object in enumerate(result["selected_objects"]):
        xs, ys = box_outline(tracked_object)
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Selected objects",
                legendgroup="Selected objects",
                showlegend=index == 0,
                fill="toself",
                line={"color": "purple"},
            ),
            row=1,
            col=1,
        )

    for trajectory, name, velocity_color, acceleration_color in (
        (reference, "Reference", "lightblue", "lightsalmon"),
        (optimized, "Optimized", "darkblue", "darkred"),
    ):
        times, velocities = trajectory_values(trajectory, "velocity_mps")
        _, accelerations = trajectory_values(trajectory, "acceleration_mps2")
        figure.add_trace(
            go.Scatter(
                x=times,
                y=velocities,
                mode="lines",
                name=f"{name} velocity",
                legendgroup=name,
                line={"color": velocity_color, "width": 3},
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=accelerations,
                mode="lines",
                name=f"{name} acceleration",
                legendgroup=name,
                line={"color": acceleration_color, "width": 3},
            ),
            row=2,
            col=2,
        )

    nominal_profile = result["nominal_control_profile"]
    nominal_times, nominal_acceleration = nominal_control_values(
        nominal_profile, "acceleration_commands_mps2"
    )
    figure.add_trace(
        go.Scatter(
            x=nominal_times,
            y=nominal_acceleration,
            mode="lines",
            name="Nominal acceleration command",
            legendgroup="Nominal control",
            line={"color": "#9467bd", "width": 3, "dash": "dash"},
        ),
        row=2,
        col=2,
    )

    odometry_type = get_message("nav_msgs/msg/Odometry")
    odometry = deserialize_message(messages["odometry"], odometry_type)
    figure.add_trace(
        go.Scatter(
            x=[0.0],
            y=[odometry.twist.twist.linear.x],
            mode="markers",
            name="Ego velocity",
            legendgroup="Ego state",
            marker={"color": "darkblue", "size": 14, "symbol": "star"},
        ),
        row=1,
        col=2,
    )
    if "acceleration" in messages:
        acceleration_type = get_message("geometry_msgs/msg/AccelWithCovarianceStamped")
        acceleration = deserialize_message(messages["acceleration"], acceleration_type)
        figure.add_trace(
            go.Scatter(
                x=[0.0],
                y=[acceleration.accel.accel.linear.x],
                mode="markers",
                name="Ego acceleration",
                legendgroup="Ego state",
                marker={"color": "darkred", "size": 14, "symbol": "star"},
            ),
            row=2,
            col=2,
        )

    steering_times, steering_angles = trajectory_values(optimized, "front_wheel_angle_rad")
    figure.add_trace(
        go.Scatter(
            x=steering_times,
            y=steering_angles,
            mode="lines",
            name="Optimized steering",
            legendgroup="Optimized",
            line={"color": output_color, "width": 3},
        ),
        row=1,
        col=3,
    )
    nominal_times, nominal_steering = nominal_control_values(
        nominal_profile, "steering_commands_rad"
    )
    figure.add_trace(
        go.Scatter(
            x=nominal_times,
            y=nominal_steering,
            mode="lines",
            name="Nominal steering command",
            legendgroup="Nominal control",
            line={"color": "#9467bd", "width": 3, "dash": "dash"},
        ),
        row=1,
        col=3,
    )
    if "steering" in messages:
        steering_type = get_message("autoware_vehicle_msgs/msg/SteeringReport")
        steering = deserialize_message(messages["steering"], steering_type)
        figure.add_trace(
            go.Scatter(
                x=[0.0],
                y=[steering.steering_tire_angle],
                mode="markers",
                name="Ego steering",
                legendgroup="Ego state",
                marker={"color": output_color, "size": 14, "symbol": "star"},
            ),
            row=1,
            col=3,
        )

    cost_breakdown = output_cost_breakdown
    cost_labels = []
    cost_values = []
    for field, label in COST_BREAKDOWN_COMPONENTS:
        value = float(cost_breakdown.get(field, 0.0))
        if math.isfinite(value) and value != 0.0:
            cost_labels.append(label)
            cost_values.append(value)
    if cost_values:
        figure.add_trace(
            go.Bar(
                x=cost_values,
                y=cost_labels,
                orientation="h",
                name="Cost component",
                legendgroup="Cost breakdown",
                showlegend=False,
                marker={"color": cost_values, "colorscale": "Viridis"},
                text=[f"{value:.4g}" for value in cost_values],
                textposition="auto",
                hovertemplate="%{y}: %{x:.6g}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    else:
        figure.add_annotation(
            text="No non-zero finite cost components",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=3,
            col=1,
        )

    total = float(cost_breakdown.get("total", 0.0))
    running_total = float(cost_breakdown.get("running_total", 0.0))
    terminal_total = float(cost_breakdown.get("terminal_total", 0.0))
    baseline_cost = float(metrics["baseline_cost"])
    timesteps = int(cost_breakdown.get("evaluated_timesteps", 0))
    cost_title = next(
        annotation
        for annotation in figure.layout.annotations
        if annotation.text == "Cost breakdown"
    )
    cost_title.text = (
        "Cost breakdown "
        f"(total={total:.4g}, running={running_total:.4g}, "
        f"terminal={terminal_total:.4g}, best sampled={baseline_cost:.4g}, "
        f"nominal={nominal_cost_text}, Δ={delta_cost_text}, "
        f"timesteps={timesteps})"
    )

    figure.update_xaxes(title_text="Map X (m)", row=1, col=1)
    figure.update_yaxes(title_text="Map Y (m)", scaleanchor="x", scaleratio=1, row=1, col=1)
    figure.update_xaxes(title_text="Time (s)", row=1, col=2)
    figure.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
    figure.update_xaxes(title_text="Time (s)", row=2, col=2)
    figure.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=2)
    figure.update_xaxes(title_text="Time (s)", row=1, col=3)
    figure.update_yaxes(title_text="Steering angle (rad)", row=1, col=3)
    figure.update_xaxes(title_text="Horizon-average cost", row=3, col=1)
    figure.update_yaxes(autorange="reversed", row=3, col=1)
    figure.update_layout(
        title=figure_title,
        template="plotly_white",
        height=1150,
        width=1800,
    )
    return figure


def make_html_report(records_by_configuration: Dict[str, List[Dict]], average_ms) -> str:
    average_text = "N/A" if average_ms is None else f"{average_ms:.3f} ms"
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        "<title>MPPI Evaluation Report</title>",
        '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>',
        "</head><body>",
        "<h1>MPPI Evaluation Report</h1>",
        f"<p><strong>Average Optimization Time:</strong> {average_text}</p>",
    ]
    for config_name, records in records_by_configuration.items():
        parts.append(f"<h2>{html.escape(config_name)}</h2>")
        for record in records:
            frame_id = html.escape(str(record["result"]["frame_id"]))
            parts.append(f"<h3>{frame_id}</h3>")
            parts.append(make_frame_figure(record).to_html(full_html=False, include_plotlyjs=False))
    parts.append("</body></html>\n")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="MCAP path or curated dataset path")
    parser.add_argument("--input-format", choices=("bag", "dataset"), default="bag")
    parser.add_argument("--topics", help="Topic configuration YAML for MCAP input")
    parser.add_argument(
        "--optimizer-config",
        action="append",
        required=True,
        type=parse_named_path,
        metavar="NAME=PATH",
    )
    parser.add_argument("--vehicle-info", required=True)
    parser.add_argument("--simulator-model", required=True)
    parser.add_argument("--mode", choices=("isolated", "chronological"), default="isolated")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--output-stride", type=int, default=1)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write a Plotly HTML report for prioritized frames",
    )
    parser.add_argument(
        "--visualize-limit",
        type=int,
        default=20,
        help="Maximum number of frame plots per configuration",
    )
    parser.add_argument("--output", required=True, help="Output path without an extension")
    arguments = parser.parse_args()

    if arguments.output_stride < 1:
        parser.error("--output-stride must be positive")
    if arguments.visualize_limit < 1:
        parser.error("--visualize-limit must be positive")

    if arguments.input_format == "bag":
        if not arguments.topics:
            parser.error("--topics is required for MCAP input")
        frame_source = McapZohSynchronizer(arguments.input, arguments.topics)
        frame_count = len(frame_source)

        def selected_frames(start, stop):
            return frame_source.iter_frames(start, stop)

    else:
        all_frames = load_dataset(arguments.input)
        frame_count = len(all_frames)

        def selected_frames(start, stop):
            return iter(all_frames[start:stop])

    stop = frame_count if arguments.stop is None else min(arguments.stop, frame_count)
    if arguments.start < 0 or arguments.start >= stop:
        parser.error("The selected frame range is empty")

    rows: List[Dict] = []
    optimization_times: List[float] = []
    visualization_records: Dict[str, List[Dict]] = {
        config_name: [] for config_name, _ in arguments.optimizer_config
    }
    for config_name, config_path in arguments.optimizer_config:
        configuration = make_configuration(
            mppi_cpp,
            config_path,
            arguments.vehicle_info,
            arguments.simulator_model,
            config_name,
        )
        if arguments.visualize:
            configuration.runtime_options.skip_if_invalid = False
        session = None
        environment_key = None
        for frame in selected_frames(arguments.start, stop):
            if not frame.is_usable:
                rows.append(
                    {
                        "frame_id": frame.frame_id,
                        "timestamp_ns": frame.timestamp_ns,
                        "config_name": config_name,
                        "error": "; ".join(frame.warnings),
                    }
                )
                continue

            next_environment_key = (
                hash(frame.messages["lanelet_map"]),
                hash(frame.messages["route"]),
            )
            if session is None or environment_key != next_environment_key:
                session = mppi_cpp.EvaluationSession(
                    frame.messages["lanelet_map"],
                    frame.messages["route"],
                    configuration,
                    arguments.mode,
                )
                environment_key = next_environment_key
            try:
                result = evaluate_frame(session, frame)
                execution_time_ms = float(result["metrics"]["execution_time_ms"])
                if math.isfinite(execution_time_ms):
                    optimization_times.append(execution_time_ms)
                if arguments.visualize:
                    candidates = visualization_records[config_name]
                    candidates.append(
                        {
                            "result": result,
                            "messages": dict(frame.messages),
                        }
                    )
                    visualization_records[config_name] = select_visualization_frames(
                        candidates, arguments.visualize_limit
                    )
                if (frame.index - arguments.start) % arguments.output_stride == 0:
                    row = {
                        "frame_id": result["frame_id"],
                        "timestamp_ns": result["timestamp_ns"],
                        "config_name": result["config_name"],
                    }
                    row.update(
                        {key: finite_or_none(value) for key, value in result["metrics"].items()}
                    )
                    rows.append(row)
            except Exception as error:
                rows.append(
                    {
                        "frame_id": frame.frame_id,
                        "timestamp_ns": frame.timestamp_ns,
                        "config_name": config_name,
                        "error": str(error),
                    }
                )
                if arguments.mode == "chronological":
                    session.reset()

    output_base = Path(arguments.output).expanduser().resolve()
    json_payload = {"schema_version": 1, "summary": summarize(rows), "frames": rows}
    atomic_write(output_base.with_suffix(".json"), json.dumps(json_payload, indent=2) + "\n")

    fieldnames = sorted({key for row in rows for key in row})
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        atomic_write(output_base.with_suffix(".csv"), stream.read())

    if arguments.visualize:
        average_ms = statistics.fmean(optimization_times) if optimization_times else None
        report = make_html_report(visualization_records, average_ms)
        atomic_write(output_base.with_suffix(".html"), report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
