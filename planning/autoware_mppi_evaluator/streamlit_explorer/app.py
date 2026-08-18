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

"""Streamlit explorer for recorded MPPI input frames."""

import math
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

from ament_index_python.packages import get_package_share_directory
from autoware_mppi_evaluator.dataset_io import save_frame
from autoware_mppi_evaluator.evaluator_config import make_configuration
from autoware_mppi_evaluator.mcap_reader import McapZohSynchronizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

try:
    from autoware_mppi_evaluator import mppi_optimizer_py as mppi_cpp

    BACKEND_ERROR = None
except ImportError as error:
    mppi_cpp = None
    BACKEND_ERROR = str(error)


def package_file(package: str, relative: str) -> str:
    try:
        return str(Path(get_package_share_directory(package)) / relative)
    except Exception:
        return ""


def configured_package_file(configuration: Dict, key: str) -> str:
    file_configuration = configuration["parameter_files"][key]
    return package_file(file_configuration["package"], file_configuration["path"])


@st.cache_resource(show_spinner="Index the selected MCAP file")
def load_synchronizer(bag: str, config: str) -> McapZohSynchronizer:
    return McapZohSynchronizer(bag, config)


def evaluate(session, frame):
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


def box_outline(
    x: float, y: float, yaw: float, length: float, width: float
) -> Tuple[List[float], List[float]]:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    corners = []
    for longitudinal, lateral in (
        (length / 2.0, width / 2.0),
        (length / 2.0, -width / 2.0),
        (-length / 2.0, -width / 2.0),
        (-length / 2.0, width / 2.0),
        (length / 2.0, width / 2.0),
    ):
        corners.append(
            (
                x + longitudinal * cosine - lateral * sine,
                y + longitudinal * sine + lateral * cosine,
            )
        )
    return [point[0] for point in corners], [point[1] for point in corners]


def object_filter_margin(configuration) -> float:
    vehicle = configuration.vehicle_params
    max_longitudinal_offset = abs(float(vehicle.ego_axle_to_box_center)) + 0.5 * float(
        vehicle.ego_length
    )
    vehicle_radius = math.hypot(max_longitudinal_offset, 0.5 * float(vehicle.ego_width))
    return vehicle_radius + float(configuration.cost_params.boundary_threshold)


def add_segments(figure: go.Figure, segments: Iterable, name: str, color: str) -> None:
    first = True
    for x0, y0, x1, y1 in segments:
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=first,
                line={"color": color, "width": 2},
                hoverinfo="skip",
            )
        )
        first = False


def add_trajectory_arrays(
    figure: go.Figure, xs: List[float], ys: List[float], name: str, color: str, dash: str
) -> None:
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=name,
            line={"color": color, "width": 3, "dash": dash},
            marker={"size": 4},
        )
    )


def add_trajectory(figure: go.Figure, trajectory: Dict, name: str, color: str, dash: str) -> None:
    points = trajectory["points"]
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    add_trajectory_arrays(figure, xs, ys, name, color, dash)


def trajectory_profile(trajectory: Dict, field: str) -> Tuple[List[float], List[float]]:
    points = trajectory["points"]
    times = [float(point["time_from_start_ns"]) / 1.0e9 for point in points]
    values = [float(point[field]) for point in points]
    return times, values


def nominal_control_values(profile: Dict, field: str) -> Tuple[List[float], List[float]]:
    values = [float(value) for value in profile[field]]
    time_step_s = float(profile["time_step_s"])
    return [index * time_step_s for index in range(len(values))], values


def add_longitudinal_profile(figure: go.Figure, trajectory: Dict, name: str, color: str) -> None:
    times, velocities = trajectory_profile(trajectory, "velocity_mps")
    _, accelerations = trajectory_profile(trajectory, "acceleration_mps2")
    figure.add_trace(
        go.Scatter(
            x=times,
            y=velocities,
            mode="lines",
            name=f"{name} velocity",
            legendgroup=name,
            line={"color": color, "width": 3},
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=accelerations,
            mode="lines",
            name=f"{name} acceleration",
            legendgroup=name,
            line={"color": color, "width": 3, "dash": "dash"},
        ),
        secondary_y=True,
    )


def add_lateral_profile(figure: go.Figure, trajectory: Dict, name: str, color: str) -> None:
    times, steering_angles = trajectory_profile(trajectory, "front_wheel_angle_rad")
    figure.add_trace(
        go.Scatter(
            x=times,
            y=steering_angles,
            mode="lines",
            name=name,
            line={"color": color, "width": 3},
        )
    )


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


def make_cost_breakdown_figure(cost_breakdown: Dict, baseline_cost: float) -> go.Figure:
    labels = []
    values = []
    for field, label in COST_BREAKDOWN_COMPONENTS:
        value = float(cost_breakdown.get(field, 0.0))
        if math.isfinite(value) and value != 0.0:
            labels.append(label)
            values.append(value)

    figure = go.Figure()
    if values:
        figure.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                name="Cost component",
                legendgroup="Cost breakdown",
                marker={"color": values, "colorscale": "Viridis"},
                text=[f"{value:.4g}" for value in values],
                textposition="auto",
                hovertemplate="%{y}: %{x:.6g}<extra></extra>",
            )
        )
    else:
        figure.add_annotation(
            text="No non-zero finite cost components",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    total = float(cost_breakdown.get("total", 0.0))
    running_total = float(cost_breakdown.get("running_total", 0.0))
    terminal_total = float(cost_breakdown.get("terminal_total", 0.0))
    timesteps = int(cost_breakdown.get("evaluated_timesteps", 0))
    figure.update_layout(
        title=(
            "Cost breakdown "
            f"(total={total:.4g}, running={running_total:.4g}, "
            f"terminal={terminal_total:.4g}, best sampled={float(baseline_cost):.4g}, "
            f"timesteps={timesteps})"
        ),
        template="plotly_white",
        height=max(360, 30 * len(labels) + 130),
        xaxis_title="Horizon-average cost",
        yaxis={"autorange": "reversed"},
        showlegend=False,
        uirevision="constant",
        margin={"l": 150, "r": 30, "t": 70, "b": 50},
    )
    return figure


COST_OVERRIDE_GROUPS = (
    (
        "Core tracking",
        (
            ("lambda_", "Lambda", 0.001, 10.0),
            ("speed_coeff", "Speed coefficient", 0.0, 10.0),
            ("track_coeff", "Tracking coefficient", 0.0, 10.0),
            ("track_terminal_scale", "Tracking terminal scale", 0.0, 0.5),
            ("heading_coeff", "Heading coefficient", 0.0, 10.0),
            ("lateral_distance_coeff", "Lateral-distance coefficient", 0.0, 10.0),
            ("lateral_yaw_error_coeff", "Lateral-yaw-error coefficient", 0.0, 10.0),
            ("remaining_distance_coeff", "Remaining-distance coefficient", 0.0, 10.0),
            ("path_overshoot_coeff", "Path-overshoot coefficient", 0.0, 10.0),
            ("track_center_coeff", "Track-center coefficient", 0.0, 10.0),
        ),
    ),
    (
        "Collision and boundaries",
        (
            ("corner_buffer_coeff", "Corner-buffer coefficient", 0.0, 10.0),
            ("corner_safe_margin", "Corner safe margin (m)", 0.0, 0.05),
            ("boundary_threshold", "Boundary threshold (m)", 0.001, 0.05),
            ("lateral_boundary_soft_margin", "Lateral soft margin (m)", 0.0, 0.05),
            ("obstacle_collision_margin", "Obstacle collision margin (m)", 0.0, 0.05),
            ("road_border_collision_margin", "Road-border collision margin (m)", 0.0, 0.05),
            ("obstacle_safe_margin", "Obstacle safe margin (m)", 0.0, 0.05),
            ("road_border_safe_margin", "Road-border safe margin (m)", 0.0, 0.05),
            ("drivable_area_safe_margin", "Drivable-area safe margin (m)", 0.0, 0.05),
            ("drivable_area_barrier_weight", "Drivable-area barrier weight", 0.0, 100.0),
            ("crash_contact_penalty", "Contact penalty", 0.0, 1000.0),
        ),
    ),
    (
        "Sampling and nominal control",
        (
            ("accel_cmd_std_dev", "Acceleration-command std. dev. (m/s²)", 0.001, 0.01),
            ("steer_cmd_std_dev", "Steering-command std. dev. (rad)", 0.001, 0.001),
            (
                "nominal_curvature_min_chord_length_m",
                "Nominal curvature minimum chord length (m)",
                0.0,
                0.1,
            ),
        ),
    ),
    (
        "Control and comfort",
        (
            ("accel_cmd_coeff", "Acceleration-command coefficient", 0.0, 10.0),
            ("steer_cmd_coeff", "Steering-command coefficient", 0.0, 10.0),
            ("steer_rate_coeff", "Steering-rate coefficient", 0.0, 100.0),
            ("lateral_acceleration_coeff", "Lateral-acceleration coefficient", 0.0, 10.0),
            ("lateral_jerk_coeff", "Lateral-jerk coefficient", 0.0, 10.0),
            ("longitudinal_jerk_coeff", "Longitudinal-jerk coefficient", 0.0, 10.0),
        ),
    ),
)

COST_OVERRIDE_FIELDS = tuple(
    field for _group_name, fields in COST_OVERRIDE_GROUPS for field in fields
)


def cost_state_key(attribute: str) -> str:
    return f"cost_override_{attribute}"


def get_cost_parameter(cost_params, attribute: str):
    value = cost_params
    for component in attribute.split("."):
        value = getattr(value, component)
    return value


def set_cost_parameter(cost_params, attribute: str, value: float) -> None:
    components = attribute.split(".")
    target = cost_params
    for component in components[:-1]:
        target = getattr(target, component)
    setattr(target, components[-1], value)


st.set_page_config(page_title="MPPI Frame Explorer", layout="wide")
st.sidebar.title("MPPI Frame Explorer")

explorer_config_path = package_file("autoware_mppi_evaluator", "config/streamlit_explorer.yaml")
default_optimizer = package_file("autoware_mppi_optimizer", "config/mppi_optimizer.param.yaml")

try:
    with Path(explorer_config_path).open(encoding="utf-8") as stream:
        explorer_config = yaml.safe_load(stream) or {}
    vehicle_path = configured_package_file(explorer_config, "vehicle_information")
    simulator_path = configured_package_file(explorer_config, "vehicle_dynamics")
except (KeyError, OSError, TypeError) as error:
    st.error(f"The explorer configuration load failed: {error}")
    st.stop()

st.sidebar.subheader("Dataset Selection")
bag_dir = st.sidebar.text_input("MCAP directory", ".")
bag_dir_path = Path(bag_dir).expanduser().resolve()
bag_path = ""

if bag_dir_path.exists() and bag_dir_path.is_dir():
    mcap_files = sorted(bag_dir_path.glob("*.mcap"))
    if mcap_files:
        selected_file = st.sidebar.selectbox(
            "Select MCAP", mcap_files, format_func=lambda p: p.name
        )
        bag_path = str(selected_file)
    else:
        st.sidebar.warning("No .mcap files found in this directory.")

optimizer_path = st.sidebar.text_input("Optimizer parameters", default_optimizer)
mode = st.sidebar.radio("Evaluation mode", ("isolated", "chronological"))

if BACKEND_ERROR:
    st.error(f"The native evaluator is unavailable: {BACKEND_ERROR}")
    st.stop()

required_parameter_paths = {
    "optimizer parameters": optimizer_path,
    "vehicle information": vehicle_path,
    "vehicle dynamics": simulator_path,
}
missing_parameter_paths = [
    name for name, path in required_parameter_paths.items() if not path or not Path(path).exists()
]
if missing_parameter_paths:
    st.info(f"Select valid files for: {', '.join(missing_parameter_paths)}.")
    st.stop()

try:
    configuration = make_configuration(
        mppi_cpp, optimizer_path, vehicle_path, simulator_path, name="interactive"
    )
except Exception as error:
    st.error(f"The parameter load failed: {error}")
    st.stop()


# Initialize session state for isolated cost overrides.
optimizer_source = (
    str(Path(optimizer_path).expanduser().resolve()),
    Path(optimizer_path).stat().st_mtime_ns,
)
vehicle_source = (
    str(Path(vehicle_path).expanduser().resolve()),
    Path(vehicle_path).stat().st_mtime_ns,
)
simulator_source = (
    str(Path(simulator_path).expanduser().resolve()),
    Path(simulator_path).stat().st_mtime_ns,
)
cost_source_changed = st.session_state.get("cost_override_source") != optimizer_source
for cost_attribute, _label, _minimum, _step in COST_OVERRIDE_FIELDS:
    state_key = cost_state_key(cost_attribute)
    if cost_source_changed or state_key not in st.session_state:
        st.session_state[state_key] = float(
            get_cost_parameter(configuration.cost_params, cost_attribute)
        )
if cost_source_changed or "skip_if_invalid" not in st.session_state:
    st.session_state.skip_if_invalid = configuration.runtime_options.skip_if_invalid
if cost_source_changed:
    st.session_state.cost_override_source = optimizer_source
if "auto_evaluate" not in st.session_state:
    st.session_state.auto_evaluate = False


@st.fragment
def render_cost_overrides():
    """Isolated fragment so adjusting sliders does not reload the main canvas."""
    st.sidebar.subheader("Cost overrides")
    for group_name, fields in COST_OVERRIDE_GROUPS:
        with st.sidebar.expander(group_name, expanded=group_name == "Core tracking"):
            for cost_attribute, label, minimum, step in fields:
                st.number_input(
                    label,
                    min_value=minimum,
                    step=step,
                    format="%.3f",
                    key=cost_state_key(cost_attribute),
                )
    st.sidebar.checkbox("Reject an invalid result", key="skip_if_invalid")
    st.sidebar.checkbox("Auto-evaluate on select", key="auto_evaluate")


render_cost_overrides()

if not bag_path or not Path(bag_path).expanduser().exists():
    st.info("Select an MCAP file to start.")
    st.stop()

try:
    synchronizer = load_synchronizer(bag_path, explorer_config_path)
except Exception as error:
    st.error(f"The MCAP index failed: {error}")
    st.stop()


@st.fragment
def render_main_explorer() -> None:
    # Apply the latest cost overrides from session state before evaluation
    for cost_attribute, _label, _minimum, _step in COST_OVERRIDE_FIELDS:
        set_cost_parameter(
            configuration.cost_params,
            cost_attribute,
            st.session_state[cost_state_key(cost_attribute)],
        )
    configuration.runtime_options.skip_if_invalid = st.session_state.skip_if_invalid
    runtime_options = configuration.runtime_options
    runtime_key = (
        bool(runtime_options.enable_debug_trajectory_log),
        str(runtime_options.debug_trajectory_log_directory),
        bool(runtime_options.ignore_obstacles),
        bool(runtime_options.ignore_drivable_area),
        bool(runtime_options.force_cold_start_each_step),
        bool(runtime_options.skip_if_invalid),
        bool(runtime_options.use_last_control_as_nominal),
        bool(runtime_options.use_temporal_mpt_as_nominal),
        bool(runtime_options.enable_input_delay_compensation),
    )

    frame_index = st.slider("Frame index", 0, len(synchronizer) - 1, len(synchronizer) // 2)
    frame = synchronizer.get_synchronized_frame(frame_index)
    for warning in frame.warnings:
        st.warning(warning)

    session_key = (
        str(Path(bag_path).expanduser().resolve()),
        mode,
        optimizer_source,
        vehicle_source,
        simulator_source,
        tuple(
            st.session_state[cost_state_key(cost_attribute)]
            for cost_attribute, _label, _minimum, _step in COST_OVERRIDE_FIELDS
        ),
        runtime_key,
        hash(frame.messages["lanelet_map"]),
        hash(frame.messages["route"]),
    )
    if st.session_state.get("session_key") != session_key:
        st.session_state.evaluator = mppi_cpp.EvaluationSession(
            frame.messages["lanelet_map"], frame.messages["route"], configuration, mode
        )
        st.session_state.session_key = session_key
        st.session_state.last_evaluated_index = -1
        st.session_state.result = None

    button_label = "Evaluate frame" if mode == "isolated" else "Replay through frame"
    needs_eval = (
        st.session_state.auto_evaluate and st.session_state.last_evaluated_index != frame_index
    )
    if st.button(button_label, type="primary", disabled=not frame.is_usable) or needs_eval:
        if frame.is_usable:
            try:
                if mode == "isolated":
                    st.session_state.result = evaluate(st.session_state.evaluator, frame)
                    st.session_state.last_evaluated_index = frame_index
                else:
                    if frame_index <= st.session_state.last_evaluated_index:
                        st.session_state.evaluator.reset()
                        st.session_state.last_evaluated_index = -1
                    progress = st.progress(0.0)
                    start = st.session_state.last_evaluated_index + 1
                    usable_count = max(1, frame_index - start + 1)
                    for offset, replay_index in enumerate(range(start, frame_index + 1)):
                        replay_frame = synchronizer.get_synchronized_frame(replay_index)
                        if replay_frame.is_usable:
                            st.session_state.result = evaluate(
                                st.session_state.evaluator, replay_frame
                            )
                        progress.progress((offset + 1) / usable_count)
                    progress.empty()
                    st.session_state.last_evaluated_index = frame_index
            except Exception as error:
                st.error(f"The evaluation failed: {error}")

    result = st.session_state.get("result")
    is_evaluated = result and result["timestamp_ns"] == frame.timestamp_ns

    reference_trajectory = (
        result["reference_trajectory"]
        if is_evaluated
        else mppi_cpp.deserialize_trajectory(frame.messages["reference_trajectory"])
    )
    original_trajectory = None
    if "original_trajectory" in frame.messages:
        original_trajectory = mppi_cpp.deserialize_trajectory(frame.messages["original_trajectory"])
    optimized_trajectory = result["optimized_trajectory"] if is_evaluated else None
    output_color = "green" if not is_evaluated or result["metrics"]["is_valid"] else "red"

    ego_velocity = None
    if "odometry" in frame.messages:
        odometry = mppi_cpp.deserialize_cdr(frame.messages["odometry"], "nav_msgs/msg/Odometry")
        ego_velocity = float(odometry["twist"]["twist"]["linear"]["x"])
    ego_acceleration = None
    if "acceleration" in frame.messages:
        acceleration = mppi_cpp.deserialize_cdr(
            frame.messages["acceleration"],
            "geometry_msgs/msg/AccelWithCovarianceStamped",
        )
        ego_acceleration = float(acceleration["accel"]["accel"]["linear"]["x"])
    ego_steering = None
    if "steering" in frame.messages:
        steering = mppi_cpp.deserialize_cdr(
            frame.messages["steering"], "autoware_vehicle_msgs/msg/SteeringReport"
        )
        ego_steering = float(steering["steering_tire_angle"])

    if not is_evaluated:
        st.info(
            'ℹ️ Preview Mode — Showing recorded bag data. Click "Evaluate" to run the MPPI optimizer.'
        )

    bev_figure = go.Figure()
    add_trajectory(bev_figure, reference_trajectory, "Reference", "gray", "dash")
    if original_trajectory is not None:
        add_trajectory(
            bev_figure,
            original_trajectory,
            "Original (Recorded)",
            "#1f77b4",
            "dot",
        )

    if is_evaluated:
        add_trajectory(bev_figure, optimized_trajectory, "Optimized", output_color, "solid")
        add_segments(bev_figure, result["road_borders"], "Road borders", "firebrick")
        add_segments(bev_figure, result["drivable_area"], "Drivable bounds", "darkorange")
        objects = result["selected_objects"]
        object_name = "Selected objects"
        object_color = "purple"
    else:
        objects = []
        if "tracked_objects" in frame.messages:
            objects = mppi_cpp.deserialize_tracked_objects_in_range(
                frame.messages["tracked_objects"],
                frame.messages["reference_trajectory"],
                object_filter_margin(configuration),
            )
        object_name = "Tracked objects"
        object_color = "gray"

    for object_index, tracked_object in enumerate(objects):
        if is_evaluated:
            xs_box, ys_box = box_outline(**tracked_object)
        else:
            xs_box, ys_box = box_outline(
                tracked_object["x"],
                tracked_object["y"],
                tracked_object["yaw"],
                tracked_object["length"],
                tracked_object["width"],
            )
        bev_figure.add_trace(
            go.Scatter(
                x=xs_box,
                y=ys_box,
                mode="lines",
                name=object_name,
                legendgroup=object_name,
                showlegend=object_index == 0,
                fill="toself",
                line={"color": object_color},
            )
        )

    bev_figure.update_layout(
        title=f"Frame {frame.timestamp_ns}",
        xaxis_title="Map X (m)",
        yaxis_title="Map Y (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        height=780,
        uirevision="constant",
    )

    longitudinal_figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_longitudinal_profile(longitudinal_figure, reference_trajectory, "Reference", "gray")
    if original_trajectory is not None:
        add_longitudinal_profile(
            longitudinal_figure,
            original_trajectory,
            "Original (Recorded)",
            "#1f77b4",
        )
    if optimized_trajectory is not None:
        add_longitudinal_profile(
            longitudinal_figure, optimized_trajectory, "Optimized", output_color
        )
        nominal_times, nominal_acceleration = nominal_control_values(
            result["nominal_control_profile"], "acceleration_commands_mps2"
        )
        longitudinal_figure.add_trace(
            go.Scatter(
                x=nominal_times,
                y=nominal_acceleration,
                mode="lines",
                name="Nominal acceleration command",
                legendgroup="Nominal control",
                line={"color": "#9467bd", "width": 3, "dash": "dash"},
            ),
            secondary_y=True,
        )

    if ego_velocity is not None:
        longitudinal_figure.add_trace(
            go.Scatter(
                x=[0.0],
                y=[ego_velocity],
                mode="markers",
                name="Ego velocity",
                marker={"color": "black", "size": 14, "symbol": "star"},
            ),
            secondary_y=False,
        )
    if ego_acceleration is not None:
        longitudinal_figure.add_trace(
            go.Scatter(
                x=[0.0],
                y=[ego_acceleration],
                mode="markers",
                name="Ego acceleration",
                marker={"color": "darkorange", "size": 14, "symbol": "star"},
            ),
            secondary_y=True,
        )
    longitudinal_figure.update_xaxes(title_text="Time (s)")
    longitudinal_figure.update_yaxes(title_text="Velocity (m/s)", secondary_y=False)
    longitudinal_figure.update_yaxes(title_text="Acceleration (m/s²)", secondary_y=True)
    longitudinal_figure.update_layout(
        title="Longitudinal profile", height=380, uirevision="constant"
    )

    lateral_figure = go.Figure()
    if original_trajectory is not None:
        add_lateral_profile(lateral_figure, original_trajectory, "Original (Recorded)", "#1f77b4")
    if optimized_trajectory is not None:
        add_lateral_profile(lateral_figure, optimized_trajectory, "Optimized", output_color)
        nominal_times, nominal_steering = nominal_control_values(
            result["nominal_control_profile"], "steering_commands_rad"
        )
        lateral_figure.add_trace(
            go.Scatter(
                x=nominal_times,
                y=nominal_steering,
                mode="lines",
                name="Nominal steering command",
                legendgroup="Nominal control",
                line={"color": "#9467bd", "width": 3, "dash": "dash"},
            )
        )
    if ego_steering is not None:
        lateral_figure.add_trace(
            go.Scatter(
                x=[0.0],
                y=[ego_steering],
                mode="markers",
                name="Ego steering",
                marker={"color": "black", "size": 14, "symbol": "star"},
            )
        )
    lateral_figure.update_layout(
        title="Lateral profile",
        xaxis_title="Time (s)",
        yaxis_title="Steering angle (rad)",
        height=380,
        uirevision="constant",
    )

    bev_column, profile_column = st.columns([3, 2])
    with bev_column:
        st.plotly_chart(bev_figure, use_container_width=True, key="mppi_bev_plot")
    with profile_column:
        st.plotly_chart(longitudinal_figure, use_container_width=True, key="mppi_longitudinal_plot")
        st.plotly_chart(lateral_figure, use_container_width=True, key="mppi_lateral_plot")

    if is_evaluated:
        cost_breakdown_figure = make_cost_breakdown_figure(
            result["cost_breakdown"], result["metrics"]["baseline_cost"]
        )
        st.plotly_chart(
            cost_breakdown_figure,
            use_container_width=True,
            key="mppi_cost_breakdown_plot",
        )

    st.subheader("Dataset curation")
    dataset_directory = st.text_input("Dataset directory", "dataset")
    tags = st.multiselect(
        "Tags",
        ("challenging", "avoidance", "cut_in", "sharp_turn", "high_speed", "kinematic_limit"),
    )
    custom_tag = st.text_input("Custom tag")
    if st.button("Save synchronized frame", disabled=not frame.is_usable):
        selected_tags = list(tags)
        if custom_tag.strip():
            selected_tags.append(custom_tag.strip())
        try:
            saved_path = save_frame(frame, synchronizer, dataset_directory, selected_tags)
            st.success(f"Saved {saved_path}")
        except Exception as error:
            st.error(f"The dataset write failed: {error}")

    st.subheader("Metrics")
    if is_evaluated:
        st.json(result["metrics"])
    else:
        st.info("Evaluate the selected frame to compute metrics.")


render_main_explorer()
