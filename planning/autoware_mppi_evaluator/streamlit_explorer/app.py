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
from dataset_io import save_frame
from evaluator_config import make_configuration
from mcap_reader import McapZohSynchronizer
import plotly.graph_objects as go
import streamlit as st

try:
    import mppi_optimizer_py as mppi_cpp

    BACKEND_ERROR = None
except ImportError as error:
    mppi_cpp = None
    BACKEND_ERROR = str(error)


def package_file(package: str, relative: str) -> str:
    try:
        return str(Path(get_package_share_directory(package)) / relative)
    except Exception:
        return ""


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


st.set_page_config(page_title="MPPI Frame Explorer", layout="wide")
st.sidebar.title("MPPI Frame Explorer")

default_topics = package_file("autoware_mppi_evaluator", "config/streamlit_explorer.yaml")
default_optimizer = package_file("autoware_mppi_optimizer", "config/mppi_optimizer.param.yaml")
default_vehicle = package_file("j6_gen2_description", "config/vehicle_info.param.yaml")
default_simulator = package_file("j6_gen2_description", "config/simulator_model.param.yaml")

bag_path = st.sidebar.text_input("MCAP path")
topics_path = st.sidebar.text_input("Topic configuration", default_topics)
optimizer_path = st.sidebar.text_input("Optimizer parameters", default_optimizer)
vehicle_path = st.sidebar.text_input("Vehicle information", default_vehicle)
simulator_path = st.sidebar.text_input("Vehicle dynamics", default_simulator)
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

st.sidebar.subheader("Cost overrides")
boundary_threshold = st.sidebar.slider(
    "Boundary threshold (m)",
    0.1,
    3.0,
    min(3.0, max(0.1, float(configuration.cost_params.boundary_threshold))),
    0.05,
)
obstacle_margin = st.sidebar.slider(
    "Obstacle margin (m)",
    0.0,
    1.0,
    min(1.0, max(0.0, float(configuration.cost_params.obstacle_collision_margin))),
    0.05,
)
border_margin = st.sidebar.slider(
    "Road-border margin (m)",
    0.0,
    1.0,
    min(1.0, max(0.0, float(configuration.cost_params.road_border_collision_margin))),
    0.05,
)
skip_if_invalid = st.sidebar.checkbox(
    "Reject an invalid result", value=configuration.runtime_options.skip_if_invalid
)
auto_evaluate = st.sidebar.checkbox("Auto-evaluate on select", value=False)

if not bag_path or not Path(bag_path).expanduser().exists():
    st.info("Select an MCAP file to start.")
    st.stop()

try:
    synchronizer = load_synchronizer(bag_path, topics_path)
except Exception as error:
    st.error(f"The MCAP index failed: {error}")
    st.stop()

configuration.cost_params.boundary_threshold = boundary_threshold
configuration.cost_params.obstacle_collision_margin = obstacle_margin
configuration.cost_params.road_border_collision_margin = border_margin
configuration.runtime_options.skip_if_invalid = skip_if_invalid

frame_index = st.slider("Frame index", 0, len(synchronizer) - 1, len(synchronizer) // 2)
frame = synchronizer.get_synchronized_frame(frame_index)
for warning in frame.warnings:
    st.warning(warning)

session_key = (
    str(Path(bag_path).expanduser().resolve()),
    mode,
    optimizer_path,
    vehicle_path,
    simulator_path,
    boundary_threshold,
    obstacle_margin,
    border_margin,
    skip_if_invalid,
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
needs_eval = auto_evaluate and st.session_state.last_evaluated_index != frame_index
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
                        st.session_state.result = evaluate(st.session_state.evaluator, replay_frame)
                    progress.progress((offset + 1) / usable_count)
                progress.empty()
                st.session_state.last_evaluated_index = frame_index
        except Exception as error:
            st.error(f"The evaluation failed: {error}")

result = st.session_state.get("result")
is_evaluated = result and result["timestamp_ns"] == frame.timestamp_ns

plot_column, metric_column = st.columns([3, 1])
with plot_column:
    figure = go.Figure()

    if is_evaluated:
        add_trajectory(figure, result["reference_trajectory"], "Reference", "gray", "dash")

        if "original_trajectory" in frame.messages:
            trajectory = mppi_cpp.deserialize_trajectory(frame.messages["original_trajectory"])
            xs = [point["x"] for point in trajectory["points"]]
            ys = [point["y"] for point in trajectory["points"]]
            add_trajectory_arrays(figure, xs, ys, "Original (Recorded)", "#1f77b4", "dot")

        output_color = "red" if result["metrics"]["was_rejected"] else "green"
        add_trajectory(figure, result["optimized_trajectory"], "Optimized", output_color, "solid")
        add_segments(figure, result["road_borders"], "Road borders", "firebrick")
        add_segments(figure, result["drivable_area"], "Drivable bounds", "darkorange")
        for object_index, tracked_object in enumerate(result["selected_objects"]):
            xs_box, ys_box = box_outline(**tracked_object)
            figure.add_trace(
                go.Scatter(
                    x=xs_box,
                    y=ys_box,
                    mode="lines",
                    name="Selected objects",
                    legendgroup="Selected objects",
                    showlegend=object_index == 0,
                    fill="toself",
                    line={"color": "purple"},
                )
            )
    else:
        st.info(
            'ℹ️ Preview Mode — Showing recorded bag data. Click "Evaluate" to run the MPPI optimizer.'
        )

        for traj_key, name, color, dash in [
            ("reference_trajectory", "Reference", "gray", "dash"),
            ("original_trajectory", "Original (Recorded)", "#1f77b4", "dot"),
        ]:
            if traj_key in frame.messages:
                trajectory = mppi_cpp.deserialize_trajectory(frame.messages[traj_key])
                xs = [point["x"] for point in trajectory["points"]]
                ys = [point["y"] for point in trajectory["points"]]
                add_trajectory_arrays(figure, xs, ys, name, color, dash)

        if "tracked_objects" in frame.messages:
            tracked_objects = mppi_cpp.deserialize_tracked_objects_in_range(
                frame.messages["tracked_objects"],
                frame.messages["reference_trajectory"],
                object_filter_margin(configuration),
            )
            for object_index, tracked_object in enumerate(tracked_objects):
                xs_box, ys_box = box_outline(
                    tracked_object["x"],
                    tracked_object["y"],
                    tracked_object["yaw"],
                    tracked_object["length"],
                    tracked_object["width"],
                )
                figure.add_trace(
                    go.Scatter(
                        x=xs_box,
                        y=ys_box,
                        mode="lines",
                        name="Tracked objects",
                        legendgroup="Tracked objects",
                        showlegend=object_index == 0,
                        fill="toself",
                        line={"color": "gray"},
                    )
                )

    figure.update_layout(
        title=f"Frame {frame.timestamp_ns}",
        xaxis_title="Map X (m)",
        yaxis_title="Map Y (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        height=700,
    )
    st.plotly_chart(figure, use_container_width=True)

with metric_column:
    st.subheader("Metrics")
    if is_evaluated:
        st.json(result["metrics"])
    else:
        st.info("Evaluate the selected frame to compute metrics.")

st.subheader("Dataset curation")
dataset_directory = st.text_input("Dataset directory", "dataset")
tags = st.multiselect(
    "Tags", ("challenging", "avoidance", "cut_in", "sharp_turn", "high_speed", "kinematic_limit")
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
