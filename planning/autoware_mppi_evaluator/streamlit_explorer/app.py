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
# limitations under the License.# mcap_reader.py
import json
from pathlib import Path

from mcap_reader import McapZohSynchronizer
import plotly.graph_objects as go
import streamlit as st
import yaml

# Optional: Import C++ Pybind module if built, otherwise mock
try:
    import mppi_optimizer_py as mppi_cpp

    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False

st.set_page_config(page_title="MPPI Frame Explorer", layout="wide")

# --- SIDEBAR: CONFIG & SLIDERS ---
st.sidebar.title("🛠️ MPPI Frame Explorer")
bag_path = st.sidebar.text_input("MCAP Bag Path", "data/test_drive.mcap")
config_path = st.sidebar.text_input("Topics Config", "topics.yaml")

st.sidebar.markdown("### 🎛️ MPPI Optimizer Sliders")
boundary_threshold = st.sidebar.slider("Boundary Threshold (m)", 0.1, 2.0, 0.5, 0.05)
obstacle_margin = st.sidebar.slider("Obstacle Margin (m)", 0.0, 1.0, 0.2, 0.05)
border_margin = st.sidebar.slider("Road Border Margin (m)", 0.0, 1.0, 0.2, 0.05)
skip_if_invalid = st.sidebar.checkbox("Skip if Invalid", value=True)


# --- LOAD DATASET ---
@st.cache_resource
def load_synchronizer(bag: str, cfg: str):
    return McapZohSynchronizer(bag, cfg)


if Path(bag_path).exists() and Path(config_path).exists():
    sync = load_synchronizer(bag_path, config_path)
    total_frames = len(sync.ref_timestamps)

    # --- TIMELINE SCRUBBER ---
    st.markdown("### ⏱️ Trajectory Timeline Scrubber")
    frame_idx = st.slider(
        "Select Frame Index",
        min_value=0,
        max_value=total_frames - 1,
        value=total_frames // 2,
        help="Jumps between reference trajectory messages over time.",
    )

    # Fetch Zero-Order-Hold synchronized frame
    frame_data = sync.get_synchronized_frame(frame_idx)

    # Render Stale Warnings
    for warn in frame_data["warnings"]:
        st.warning(f"⚠️ {warn}")

    # --- MPPI OPTIMIZATION ---
    opt_trajectory_x, opt_trajectory_y = [], []
    was_rejected = False

    if HAS_CPP_BACKEND:
        mppi = mppi_cpp.MppiInterface()
        params = mppi_cpp.CostParams()
        params.boundary_threshold = boundary_threshold
        params.obstacle_collision_margin = obstacle_margin
        params.road_border_collision_margin = border_margin
        mppi.set_cost_params(params)

        # Execute fast CUDA optimize call
        # result = mppi.optimize(...)
        # Extract x, y points from result

    # --- BEV CANVAS (PLOTLY) ---
    col_plot, col_meta = st.columns([3, 1])

    with col_plot:
        fig = go.Figure()

        # 1. Plot Reference Trajectory (Gray)
        # Extracting mock x/y from reference trajectory points
        ref_x = [p.pose.position.x for p in frame_data["reference_trajectory"].points]
        ref_y = [p.pose.position.y for p in frame_data["reference_trajectory"].points]
        fig.add_trace(
            go.Scatter(
                x=ref_x,
                y=ref_y,
                mode="lines+markers",
                name="Reference Trajectory",
                line={"color": "gray", "width": 2, "dash": "dash"},
            )
        )

        # 2. Plot Optimized Trajectory (Green / Red if Rejected)
        if opt_trajectory_x:
            color = "red" if was_rejected else "green"
            fig.add_trace(
                go.Scatter(
                    x=opt_trajectory_x,
                    y=opt_trajectory_y,
                    mode="lines",
                    name="Optimized Trajectory",
                    line={"color": color, "width": 4},
                )
            )

        # 3. Plot Ego Odometry Footprint
        if frame_data["odometry"]:
            ego_x = frame_data["odometry"].pose.pose.position.x
            ego_y = frame_data["odometry"].pose.pose.position.y
            fig.add_trace(
                go.Scatter(
                    x=[ego_x],
                    y=[ego_y],
                    mode="markers",
                    name="Ego Vehicle",
                    marker={"color": "blue", "size": 14, "symbol": "arrow-up"},
                )
            )

        fig.update_layout(
            title=f"BEV Canvas — Timestamp: {frame_data['timestamp_ns']}",
            xaxis_title="X (map)",
            yaxis_title="Y (map)",
            yaxis={"scaleanchor": "x", "scaleratio": 1},  # Equal aspect ratio
            height=650,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- METADATA & EXPORT TOOLING ---
    with col_meta:
        st.markdown("### 💾 Save & Curate")
        st.markdown(f"**Timestamp:** `{frame_data['timestamp_ns']}`")

        tags = st.multiselect(
            "Frame Tags",
            ["cut_in", "sharp_turn", "tight_borders", "high_speed", "emergency_stop"],
            default=["sharp_turn"],
        )
        custom_tag = st.text_input("Custom Tag")
        if custom_tag:
            tags.append(custom_tag)

        if st.button("📥 Save Frame to Dataset", type="primary", use_container_width=True):
            out_dir = Path("dataset/curated")
            out_dir.mkdir(parents=True, exist_ok=True)

            frame_id = f"frame_{frame_data['timestamp_ns']}"
            file_path = out_dir / f"{frame_id}.json"

            # Save raw ROS 2 input messages as JSON
            export_payload = {
                "frame_id": frame_id,
                "timestamp_ns": frame_data["timestamp_ns"],
                "tags": tags,
                "reference_trajectory": "...",  # Serialized JSON string of ROS msg
                "odometry": "...",
                "tracked_objects": "...",
            }

            with open(file_path, "w") as f:
                json.dump(export_payload, f, indent=2)

            # Append entry to dataset manifest.yaml
            manifest_path = Path("dataset/manifest.yaml")
            manifest_data = {"frames": []}
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    manifest_data = yaml.safe_load(f) or {"frames": []}

            manifest_data["frames"].append({"id": frame_id, "path": str(file_path), "tags": tags})

            with open(manifest_path, "w") as f:
                yaml.dump(manifest_data, f)

            st.success(f"Saved `{frame_id}.json` and updated `manifest.yaml`!")
else:
    st.info(
        "👈 Please select a valid MCAP bag and topic configuration YAML in the sidebar to begin."
    )
