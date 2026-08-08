# MPPI Frame Explorer

An interactive web-based visualization and dataset curation tool for the Autoware **First-Order Dubins MPPI Trajectory Planner**.

Built with **Streamlit**, **Plotly**, and **pybind11**, this application bridges the gap between raw `.mcap` ROS 2 driving logs and deterministic, isolated C++ unit test / evaluation datasets. It enables autonomous driving engineers to inspect reference trajectories, evaluate MPPI configurations on-the-fly via C++17/CUDA bindings, and curate edge-case test frames with synchronized multi-topic state.

---

## Features

- **MCAP Bag Exploration:** Direct zero-order-hold (ZOH) indexing and scrubbing of Autoware `.mcap` ROS 2 recordings without manual pre-conversion.
- **Zero-Order-Hold Synchronization:** Uses the `reference_trajectory` timestamp as the trigger clock and retrieves the latest available odometry, tracked objects, and road borders within user-configured stale thresholds.
- **Live C++/CUDA Re-Optimization:** Embeds the native C++17 `FirstOrderDubinsMppiInterface` via Python bindings (`pybind11`), enabling ~1–3 ms live trajectory re-optimization when adjusting GUI sliders.
- **Interactive BEV Canvas:** Plotly-powered Bird's-Eye View displaying:
- Ego vehicle pose and odometry footprint
- Raw Reference Trajectory (gray dashed)
- Optimized MPPI Trajectory (green, or red if rejected by safety margins)
- Tracked obstacles and road border geometry

- **One-Click Dataset Curation:** Export synchronized frames as individual JSON test fixtures (`frame_<timestamp>.json`) and automatically register them with tags into `dataset/manifest.yaml`.

---

## Architecture

```text
                  ┌─────────────────────────────────────┐
                  │          topics.yaml                │
                  │   (User-Configurable ROS 2 Topics)  │
                  └──────────────────┬──────────────────┘
                                     │
┌─────────────────────────┐          ▼           ┌────────────────────────┐
│   MCAP Recording (.bag) │ ──► McapZohSync ───► │ Streamlit UI / Plotly  │
└─────────────────────────┘     (mcap_reader.py) │   • Timeline Scrubber  │
                                                 │   • Parameter Sliders  │
                                                 │   • BEV Canvas         │
                                                 └───────────┬────────────┘
                                                             │
                                                             ▼
                                                 ┌────────────────────────┐
                                                 │ pybind11 C++/CUDA Mod  │
                                                 │  mppi_optimizer_py     │
                                                 └───────────┬────────────┘
                                                             │
                                                             ▼
                                                 ┌────────────────────────┐
                                                 │ Curated Dataset Output │
                                                 │  • /dataset/curated/*.json
                                                 │  • /dataset/manifest.yaml
                                                 └────────────────────────┘

```

### Key Components

- `app.py`: Streamlit frontend providing the timeline scrubber, parameter controls, BEV plotting, and dataset export buttons.
- `mcap_reader.py`: Implements `McapZohSynchronizer` using the `rosbags` library to index timestamps and synchronize asynchronous ROS 2 topics with data-lag monitoring.
- `pybind_mppi.cpp`: C++ wrapper exposing `FirstOrderDubinsMppiInterface`, cost parameters, and runtime options to Python.
- `topics.yaml`: External configuration file mapping standard or custom ROS 2 topic names to logical MPPI input structures.

---

## Prerequisites

- **OS:** Ubuntu 22.04 / 24.04 (Linux x86_64 or ARM64)
- **C++ Standard:** C++17
- **GPU Runtime:** NVIDIA CUDA Toolkit (required for GPU MPPI rollouts)
- **Python:** 3.10+
- **ROS 2:** Humble / Jazzy (or standard Autoware message definitions)

---

## Installation & Setup

### 1. Clone & Install Python Dependencies

```bash
git clone https://github.com/your-org/mppi_frame_explorer.git
cd mppi_frame_explorer

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

```

#### Example `requirements.txt`

```text
streamlit>=1.30.0
plotly>=5.18.0
pyyaml>=6.0
rosbags>=0.9.15
pybind11>=2.11.0

```

### 2. Build C++/CUDA PyBind11 Module

Ensure your Autoware environment or ROS 2 workspace is sourced so dependency headers (e.g., `autoware_planning_msgs`) are available:

```bash
mkdir build && cd build
cmake .. -DBUILD_PYBIND_MODULE=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Add the compiled library (.so) to your Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..

```

---

## Configuration

Modify `topics.yaml` to match the topic names recorded in your `.mcap` files:

```yaml
topics:
  reference_trajectory: "/planning/scenario_planning/trajectory"
  odometry: "/localization/kinematic_state"
  tracked_objects: "/perception/object_recognition/tracking/objects"
  road_borders: "/planning/scenario_planning/lane_driving/behavior_planning/path_with_lane_id"
  vehicle_status: "/vehicle/status/steering_status"

thresholds:
  stale_warning_ms: 100.0 # Display UI warning if perception/odometry lags target timestamp
```

---

## Usage

### 1. Launch the Explorer

```bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/build
streamlit run app.py

```

### 2. Exploring & Tuning

1. **Load Bag:** In the sidebar, specify the path to your `.mcap` file and `topics.yaml`.
2. **Scrub Timeline:** Use the **Trajectory Timeline Scrubber** to move across reference trajectory messages. The application automatically pulls synchronized odometry, object, and border states for that exact timestamp.
3. **Adjust MPPI Parameters:** Drag the sliders for:
   - **Boundary Threshold (`boundary_threshold`)**
   - **Obstacle Collision Margin (`obstacle_collision_margin`)**
   - **Road Border Margin (`road_border_collision_margin`)**
4. **Inspect BEV Plot:** Watch the optimized trajectory dynamically re-compute and display over the reference path and obstacle bounding boxes.

### 3. Saving a Curated Frame

1. Apply relevant categorical tags from the UI (e.g., `cut_in`, `sharp_turn`, `tight_borders`, `emergency_stop`).
2. Click **Save Frame to Dataset**.
3. The tool generates an isolated input fixture at `dataset/curated/frame_<timestamp_ns>.json` and registers the path and tags in `dataset/manifest.yaml`.

---

## Dataset Format

Saved JSON frames are self-contained and directly digestible by the offline `MppiBatchEvaluator`:

```json
{
  "frame_id": "frame_1723107284000000000",
  "timestamp_ns": 1723107284000000000,
  "tags": ["sharp_turn", "tight_borders"],
  "reference_trajectory": { ... },
  "odometry": { ... },
  "tracked_objects": { ... },
  "road_borders": [ ... ]
}

```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.
