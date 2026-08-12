# Autoware MPPI Evaluator

This package evaluates `autoware_mppi_optimizer` against synchronized frames from ROS 2 MCAP files.

It provides a native batch API, a command-line evaluator, and an interactive Streamlit explorer.

## Input contract

The recorded MPPI reference trajectory defines the evaluation clock.

The evaluator selects the newest input message whose bag timestamp does not exceed the reference timestamp.

The required inputs are:

- `autoware_planning_msgs/msg/Trajectory`
- `nav_msgs/msg/Odometry`
- `autoware_perception_msgs/msg/TrackedObjects`
- `autoware_map_msgs/msg/LaneletMapBin`
- `autoware_planning_msgs/msg/LaneletRoute`

Acceleration and steering reports are optional.

The default topic configuration uses the diffusion planner MPPI reference debug topic.

This topic preserves the exact trajectory that entered the recorded MPPI call.

See `config/streamlit_explorer.yaml` for all topic names and stale-data limits.

## Map preparation

The package constructs `ExtendedRouteHandler` from the recorded map and route.

It creates the extended route map once for each map and route pair.

It queries nearby road borders and drivable-area boundaries for every reference trajectory.

The query margin contains the vehicle front extent plus one configurable meter.

Chronological mode also uses `TrackedObjectSelector` from `autoware_avoidance_target_detector`.

## Evaluation modes

`isolated` creates a new MPPI optimizer for each frame.

This mode removes warm-start state and permits random frame access.

This mode passes all tracked objects because one frame cannot reconstruct the detector history.

`chronological` preserves the MPPI warm start and the tracked-object filter state.

Frames must use timestamps in strict ascending order for this mode.

The explorer replays earlier frames when the selected index moves backward.

## Parameters

Provide these ROS parameter files:

- The MPPI optimizer parameter file
- The vehicle information parameter file
- The simulator model parameter file

The evaluator converts vehicle dimensions with the same formulas as `makeVehicleParams()`.

The native parameter structures provide defaults when a file omits a value.

## Build

Install the two Python user-interface dependencies:

```bash
python3 -m pip install -r src/tools/planning/autoware_mppi_evaluator/requirements.txt
```

Build the package in a sourced Autoware workspace with CUDA and TensorRT available:

```bash
colcon build --symlink-install --packages-up-to autoware_mppi_evaluator
source install/setup.bash
```

## Explorer

Start the explorer:

```bash
ros2 run autoware_mppi_evaluator mppi_streamlit_explorer.py
```

Select the MCAP directory and optimizer parameters in the sidebar.

The explorer reads its topic, vehicle-information, and vehicle-dynamics configuration from
`config/streamlit_explorer.yaml`.

Use `Evaluate frame` for an isolated frame.

Use `Replay through frame` for production-like history.

## Batch command

Evaluate one configuration:

```bash
ros2 run autoware_mppi_evaluator mppi_evaluate_bag.py BAG_PATH \
  --topics TOPICS.yaml \
  --optimizer-config baseline=MPPI.yaml \
  --vehicle-info VEHICLE_INFO.yaml \
  --simulator-model SIMULATOR_MODEL.yaml \
  --mode chronological \
  --output results/mppi
```

Repeat `--optimizer-config NAME=PATH` to compare configurations.

The command writes a frame CSV file and a JSON file with aggregate latency and rejection counts.

`--output-stride` reduces output rows but still evaluates every chronological frame.

Add `--visualize` to write a Plotly report to `<output>.html`. The report prioritizes invalid
or rejected frames, then fills the remaining `--visualize-limit` slots per configuration with
the valid frames that have the largest cross-track error. Visualization forces
`skip_if_invalid=false` so the report can display invalid MPPI candidate trajectories instead of
the fallback reference trajectory.

Evaluate a curated dataset:

```bash
ros2 run autoware_mppi_evaluator mppi_evaluate_bag.py dataset \
  --input-format dataset \
  --optimizer-config baseline=MPPI.yaml \
  --vehicle-info VEHICLE_INFO.yaml \
  --simulator-model SIMULATOR_MODEL.yaml \
  --output results/curated
```

## Dataset format

The explorer stores each curated frame as schema-versioned JSON.

Schema version 2 keeps dynamic ROS messages as base64-encoded CDR in each frame JSON.

Lanelet-map and route CDR payloads are content-addressed, gzip-compressed, and stored once under
`blobs/`. Frame files reference an environment hash, and the loader shares the decoded static
messages between frames.

The manifest stores topic names, message types, timestamps, tags, and environment metadata.

Manifest updates replace duplicate frame identifiers and use atomic file replacement.

## Metrics

The evaluator reports:

- MPPI execution time
- Baseline cost
- Rejection status and crash status (`0` valid, `1` lateral bound, `2` obstacle,
  `3` road border)
- Effective sample size
- Maximum importance weight
- Maximum and mean cross-track error
- Maximum lateral acceleration
- Maximum acceleration-command rate
- Maximum steering-state rate
- Obstacle and boundary clearance

The acceleration trajectory field contains an MPPI command, not the simulated acceleration state.

The obstacle clearance uses conservative circumscribed circles around each vehicle box.

The boundary clearance uses a conservative circle around the centered ego vehicle box.
