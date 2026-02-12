# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

This is a ROS 2 package for analyzing planning behavior by comparing manual driving data against Autoware planning outputs. It processes rosbag data to calculate driving metrics and scores for both human drivers and autonomous system trajectories.

The package is part of the `autoware_tools` repository, which contains development, debugging, and analysis tools for Autoware.

## Build System

This package uses ROS 2 with ament_cmake and autoware_cmake build tools.

### Building

From the workspace root (pilot-auto.x2):
```bash
colcon build --packages-select autoware_planning_data_analyzer --symlink-install
```

### Running

```bash
source install/setup.bash
ros2 launch autoware_planning_data_analyzer behavior_analyzer.launch.xml bag_path:=/path/to/rosbag
```

### Testing

```bash
colcon test --packages-select autoware_planning_data_analyzer
```

## Code Architecture

### Core Components

**BehaviorAnalyzerNode** (`node.hpp`, `node.cpp`)
- Main ROS 2 node that orchestrates the analysis
- Reads rosbag data and manages playback via services (play, rewind)
- Publishes metrics, scores, and visualization markers
- Runs grid search for optimal weight parameters via the `weight_grid_search` service

**Data Structures** (`data_structs.hpp`, `data_structs.cpp`)
- `BagData`: Manages timestamped message buffers for all input topics
- `ManualDrivingData`: Processes actual driven trajectory from odometry/acceleration/steering
- `TrajectoryData`: Processes candidate planned trajectories
- `SamplingTrajectoryData`: Collection of sampled trajectories with optimization methods
- `DataSet`: Combines manual and sampled data for comparison

**Buffer System**
- Template-based `Buffer<T>` for time-synchronized message storage
- Specialized implementations for `TFMessage` and `SteeringReport`
- 20-second rolling window of historical data

### Metrics and Scoring

**Metrics** (METRIC enum)
- LATERAL_ACCEL: Lateral acceleration magnitude
- LONGITUDINAL_ACCEL: Longitudinal acceleration
- LONGITUDINAL_JERK: Rate of change of longitudinal acceleration
- TRAVEL_DISTANCE: Total distance traveled
- MINIMUM_TTC: Minimum time-to-collision with predicted objects

**Scores** (SCORE enum)
- LATERAL_COMFORTABILITY: Comfort in lateral motion
- LONGITUDINAL_COMFORTABILITY: Comfort in longitudinal motion
- EFFICIENCY: Travel efficiency metric
- SAFETY: Collision risk assessment

### Topic Configuration

Input topics are defined in `TOPIC` struct:
- TF: `/tf`
- ODOMETRY: `/localization/kinematic_state`
- ACCELERATION: `/localization/acceleration`
- OBJECTS: `/perception/object_recognition/objects`
- TRAJECTORY: `/planning/trajectory`
- STEERING: Vehicle steering reports

### Trajectory Sampling

The package uses `autoware_frenet_planner` and `autoware_path_sampler` to generate candidate trajectories with different lateral/longitudinal target states. These are compared against the actual driven trajectory to find optimal planning weights.

## ROS 2 Parameter Management

This package follows standard Autoware parameter patterns. When adding new parameters:

1. Add to `Parameters` struct in `data_structs.hpp`
2. Add `declare_parameter` call in node constructor (`node.cpp`)
3. Update `config/behavior_analyzer.param.yaml` with default value
4. Update `config/behavior_analyzer.json` schema if using parameter validation

Current parameters:
- `resample_num`: Number of resampling points for trajectories
- `time_resolution`: Time step for trajectory analysis
- `weight.*`: Scoring weights for different metrics
- `target_state.*`: Frenet target state parameters for trajectory sampling
- `grid_search.*`: Grid search configuration for weight optimization

## Code Quality

### Pre-commit Hooks

Run before committing:
```bash
pre-commit run --all-files
```

Key checks:
- clang-format: C++ formatting
- cpplint: C++ style checking
- prettier: XML/YAML formatting
- yamllint: YAML validation

### Header Guards

Use `#ifndef FILENAME_HPP_` format with ROS include guard conventions.

### Include Order

1. Local headers
2. Autoware headers
3. ROS headers
4. System/STL headers

## Dependencies

Critical dependencies:
- `autoware_frenet_planner`: Frenet frame trajectory generation
- `autoware_path_sampler`: Path sampling utilities
- `autoware_route_handler`: Route and lanelet map handling
- `autoware_motion_utils`: Trajectory interpolation and calculations
- `autoware_vehicle_info_utils`: Vehicle geometry parameters
- `rosbag2_cpp`: Rosbag reading interface

## Service Interface

Available services:
- `~/play`: Start/stop playback (SetBool)
- `~/rewind`: Reset playback to beginning (Trigger)
- `~/weight_grid_search`: Run weight optimization (Trigger)

## Output Topics

- `~/output/manual_metrics`: Metrics from human driving
- `~/output/system_metrics`: Metrics from Autoware planning
- `~/output/manual_score`: Scores from human driving
- `~/output/system_score`: Scores from Autoware planning
- `~/marker`: Visualization markers for RViz
