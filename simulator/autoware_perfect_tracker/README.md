# Autoware Perfect Tracker

## 1. Introduction

The `autoware_perfect_tracker` is a physics-free simulator for Autoware. Its primary purpose is to provide a "perfect" vehicle that flawlessly follows a given trajectory. This is Useful for developing and debugging planning and control modules without the complexities and noise of a full physics-based simulator.

The tracker works by subscribing to the planned trajectory and updating the vehicle's state.

## 2. Core Logic

The state of the simulated vehicle is updated in a timer callback. The logic can be summarized in the following steps:

1. **Select a Trajectory**: The tracker maintains a short history of received trajectories. Based on the `n_delay_steps` parameter, it selects a trajectory to follow. This can be used to simulate system latency.

2. **Find the Closest Point**: It finds the point on the selected trajectory that is closest to the vehicle's current 2D position (x, y). Let's call this point `p_target`.

3. **State Update**: The vehicle's kinematic state is updated based on `p_target` and a simple time step integration.
   - **Velocity, Acceleration**: The vehicle's longitudinal velocity, heading rate (angular velocity), steering angle, and acceleration are instantly set to the values from `p_target`.
     - `v_t = p_target.longitudinal_velocity_mps`
     - `ω_t = p_target.heading_rate_rps`

   - **Position and direction Update**: The vehicle's 2D pose is moved forward using **Euler integration** with the target velocity from the _previous_ step. The new yaw is snapped directly to the yaw of `p_target`.
     - `x_{t+1} = x_t + v_t * cos(yaw_t) * dt`
     - `y_{t+1} = y_t + v_t * sin(yaw_t) * dt`
     - `z_{t+1} = p_target.z`
     - `yaw_{t+1} = p_target.yaw`
     - `pitch_{t+1} = calculated slope from map info`

## 3. Perfect Following with `n_lookahead_points`

By default the tracker moves the vehicle forward with **Euler integration** (Section 2), so on curves the vehicle can drift slightly off the trajectory. If you want the vehicle to stay **exactly on the trajectory**, enable perfect following via the `n_lookahead_points` parameter.

- **Logic**: Each step the tracker finds the closest trajectory point, then **snaps the vehicle directly onto the trajectory point `n_lookahead_points` ahead of it**.
  - `x_{t+1} = p_ahead.x`
  - `y_{t+1} = p_ahead.y`
  - `yaw_{t+1} = p_ahead.yaw`
  - `z_{t+1}` : unchanged from the default logic (interpolated along the trajectory)
  - `pitch_{t+1}` : unchanged from the default logic (calculated from map slope)
  - velocity / heading rate / steering / acceleration are reported from `p_ahead`.
- **Parameter**:
  - `n_lookahead_points < 0` (default `-1`): **disabled**. Uses the physics-mimic Euler integration described in Section 2.
  - `n_lookahead_points = 0`: snap onto the closest point.
  - `n_lookahead_points = N`: snap onto the point `N` indices ahead of the closest point.
- **Note**: In this mode the forward progress per step is roughly `N` point-spacings per `0.1 s` cycle, so the actual travel speed is determined by `N` and the trajectory point spacing rather than by the planned `longitudinal_velocity_mps` (the planned velocity is still published on the report topics).

## 4. Flexible Delay with `n_delay_steps`

This package includes a feature to simulate system latency via the `n_delay_steps` parameter.

- **Logic**: The tracker stores the last 20 trajectories it received. `n_delay_steps` is the index into this history buffer (0 being the most recent).
  - `n_delay_steps = 0` (default): Use the latest trajectory. No delay.
  - `n_delay_steps = 1`: Use the second-to-last trajectory received.
  - ...and so on.
- **Use Case**: This is useful for testing the robustness of the planner and controller to delays in the system.

## 5. Inputs / Outputs

### 5.1 Subscribed Topics

| Name                            | Type                                          | Description                                                                      |
| :------------------------------ | :-------------------------------------------- | :------------------------------------------------------------------------------- |
| `/planning/trajectory`          | `autoware_planning_msgs/Trajectory`           | **[Key]** The target trajectory the vehicle must follow.                         |
| `/map/vector_map`               | `autoware_map_msgs/LaneletMapBin`             | **[Key]** Lanelet2 map data used to calculate vehicle pitch based on road slope. |
| `/initialpose3d`                | `geometry_msgs/PoseWithCovarianceStamped`     | **[Key]** Input to initialize or reset the vehicle's pose.                       |
| `/initialtwist`                 | `geometry_msgs/TwistStamped`                  | Input to initialize the vehicle's velocity.                                      |
| `/planning/turn_indicators_cmd` | `autoware_vehicle_msgs/TurnIndicatorsCommand` | Commands for turn signals.                                                       |
| `/planning/hazard_lights_cmd`   | `autoware_vehicle_msgs/HazardLightsCommand`   | Commands for hazard lights.                                                      |

### 5.2 Published Topics

| Name                         | Type                                       | Description                                                                  |
| :--------------------------- | :----------------------------------------- | :--------------------------------------------------------------------------- |
| `output/odometry`            | `nav_msgs/Odometry`                        | **[Key]** The simulated vehicle pose and twist in the map frame.             |
| `output/twist`               | `autoware_vehicle_msgs/VelocityReport`     | **[Key]** Current vehicle velocity and heading rate.                         |
| `output/steering`            | `autoware_vehicle_msgs/SteeringReport`     | **[Key]** Current tire steering angle.                                       |
| `output/gear_report`         | `autoware_vehicle_msgs/GearReport`         | **[Key]** Current gear (Drive/Reverse), calculated from trajectory velocity. |
| `output/control_mode_report` | `autoware_vehicle_msgs/ControlModeReport`  | Reports the current control mode (e.g., AUTONOMOUS).                         |
| `output/acceleration`        | `geometry_msgs/AccelWithCovarianceStamped` | Current acceleration derived from the trajectory target point.               |
| `output/pose`                | `geometry_msgs/PoseWithCovarianceStamped`  | The current pose of the vehicle (subset of odometry).                        |

## 6. How to Use

### (1) Git clone

clone this repo under `path/to/universe/simulator/`

### (2) Launch Integration

Please apply the following changes to `path/to/universe/launch/tier4_simulator_launch/launch/simulator.launch.xml`

```diff
@@ -2,6 +2,8 @@
 <launch>
+  <arg name="simulator_type" default="simple_planning_simulator"/>
+  <arg name="n_delay_steps" default="0" description="delay step for perfect tracker"/>
+  <arg name="n_lookahead_points" default="-1" description="perfect following lookahead points (negative disables it)"/>
   <arg name="fault_injection_param_path"/>
   <arg name="obstacle_segmentation_ground_segmentation_elevation_map_param_path"/>
   <arg name="laserscan_based_occupancy_grid_map_param_path"/>
@@ -155,14 +157,22 @@

   <group if="$(var launch_dummy_vehicle)">
-    <arg name="simulator_model" default="$(var vehicle_model_pkg)/config/simulator_model.param.yaml" description="path to the file of simulator model"/>
-    -    <let name="motion_publish_mode" value="pose_only" if="$(eval '&quot;$(var localization_sim_mode)&quot;==&quot;pose_twist_estimator&quot;')"/>
-    <let name="motion_publish_mode" value="full_motion" unless="$(eval '&quot;$(var localization_sim_mode)&quot;==&quot;pose_twist_estimator&quot;')"/>
-    <include file="$(find-pkg-share autoware_simple_planning_simulator)/launch/simple_planning_simulator.launch.py">
-      <arg name="vehicle_info_param_file" value="$(var vehicle_info_param_file)"/>
-      <arg name="simulator_model_param_file" value="$(var simulator_model)"/>
-      <arg name="initial_engage_state" value="$(var initial_engage_state)"/>
-      <arg name="raw_vehicle_cmd_converter_param_path" value="$(var raw_vehicle_cmd_converter_param_path)"/>
-      <arg name="motion_publish_mode" value="$(var motion_publish_mode)"/>
-    </include>
+    <group if="$(eval &quot;'$(var simulator_type)'=='simple_planning_simulator'&quot;)">
+      <arg name="simulator_model" default="$(var vehicle_model_pkg)/config/simulator_model.param.yaml" description="path to the file of simulator model"/>
+      <let name="motion_publish_mode" value="pose_only" if="$(eval '&quot;$(var localization_sim_mode)&quot;==&quot;pose_twist_estimator&quot;')"/>
+      <let name="motion_publish_mode" value="full_motion" unless="$(eval '&quot;$(var localization_sim_mode)&quot;==&quot;pose_twist_estimator&quot;')"/>
+      <include file="$(find-pkg-share autoware_simple_planning_simulator)/launch/simple_planning_simulator.launch.py">
+        <arg name="vehicle_info_param_file" value="$(var vehicle_info_param_file)"/>
+        <arg name="simulator_model_param_file" value="$(var simulator_model)"/>
+        <arg name="initial_engage_state" value="$(var initial_engage_state)"/>
+        <arg name="raw_vehicle_cmd_converter_param_path" value="$(var raw_vehicle_cmd_converter_param_path)"/>
+        <arg name="motion_publish_mode" value="$(var motion_publish_mode)"/>
+      </include>
+    </group>
+
+    <group if="$(eval &quot;'$(var simulator_type)'=='perfect_tracker'&quot;)">
+      <include file="$(find-pkg-share autoware_perfect_tracker)/launch/perfect_tracker.launch.xml">
+        <arg name="n_delay_steps" value="$(var n_delay_steps)"/>
+        <arg name="n_lookahead_points" value="$(var n_lookahead_points)"/>
+      </include>
+    </group>
+
   </group>
 </launch>
```

### (3) Build

To build the project with optimized release settings, use the following command within the `pilot-auto` dir:

```bash
colcon build --symlink-install --packages-select autoware_perfect_tracker --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

if run into deprecated warnings, use the below

```bash
colcon build --symlink-install --packages-select autoware_perfect_tracker --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations"
```

### (4) Command on CLI

To run the perfect tracker, set the `simulator_type` argument to `perfect_tracker` when launching `planning simulator`. You can also specify the delay using the `n_delay_steps`(`int`) argument with one step = 0.1s (if not added, `0` delay as default).

```bash
ros2 launch autoware_launch planning_simulator.launch.xml \
  map_path:=/path/to/your/map \
  vehicle_model:=your_vehicle_model \
  sensor_model:=your_sensor_model \
  simulator_type:=perfect_tracker \
  n_delay_steps:=int
```

Example command: 0.3s delay

```bash
ros2 launch autoware_launch planning_simulator.launch.xml \
  map_path:=$HOME/autoware_map/shinagawa_odaiba \
  vehicle_model:=lexus \
  sensor_model:=aip_xx1 \
  simulator_type:=perfect_tracker \
  n_delay_steps:=3
```
