# Frame Geometry Output

With `export_frame_geometry:=true`, the open-loop evaluator writes two additional files next to
`time_step_based_trajectory_result.jsonl`. They contain the positions needed to place the
per-frame ADE/FDE on the map (e.g. for location-based error maps). The existing outputs are not
changed.

```sh
ros2 launch autoware_planning_data_analyzer planning_data_analyzer.launch.xml \
  bag_path:=$HOME/my_bag output_dir:=$HOME/result vehicle_model:=j6_gen2 \
  export_frame_geometry:=true
```

| File                                              | Content                                                          |
| ------------------------------------------------- | ---------------------------------------------------------------- |
| `time_step_based_trajectory_frame_geometry.jsonl` | One line per evaluated frame: ego state, predicted trajectory, GT |
| `time_step_based_trajectory_route.json`           | Route centerline of the whole drive and its lanelets              |

## Conventions

- Coordinates are in the `map` frame [m] (2D). yaw is [rad] in (-π, π], time [s], velocity [m/s].
- Curvature is signed [1/m]; a left turn is positive.
- Point sequences are stored column-wise (`{"x": [...], "y": [...]}`).
- Values that cannot be computed are `null` (e.g. a point farther than
  `max_route_lateral_offset_m` from the route).
- The N-th line of `frame_geometry.jsonl` is the same frame as the N-th line of
  `time_step_based_trajectory_result.jsonl` (`timestamp_sec` equals `Stamp.ROS`).

## Route of a drive

A drive often consists of several routes (e.g. a bus gets a new route at every stop). All route
messages in the bag are joined in time order into one centerline:

1. Each route is active from its `header.stamp` until the next route. The first route is also
   used before its stamp.
2. Every ego / GT position is assigned to the route active at its time.
3. Each route is clipped to the part the ego drove while it was active, and the clipped routes
   are concatenated. A gap between routes is counted by its straight-line distance.

`route_s` is therefore the arc length along the driven route, and points are projected only onto
the route active at their time, so the same road driven twice (laps) gets different `route_s`.

## time_step_based_trajectory_frame_geometry.jsonl

```json
{
  "frame_index": 100,
  "timestamp_sec": 1792465267.385,
  "bag_timestamp_sec": 1792465267.385,
  "valid": true,
  "control_mode": "AUTONOMOUS",
  "ego": {
    "x": 50026.886, "y": 9792.713, "yaw": -3.0137, "v": 3.115,
    "route_s": 1151.067, "route_lateral_offset": 0.021,
    "lanelet_id": 83907, "route_curvature": -0.00056, "route_index": 4
  },
  "predicted": {"t": [...], "x": [...], "y": [...], "yaw": [...], "v": [...], "curvature": [...]},
  "ground_truth": {
    "x": [...], "y": [...], "yaw": [...],
    "route_s": [...], "route_lateral_offset": [...], "lanelet_id": [...],
    "route_curvature": [...], "route_index": [...]
  },
  "horizons": {
    "1.0s": {"index": 9, "t": 1.0, "ade": 0.068, "fde": 0.1},
    "2.0s": {"index": 19, "t": 2.0, "ade": 0.155, "fde": 0.388},
    "4.0s": {"index": 39, "t": 4.0, "ade": 0.535, "fde": 1.378},
    "full": {"index": 39, "t": 4.0, "ade": 0.535, "fde": 1.378}
  }
}
```

| Field                         | Unit        | Description                                                                                                            |
| ----------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `frame_index`                 | -           | 0-based frame number (line number of `time_step_based_trajectory_result.jsonl`)                                        |
| `timestamp_sec`               | s           | `header.stamp` of the trajectory (= `Stamp.ROS`)                                                                       |
| `bag_timestamp_sec`           | s           | Bag time (= `Stamp.System`)                                                                                            |
| `valid`                       | -           | At least one horizon was evaluated (= `Result.Success`)                                                                |
| `control_mode`                | -           | Last `/vehicle/status/control_mode` at or before the frame: `AUTONOMOUS`, `MANUAL`, ... `UNKNOWN` if none              |
| `ego.x/y/yaw/v`               | m, rad, m/s | Ego state at the inference time (synchronized kinematic state = GT point 0)                                            |
| `ego.route_s`                 | m           | Arc length of the projection onto the drive route                                                                      |
| `ego.route_lateral_offset`    | m           | Signed distance from the route centerline (left is positive)                                                           |
| `ego.lanelet_id`              | -           | Route lanelet at the projection                                                                                        |
| `ego.route_curvature`         | 1/m         | Route centerline curvature at the projection                                                                           |
| `ego.route_index`             | -           | Index of the route (`routes[].index` of route.json) at the projection                                                  |
| `predicted.t`                 | s           | `time_from_start` of each point (after truncation by `trajectory_evaluation_horizon`)                                  |
| `predicted.x/y/yaw/v`         | m, rad, m/s | Predicted trajectory points                                                                                            |
| `predicted.curvature`         | 1/m         | Curvature of the predicted trajectory (three points about 1 m apart; `null` when the trajectory is too short / stopped) |
| `ground_truth.*`              |             | Actual ego poses at the same `time_from_start` as the predicted points, with the same route attributes as `ego`        |
| `horizons.<key>.index`        | -           | Index of the point used for the horizon in `predicted` / `ground_truth`                                                |
| `horizons.<key>.t/ade/fde`    | s, m        | Same values as `time_step_based_trajectory_result.jsonl`                                                               |

### Where to place the error of a frame

- At the inference time (default of the report): `ego.x/y`, `ego.route_s`
- At the evaluated time: `ground_truth.x/y[index]`, `ground_truth.route_s[index]`
  with `index = horizons.<key>.index`

The two differ by about (vehicle speed × horizon), e.g. 33 m for 30 km/h and a 4.0 s horizon.

## time_step_based_trajectory_route.json

```json
{
  "frame_id": "map",
  "resample_interval_m": 1.0,
  "length_m": 4002.313,
  "centerline": {
    "s": [...], "x": [...], "y": [...], "yaw": [...],
    "curvature": [...], "lanelet_id": [...], "route_index": [...]
  },
  "routes": [
    {"index": 0, "uuid": "0cc1beb2...", "active_from_sec": 1792464855.2, "used": true,
     "s_start": 0.0, "s_end": 63.483}
  ],
  "lanelets": [
    {"id": 1234, "route_index": 0, "s_start": 0.0, "s_end": 42.3,
     "turn_direction": "straight", "is_intersection": true, "speed_limit_mps": 8.333}
  ]
}
```

| Field                     | Description                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `centerline`              | Preferred lanelets of each route resampled every `resample_interval_m`. Curvature uses a 3 m baseline |
| `routes`                  | Route messages in time order (consecutive duplicates removed). `used` is false if never driven       |
| `lanelets`                | Arc length range of each lanelet on the centerline, for sectioning (e.g. intersections)              |
| `lanelets.turn_direction` | `turn_direction` attribute of the lanelet (`straight`, `left`, `right`), or `none`                   |
| `lanelets.is_intersection`| true if the lanelet has a `turn_direction` attribute                                                 |

## Parameters

| Parameter                                          | Default | Description                                                  |
| -------------------------------------------------- | ------- | ------------------------------------------------------------ |
| `open_loop.frame_geometry.enabled`                 | false   | Set by the `export_frame_geometry` launch argument           |
| `open_loop.frame_geometry.position_decimals`       | 3       | Decimal digits of x, y, route_s, route_lateral_offset, ade, fde |
| `open_loop.frame_geometry.angle_decimals`          | 4       | Decimal digits of yaw                                        |
| `open_loop.frame_geometry.curvature_decimals`      | 5       | Decimal digits of curvature                                  |
| `open_loop.frame_geometry.route_resample_interval_m` | 1.0   | Resampling interval of the route centerline [m]              |
| `open_loop.frame_geometry.max_route_lateral_offset_m` | 5.0  | Points farther than this from the route get `null` route attributes [m] |

For a 25 minute drive (about 15,000 frames with 40 points each), `frame_geometry.jsonl` is about
70 MB.

## Error map report

`plot_error_map.py` bins the per-frame ADE/FDE by route arc length and writes one PNG per
(metric, horizon): the route colored by the mean error of each bin, and the mean / max error of
each bin along the route (intersection lanelets are shaded).

```sh
ros2 run autoware_planning_data_analyzer plot_error_map.py $HOME/result
```

Output: `<input_dir>/error_map/{ade,fde}_{1.0s,2.0s,4.0s}_{inference,evaluated}.png`

| Option           | Default             | Description                                                                         |
| ---------------- | ------------------- | ----------------------------------------------------------------------------------- |
| `--position`     | `inference`         | Place the error at the ego position at the inference time, or `evaluated`: the GT position at t + horizon |
| `--horizons`     | `1.0s 2.0s 4.0s`    | Horizon keys                                                                        |
| `--metrics`      | `ade fde`           | Metrics to plot                                                                     |
| `--bin-size`     | `5.0`               | Bin size along the route [m]                                                        |
| `--control-mode` | `AUTONOMOUS`        | Use only frames in this control mode, or `all`                                      |
| `--vmax`         | 99th percentile     | Upper limit of the map color scale [m]; set it to compare images with the same scale |
| `--label`        | input directory name | Run name shown in the title                                                        |
| `--output-dir`   | `<input_dir>/error_map` | Output directory                                                                |
