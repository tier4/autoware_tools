# Setup

## Hardware

- NVIDIA GPU, ≥ 8 GB. Each parallel job = one Autoware stack + one TensorRT engine (~2.5 GB). Start with `max_workers: 1`.
- Fast local disk for `results_root` (~1 GB per job). Recordings are 20–80 GB each.
- ≥ 32 GB RAM (the reproducer loads a whole recording).
- Never `kill -9` a run: killed CUDA processes leak GPU memory until reboot. Use Ctrl-C.

## Workspace

| Package | Note |
|---|---|
| `dp_multi_eval` | this tool (`src/tools/planning`) |
| `planning_debug_tools` | the original `perception_reproducer.py`, unmodified; `scene_driver.py` runs it. Required. |
| `diffusion_planner_batch_eval` | route helper |
| `autoware_launch`, planning simulator, `autoware_diffusion_planner` | only `major_version` 4 weights load |
| `<vehicle_model>_description` | its `vehicle_info.param.yaml` is the ego's body everywhere; without it the scene warns and uses a default body |
| `autoware_trajectory_modifier` | must be in the launch tree — on this workspace: universe branch `feat/cherry-pick-poincloud-fusion` (`beta/v0.41` lacks it → "planning simulator exited early") |
| `topic_tools`, `rosbag2` (sqlite3) | recording |

Python modules (`yaml`, `numpy`, `scipy`, `matplotlib`, `lanelet2`, `autoware_lanelet2_extension_python`), `ffmpeg` and `streamlit` (the GUI) are declared in `package.xml`: the `rosdep` line below installs them.

| Rule | Why |
|---|---|
| run from a shell with no virtualenv active (`deactivate`) | a virtualenv hides the system modules; `run_all.py` refuses to start and names what is missing |

```bash
rosdep install --from-paths src/tools/planning/dp_multi_eval --ignore-src -y
colcon build --packages-select dp_multi_eval planning_debug_tools diffusion_planner_batch_eval   # --symlink-install optional
source install/setup.bash
ros2 run dp_multi_eval scene_driver.py --help | grep -c window-clock                  # 1
ros2 pkg list | grep -c autoware_trajectory_modifier                                  # 1
```

## Map

`map_path` = the vehicle's Hiratsuka map directory (v206): `lanelet2_map.osm`,
`map_projector_info.yaml`, `stop_points.csv` (`name,x,y,yaw`), `pointcloud_map.pcd`.
The older v106 map cannot route into the bus bays.

## Models

`models:` entries are `{name, model_config}`. `model_config` is a directory with
`diffusion_planner.onnx` + `args.json`, an `ml_planner.onnx` file (needs
`autoware_ml_planner`; such a job launches with `planning_setting: ml_planner`
on its own), or a `.param.yaml`. A param YAML is generated into
`~/.cache/dp_multi_eval/model_params/`. The name is the
dashboard column and part of every job id.

## Recordings

`rosbag_dir/<bag>/metadata.yaml + *.db3` (one extra folder level is allowed).
Run `ros2 bag reindex <bag> -s sqlite3` if `metadata.yaml` is missing.
Needed topics: ego pose, tracked objects, traffic signals, turn indicators
(see `config/topics.yaml`). Manually driven recordings are fine.

## Where things land

| Path | |
|---|---|
| `results_root/` | manifest, dashboard, videos, per-job folders |
| `~/.cache/dp_multi_eval/{model_params,bag_derived}/` | generated param YAMLs, per-bag caches (safe to delete) |
| `/tmp/dp_multi_eval_overlays/<sha1>/` | per-job launch overlay; never auto-cleaned |

## Shared machines

Each worker uses its own `ROS_DOMAIN_ID` (`domain_id_start` + i). A job kills
leftover evaluation processes on its domain and the machine's tracker nodes.
One batch per host; distinct `domain_id_start` per user.
