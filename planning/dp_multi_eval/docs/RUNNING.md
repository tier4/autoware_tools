# Running

## Config (`config/example_pipeline.yaml`)

Unknown keys are ignored silently — the accepted list is `PIPELINE_KNOBS` in `job_config.py`.

| Key | Default | |
|---|---|---|
| `rosbag_dir`, `results_root`, `map_path`, `models` | — | required |
| `bag_whitelist`, `scenario_whitelist` | all | run only these paths under the folder (the GUI writes them) |
| `max_workers` | 2 | parallel jobs (GPU-bound) |
| `domain_id_start` | 10 | first `ROS_DOMAIN_ID` |
| `vehicle_model`, `sensor_model`, `vehicle_id` | `lv828l`, `aip_x2_gen2`, — | psim launch |
| `disable_long_stop_fixer`, `match_vehicle_dynamics`, `indicator_merge` | on | stack overlay ([method](EVALUATION_METHOD.md#stack-overlay)) |
| `render_video`, `render_video_on_failure`, `render_workers` | on, on, 1 | preview videos |
| `video_fps`, `video_sample_dt` | 8, 0.2 | video speed = fps × dt (1.6×; the example's 6/0.5 = 3×) |
| `video_view_frame`, `video_view_range_m`, `show_planning_factors` | `base_link`, 40, on | |
| `psim_startup_sec`, `planner_warmup_timeout_sec`, `route_setup_timeout_sec`, `auto_engage_timeout_sec` | 120, 900, 90, 60 | waits |
| `bag_load_base_sec`, `bag_load_duration_factor`, `bag_load_timeout_cap_sec` | 120, 1.0, 3600 | reproducer load budget = base + duration × factor |
| `gpu_min_free_mib`, `gpu_wait_timeout_sec` | 2500, 600 | GPU gate |
| `multi_goal`, `multi_goal_source` | `auto`, `auto` | legs from the recorded route / goal topic / dwells; bus-stop dwells become leg goals; `true` = `auto`, `false` = whole bag one leg |
| `multi_goal_stop_order`, `stop_points_csv` | —, `stop_points.csv` | the written order: only for `multi_goal_source: stop_points`. A cut recording's destination comes from the vehicle's recorded routes (its own, else the other recordings' in `rosbag_dir`) |
| `leg_cuts` | `auto` | also cut legs at every recorded stand ≥ 3 s ([method](EVALUATION_METHOD.md#legs)); `off` = bus-stop legs only |
| `thresholds`, `topics_yaml`, `vehicle_info_yaml` | package `config/`; vehicle info from `<vehicle_model>_description` | |
| `live_web_monitor` | on | live panel on the dashboard |

Legacy knobs (`stuck_*`, `resync_lag_sec`, `npc_reaction*`, `end_condition`,
`route_timeout_sec`, …) belong to the old drive loop and do nothing while
`window_eval` is on. `goal_backoff_steps_m`, `stop_fallback_max_dist_m`,
`route_total_timeout_sec` are accepted but no longer read.

## Commands

```bash
ros2 run dp_multi_eval run_all.py -c <yaml> --dry_run          # write manifest, launch nothing
ros2 run dp_multi_eval run_all.py -c <yaml>                    # run all pending jobs
ros2 run dp_multi_eval run_all.py -c <yaml> --only H95         # jobs whose id contains H95 (bag or model)
ros2 run dp_multi_eval run_all.py -c <yaml> --max_workers 1
```

`run_all.py` = merge manifest → run pending → one retry of failures → dashboard.
Exit 0 = no failures, 1 = failures, 130 = interrupted.

## Resume, rerun, archive

- The manifest (`results_root/manifest.json`) is the source of truth. `done` jobs with an `output/` bag are never re-run; jobs the yaml no longer mentions are kept.
- To redo a job, move its folder and video into `results_root/_archive/<date>_<reason>/` and run again (with `--only <bag>`):

```bash
mv $RES/$MODEL/${BAG} $RES/_archive/$(date +%F)_reason/
mv $RES/videos/${MODEL}__${BAG}_preview.mp4 $RES/_archive/$(date +%F)_reason/
ros2 run dp_multi_eval run_all.py -c <yaml> --only $BAG
```

- One canonical `results_root` per evaluation set; superseded results only under `_archive/`.

## Stopping

Ctrl-C in the `run_all.py` terminal, or `kill -INT|-TERM <pid>`. Simulators are
stopped, interrupted jobs go back to `pending`, rerun resumes.
Do not `pkill -9 -f …`: it leaks GPU memory and can kill your own shell. Leftovers:
`pgrep -af "planning_simulator|scene_driver|ros2 bag record"` → kill by PID.

## Rescore (no re-simulation)

```bash
ros2 run dp_multi_eval rescore.py -c <yaml>                       # all done jobs, in place, dashboard rebuilt
ros2 run dp_multi_eval rescore.py -c <yaml> --only H95 --dry-run  # writes metrics_rescored.json only
```

Prints `before -> after` per job. Use after a metric rule change; bit-identical
`metrics.json` is the regression test for metric refactors.

## Dashboard

Serve the results directory (`python3 -m http.server 8765 --directory <results_root>`);
under `file://` the page cannot auto-reload or show the live panel.

## One job by hand

```bash
ros2 run dp_multi_eval run_single_job.py --model_config <dir|yaml> --bag_path <bag> \
  --output_dir /tmp/dp_one --domain_id 10 --map_path <map> --window_eval
```

Metrics/video are pipeline post-processing steps; run `compute_metrics.py` /
`render_video.py` on the folder afterwards.
