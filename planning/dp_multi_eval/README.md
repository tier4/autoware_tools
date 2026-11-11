# dp_multi_eval

Closed-loop evaluation of Diffusion Planner models on recorded Isuzu Erga
drives. Each model drives every recording in the planning simulator while the
recorded traffic, lights and indicators are replayed around it. Output: one
dashboard, models side by side, a verdict and a video per run.

## Quick start

| Needed | |
|---|---|
| the vehicle's Autoware workspace, built | universe and launcher at its `autoware.repos` pins; this repository at `src/tools`, on this branch |
| recordings, the vehicle's map, model weights | not in this repository — what each must hold: [SETUP](docs/SETUP.md) |
| NVIDIA GPU ≥ 8 GB, ≥ 32 GB RAM | |

```bash
# from the workspace root, no virtualenv active
rosdep install --from-paths src/tools/planning/dp_multi_eval src/tools/planning/planning_debug_tools src/tools/planning/diffusion_planner_batch_eval --ignore-src -y
colcon build --packages-select dp_multi_eval planning_debug_tools diffusion_planner_batch_eval   # again after every pull
source install/setup.bash
cp src/tools/planning/dp_multi_eval/config/example_pipeline.yaml ~/dp_multi_eval_pipeline.yaml
# edit rosbag_dir, results_root, map_path, models
ros2 run dp_multi_eval run_all.py -c ~/dp_multi_eval_pipeline.yaml --dry_run   # checks the config, launches nothing
ros2 run dp_multi_eval run_all.py -c ~/dp_multi_eval_pipeline.yaml             # run
python3 -m http.server 8765 --directory <results_root>                         # http://localhost:8765/dashboard.html
```

A job takes about twice its recording's length (a 4-minute recording: ~8 minutes).
Re-running resumes. `--only <bag|model>` runs a subset. Ctrl-C stops cleanly.
`ros2 run dp_multi_eval rescore.py -c <yaml>` re-judges finished runs after a metric change.

## Docs

| | |
|---|---|
| [SETUP](docs/SETUP.md) | what the machine needs |
| [RUNNING](docs/RUNNING.md) | config keys, commands, reruns, stopping, rescore |
| [EVALUATION_METHOD](docs/EVALUATION_METHOD.md) | how a recording becomes a test |
| [METRICS](docs/METRICS.md) | what is measured, what fails a run |
| [RESULTS](docs/RESULTS.md) | result files and the dashboard |
| [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) | symptoms → fixes |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | modules and design decisions |
| [LEGACY](docs/LEGACY.md) | GUI, scenario simulator, old drive loop |

## Check without recordings

```bash
source install/setup.bash                                            # the tests need a ROS 2 environment
cd src/tools/planning/dp_multi_eval && python3 tests/run_tests.py    # 0 failed; the map test skips without the map
```

`colcon test --packages-select dp_multi_eval` runs the lint. Maintainers: see `package.xml`.
