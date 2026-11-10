# dp_multi_eval

Closed-loop evaluation of Diffusion Planner models on recorded Isuzu Erga
drives. Each model drives every recording in the planning simulator while the
recorded traffic, lights and indicators are replayed around it. Output: one
dashboard, models side by side, a verdict and a video per run.

## Quick start

```bash
source install/setup.bash
cp src/tools/planning/dp_multi_eval/config/example_pipeline.yaml ~/dp_multi_eval_pipeline.yaml
# edit rosbag_dir, results_root, map_path, models
ros2 run dp_multi_eval run_all.py -c ~/dp_multi_eval_pipeline.yaml --dry_run   # manifest only
ros2 run dp_multi_eval run_all.py -c ~/dp_multi_eval_pipeline.yaml             # run
python3 -m http.server 8765 --directory <results_root>                         # open /dashboard.html
```

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

Tests: `python3 tests/run_tests.py` (source the workspace to include the map/overlay
integration tests; they skip otherwise). Maintainer: see `package.xml`.
