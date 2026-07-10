# Operations guide

Shared reference for `dp_multi_eval`. For the main workflows, see:

- **Headless CLI batch:** [HEADLESS.md](HEADLESS.md)
- **Streamlit GUI:** [GUI_TESTING.md](GUI_TESTING.md)

---

## Prerequisites

- Built Autoware workspace with `autoware_launch`, `diffusion_planner_batch_eval`, planning simulator
- GPU with enough memory for TensorRT diffusion planner (test with `max_workers: 1` first)
- Scenario bags: folders with `metadata.yaml` or `.db3` files (e.g. `hiratsuka/ID1`, …)
- Map at `map_path` must match scenario coordinates (Hiratsuka bags + Hiratsuka map, etc.)
- Optional: `pip install streamlit` for GUI

---

## Build

```bash
cd <workspace>
colcon build --packages-select dp_multi_eval diffusion_planner_batch_eval --allow-overriding dp_multi_eval
source install/setup.bash
```

---

## Configuration

```bash
cp src/tools/planning/dp_multi_eval/config/example_pipeline.yaml ~/dp_multi_eval_pipeline.yaml
```

### Important fields

| Key | Typical value | Notes |
|-----|---------------|-------|
| `rosbag_dir` | path to scenario parent folder | Subdirs `ID1`, `ID2`, … |
| `results_root` | writable output directory | Prefer local disk or fast external SSD |
| `map_path` | `/opt/autoware/maps` | Must match bag geography |
| `vehicle_model` / `sensor_model` / `vehicle_id` | e.g. `lv828l`, `aip_x2_gen2`, `6_lv828l` | Passed to psim launch |
| `max_workers` | **1** (recommended) | Each worker needs GPU + CPU + unique `ROS_DOMAIN_ID` |
| `domain_id_start` | `10` | Workers use `10`, `11`, … |
| `route_timeout_sec` | `300` | Fail driving phase if route never `ARRIVED` |
| `psim_startup_sec` | `120` | Wait for planning stack services/nodes |
| `render_video` | `true` / `false` | `false` speeds batch; render failures on demand |
| `video_sample_dt` | `0.5` | Lower = smoother video, slower post-processing |
| `models` | list of `{name, model_config}` | `model_config` = `.param.yaml` or ONNX directory |

GUI runs write a per-run copy to `{results_root}/{run_name}/pipeline_config.yaml` and may add `bag_whitelist` for selected scenarios only.

---

## Running

### Streamlit GUI

```bash
pip install streamlit   # once
ros2 run dp_multi_eval app.py
```

1. **Setup & Run** — pick folders, select models/bags, **Start evaluation**
2. **Live Progress** — counts, running phases, failures
3. **Results** — embedded `dashboard.html`

Sidebar **Pipeline config file** defaults to `~/dp_multi_eval_pipeline.yaml`.

### CLI

```bash
# Dry-run: validate manifest, no psim
ros2 run dp_multi_eval run_all.py -c ~/dp_multi_eval_pipeline.yaml --dry_run

# Full run
ros2 run dp_multi_eval run_all.py -c ~/dp_multi_eval_pipeline.yaml --max_workers 1

# Resume a GUI-started run (skips done jobs)
ros2 run dp_multi_eval run_all.py \
  -c /path/to/results/run_YYYYMMDD_HHMMSS/pipeline_config.yaml \
  --max_workers 1
```

### Single job (debug)

```bash
ros2 run dp_multi_eval run_single_job.py \
  --model_config /opt/autoware/mlmodels/diffusion_planner_for_erga_hiratsuka_june_26 \
  --bag_path /path/to/hiratsuka/ID1 \
  --output_dir /tmp/dp_test_ID1 \
  --domain_id 10 \
  --map_path /opt/autoware/maps
```

---

## Stopping a run

### GUI

**Live Progress**, **Setup & Run**, or **Results** (while active):

1. Check **Confirm stop — kills orchestrator and any running simulators**
2. Click **Stop evaluation**

Stops `run_all`, cleans leftover psim/reproducer/bag-record processes, marks running manifest jobs `cancelled_by_user`, rebuilds dashboard, clears `.gui_run_state.json`.

### Terminal

```bash
RESULTS=/path/to/dp_multi_eval_results
PID=$(python3 -c "import json; print(json.load(open('$RESULTS/.gui_run_state.json'))['pid'])")
kill -TERM -$PID
sleep 3
kill -KILL -$PID 2>/dev/null
pkill -f "planning_simulator.launch"
pkill -f "perception_reproducer.py"
pkill -f "ros2 bag record"
rm -f "$RESULTS/.gui_run_state.json"
```

Stale Autoware processes on `ROS_DOMAIN_ID=10` cause `planning_stack_not_ready` for every scenario. Always stop cleanly or run cleanup before a new batch.

---

## After a run finishes

1. Open `{results_root}/{run_name}/dashboard.html` (or GUI **Results** tab)
2. Review pass rate, collisions, OOB, stuck, goal error
3. Click cells for detail; open `preview.mp4` for failures
4. Retry failed only: `run_all.py -c .../pipeline_config.yaml` (done jobs skipped; one built-in retry pass)
5. Archive: zip run folder (`manifest.json`, `dashboard.html`, `metrics.json`, `preview.mp4`)

### Failure categories

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `planning_stack_not_ready` | Polluted `ROS_DOMAIN_ID`, slow psim start | Clean processes; increase `psim_startup_sec` |
| `route setup failed: planned route is empty` | Map/bag/model mismatch | Align model, map, scenario set |
| `route_not_arrived` | Planner stuck / timeout | Watch `preview.mp4`; tune model or `route_timeout_sec` |
| Suspicious PASS | Metric edge case | Re-read `metrics.json` and video |

---

## Metrics and pass criteria

Configured in `config/thresholds.yaml`:

| Metric | Default threshold | Pass impact |
|--------|-------------------|-------------|
| Stuck | ≤ 0.05 m/s for ≥ 45 s (away from goal) | **Fail** |
| Collision | min distance < 0.1 m to vehicle-class NPCs | **Fail** |
| Out of boundary | any corner outside Lanelet2 drivable area | **Fail** (if map loaded) |
| Goal stop precision | avg position error when speed ≤ 0.2 m/s within 2 m of goal | **Fail** if flagged |

Collision skips `UNKNOWN` / `PEDESTRIAN` and zero-size objects. Goal precision averages valid low-speed samples near goal; stuck scenes use `excluded_stuck`.

Recompute without resim:

```bash
ros2 run dp_multi_eval compute_metrics.py \
  --bag_path /path/to/job/output \
  --map_path /opt/autoware/maps \
  --goal_pose X Y YAW \
  --thresholds $(ros2 pkg prefix dp_multi_eval)/share/dp_multi_eval/config/thresholds.yaml \
  --output /path/to/job/metrics.json
```

---

## Parallelism

| `max_workers` | Guidance |
|---------------|----------|
| 1 | **Default** — stable, no domain contention |
| 2 | Only with confirmed GPU/CPU headroom |
| 3+ | Not recommended on one GPU |

Each worker uses a distinct `ROS_DOMAIN_ID`. Shared-GPU parallel runs often cause jitter, false OOB, and route timeouts.

---

## Troubleshooting

```bash
pgrep -af "dp_multi_eval|planning_simulator|perception_reproducer"
ROS_DOMAIN_ID=10 ros2 node list | sort | uniq -d   # duplicates = polluted domain
tail -f /path/to/results/run_*/run_all.log
less /path/to/results/run_*/{model}/{ID}/psim_launch.log
```
