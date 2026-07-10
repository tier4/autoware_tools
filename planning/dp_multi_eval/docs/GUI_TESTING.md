# GUI testing guide (`app.py`)

**Ask:** please run through this checklist on a machine with Autoware + GPU + Hiratsuka (or similar) scenario bags, and report pass/fail per test case.

**Code under test:** `dp_multi_eval/app.py` (Streamlit)  
**Related:** orchestrator stop/cleanup in `process_utils.py`; embedded `dashboard.html` from `build_dashboard.py`

---

## Before you start

```bash
# Build
cd <workspace>
colcon build --packages-select dp_multi_eval diffusion_planner_batch_eval --allow-overriding dp_multi_eval
source install/setup.bash

# Config (edit paths to your machine)
cp src/tools/planning/dp_multi_eval/config/example_pipeline.yaml ~/dp_multi_eval_pipeline.yaml

# GUI dependency
pip install streamlit

# Clean slate — no leftover eval processes
pkill -f "dp_multi_eval.run_all" 2>/dev/null || true
pkill -f "planning_simulator.launch" 2>/dev/null || true
rm -f /path/to/your/results/.gui_run_state.json
```

**Recommended test matrix (small, ~15–30 min):**

| Setting | Value |
|---------|-------|
| Models | 1 (Hiratsuka-matched ONNX) |
| Scenarios | 2 (e.g. ID1 + ID2) |
| Parallel workers | **1** |
| `map_path` in pipeline yaml | Must match bag geography |

**Launch:**

```bash
ros2 run dp_multi_eval app.py
```

Browser opens at `http://localhost:8501` (default).

---

## Test cases

Copy this table into your PR comment or issue and fill in **Pass / Fail / Notes**.

| ID | Test | Steps | Expected | Pass? | Notes |
|----|------|-------|----------|-------|-------|
| G1 | App launches | Run `ros2 run dp_multi_eval app.py` | Page loads; title “Planner evaluation”; 3 tabs visible | | |
| G2 | Sidebar config | Set **Pipeline config file** to `~/dp_multi_eval_pipeline.yaml` | Models / scenarios / results paths populate from yaml | | |
| G3 | Folder browse | Click `…` next to each sidebar folder field | Native folder picker opens; path updates after selection | | |
| G4 | Model discovery | Set **Models folder** to `/opt/autoware/mlmodels` (or your path) | Setup tab lists ONNX dirs / `*.param.yaml` as checkboxes | | |
| G5 | Scenario discovery | Set **Scenarios folder** to hiratsuka parent | ID* folders appear; **Select all scenarios** checks all | | |
| G6 | Start validation | Click **Start evaluation** with nothing selected | Error: select at least one model / scenario | | |
| G7 | Start run | 1 model, 2 scenarios, workers=1, unique run name → **Start evaluation** | Success message; “run already in progress” on Setup; `.gui_run_state.json` created | | |
| G8 | Double-start blocked | While run active, try **Start evaluation** again | Cannot start second run; stop controls shown instead | | |
| G9 | Live progress | **Live Progress** tab → **Refresh now** | Progress bar moves; Completed / Running / Failed metrics update; manifest path shown | | |
| G10 | Running detail | During sim, check **Currently running** table | Shows scenario, phase (`psim_startup`, `simulation`, etc.), total_elapsed, phase_elapsed, domain | | |
| G11 | Failed table | If any scenario fails, check **Failed scenarios** | scenario, model, error (truncated) columns populated | | |
| G12 | Results embed | **Results** tab while run in progress | Latest `dashboard.html` embedded; auto-refresh note in caption; cells update after browser refresh of HTML | | |
| G13 | Stop — disabled | **Live Progress** → **Stop evaluation** without checkbox | Button disabled until confirm checked | | |
| G14 | Stop — mid-run | Confirm checkbox → **Stop evaluation** | Success/info message; run stops within ~30s; no new scenarios start | | |
| G15 | Stop cleanup | After G14: `pgrep -af planning_simulator`; `ROS_DOMAIN_ID=10 ros2 node list \| wc -l` | No planning_simulator; node count low / no duplicate node names | | |
| G16 | State file cleared | After stop: `ls {results_root}/.gui_run_state.json` | File absent; Setup allows **Start evaluation** again | | |
| G17 | Cancelled jobs | Open `{run_name}/manifest.json` after mid-run stop | Previously `running` jobs → `failed`, error `cancelled_by_user` | | |
| G18 | Dashboard after stop | Open `{run_name}/dashboard.html` | Running cells gone or show failed; no infinite “RUNNING” | | |
| G19 | Stop from Setup tab | Start 2-scenario run; stop from **Setup & Run** tab (not Live Progress) | Same cleanup as G14–G18 | | |
| G20 | Stop from Results tab | Start run; stop from **Results** tab | Same cleanup as G14–G18 | | |
| G21 | Full completion | Run 2 scenarios to completion (no manual stop) | Live Progress: 2/2 completed; Results shows PASS/FAIL cells; `preview.mp4` exists per done job | | |
| G22 | Log file | During/after run: `tail {results_root}/{run_name}/run_all.log` | Orchestrator output present; no silent hang | | |

---

## Verification commands (after stop or run)

```bash
RESULTS=/path/to/dp_multi_eval_results
RUN=run_YYYYMMDD_HHMMSS   # your run name

# GUI state
cat "$RESULTS/.gui_run_state.json" 2>/dev/null || echo "OK: no lock file"

# Orchestrator still running?
pgrep -af "dp_multi_eval.run_all"

# Orphan simulators?
pgrep -af "planning_simulator.launch"

# ROS domain pollution (should be minimal after clean stop)
source install/setup.bash
ROS_DOMAIN_ID=10 ros2 node list 2>/dev/null | sort | uniq -d

# Per-run artifacts
ls "$RESULTS/$RUN/manifest.json" "$RESULTS/$RUN/dashboard.html"
ls "$RESULTS/$RUN"/*/ID*/job_status.json 2>/dev/null | head
```

---

## What to report back

1. Filled test table (G1–G22) with Pass/Fail and notes  
2. Machine info: GPU model, `max_workers` used, scenario count  
3. On failure: screenshot of Streamlit tab + attach:
   - `{results_root}/.gui_run_state.json` (if present)
   - `{results_root}/{run_name}/run_all.log` (last ~100 lines)
   - Output of verification commands above  

**File bugs against:** `app.py` for UI/start/stop/state; `run_single_job.py` / orchestrator if sim fails after GUI correctly started the run.

---

## Known GUI limitations (not bugs unless behavior differs)

- **Live Progress** does not auto-refresh in all Streamlit versions — user must click **Refresh now**
- **Results** embedded HTML auto-refreshes every 5s internally, but the Streamlit frame does not reload until you revisit the tab or refresh Streamlit
- Stop button is only in Streamlit, not inside the embedded `dashboard.html` iframe
- Folder picker (`…`) requires a display / Zenity (may not work over pure SSH without X forwarding)

---

## Optional stretch tests

| Test | Steps | Expected |
|------|-------|----------|
| Resume after partial run | Stop mid-run; CLI: `ros2 run dp_multi_eval run_all.py -c .../pipeline_config.yaml --max_workers 1` | Skips `done` jobs; GUI state independent of CLI resume |
| Many scenarios | 10+ scenarios, workers=1 | UI stays responsive; progress bar advances |
| workers=2 | Two workers on single GPU | Document instability; note if `planning_stack_not_ready` spikes |

Maintainer: Takahiko Hasegawa
