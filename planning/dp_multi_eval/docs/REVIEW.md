# Review guide

For engineers reviewing `dp_multi_eval` before merge or before trusting batch results.

**Entry point:** [README.md](../README.md)  
**Headless pipeline:** [HEADLESS.md](HEADLESS.md) ← **main feature doc / engineer testing**  
**GUI testing:** [GUI_TESTING.md](GUI_TESTING.md)  
**Operations:** [OPERATIONS.md](OPERATIONS.md)  
**Design:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Scope of this change

New package under `src/tools/planning/dp_multi_eval/` that:

- Orchestrates M×N diffusion-planner eval jobs (models × scenario bags)
- Runs planning simulator headlessly with per-job ONNX overlay
- Records rosbags, computes offline metrics, renders preview video
- Exposes HTML dashboard + Streamlit GUI with start/stop controls

Reuses `diffusion_planner_batch_eval` for route setup — does not replace it for interactive work.

---

## Review checklist

### Correctness

- [ ] Per-job AMENT overlay does not leak ONNX paths across workers or consecutive runs
- [ ] Route setup → perception reproducer → engage ordering matches `batch_eval` expectations
- [ ] Manifest resume skips only `done` jobs that have a real `output/` bag (not dry-run placeholders)
- [ ] Metrics `pass` logic matches planning QA intent (stuck, collision, OOB, goal-stop)
- [ ] Collision / OOB geometry handles degenerate polygons and zero-size objects
- [ ] Goal-stop precision averages multiple samples; stuck scenes excluded (`excluded_stuck`)

### Operations / safety

- [ ] Stop evaluation kills orchestrator + orphan sim processes without leaving polluted `ROS_DOMAIN_ID`
- [ ] `.gui_run_state.json` prevents double-start; cleared on stop and after process exit
- [ ] `max_workers: 1` documented as default; parallel risks called out
- [ ] Map / bag / model alignment documented (empty route = mismatch symptom)

### Code quality

- [ ] No secrets or machine-specific paths committed in `config/example_pipeline.yaml`
- [ ] Log files use file handles not PIPE (avoids psim deadlock)
- [ ] Orchestrator refreshes dashboard during run; cancelled jobs update manifest

### Tests / validation

- [ ] **Headless:** [HEADLESS.md](HEADLESS.md) checklist H1–H6
- [ ] **GUI:** [GUI_TESTING.md](GUI_TESTING.md) checklist G1–G22 (especially G14–G18 stop/cleanup)
- [ ] Dry-run path exercised: `run_all.py --dry_run`
- [ ] At least one single-scenario happy path on Hiratsuka (ID1 or similar)
- [ ] Stop button tested mid-run (no orphan `planning_simulator` on domain 10)

---

## Files to read first

| Priority | File | Why |
|----------|------|-----|
| 1 | `run_single_job.py` | Core sim lifecycle, readiness waits, route setup |
| 2 | `ament_overlay.py` | Parallel ONNX swap — highest regression risk |
| 3 | `compute_metrics.py` | Pass/fail semantics |
| 4 | `run_orchestrator.py` | Resume, post-processing, worker pool |
| 5 | `app.py` | GUI start/stop, state file |
| 6 | `process_utils.py` | Stop/cleanup helpers |

---

## Known limitations (accept or file issues)

- Single shared GPU + `max_workers > 1` is unstable in practice
- HTML `dashboard.html` is read-only; stop control is in Streamlit only
- Route-setup failures for map/bag mismatch are not auto-retried with different maps
- Early integration: thresholds in `config/thresholds.yaml` may need alignment with fleet QA

---

## Bug reports

Attach:

- `pipeline_config.yaml` for the run
- `run_all.log`
- `manifest.json` entries for failed `job_id`
- `job_status.json` + tail of `psim_launch.log` for one failing scenario
- `metrics.json` + `preview.mp4` if the job completed driving but failed metrics

Maintainer: Takahiko Hasegawa (`package.xml`).
