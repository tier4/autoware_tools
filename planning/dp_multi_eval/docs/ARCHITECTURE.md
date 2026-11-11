# Architecture

```
pipeline.yaml ─► run_all.py ─► manifest.json ─► run_orchestrator (worker pool, domain pool, GPU gate)
                                                      │ per job, own ROS_DOMAIN_ID
                                                      ▼
   ament_overlay ─► planning_simulator ◄──► scene_driver ─► perception_reproducer (original, -r 0)
                          │                        ▲  scene_time / hold / objects  (harness topics)
                    routing, multi_goal ─► window_eval (legs: hand-over, reveal, divergence)
                          │
                    ros2 bag record ─► output/ ─► compute_metrics ─► metrics.json ─► build_dashboard
                                                 render_video    ─► videos/*.mp4  ─►   dashboard.html
```

## One job

1. Overlay: clone `autoware_launch` share to `/tmp/dp_multi_eval_overlays/<sha1>/`, swap the model param, fixer rules off, indicator topic, delayed vehicle model; prepend to `AMENT_PREFIX_PATH`.
2. GPU gate; kill leftovers on this domain.
3. Launch psim and the reproducer in parallel (bag loads while psim boots).
4. Wait for services; check the planner loaded weights.
5. Goals (`multi_goal.py`) and route (`routing.py`, candidate ladder) → `route_goals.json`.
6. Perception ready → object throttle → `ros2 bag record` → first trajectory (engine build) → engage.
7. `window_eval.run_legs`: per leg hold + jump + place + follow 3.5 s + hand-over, reveal at departure, divergence checks, arrival → `windows.json`.
8. Teardown in reverse; `job_status.json`. Orchestrator: `goal_pose.json`, metrics, video (separate pool), diagnosis.

## Modules

| Module | Role |
|---|---|
| `run_all.py`, `rescore.py`, `generate_manifest.py` | entry points; manifest merge; job ids; model/bag discovery |
| `run_orchestrator.py` | pool, free-domain pool, 15 s dashboard refresh, SIGINT/SIGTERM, post-processing |
| `job_config.py` | `JobConfig` (every knob + reason), `PIPELINE_KNOBS` |
| `run_single_job.py`, `job_waiters.py`, `process_utils.py` | job lifecycle, readiness waits, process groups / cleanup / GPU |
| `ament_overlay.py`, `model_config.py` | overlay rewrites; model → param yaml |
| `routing.py`, `route_control.py`, `multi_goal.py`, `dwell_profile.py` | goals and routes |
| `window_eval.py`, `lag.py` | leg harness; lag vs recording, wall→scene time |
| `compute_metrics.py`, `stop_go.py`, `geometry.py`, `bag_reader.py`, `bag_cache.py`, `diagnose.py` | scoring |
| `render_video.py`, `build_dashboard.py`, `job_status.py`, `live_monitor.py` | outputs |
| `resync.py`, `fixer_replay.py`, `app.py`, `scenario_sim.py` | legacy ([LEGACY](LEGACY.md)) |
| `scene_driver.py`, `scene_clock.py`, `reactive_shield.py`, `shield_map.py`, `prebirth.py`, `scene_audit.py`, `scene_bench.py` | scene player: runs the original `perception_reproducer` and feeds it the clock's instant; object gate with the reactive shield (map, lights, ego plan) objects the recording saw late, the audit of what is published, and the offline bench (`build_scene` builds the same pipeline for both); `shield_log.jsonl` |

Tests: `python3 tests/run_tests.py` runs every `tests/test_*.py` here
(plain scripts; PASS / SKIP / FAIL table). Unit tests
need no ROS; `test_ament_overlay` and `test_light_ahead_map` need the sourced
workspace (and the map) and skip otherwise; `test_sigterm_stop` runs the
orchestrator in a child process with a fake job.

## Decisions

- **Original reproducer, clock outside** — `planning_debug_tools` stays unmodified. With `-r 0` it shows the frame nearest the ego pose it reads; `scene_driver` remaps that input and feeds the recorded pose of the instant its clock picks.
- **Overlay, not remaps** — psim reads the planner param by share path; parallel models need parallel share trees. `src/autoware` untouched.
- **The recording is the timeline** — legs, hand-over, reveal, 1× clock ([method](EVALUATION_METHOD.md)).
- **The vehicle and the line from their own data** — the ego's body from `<vehicle_model>_description`'s vehicle_info; the stop a cut recording was heading for from the routes the vehicle recorded (its own, else the other recordings' sequence). A written stop order had two stops out of place.
- **Free-domain pool** — a domain returns only when its job ends; round-robin once started a job on a domain still in use and its cleanup killed the running simulator.
- **GPU gate** — two engines built at once → OOM → a planner that stays up silent and reads as a stall.
- **One bag read, video on its own pool** — rendering on the simulator worker cost 264 s per job.
- **Reference ego cache** — recordings are 37–79 GB.
- **Objects recorded at 10 Hz** — the raw stream was 88 % of the output bag.
- **Manifest merge keeps unmentioned jobs** — a subset yaml once dropped 30 finished jobs.
- **All metrics offline** — rules change without re-driving; refactors verified by bit-identical `metrics.json`.
