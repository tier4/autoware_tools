# Results

## Files

```
<results_root>/
  manifest.json          job matrix + status (resume source of truth)
  dashboard.html         the report            dashboard_live.json  reload signal
  live_drive.json        running job's ego      videos/<job id>_preview.mp4
  _archive/              superseded results (by hand)
  <model>/<bag>/
    job_status.json      status, phase, timings, interventions, diagnosis, build (versions, repo commits)
    goal_pose.json       goal the metrics score against
    route_goals.json     goals routing accepted
    windows.json         legs as driven: segments, departures, jumps, ahead
    metrics.json         all metric blocks + run_status/clean_run/diagnosis
    output/              recorded rosbag2 (absent = no real output)
    psim_launch.log  reproducer.log  record.log  objects_throttle.log
```

Job id: `<model>__<bag>`. Caches:
`~/.cache/dp_multi_eval/`.

The output bag holds the ego, the scene (10 Hz), the plan, both sides of the
trajectory modifier, planning factors, signals, indicators, control commands —
enough to recompute every metric (`rescore.py`).

## Video

Top: map, ego (red), recorded ego at the same scene time (dashed purple), trail,
planned trajectory (orange), NPCs (green), virtual walls (`obst_stop` magenta =
stack, `diff_plan` yellow = model, `stop_fix` cyan), blinkers, goal. Red frame
= failed run. Bottom: ego speed vs recorded speed.
Speed-up = `video_fps × video_sample_dt` (defaults 1.6×; example config 3×):
sim time `t` is at `t / (fps × dt)` s.

## Dashboard

- **Models side by side** — one column per model: success, legs, contacts, bus stops, departure delay, late departures, stops, brakes, jerk, proximity, near misses, pace, stub share. Hover a row for its definition.
- **Model comparison** (≥ 2 models) — bags × models grid.
- **Runs** — per bag, one card per model. Pills: `PASS`, `2/3 legs passed` (orange = one short), `FAIL`, `(assisted)`, `RUNNING/QUEUED/SCORING`, `INCOMPLETE/NO RUN`. Inside: diagnosis, metric chips, video, raw numbers.
  Hover **legs** for per-leg detail (`stops 3.7s at t=30s`, `waited at red …`, `did not depart …`, re-inits); hover **proximity** for near misses.
- **Live drive** — shown while a job drives (`leg i/N`, last leg `final`).
- **Goal stop precision** — stop pose error per goal.

Serve over HTTP for auto-reload and the live panel ([RUNNING](RUNNING.md#dashboard)).

## Reading a failed leg

Legs tooltip → the reason and time → video at `t/(fps×dt)` → the bag
(`planning_factors` says who asked for the stop; `traffic_signals` what the
light showed; `objects_throttled` what the model saw) → `windows.json` for the
harness's view.
