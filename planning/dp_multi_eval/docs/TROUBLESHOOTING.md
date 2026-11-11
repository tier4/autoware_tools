# Troubleshooting

Start with the job's `diagnosis` (dashboard card / `job_status.json`), then the logs in the job folder.

## Before driving

| Symptom | Cause → fix |
|---|---|
| `planning simulator exited early` | launch tree lacks `autoware_trajectory_modifier` → source the universe checkout that has it |
| `planning_stack_not_ready` | leftover psim on the same domain, or slow machine → `pgrep -af planning_simulator`, kill by PID; raise `psim_startup_sec` |
| `Unsupported major_version: 5` | v5 checkpoint → use a v4 export |
| `Model not loaded` / `Failed to setup TRT engine` | bad model paths or no GPU memory → check `~/.cache/dp_multi_eval/model_params/<name>.param.yaml`, free the GPU |
| `planner would not fit on the GPU` / `Cuda Runtime (out of memory)` | GPU busy or leaked → `max_workers: 1`; reboot if `nvidia-smi` shows ownerless memory |
| `planning simulator not ready … exited early (code=1): package 'X' not found` | X is not built in this workspace or the checkout lacks it: `colcon build --symlink-install --packages-select X` (the GUI need not restart) |
| `planned route is empty` / `Goal's footprint exceeds lane!` | wrong map version → use the vehicle's map (v206) |
| perception-ready timeout, `reproducer.log` still growing | huge bag loading → wait (budget extends); reindex bags without `metadata.yaml` |

## While driving

| Symptom | Reading |
|---|---|
| leg `stale` | psim died → `psim_launch.log` tail, RAM |
| leg `timeout` | never reached the arrival pose → video; `stall` / `did_not_depart` |
| many `not_in_control` contacts | the stack held the ego with stop stubs → `stack_intervention` (should be ~0 with the fixer rules off) |

## Verdicts

| Symptom | Reading |
|---|---|
| short stop at a junction fails a leg, video looks fine | red light the recording did not meet → excused since 2026-09-16; `rescore.py` older results |
| same weights pass/fail across runs | run-to-run variance → judge on ≥ 2 runs |
| a car stops behind or in front of the standing ego (video) | the reactive shield brakes it to stop 3 m clear of the ego (1 m for VRUs) along its own path; it moves on when the ego clears its way |
| a car waits at a red light it drove through in the recording | it is more than 2 s behind its recording (the shield held it) and meets the next red; it goes on the green |
| a leg is `inconclusive` | a stuck stretch was the scene's (`scene_limited` in the leg): a held NPC stood on the ego's way, or the scene had run past the recording's end |
| an NPC appears from nothing, stands mid-road or vanishes near the ego (video) | look at `scene_audit` in metrics.json: it lists each such event with its cause (`why`/`src`); a `blip` is the recording's own noise |
| a pedestrian or car swerves round the standing ego (video) | deadlock detour: it waited ≥ 3 s on the ego's way while the ego stood (`held` reason `deadlock` if no detour fitted) |
| a shield change looks better on one bag | run `scene_bench.py --all --baseline <old.json>`; a change goes in only without regressions |
| NPC teleports into the ego | `replay_jump`, not gating |
| a yaml knob does nothing | not in `PIPELINE_KNOBS` or misspelt |
| passing run shows `(assisted)` | a re-init/nudge happened → `segments` in the leg |
| card says QUEUED but the job runs | flips to RUNNING when `job_status.json` gets a phase (≤ 15 s) |
| dashboard never reloads | opened via `file://` → serve with `http.server` |

## Machine

| Symptom | Fix |
|---|---|
| GPU memory used by no process | a SIGKILLed CUDA process → reboot; stop runs with Ctrl-C |
| `/tmp` full | delete `/tmp/dp_multi_eval_overlays/` between campaigns |
| shell killed (`exit 144`) | a `pkill -f` pattern matched your shell → kill by PID |
| code edits have no effect | stale install → `--symlink-install`, or `PYTHONPATH=<pkg dir>` |
| every job fails with `No module named 'dp_multi_eval.…'` | the checkout changed under the install (a pull, a branch switch): rebuild the three packages ([setup](SETUP.md)); never switch branches while a run is going |
