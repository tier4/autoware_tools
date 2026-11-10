# Metrics

`metrics.json` per job (`compute_metrics.py`, `stop_go.py`). **Gating** blocks
decide `pass`; everything else is reported. Thresholds: `config/thresholds.yaml`,
copied into `meta.thresholds`.

## Verdict

A run passes when every leg passed (`windows`), no contact frame is
`ego_caused` (`collision_rate`), the ego drove over no road border
(`out_of_boundary`, [below](#road-borders)), every recorded bus stop was served
(`bus_stop_service`), and harsh brakes ≤ recording + 1 (`comfort`).
Runs without `windows.json` (legacy) are judged by the whole-bag blocks instead.
`clean_run` = no harness intervention; a passing run with one shows as **assisted**.

## Legs (`windows`) — gating

A leg fails on any of:

| Rule | Fails when |
|---|---|
| not judged | `ended_by` ∉ {`arrived`, `window_end`} (timeout, stale, abandoned, route/engage failed) |
| stalled | the harness re-initialised a standing ego (runs before 2026-10-28; now noted as `stall`, the stuck rule judges) |
| stuck | standing (≤ `stuck_speed_mps` 0.05) ≥ `stuck_duration_sec` 45 s more than 12 m from the leg goal, not at a red light, not where the recording waited too, and ≥ 45 s of it unjustified (`stuck`; else `stuck_justified`) |
| mutual_wait | standing ≥ 10 s with no reason (waits) while an NPC the shield held for the ego (`ego`, `plan`) stands ahead of it (no further back than 1 m behind its front) within 15 m: the two waited for each other; a real one waits for a bus in its way too (`mutual_waits`). Not at the leg's goal (the stop dwell), nor where the recording waited too |
| did_not_complete | a window closed short of the leg's end — the recording's end or a `slow` re-init — with the ego standing without a reason (< 0.5 m/s, last 2 s unjustified) or ≥ 15 s behind the recording once its justified waits are taken off (`goal_missed`, `incomplete.unexplained_lag_sec`); fails the leg. A stand the recording's end cut short is judged on its live part (a stop begun after the end: on what stood still) and fails only after ≥ 5 s without a reason, else the leg is `inconclusive` (`recording_ran_out`) |
| scene_limited | a stuck stretch or long stop for most of which the ego faced an NPC the shield could not lead round it on its recorded way (`deadlock`), or the scene stood past the recording's end (`shield_log.jsonl`); reported, not gating — the leg is `inconclusive`. An NPC held for the ego (`ego`, `plan`) waits for it as a real one would: no limit, and no justified wait for the ego (a mutual wait fails) |
| scene_audit | the scene player's own defects near the ego, from the published frames (`scene_audit.py`): `stand_in_zone` (an NPC standing in a junction or on a crosswalk longer than its recording did, with the rule that held it), `pop_in` (appeared within 15 m; `blip` = a recorded track under 1 s), `vanish` (left while waiting beside the ego), `jump`; reported, not gating |
| windows (per leg) | each segment between re-inits: start, length, how it closed; closing by a re-init is no verdict |
| planned cut | a window closed at a cut (`closes: cut`) has no arrival or goal stop (`not_required_cut`); standing at the cut is no failure, unexplained lag ≥ 15 s is (`did_not_complete`). `placed` / `seamless` / `scene_jumped` say how the next opened; `cut_placements` counts placements. `cut_pass`: the ego drove on past the stand — seconds of the recorded wait skipped and the light it drove under (`red` / `not_red`); reported, not gating (`cut_red_passes`). A contact beginning within 4 s of that scene jump with an object whose body was within 15 m of the ego's then is the jump's (collision `scene_jump`, not gating); the leg is not judged from there on and is `inconclusive` (`scene_jump_contact`) |
| goal missed | the ego came within 5 m of the leg goal without ever arriving (drove past), or an intermediate leg's late grace (30 s) ran out (`goal_missed`). Late at a final leg's end (10 s grace) is reported, not failing |
| goal stop | at a bus stop (a map stop within 12 m of the leg goal), the ego came to rest > 2 m off it (position, lateral or longitudinal; `goal_tolerance_m`). Where the recording only waited in traffic the error is reported, not failing (`goal_stop.gating`) |

Justified waits (`waits.py`): every stop ≥ 1 s is judged per 0.5 s; stops ≥ 3 s are listed in `waits`.

| Reason | When |
|---|---|
| `red_light` | the light ahead is red or amber (not once the ego's front is 2 m past its stop line) |
| `vru` | a pedestrian or cyclist within 1 m of the ego's swept way, up to 25 m ahead of its front, or entering it within 3 s |
| `crosswalk` | a pedestrian or cyclist on, or within 2 m of, a map crosswalk the way crosses ahead |
| `vehicle` | a vehicle ≥ 0.5 m inside the swept way, up to 15 m ahead (a queue) |
| `crossing` | a moving vehicle (≥ 1 m/s, ≥ 30° off the way) entering it within 3 s, up to 30 m ahead |

The way is the recorded path (the ego's route), so a vehicle the recording drove round is no reason; nor is an object the shield held for the ego (it was waiting for the ego). `credit_sec` = the lag the recording gained during the stop × its justified share; it comes off the lag in `did_not_complete`.

Reported only: waits the ego drove on from — stops ≥ 3 s the recording did not make
(`long_stops`), departures missed by ≥ 15 s (`did_not_depart`), start-brake cycles —
plus arrival delay (arriving late is not a failure), re-inits, `ahead`, `stall`, stops
< 3 s (`hiccups`). Waits stopped gating on 2026-10-28 (H82, H39 gave way to oncoming
cars and completed the leg): stopping is not a failure, being stuck is.

**Stops.** Speed < 0.1 m/s for ≥ 0.5 s; a < 1 s blip does not split a stop. Stops
within 12 m of a leg goal, the spawn, or a place the recording stood ≥ 3 s are
planned (one per zone). Further excuses:

- **waited like the recording** — the recording stood within 12 m, overlapping in scene time.
- <a id="red-light-rules"></a>**red light** (`red_light_waits`, `red_light_brakes`) — the light of the ego's lane (lanelet under the ego or 10 m ahead, same heading) was red/amber for ≥ 50 % of the stop, or during the braking phase of a cycle. Correct driving regardless of what the recording met (H95: the ego arrived 9 s early, waited 3.7 s at a red the recording passed on green).

Per-leg fields: `departure_delay_sec`, `arrival_delay_sec`, `drive` (`ego_sec/rec_sec/ratio`,
summed into `windows.drive_ratio` — the pace figure), `goal_stop`, `stall`, `ahead`,
`segments`, `stop_go.{long_stops,hiccups,red_light_waits,red_light_brakes}`.

## <a id="road-borders"></a>Road borders (`out_of_boundary`) — gating on the ego's crossings

Ego footprint over a `road_border`/`curbstone`, every 0.2 s, grouped into `episodes`.
An episode is the harness's (`placement_episodes`, reported) when it starts before the
first hand-over, between a leg's start and 1 s after its first hand-over, within
`lead_sec` + 0.5 s before to 1 s after any hand-over, or within 1 s of a ≥ 2 m pose jump — the ego placed on the recorded bus in its bay can touch the
border. An episode where the recording itself crossed within 3 m (`recorded_too_episodes`,
reported; `recorded_rate` = the recording's own share) is the map's: with the real body the
recorded bus crosses the mapped curb on up to 19 % of its frames. Any other
(`ego_caused_episodes`) fails the run.

The ego is measured with the vehicle's real body (`vehicle_info_yaml`, else
`<vehicle_model>_description`'s): lv828l 10.88 × 2.87 m.

## Contacts (`collision_rate`) — gating on `ego_caused`

Ego footprint within 0.1 m of a vehicle-class object or a pedestrian (not UNKNOWN),
sampled every 0.2 s. Attribution, in order:

| Class | Meaning | Gates |
|---|---|---|
| `replay_jump` | object unseen > 2 s, moved > 20 m/s between samples (teleport), or first seen ≤ 2 s before within 15 m of the ego (born at the ego: objects appear where the recorded bus first saw them) | no |
| `not_in_control` | ego ≤ 0.25 m/s while the stack fed stop stubs | no |
| `struck_while_stopped` | ego ≤ 0.25 m/s otherwise | no |
| `ego_caused` | ego moving, object standing (< 0.5 m/s) — any side, on or off the recorded path | **yes** |
| `out_of_envelope` | ego > 5 m from any recorded pose | no |
| `through_ego` | a withheld object, ego ≥ 2 s behind (runs before 2026-10-28; objects are no longer withheld) | no |
| `ego_caused` | ego moving, object within a 60° cone ahead | **yes** |
| `cut_in` | object beside/behind a moving ego | no |

`proximity` (reported): closest approach while moving, seconds < 1 m, near misses < 0.5 m.

## Comfort — gating (harsh brakes)

Harsh brake: decel ≤ −2 m/s² for ≥ 0.3 s. Harsh jerk: |j| ≥ 2.5 m/s³ for ≥ 0.2 s
(reported), plus `max_decel_mps2`, `jerk_p95_mps3`. Hand-overs excluded; the
recording scored the same way. Fails at run harsh brakes > recording + 1.

## Bus stops (`bus_stop_service`) — gating

For each `stop_points.csv` stop the recording stood at (≥ 3 s within 12 m, same
heading): `served` if the ego stopped with |lat| ≤ 1.5 m, |lon| ≤ 5 m; else
`in_lane` / `passed`. Fails when any is not served.

## Reported

| Block | |
|---|---|
| `stuck_rate`, `stop_go`, `goal_stop_precision`, `resync` | whole-bag figures; gating only without `windows.json` |
| `reference_speed` | ego vs recorded speed at the same place; `time_ratio` |
| `replay_validity` | share of samples within 5 m of the recorded path |
| `stack_intervention` | share of cycles the controller got stop stubs, longest stretch, `green_latched_sec` |
| `objects_through_ego` | withheld objects (runs before 2026-10-28) |
| `run_status`, `interventions`, `clean_run`, `diagnosis` | how the run ended; harness assistance; one-paragraph reading |

## Thresholds

`thresholds.yaml`: `stuck_speed_mps` 0.05, `stuck_duration_sec` 45, `goal_*_tolerance_m` 2,
`collision_distance_m` 0.1, `oob_boundary_types` road_border+curbstone, `sample_dt` 0.2.
In code: cone 60°, envelope 5 m / 50 %, harsh decel 2.0, jerk 2.5, near miss 0.5 m,
`STOP_GATE_SEC` 3, `WAITED_LIKE_M` 12, `RED_WAIT_MIN_FRAC` 0.5, `DEPART_LAG_SEC` 15, `JUMP_MPS` 20.

## Reading a model

Compare on ≥ 2 runs. Look at legs, contacts, pace,
departure delay; `stub` should be ~0. For a failed leg: legs tooltip → video at
that time → `planning_factors` in the bag (`diff_plan` = model's stop, `obst_stop` = stack's guard).

<a id="open-questions"></a>
## Open questions

- Waiting for a queue the recording did not meet is not excused (red lights are).
