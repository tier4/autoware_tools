# Evaluation method

The scene is an open-loop replay of a recorded drive: objects, lights and
indicators repeat what they did and cannot react to the ego. The replay is
valid only while the ego is where and when the recorded bus was. Everything
below keeps it there, or reports when it is not.

## Player

`planning_debug_tools`' original `perception_reproducer.py` (unmodified) publishes
the scene. `scene_driver.py` runs it with `-r 0` (it shows the frame nearest the
ego pose it reads) and feeds it the recorded pose of the instant the clock picks
(`scene_clock.py`). Objects pass through the driver on the way out: objects off,
recording end (stationary only) and the reactive shield.

## Legs

`window_eval.py` cuts the recording at the recorded **departures** into legs
(`start`, `departure`…, `final`). Every model faces every leg from the same
recorded state. Only a wait at a bus stop (a map stop within 12 m) or the
recording's end is a leg goal; a wait on the road (traffic, a parked car) is
no goal.

**Planned cuts** (`leg_cuts: auto`, default). A leg is also cut into windows
where the recorded vehicle stood ≥ 3 s, at its recorded departure
(`find_cuts`), so a model late at one junction drives the next from the
recorded state.

| Rule | Value |
|---|---|
| no cut | where the recording opens or ends standing, within 15 m of a leg goal, at a hole in the recording |
| creeping queue | stands within 15 m in a row = one cut, at the last departure |
| shortest window | 10 s |
| close | when the scene reaches the departure; no arrival, no goal stop |
| open, ego in step (lag ≤ 3 s, ≤ 4 m off the track) | nothing moves: a scoring boundary only (`seamless`) |
| open, ego behind or off the track | placed where the recording stood 3.5 s before the departure, handed over along it (`placed`) |
| ego drives on ≥ 3 m past the stand without waiting | the window closes there; the scene jumps to the recorded departure, objects off 0.25 s around it, the ego untouched (`scene_jumped`) |
| objects | on from the start (no empty-scene departure) |

1. **Hand-over.** The ego is driven along the recording for 3.5 s (the model's
   history window) and handed to the model at the recorded speed.
2. **Departure.** A leg starting at a stop opens with **no objects**; the
   model's own set-off is measured (`departure_delay_sec`). When the ego moves
   (0.5 m/s) the scene jumps to the recorded departure instant and the objects
   appear as they were then.
3. **Arrival.** Stopped within 5 m of the leg's arrival pose, or
   `RouteState.ARRIVED`, or passing within 6 m for a drive-through (cut) final
   leg. Arriving late is reported, not failed.

## Scene clock

After the departure the scene follows the ego (`scene_clock.py`):

| Ego | Scene |
|---|---|
| driving | real time from the departure instant: never slower (no frozen pedestrians under a creeping ego), never faster (no fast-forwarded traffic); the scene re-syncs to the recording at each leg's departure |
| standing | the recording plays on at 1× from the scene's current instant — it never jumps to the ego's spot (≤ 6 s hold at a recorded departure; none with planned cuts) |
| an NPC heading into the ego | judged against the ego's real body (+0.25 m; pedestrians and cyclists +0.15 m), waits on its own recorded path (`reactive_shield.py`): it brakes at ≤ 2.5 m/s² to stand 3 m off (1 m for pedestrians and cyclists; an oncoming one from where the ego will have stopped at 2 m/s²), pulls away at ≤ 1.5 m/s², and never passes through. It also brakes for an NPC ahead going its way that the shield has held back (to its speed, 2 m behind; 0.5 m for pedestrians and cyclists), so a queue behind the ego never collapses into itself. Standing with its way blocked within 5 m (8 m for pedestrians and cyclists), it does not set off; within 15 m of the ego it does not catch up on its delay. It keeps to the scene's time otherwise. Nothing is hidden: a car the ego drives into is a contact |
| the ego's plan, the map and the lights | a standing vehicle does not set off while the ego's 5 s plan crosses its way, once the ego is inside a junction (never pedestrians or cyclists: the bus gives way to them). With the map (`shield_map.py`): an NPC more than 2 s behind its recording stops at the stop line unless the recorded light it now meets lets it go (green, a green arrow for its turn, or the state it crossed under); an NPC of any class held for the ego waits before a junction or crosswalk it has not entered, not inside it (a pedestrian at the kerb, a cyclist at its line); only one queueing behind the ego's tail in its lane keeps its place. A track the tracker opened on top of an older one (an ID switch) runs no earlier than that one; at the recording's end only objects standing by their last 1 s of poses stay |
| objects the recording saw late | the recorded tracker reports an object from the moment the bus first saw it; a lagging ego is nearer, so it is shown earlier, once the ego is as near as the bus first was (`prebirth.py`). Its past is only what physics gives: first seen standing, it stood there (as long before as it stands after, ≤ 30 s); first seen moving, it came along that motion (≤ 8 s, ≤ 30 m); UNKNOWN on the move, creepers and tracks under 1 s get none. Never a mover frozen at its first pose. A past that would have come within 0.3 m of the recorded vehicle, where that was then, is cut to after it passed (the recorded vehicle met nobody there). The past is part of its track, so the shield moves it like any NPC. A track that begins inside the ego, or driving within its standoff, is born at its first pose out of the ego's way |

## NPC motion contract

What every NPC obeys, whatever its class; `scene_audit.py` checks it on each published frame and lists what breaks it (`scene_audit` in the metrics).

| Rule | Why |
|---|---|
| it is shown on its recorded path, only its timing changes — except to go round a standing ego (deadlock, below) | no invented manoeuvres |
| it waits only for: the ego's body on its way, where the ego will be when it gets there (vehicles, from the ego's timed plan), the ego's plan (standing vehicles, ego in a junction), a red light it now meets late, a waiting NPC ahead | each wait has a cause a road user has |
| who goes first: a vehicle ahead on the ego's way, going its way, never waits for it; pedestrians and cyclists keep to the ego's body only, never to its plan | the bus follows the car ahead and gives way to people |
| it waits where a road user may wait: before a junction, a crosswalk or the ego's planned way, not inside | a wait inside is a deadlock with the ego |
| it never stands where its recording moved, except for those waits | no pedestrian frozen mid-road |
| it comes into view no nearer than the recording first had it, and leaves where its own (delayed) track ends | no pop-ins, no car vanishing where it waited |
| its reported speed is the speed it is shown at | the model reads the twist |

Deadlock: the ego and an NPC held for it have both stood 3 s, the NPC on the ego's way. Its way ahead is rewritten round the ego — at the margin if its recorded way does not touch the ego, else shifted sideways (≤ 4 m people, ≤ 2.5 m vehicles, smooth, ≥ 0.5 m off the ego). If no detour fits, its hold is reported `deadlock` and the leg is `scene_limited`.

Every rule is checked offline on all recordings, without a planner (`scene_bench.py`): the ego is driven along the recorded path by programs (as recorded, 15 s late, 8 s early, standing 20 s in every junction) with a generic obstacle stop; contacts, deadlocks, NPCs held inside junctions, distance from the recording and the audit's findings are compared against a baseline.

## Divergence

Checked every 0.25 s, must persist 1 s.

| Kind | Condition | Action |
|---|---|---|
| `stall` | standing ≥ 15 s past a recorded departure while the recorded bus moves | noted; the stuck rule judges (the scene plays on) |
| `slow` | driving lag (the recording's waits after the ego's spot taken out) grew ≥ 8 s since set-off, or ≥ 20 s — the recorded vehicle is ≥ 40 m ahead — and no shown object has been within 30 m of the ego for 3 s | re-init |
| `ahead` | ≥ 8 s ahead of the recording | noted |
| `lateral` | ≥ 4 m off the recorded track | re-init |

Re-init = hand-over again at the scene's current time; ≤ 8 per leg. A re-init
is a window boundary: each segment is scored on its own (`windows`), the
re-initialisation never. Closing by a re-init is no verdict: a segment
closed by `slow` is judged like any window end (`did_not_complete` on lag the
waits do not explain); one closed by `lateral` or by a recorded object through
the ego makes the leg `inconclusive`. `slow` waits for the scene to empty
around the ego: a late ego still among the recording's objects is in its
scene (the scene driver reports the nearest shown object,
`/perception_reproducer/nearest_object_m`).
Leg wall budget: 1.5 × window + 210 s.

## Lights and indicators

Traffic lights replay on the same clock as the objects (red-light rules in
[METRICS](METRICS.md#red-light-rules)). Recorded turn
indicators are replayed and merged with the model's own command into one
stream (`/dp_multi_eval/turn_indicators_status`) that the model reads —
departures in this fleet are always signalled and the model is conditioned on it.

## Stack overlay

Per-job copy of the `autoware_launch` share tree on `AMENT_PREFIX_PATH`;
`src/autoware` is never modified.

| Knob | Effect | Why |
|---|---|---|
| model swap | job's `diffusion_planner.param.yaml` | parallel models |
| `disable_long_stop_fixer` | StopPointFixer stub rules off (obstacle_stop stays) | the fixer replaced every pull-away plan with a stop stub; measures the model, not the gate |
| `match_vehicle_dynamics` | psim `DELAY_STEER_ACC_GEARED` with the description's delays | the ideal model turned waits into 1–1.6 m/s creep bursts |
| `indicator_merge` | model reads the merged indicator stream | two writers interleaved LEFT/DISABLE on the model's input |

The output bag records both sides of the trajectory modifier, so "the model
stopped" and "the stack stopped it" are separable offline (`stack_intervention`).

## Limits

- Faster-than-recording egos run out of recording; only stationary objects are kept past the end.
- Recorded `UNKNOWN` objects have zero twist and are ignored by the model.
- Run-to-run variance is real (identical weights: 5/12 bags differed between two runs). Judge on ≥ 2 runs.
