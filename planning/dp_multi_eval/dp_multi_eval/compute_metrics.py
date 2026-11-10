"""Phase 2 — offline metrics from an output bag."""

from __future__ import annotations

import argparse
import json
import bisect
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.bag_reader import (
    BagSeries,
    EgoSample,
    downsample_ego,
    load_bag_series,
    load_reference_ego,
)
from dp_multi_eval.geometry import (
    box_local,
    load_vehicle_footprint,
    polygon_distance,
    transform_local_polygon,
)
from dp_multi_eval.topics import TopicSet, load_topics

# Match diffusion_planner_batch_eval npc_collision_check.DEFAULT_NPC_LABELS
# Contacts that count. PEDESTRIAN joined on 2026-10-28 (operator): hitting a
# person failed nothing before; UNKNOWN stays out (tracker noise).
NPC_COLLISION_LABELS = frozenset(
    {"CAR", "TRUCK", "BUS", "TRAILER", "MOTORCYCLE", "BICYCLE", "PEDESTRIAN"}
)


@dataclass
class Thresholds:
    stuck_speed_mps: float = 0.05
    stuck_duration_sec: float = 45.0
    goal_stop_speed_mps: float = 0.2
    goal_tolerance_m: float = 2.0
    goal_lateral_tolerance_m: float = 2.0
    goal_longitudinal_tolerance_m: float = 2.0
    # Clearance to road_border/curbstone: OOB if footprint crosses or dist < margin.
    oob_margin_m: float = 0.0
    oob_search_radius_m: float = 80.0
    collision_distance_m: float = 0.1
    sample_dt: float = 0.2
    # Lanelet2 LineString ``type`` attributes treated as uncrossable borders.
    oob_boundary_types: tuple[str, ...] = ("road_border", "curbstone")
    # Reference-speed comparison. The recording is the reference: it is the same
    # vehicle on the same road, so a large sustained difference is a finding about
    # the planner, not about the scenario. Samples are paired by position, not by
    # time — the replay reaches a place at its own moment.
    ref_speed_match_radius_m: float = 3.0
    # Flag on the 90th percentile of the signed difference, not the median. The
    # median is pulled down by the stretches where both are stopped or crawling and
    # agree; what matters is how much faster the replay runs where it runs fast,
    # which is where a speed difference turns into a conflict.
    ref_speed_p90_tolerance_mps: float = 2.5
    ref_speed_median_tolerance_mps: float = 1.5
    # Ignore pairs where both are essentially stopped; a dwell is not a speed error.
    ref_speed_min_speed_mps: float = 0.5

    # --- stack_intervention ---
    # StopPointFixer's replacement is 3 points; a plan is 80. Anything <= 5 is a
    # forced stop, not a plan.
    stub_max_points: int = 5
    # Trajectory publishes at ~5 Hz here; a 3 s gap means real plans got through.
    stub_gap_sec: float = 3.0
    # Calibrated on the observed split: completed run 7.7% / 7.4 s, stalled runs
    # 69.8% / 79.2 s and 60%+ / 66.4 s. A quarter of the run, or one unbroken
    # 20 s stretch, is the latch — a light or a lead vehicle releases sooner.
    stub_ratio_flag: float = 0.25
    forced_stop_flag_sec: float = 20.0
    stub_moving_speed_mps: float = 0.25
    # Half-angle of the cone ahead of the ego. A contact inside it is something
    # the ego closed on; outside it, something that came into the ego's side or
    # rear, which an open-loop replay produces whenever the ego is not where the
    # recording had it.
    collision_ahead_cone_deg: float = 60.0

    # --- replay validity envelope ---
    # An open-loop replay only tests something where the ego is somewhere the
    # recording actually went, at a speed the recorded traffic was arranged
    # around. Outside that, replayed NPCs and the ego occupy the same space
    # because neither can react to the other -- which is a property of the
    # method, not a collision. Measured: one contact sat 19.9 m from the nearest
    # recorded pose, where the recorded ego had been stopped; another was on the
    # corridor but running 4.4x the recorded speed.
    envelope_offset_m: float = 5.0
    # Below this share of comparable samples the replay is not judging the model,
    # and saying so is more useful than a verdict drawn from the rest.
    envelope_min_coverage: float = 0.5

    # --- comfort (jerk / harsh brake) ---
    # Standing passengers: a bus brakes harder than 2 m/s^2 only when it has
    # to, and a jerk above 2.5 m/s^3 is felt as a lurch. Events last at least
    # comfort_event_min_sec so single-sample spikes of a 50 Hz odometry do
    # not count; the recording is scored the same way and is the reference.
    harsh_decel_mps2: float = 2.0
    harsh_jerk_mps3: float = 2.5
    comfort_event_min_sec: float = 0.3
    # --- proximity ---
    proximity_close_m: float = 1.0
    # A pass closer than this is listed with its time and place. The contact
    # threshold is 0.1 m; between the two sits the brush past that looks like a
    # collision on the video and used to leave no trace in the report.
    proximity_near_miss_m: float = 0.5


METRIC_DESCRIPTIONS = {
    "bus_stop_service": (
        "For each map bus stop the recording stood at: did the ego stop in the bay at the sign "
        "(|lat| <= 1.5 m, |lon| <= 5 m, >= 3 s)? served / in_lane / passed. Flags when any "
        "recorded stop was not served."
    ),
    "replay_validity": (
        "Share of samples where the ego is within envelope_offset_m of the recorded "
        "path. "
        "Outside it the replayed traffic belongs to a situation the ego is no longer "
        "in, and overlaps between ego and NPC are a property of open-loop replay "
        "rather than of the planner. Flags below envelope_min_coverage: the run "
        "cannot be judged from this replay."
    ),
    "collision_rate_attribution": (
        "Each contact is attributed: ego_caused (ego under its own control closing "
        "on something within collision_ahead_cone_deg ahead), cut_in (ego moving, "
        "contact from the side or rear), through_ego (a recorded object the "
        "reproducer withheld from the model because its recorded path ran through "
        "the ego's body; see objects_through_ego), struck_while_stopped, and not_in_control "
        "(ego stopped while the stack was handing the controller stop points). Only "
        "ego_caused gates the run; the others are properties of an open-loop replay."
    ),
    "stack_intervention": (
        "Fraction of published trajectories that are StopPointFixer stop points "
        "(<= stub_max_points points) instead of the model's plan, and the longest "
        "unbroken stretch of them. Separates 'the model stopped' from 'the model "
        "was stopped'. Flags past stub_ratio_flag or forced_stop_flag_sec; "
        "reported, never gating."
    ),
    "stop_go": (
        "Unplanned stops, creep time (0.05-1 m/s) and start-brake cycles outside leg "
        "goals and the places the recording itself waited, against the recording "
        "scored the same way. Flags when the run stops, creeps or cycles clearly more "
        "than the recording did. Gating."
    ),
    "objects_through_ego": (
        "Recorded objects whose box overlapped the ego footprint and were withheld "
        "from the model for those frames (the replay is open-loop: an NPC that "
        "passed where the recorded bus no longer was drives through a slower ego). "
        "Reported, never gating; each event is a place where the scene and the ego "
        "had parted ways."
    ),
    "scene_audit": (
        "The scene player's own defects, found on what it published (scene_audit.py): an "
        "NPC standing in a junction or on a crosswalk longer than its recording stood there "
        "(stand_in_zone, with the rule that held it), appearing out of nothing within 15 m "
        "of the ego (pop_in; a recorded track under 1 s is a blip, counted apart), leaving "
        "while it waited beside the ego (vanish) or stepping more than a frame allows (jump). "
        "Reported, never gating: a finding is a fault of the replay near the ego, to read a "
        "stop or a hesitation there with."
    ),
    "resync": (
        "Times the harness set the ego back onto the recording because it had fallen "
        "resync_lag_sec behind it (it stood where the recorded vehicle drove, or drove "
        "slower): the ego is placed at the recorded pose, driven along the recorded "
        "track for the model's history window and handed back at the recorded speed; "
        "the NPCs are never touched. Each event is a segment the model did not drive "
        "the way the recording did. Flags when there is any. Gating."
    ),
    "comfort": (
        "Longitudinal comfort from the ego's speed (smoothed over 0.3 s): harsh brakes "
        "(deceleration beyond harsh_decel_mps2 for comfort_event_min_sec) and harsh jerk "
        "(|jerk| beyond harsh_jerk_mps3), with the maximum deceleration and the 95th "
        "percentile of |jerk|. The harness's hand-overs are excluded; the recording is "
        "scored the same way. Flags when the run brakes harshly more often than the "
        "recording plus one. Gating."
    ),
    "proximity": (
        "Closest a replayed vehicle came to the ego's footprint while the ego was moving "
        "(objects the reproducer withheld for driving through the ego excluded), the time "
        "spent within proximity_close_m, and every approach closer than "
        "proximity_near_miss_m with its time, place and object. Reported, never gating; "
        "collision_rate gates."
    ),
    "success_rate": (
        "The run's verdict and its legs: passed (all gating metrics clear), legs passed / "
        "legs total. The dashboard aggregates these per model into a success rate."
    ),
    "windows": (
        "Leg-window evaluation (window_eval.py): the recording is cut at the recorded "
        "departures into the route's legs; each leg starts from the recorded state and "
        "the scene plays untouched. With planned cuts (leg_cuts auto) a leg is also cut where the "
        "recorded vehicle stood 3 s or more, at its departure: such a window has no arrival and no "
        "goal stop, it is judged on stuck and on lag its waits do not explain; the next opens with the "
        "objects on, the ego left as it is when in step with the recording, else placed on it (placed). "
        "An ego that drives on past the stand without waiting closes the window there and the scene "
        "jumps to the recorded departure (scene_jumped; cut_pass: the wait skipped and the light it drove under). "
        "A contact beginning within 4 s of that jump with an object whose body was then within 15 m of the ego's is the jump's (collision "
        "scene_jump, not gating); the leg is not judged from there on and is inconclusive (scene_jump_contact). A leg passes when the ego reached the leg's arrival "
        "pose or not — arriving late is reported, not failed — made no stop the recording "
        "did not make (a stop where the recording waited too, or under a red light ahead, "
        "is a wait) and left every recorded stop within 15 s of the recording. Falling "
        "behind the recording with no recorded object left around the ego, leaving its track, "
        "or a recorded object driving into the ego, re-initialises the ego on the recording, and "
        "the reset is a window boundary: each segment is scored on its own (windows), the "
        "re-initialisation never; closing by a reset is no verdict, one closed because the scene "
        "stopped being the ego's (through, lateral) makes the leg inconclusive. Running ahead is "
        "noted and the drive goes on. A stand the recording's end cut short is judged on the live recording (a stop begun after the end, on what stood "
        "still): it fails only after 5 s without a reason there, else the leg is inconclusive "
        "(recording_ran_out). Departure delay is the ego's first movement against the recorded "
        "departure. Every stop is explained (waits): seconds with a red light ahead, a walker or "
        "cyclist on or entering the ego's way, a vehicle on its way or cross traffic entering it "
        "are justified. A window that closes short of the leg's end on an ego standing without "
        "such a reason, or that closes (at the recording's end or a slow reset) 15 s or "
        "more behind the recording once its justified waits are taken off, is not completed "
        "(goal_missed did_not_complete): failed. A stuck stretch whose "
        "unjustified part is under the stuck limit is a wait (stuck_justified). "
        "A stuck stretch is scene_limited, not failed, when for most of it the "
        "ego faced an NPC the shield could not lead round it on its way (deadlock: a real one would "
        "go round) or the scene stood past the recording's end; an NPC merely waiting for the ego "
        "(held for it) is no limit and no reason to wait: standing 10 s or more with no reason while "
        "one stands ahead of the ego within 15 m — not at the leg's goal, nor where the recording "
        "waited too — is a mutual wait (mutual_waits) and fails the leg. A light whose "
        "stop line the ego's front passed by 2 m is behind it, no reason. A scene-limited leg "
        "is inconclusive. Gating."
    ),
    "reference_speed": (
        "Ego speed compared against the source recording at the same place. Each replay "
        "sample is paired with the nearest recorded ego pose within "
        "ref_speed_match_radius_m; pairs where both are below ref_speed_min_speed_mps "
        "are dropped. Reports the median and mean signed difference [m/s] and the "
        "traversal-time ratio. Fails when the median absolute difference exceeds "
        "ref_speed_median_tolerance_mps."
    ),
    "stuck_rate": (
        "Fraction of scenario time the ego is nearly stopped (speed ≤ stuck_speed_mps) "
        "away from the goal for contiguous stretches ≥ stuck_duration_sec. "
        "rate = total_stuck_duration / bag_duration."
    ),
    "collision_rate": (
        "Fraction of downsampled ego frames where the ego footprint is within "
        "collision_distance_m of an NPC (CAR/TRUCK/BUS/TRAILER/MOTORCYCLE/BICYCLE/PEDESTRIAN). "
        "rate = colliding_frames / ego_frames. UNKNOWN is excluded (tracker noise)."
    ),
    "out_of_boundary": (
        "Fraction of downsampled ego frames where the ego footprint crosses an uncrossable "
        "Lanelet2 LineString (type in oob_boundary_types, default road_border + curbstone), "
        "or comes within oob_margin_m of one. "
        "rate = oob_frames / ego_frames. Requires a loaded lanelet map."
    ),
    "goal_stop_precision": (
        "Average stop pose error in the goal frame over low-speed samples "
        "(speed ≤ goal_stop_speed_mps) within goal_tolerance_m of the goal. "
        "Reports position, lateral, longitudinal [m] and heading [deg]. "
        "Fails when |lateral|, |longitudinal|, or position exceeds the configured tolerances. "
        "Scenes flagged as stuck are excluded (status=excluded_stuck)."
    ),
}


def load_thresholds(path: Path | None) -> Thresholds:
    if path is None or not path.expanduser().is_file():
        return Thresholds()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in Thresholds.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            continue
        if key == "oob_boundary_types":
            if isinstance(value, (list, tuple)):
                kwargs[key] = tuple(str(v) for v in value)
            else:
                kwargs[key] = (str(value),)
        else:
            kwargs[key] = float(value)
    return Thresholds(**kwargs)


def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def goal_frame_errors(
    ego_x: float, ego_y: float, ego_yaw: float, goal_x: float, goal_y: float, goal_yaw: float
) -> tuple[float, float, float, float]:
    """Return lateral [m], longitudinal [m], position [m], heading [deg] in goal frame."""
    dx, dy = ego_x - goal_x, ego_y - goal_y
    pos = math.hypot(dx, dy)
    cos_g, sin_g = math.cos(goal_yaw), math.sin(goal_yaw)
    longitudinal = dx * cos_g + dy * sin_g
    lateral = -dx * sin_g + dy * cos_g
    heading = math.degrees(normalize_angle(ego_yaw - goal_yaw))
    return lateral, longitudinal, pos, heading


BUS_STOP_SERVED_LAT_M = 1.5
BUS_STOP_SERVED_LON_M = 5.0
BUS_STOP_NEAR_M = 12.0
BUS_STOP_DWELL_SEC = 3.0


def map_bus_stops(map_path) -> list[dict[str, str]] | None:
    """The map's named stops (stop_points.csv rows: name, x, y, yaw), or None
    without a map or a stop list."""
    if map_path is None:
        return None
    csv_path = Path(map_path) / "stop_points.csv"
    if not csv_path.is_file():
        return None
    import csv as _csv

    with csv_path.open(newline="", encoding="utf-8") as fh:
        return [row for row in _csv.DictReader(fh) if (row.get("name") or "").strip()]


def _stops_with_offsets(samples: list[EgoSample], gx: float, gy: float, gyaw: float,
                        speed_mps: float, min_sec: float):
    """Stops (>= min_sec below speed_mps) within BUS_STOP_NEAR_M of the stop,
    as (duration, lon, lat) in the stop's frame."""
    out = []
    i, n = 0, len(samples)
    while i < n:
        if samples[i].speed_mps > speed_mps:
            i += 1
            continue
        j = i
        while j < n and samples[j].speed_mps <= speed_mps:
            j += 1
        dur = samples[j - 1].stamp_sec - samples[i].stamp_sec
        mid = samples[(i + j - 1) // 2]
        # heading must agree with the stop's (the return-trip stop of the same
        # name sits 10 m away on the opposite kerb, facing the other way)
        dyaw = abs((mid.yaw_rad - gyaw + math.pi) % (2 * math.pi) - math.pi)
        if dur >= min_sec and math.hypot(mid.x - gx, mid.y - gy) <= BUS_STOP_NEAR_M and dyaw <= math.pi / 2:
            dx, dy = mid.x - gx, mid.y - gy
            lon = dx * math.cos(gyaw) + dy * math.sin(gyaw)
            lat = -dx * math.sin(gyaw) + dy * math.cos(gyaw)
            out.append((dur, lon, lat))
        i = j
    return out


def metric_bus_stop_service(
    series: BagSeries,
    reference: BagSeries | None,
    map_path: Path | None,
    thr: Thresholds,
) -> dict[str, Any]:
    """Did the ego serve the bus stops the recording served?

    For every map stop (stop_points.csv) where the recorded vehicle stood for
    BUS_STOP_DWELL_SEC within BUS_STOP_NEAR_M, the ego's best stop near that
    pose is measured in the stop's frame: served = |lat| <= 1.5 m and
    |lon| <= 5 m (in the bay, at the sign); in_lane = stopped nearby but
    outside the bay; passed = never stopped within 12 m. Flags when any
    recorded stop was not served — on the vehicle a bus stop is the goal.
    """
    empty = {"flagged": False, "stops": [], "served": 0, "recorded": 0,
             "status": "no_reference", "description": METRIC_DESCRIPTIONS["bus_stop_service"]}
    if reference is None or not reference.ego or not series.ego or map_path is None:
        return empty
    rows = map_bus_stops(map_path)
    if rows is None:
        return {**empty, "status": "no_stop_points"}
    stops_out = []
    for row in rows:
        gx, gy, gyaw = float(row["x"]), float(row["y"]), float(row["yaw"])
        rec = _stops_with_offsets(reference.ego, gx, gy, gyaw, thr.goal_stop_speed_mps, BUS_STOP_DWELL_SEC)
        if not rec:
            continue
        # the spawn: the ego starts there, nothing to serve
        e0 = series.ego[0]
        if math.hypot(e0.x - gx, e0.y - gy) <= BUS_STOP_NEAR_M:
            continue
        passed_by = any(math.hypot(e.x - gx, e.y - gy) <= 30.0 for e in series.ego)
        if not passed_by:
            continue
        mine = _stops_with_offsets(series.ego, gx, gy, gyaw, thr.goal_stop_speed_mps, BUS_STOP_DWELL_SEC)
        best = min(mine, key=lambda t: abs(t[2]) + 0.2 * abs(t[1]), default=None)
        if best is None:
            verdict = "passed"
        elif abs(best[2]) <= BUS_STOP_SERVED_LAT_M and abs(best[1]) <= BUS_STOP_SERVED_LON_M:
            verdict = "served"
        else:
            verdict = "in_lane"
        stops_out.append({
            "name": row["name"].strip(),
            "recorded_dwell_sec": round(max(t[0] for t in rec), 1),
            "ego_dwell_sec": round(best[0], 1) if best else 0.0,
            "lon_m": round(best[1], 2) if best else None,
            "lat_m": round(best[2], 2) if best else None,
            "verdict": verdict,
        })
    served = sum(1 for s in stops_out if s["verdict"] == "served")
    return {
        "flagged": bool(stops_out) and served < len(stops_out),
        "stops": stops_out,
        "served": served,
        "recorded": len(stops_out),
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["bus_stop_service"],
    }


RECORDED_WAIT_RADIUS_M = 25.0
RECORDED_WAIT_MATCH = 0.5   # a recorded dwell at least this fraction of the ego's stop explains it


def recorded_wait_near(reference_ego: list[EgoSample] | None, x: float, y: float, thr: Thresholds) -> float:
    """Seconds the recorded vehicle itself stood (speed <= stuck_speed) within
    RECORDED_WAIT_RADIUS_M of this spot — a red light, a queue, a bus stop."""
    if not reference_ego:
        return 0.0
    total = 0.0
    prev = None
    for e in reference_ego:
        if prev is not None and e.speed_mps <= thr.stuck_speed_mps and prev.speed_mps <= thr.stuck_speed_mps:
            if math.hypot(e.x - x, e.y - y) <= RECORDED_WAIT_RADIUS_M:
                total += max(0.0, e.stamp_sec - prev.stamp_sec)
        prev = e
    return total


def find_stuck_events(
    samples: list[EgoSample],
    goal_x: float,
    goal_y: float,
    thr: Thresholds,
    reference_ego: list[EgoSample] | None = None,
) -> list[dict[str, Any]]:
    """Contiguous near-zero speed away from goal lasting >= stuck_duration_sec.

    Each event carries ``recorded_wait_sec``: how long the recorded vehicle
    stood within 25 m of the same spot. When that covers at least half of the
    ego's stop the event is a wait the recording made too (a red light the
    recorded vehicle also sat through: 57 s against the recording's 56 s on
    2026-04-28) and is marked ``waited_like_recording`` — reported, not a
    stall. A stop where the recording rolled through stays a stall.
    """
    events: list[dict[str, Any]] = []
    if not samples:
        return events

    i = 0
    n = len(samples)
    while i < n:
        if samples[i].speed_mps > thr.stuck_speed_mps:
            i += 1
            continue
        j = i
        while j < n and samples[j].speed_mps <= thr.stuck_speed_mps:
            j += 1
        dur = samples[j - 1].stamp_sec - samples[i].stamp_sec
        mid = samples[(i + j - 1) // 2]
        dist_goal = math.hypot(mid.x - goal_x, mid.y - goal_y)
        near_goal = dist_goal <= thr.goal_tolerance_m
        if dur >= thr.stuck_duration_sec and not near_goal:
            rec_wait = recorded_wait_near(reference_ego, mid.x, mid.y, thr)
            events.append(
                {
                    "start_sec": samples[i].stamp_sec,
                    "end_sec": samples[j - 1].stamp_sec,
                    "duration_sec": dur,
                    "distance_to_goal_m": dist_goal,
                    "recorded_wait_sec": rec_wait,
                    "waited_like_recording": bool(rec_wait >= RECORDED_WAIT_MATCH * dur),
                }
            )
        i = j
    return events


def metric_stuck(
    series: BagSeries,
    goal_x: float,
    goal_y: float,
    thr: Thresholds,
    reference: BagSeries | None = None,
) -> dict[str, Any]:
    """Stuck time rate: near-zero speed away from goal lasting > stuck_duration_sec,
    excluding waits the recording made at the same place (see find_stuck_events)."""
    samples = series.ego
    duration = series.duration_sec
    if not samples:
        if not series.velocity_mps:
            return {
                "flagged": False,
                "rate": 0.0,
                "duration_sec": 0.0,
                "bag_duration_sec": 0.0,
                "events": [],
                "recorded_waits": [],
                "recorded_wait_sec": 0.0,
                "status": "no_ego_data",
                "description": METRIC_DESCRIPTIONS["stuck_rate"],
            }

    all_events = find_stuck_events(
        samples, goal_x, goal_y, thr, reference_ego=(reference.ego if reference else None)
    )
    events = [e for e in all_events if not e["waited_like_recording"]]
    recorded_waits = [e for e in all_events if e["waited_like_recording"]]
    total = sum(e["duration_sec"] for e in events)
    rate = (total / duration) if duration > 1e-9 else 0.0
    # Also total every stalled stretch away from the goal, however short. The run
    # itself gives up on a cumulative budget (stuck_abort_sec) while this metric
    # only counted contiguous stretches past stuck_duration_sec, so a job abandoned
    # for stalling could report stuck_rate 0.0: each repositioning split the stall
    # into pieces below the threshold. The two now measure the same thing, one
    # flagging sustained stalls and one accounting for all of them.
    stalled_total = 0.0
    i, n = 0, len(samples)
    while i < n:
        if samples[i].speed_mps > thr.stuck_speed_mps:
            i += 1
            continue
        j = i
        while j < n and samples[j].speed_mps <= thr.stuck_speed_mps:
            j += 1
        mid = samples[(i + j - 1) // 2]
        if math.hypot(mid.x - goal_x, mid.y - goal_y) > thr.goal_tolerance_m:
            stalled_total += samples[j - 1].stamp_sec - samples[i].stamp_sec
        i = j
    stalled_rate = (stalled_total / duration) if duration > 1e-9 else 0.0
    return {
        "flagged": bool(events),
        "rate": rate,
        # every stall away from the goal, not only sustained ones
        "stalled_rate": stalled_rate,
        "stalled_sec": stalled_total,
        "duration_sec": total,
        "bag_duration_sec": duration,
        "events": events,
        # long stops the recorded vehicle also made here (red lights, queues)
        "recorded_waits": recorded_waits,
        "recorded_wait_sec": sum(e["duration_sec"] for e in recorded_waits),
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["stuck_rate"],
    }


def metric_stop_go(bag_path: Path, series: BagSeries, reference: BagSeries | None, goal_x: float, goal_y: float) -> dict[str, Any]:
    """See dp_multi_eval.stop_go. Leg goals come from route_goals.json beside the bag."""
    import json as _json

    from dp_multi_eval.stop_go import stop_go

    out: dict[str, Any] = {"status": "ok", "description": METRIC_DESCRIPTIONS["stop_go"]}
    if not series.ego:
        out["status"] = "no_ego"
        return out
    goals = [(goal_x, goal_y)]
    try:
        payload = _json.loads((Path(bag_path).parent / "route_goals.json").read_text(encoding="utf-8"))
        goals += [(float(g["x"]), float(g["y"])) for g in payload.get("goals", [])]
    except Exception:  # noqa: BLE001
        pass
    res = stop_go(series.ego, reference.ego if reference else None, goals)
    out.update(res)
    if reference is None:
        out["status"] = "no_reference"
    return out


def metric_objects_through_ego(bag_path: Path, series: BagSeries) -> dict[str, Any]:
    """Footprint suppressions logged by the reproducer (reproducer.log beside the bag)."""
    import re

    out: dict[str, Any] = {
        "flagged": False,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["objects_through_ego"],
        "count": 0,
        "total_sec": 0.0,
        "events": [],
    }
    log = Path(bag_path).parent / "reproducer.log"
    if not log.is_file():
        out["status"] = "no_log"
        return out
    base = series.ego[0].stamp_sec if series.ego else None
    start = re.compile(r"SUPPRESS #\d+: object (\w+) (?:inside the ego footprint|recorded path enters the ego footprint in [\d.]+ s) at \(([-\d.]+), ([-\d.]+)\)(?: t=([\d.]+))?")
    end = re.compile(r"SUPPRESS end: object (\w+) clear of the ego(?: t=([\d.]+))?")
    open_events: dict[str, dict[str, Any]] = {}
    for line in log.read_text(errors="ignore").splitlines():
        m = start.search(line)
        if m:
            t = float(m.group(4)) if m.group(4) else None
            ev = {"object": m.group(1), "x": float(m.group(2)), "y": float(m.group(3)),
                  "t_rel": (round(t - base, 1) if (t is not None and base is not None) else None),
                  "duration_sec": None}
            open_events[m.group(1)] = ev
            out["events"].append(ev)
            continue
        m = end.search(line)
        if m and m.group(1) in open_events:
            ev = open_events.pop(m.group(1))
            if m.group(2) and ev["t_rel"] is not None and base is not None:
                ev["duration_sec"] = round(float(m.group(2)) - base - ev["t_rel"], 1)
    out["count"] = len(out["events"])
    out["total_sec"] = round(sum(e["duration_sec"] or 0.0 for e in out["events"]), 1)
    # Vehicles that braked for the ego (reactive_agents.py) and those dropped
    # for falling too far behind their recorded schedule.
    text = log.read_text(errors="ignore")
    out["yielding"] = len(re.findall(r"^YIELD #\d+:", text, re.M))
    out["dropped"] = len(re.findall(r"^DROP:", text, re.M))
    return out


def metric_resync(bag_path: Path, series: BagSeries) -> dict[str, Any]:
    """Re-sync events written by the harness (resyncs.json beside the bag)."""
    out: dict[str, Any] = {
        "flagged": False,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["resync"],
        "count": 0,
        "events": [],
    }
    path = Path(bag_path).parent / "resyncs.json"
    if not path.is_file():
        return out
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        out["status"] = "unreadable"
        return out
    base = series.ego[0].stamp_sec if series.ego else None
    for ev in raw:
        before = ev.get("ego_before") or {}
        start = ev.get("recorded_start") or {}
        end = ev.get("recorded_end") or {}
        t_wall = ev.get("t_wall")
        out["events"].append(
            {
                "t_rel": (round(float(t_wall) - base, 1) if (t_wall is not None and base is not None) else None),
                "lag_sec": ev.get("lag_sec"),
                "x": before.get("x"),
                "y": before.get("y"),
                "ego_v": before.get("v"),
                "to_x": start.get("x"),
                "to_y": start.get("y"),
                "recorded_v": start.get("v"),
                "handed_back_v": end.get("v"),
                "jump_m": ev.get("jump_m"),
                "kind": "stall" if (before.get("v") or 0.0) < 0.5 else "slow",
            }
        )
    out["count"] = len(out["events"])
    out["flagged"] = out["count"] > 0
    return out


def _bus_stop_at(stops, x: float, y: float) -> str | None:
    """The map stop a leg goal belongs to (within BUS_STOP_NEAR_M), if any."""
    for row in stops or []:
        if math.hypot(float(row["x"]) - x, float(row["y"]) - y) <= BUS_STOP_NEAR_M:
            return str(row["name"])
    return None


def _segment_windows(segs: list[dict], leg_wall_end) -> list[tuple[float, float, str]]:
    """(wall from, wall to, how it closed) of each segment of a leg: from its
    hand-over to its end. Runs before 2026-11-10 have no segment wall_end: a
    segment then ends LEAD_SEC + 2 s before the next hand-over (the reset and
    the lead along the recording)."""
    from dp_multi_eval.window_eval import LEAD_SEC

    out = []
    for i, s in enumerate(segs):
        a = s.get("wall_handover")
        if a is None:
            continue
        b = s.get("wall_end")
        if b is None:
            nxt = segs[i + 1].get("wall_handover") if i + 1 < len(segs) else None
            b = (float(nxt) - LEAD_SEC - 2.0) if nxt is not None else leg_wall_end
        if b is None:
            continue
        out.append((float(a), max(float(a), float(b)), str(s.get("ended_by") or "")))
    return out


def _in_windows(t: float, wins) -> bool:
    return not wins or any(a <= t <= b for a, b, _ in wins)


def metric_windows(
    bag_path: Path,
    series: BagSeries,
    reference: BagSeries | None,
    scene_time=None,
    topics: TopicSet | None = None,
    map_path: Path | None = None,
    thr: Thresholds | None = None,
    footprint=None,
    scene_jump_contacts: list[dict] | None = None,
) -> dict[str, Any]:
    """Per-leg verdicts from windows.json (window_eval.py) beside the bag."""
    thr = thr or Thresholds()
    from dp_multi_eval.stop_go import stop_go
    from dp_multi_eval import waits as W

    # The light ahead of a stopped ego, read once and only when a leg has a stop
    # to judge: the map and the replayed signals are per bag, not per leg.
    lights: dict[str, Any] = {}

    def light_states():
        if "states" not in lights:
            lights["states"] = _light_states(bag_path, topics or TopicSet())
        return lights["states"]

    bus_stops = map_bus_stops(map_path)

    def shield_map():
        if "smap" not in lights:
            try:
                from dp_multi_eval.shield_map import load_shield_map

                lights["smap"] = load_shield_map(map_path)
            except Exception:  # noqa: BLE001
                lights["smap"] = None
        return lights["smap"]

    def light_group(x, y, yaw):
        if "map" not in lights:
            try:
                lights["map"] = _load_lanelet_map(map_path)
            except Exception:  # noqa: BLE001
                lights["map"] = None
        group = _light_ahead(lights["map"], x, y, yaw) if lights["map"] is not None else None
        # Past its stop line the light is behind the ego: a red there is no
        # reason to stand inside the junction (H78, 2026-11-10: 18.7 s mid-turn
        # counted as a red-light wait).
        smap = shield_map() if group is not None else None
        if smap is not None and _past_stop_line(smap.stop_lines, group, x, y, yaw, W.body_from_footprint(footprint).front_m):
            return None
        return group

    def light_red(t, x, y, yaw, known=False):
        """Is the light ahead red or amber at t? ``known``: None instead of
        False when no light governs the ego there."""
        states = light_states()
        if not states:
            return None if known else False
        key = (round(x, 1), round(y, 1), round(yaw, 2))
        groups = lights.setdefault("groups", {})
        if key not in groups:
            groups[key] = light_group(x, y, yaw)
        if groups[key] is None:
            return None if known else False
        times = lights.get("times")
        if times is None or len(times) != len(states):
            times = lights["times"] = [t_ for t_, _ in states]
        j = bisect.bisect_right(times, t) - 1
        return j >= 0 and states[j][1].get(groups[key]) in (1, 2)  # RED, AMBER

    def explain(t0, t1, x, y, yaw):
        """Why the ego stood (t0, t1) at (x, y, yaw): waits.explain_stop."""
        if "frames" not in lights:
            lights["frames"] = W.ObjectFrames(series.objects) if series.objects else None
            lights["ref"] = W.decimate_ref(reference.ego) if (reference and reference.ego) else []
            lights["body"] = W.body_from_footprint(footprint)
            smap = shield_map()
            lights["crosswalks"] = smap.crosswalks if smap is not None else []
            if "log" not in lights:
                lights["log"] = _shield_log(Path(bag_path).parent / SHIELD_LOG)
            lights["held"] = W.held_index(lights["log"][0])
        return W.explain_stop(t0, t1, x, y, yaw, frames=lights["frames"], ref=lights["ref"],
                              body=lights["body"], light_red=light_red, crosswalks=lights["crosswalks"],
                              held_for_ego=lights["held"])

    def mutual(t0, t1, x, y, yaw):
        """(seconds, object ids) the ego stood with no reason while NPCs the
        shield held for it stood AHEAD of it within MUTUAL_NEAR_M
        (waits.mutual_wait_sec); the ids most often near first. One behind the
        ego's front is queueing behind it, not what the ego waits for: a car
        behind a bus at its stop made every stop dwell a "mutual wait"
        (two-model run, 2026-11-10: 10 of 10 flags wrong)."""
        explain(t0, t0, x, y, yaw)   # loads the context once
        holds = [h for h in lights["log"][0] if h.get("reason") in ("ego", "plan")]
        if not holds:
            return 0.0, None
        ht = [float(h["t"]) for h in holds]
        cx, cy = x + EGO_CENTER_OFFSET_M * math.cos(yaw), y + EGO_CENTER_OFFSET_M * math.sin(yaw)
        c_, s_ = math.cos(yaw), math.sin(yaw)
        ahead_from = lights["body"].front_m - MUTUAL_BEHIND_FRONT_M
        seen: dict[str, int] = {}

        def held_near(t):
            lo, hi = bisect.bisect_left(ht, t - 0.6), bisect.bisect_right(ht, t + 0.6)
            near = {h["id"] for h in holds[lo:hi]
                    if math.hypot(float(h["x"]) - cx, float(h["y"]) - cy) <= MUTUAL_NEAR_M
                    and (float(h["x"]) - x) * c_ + (float(h["y"]) - y) * s_ >= ahead_from}
            for oid in near:
                seen[oid] = seen.get(oid, 0) + 1
            return bool(near)

        sec = W.mutual_wait_sec(t0, t1, x, y, yaw, held_near=held_near, frames=lights["frames"], ref=lights["ref"],
                                body=lights["body"], light_red=light_red, crosswalks=lights["crosswalks"],
                                held_for_ego=lights["held"])
        return sec, sorted(seen, key=lambda k: -seen[k])

    def explain_tail(t0, t1, x, y, yaw):
        """Seconds the ego had stood without a reason at t1 (waits.unjustified_tail_sec)."""
        explain(t0, t0, x, y, yaw)   # loads the context once
        return W.unjustified_tail_sec(t0, t1, x, y, yaw, frames=lights["frames"], ref=lights["ref"],
                                      body=lights["body"], light_red=light_red, crosswalks=lights["crosswalks"],
                                      held_for_ego=lights["held"])

    out: dict[str, Any] = {
        "flagged": False,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["windows"],
        "legs": [],
        "legs_total": 0,
        "legs_passed": 0,
    }
    path = Path(bag_path).parent / "windows.json"
    if not path.is_file():
        out["status"] = "no_windows"
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        out["status"] = "unreadable"
        return out
    ego = series.ego
    ref = reference.ego if reference else []
    from dp_multi_eval.lag import recorded_lag_sec, reference_track

    ref_track = reference_track(ref) if ref else None
    rec_end = ref[-1].stamp_sec if ref else None
    base_wall = ego[0].stamp_sec if ego else None
    base_rec = ref[0].stamp_sec if ref else None
    for leg in payload.get("legs", []):
        segs = leg.get("segments") or []
        first = segs[0] if segs else None
        wall_h = first.get("wall_handover") if first else None
        wall_end = leg.get("wall_end")
        gaps = [(float(g[0]), float(g[1])) for g in (leg.get("gaps") or [])]
        # the whole leg, holes included: where the ego came to rest at its goal
        # may well lie inside a hole it drove through (2026-08-24)
        # A reset is a window boundary (operator, 2026-11-10): each segment is
        # scored from its hand-over to the moment it ended, never across the
        # re-initialisation (the ego driven onto the recording, 3.5 s of lead).
        wins = _segment_windows(segs, wall_end)
        ego_leg = (
            [e for e in ego if wall_h <= e.stamp_sec <= wall_end and _in_windows(e.stamp_sec, wins)]
            if (wall_h is not None and wall_end is not None)
            else []
        )
        ego_slice = (
            [
                e for e in ego
                if wall_h <= e.stamp_sec <= wall_end and _in_windows(e.stamp_sec, wins)
                # inside a hole in the recording the ego drove by itself; its
                # stops there (a light frozen in the nearest frame) are no verdict
                and not (gaps and scene_time is not None and any(g0 <= scene_time(e.stamp_sec) < g1 for g0, g1 in gaps))
            ]
            if (wall_h is not None and wall_end is not None)
            else []
        )
        # A window opened at a planned cut starts at the recorded departure; the
        # recorded stand before it belongs to it too (the ego may still wait there).
        ref_from = min(float(leg["t_start"]), float(leg.get("cut_arrive_t") or leg["t_start"]))
        ref_slice = [r for r in ref if ref_from <= r.stamp_sec <= leg["t_end"]]
        goals_xy = [tuple(leg["arrival"][:2])] if leg.get("arrival") else []
        sg = stop_go(ego_slice, ref_slice, goals_xy) if ego_slice else None
        red_waits: list[list] = []
        if sg and sg["run"].get("stops_at") and ref_slice:
            # Stops the recording made too, where and when: planned, whatever
            # the zone bookkeeping says. So is a stop at a red light, whether or
            # not the recording caught that red.
            from dp_multi_eval.stop_go import WAIT_MIN_S, _stops
            rec_waits = [(a.stamp_sec, b.stamp_sec, a.x, a.y) for a, b in _stops(ref_slice, min_s=WAIT_MIN_S)]
            by_key = {
                (round(a.stamp_sec - ego_slice[0].stamp_sec), round(b.stamp_sec - a.stamp_sec, 1)): (a.stamp_sec, b.stamp_sec, a.x, a.y, a.yaw_rad)
                for a, b in _stops(ego_slice)
            }
            keyed = [(tuple(st), by_key.get((int(st[0]), float(st[1])))) for st in sg["run"]["stops_at"]]
            valued = [(st, v) for st, v in keyed if v]
            like = _waited_like_recording([v[:4] for _, v in valued], rec_waits, scene_time)
            red = _at_red_light([v for _, v in valued], light_states(), light_group) if valued else []
            excused = {st for (st, _), l, r in zip(valued, like, red) if l or r}
            red_waits = [list(st) for (st, _), r in zip(valued, red) if r]
            kept = [st for st, _ in keyed if st not in excused]
            if len(kept) != len(keyed):
                sg["run"]["stops_at"] = [list(st) for st in kept]
                sg["run"]["unplanned_stops"] = len(kept)
                sg["run"]["unplanned_stop_sec"] = round(sum(float(st[1]) for st in kept), 1)
        # Braking to a standstill under a red light ahead is not a start-brake
        # cycle against the model either (H95 odaiba: stopped 4 s at the red,
        # crept to the line, stopped again as it turned green — 2 cycles > 0 + 1).
        red_brakes = 0
        if sg and sg["run"].get("cycles_at") and ego_slice:
            base = ego_slice[0].stamp_sec
            cyc = [(base + float(c[0]), base + float(c[1]), float(c[2]), float(c[3]), float(c[4])) for c in sg["run"]["cycles_at"]]
            red_brakes = sum(_at_red_light(cyc, light_states(), light_group))
            sg["run"]["start_brake_cycles"] -= red_brakes
        diverged = [
            {
                "kind": str(s.get("ended_by", "")).split(":", 1)[-1],
                "t_rel": (round(float(s["t_end"]) - base_rec, 1) if (s.get("t_end") is not None and base_rec is not None) else None),
                "at": s.get("at"),
                "lag_sec": s.get("lag_sec"),
                "ego_v": s.get("ego_v"),
            }
            for s in segs
            if str(s.get("ended_by") or "").startswith("diverged")
        ]
        delay = None
        if leg.get("depart_t") is not None and first is not None and ego_slice:
            expected = float(first["wall_handover"]) + (float(leg["depart_t"]) - float(first["t_handover"]))
            move = next(
                (e.stamp_sec for e in ego_slice if e.speed_mps >= 0.5 and e.stamp_sec >= float(first["wall_handover"]) + 0.5),
                None,
            )
            delay = None if move is None else round(move - expected, 1)
        # A leg part that ends where a hole in the recording opens has no pose to
        # reach; only a part with an arrival pose can fail for not reaching it.
        reached = leg.get("arrived_t") is not None or leg.get("arrival") is None
        arrival_delay = (
            round(float(leg["arrived_t"]) - float(leg["arrive_t"]), 1)
            if (reached and leg.get("arrive_t") is not None)
            else None
        )
        # Divergence is not a verdict (operator, 2026-09-10): a leg is judged on
        # reaching its pose and on the stops it made; re-initialisations are
        # reported beside it.
        # A wait the ego drives on from is not a failure (operator, 2026-10-28):
        # stops of STOP_GATE_SEC or more the recording did not make (long_stops)
        # and departures DEPART_LAG late (did_not_depart) are reported, not
        # gating. H82 failed for giving way to an oncoming car 13 m and a few
        # seconds off where the recording gave way. Start-brake cycles are
        # reported too (operator, 2026-10-28): H39 gave way twice, 2.8 s and
        # 2.5 s, to an oncoming car right after its departure and completed the
        # leg — stopping is not a failure, being stuck is. A stall the harness
        # had to re-initialise (runs before 2026-10-28) still fails the leg.
        def _long(block):
            return [st for st in ((block or {}).get("stops_at") or []) if float(st[1]) >= STOP_GATE_SEC]
        long_stops = _long((sg or {}).get("run")) if sg else []
        hiccups = [st for st in (((sg or {}).get("run") or {}).get("stops_at") or []) if float(st[1]) < STOP_GATE_SEC]
        late = [d for d in (leg.get("departures") or []) if (d.get("delay_sec") or 0.0) >= 15.0]
        stalled = any(d.get("kind") == "stall" for d in diverged)
        stuck = _leg_stuck(ego_slice, ref_slice, leg.get("arrival"), thr, scene_time, light_states, light_group)
        # Stops the scene made: facing an NPC the shield held on the ego's way,
        # or past the recording's end. Reported, not gating; the leg is
        # inconclusive when one of them would have failed it.
        scene_limited: list[dict] = []
        if ego_slice and (stuck or long_stops):
            if "log" not in lights:
                lights["log"] = _shield_log(Path(bag_path).parent / SHIELD_LOG)
            holds, ends = lights["log"]
            base = ego_slice[0].stamp_sec
            for kind, block in (("stuck", stuck), ("long_stop", long_stops)):
                for st in block:
                    why = _scene_limited(base + float(st[0]), base + float(st[0]) + float(st[1]),
                                         ego_slice, ref_slice, holds, ends)
                    if why is not None:
                        scene_limited.append({"kind": kind, "t_rel": float(st[0]), "sec": float(st[1])} | why)
            limited = {(round(d["t_rel"], 1), round(d["sec"], 1)) for d in scene_limited if d["kind"] == "stuck"}
            stuck = [st for st in stuck if (round(float(st[0]), 1), round(float(st[1]), 1)) not in limited]
        inconclusive = any(d["kind"] == "stuck" for d in scene_limited)
        # Why the ego stood (waits.py, 2026-11-10): a red light, a walker or
        # cyclist on or entering its way, a vehicle on its way, cross traffic.
        # The stop rules and the lag judge only what is left unexplained: the
        # model is measured on correct driving, not on keeping up with the
        # recording (H78: 28.9 s for a crosswalk group, then failed).
        waits: list[dict] = []
        if ego_slice:
            from dp_multi_eval.stop_go import _stops as _standstills

            base = ego_slice[0].stamp_sec
            for a, b in _standstills(ego_slice, min_s=1.0):
                why = explain(a.stamp_sec, b.stamp_sec, a.x, a.y, a.yaw_rad)
                why["credit_sec"] = _wait_credit(why, a, b, ref_track, scene_time)
                waits.append({"t0": a.stamp_sec, "t_rel": round(a.stamp_sec - base, 1)} | why)
        # A mutual wait fails the leg (operator, 2026-11-10): the ego stood with
        # no reason while an NPC held for it — waiting for the ego, as a real
        # one would for a bus in its way — stood near (H78: 18.7 s mid-turn, a
        # car at its line 6 m off, the bus 2.5 m inside the recorded line).
        # Not a stand the leg asks for: the dwell at its goal, or a wait the
        # recording made at the same place and time.
        mutual_waits: list[dict] = []
        if ego_slice:
            from dp_multi_eval.stop_go import WAIT_MIN_S as _WAIT_MIN_S

            stamps_m = [e.stamp_sec for e in ego_slice]
            base = ego_slice[0].stamp_sec
            rec_stands = [(a.stamp_sec, b.stamp_sec, a.x, a.y) for a, b in _standstills(ref_slice, min_s=_WAIT_MIN_S)]
            arr = leg.get("arrival")
            for w_ in waits:
                if w_["sec"] < STOP_GATE_SEC:
                    continue
                e0 = ego_slice[min(bisect.bisect_left(stamps_m, w_["t0"]), len(ego_slice) - 1)]
                if arr and math.hypot(e0.x - float(arr[0]), e0.y - float(arr[1])) <= GOAL_ZONE_M:
                    continue
                if _waited_like_recording([(w_["t0"], w_["t0"] + w_["sec"], e0.x, e0.y)], rec_stands, scene_time)[0]:
                    continue
                sec, oids = mutual(w_["t0"], w_["t0"] + w_["sec"], e0.x, e0.y, e0.yaw_rad)
                if sec >= MUTUAL_WAIT_SEC:
                    mutual_waits.append({"t_rel": w_["t_rel"], "sec": w_["sec"], "mutual_sec": sec, "object_ids": oids})
        justified_stuck: list[list[float]] = []
        if stuck and ego_slice:
            base = ego_slice[0].stamp_sec
            stamps_s = [e.stamp_sec for e in ego_slice]
            kept_stuck = []
            for st in stuck:
                t0 = base + float(st[0])
                e0 = ego_slice[min(bisect.bisect_left(stamps_s, t0), len(ego_slice) - 1)]
                why = explain(t0, t0 + float(st[1]), e0.x, e0.y, e0.yaw_rad)
                if float(st[1]) - why["justified_sec"] < thr.stuck_duration_sec:
                    justified_stuck.append(list(st) + [why["justified_sec"]])
                else:
                    kept_stuck.append(st)
            stuck = kept_stuck
        # Arriving late is not a failure (operator, 2026-09-10): it is reported
        # with the arrival delay. Missing the goal is (operator, 2026-10-28):
        # driving within ARRIVE_M of it without ever arriving (a drive past), or
        # an intermediate leg's late grace running out (end_dist_m is set then).
        # A final leg has only FINAL_GRACE_SEC after the recording's end, so an
        # ego still short of it then is late, not missing (04-28 leg 3: 27.5 m
        # out at 1.6 m/s when the window closed).
        goal_missed = None
        if not reached and leg.get("arrival") and ego_leg:
            ax, ay = float(leg["arrival"][0]), float(leg["arrival"][1])
            if min(math.hypot(e.x - ax, e.y - ay) for e in ego_leg) <= ARRIVE_M:
                goal_missed = "drove_past"
            elif leg.get("end_dist_m") is not None:
                goal_missed = "late_grace_ran_out"
        # A window that closed on an ego standing, or far behind the
        # recording, short of the leg's end is a leg not completed (operator,
        # 2026-11-09): H95's 60 s recording ended with the ego waiting behind
        # a stopped truck the recorded bus had driven round, 22 s behind it,
        # and the leg passed — no stop reached the 45 s stuck rule.
        # Every window is judged where it closed (operator, 2026-11-10: a reset
        # is a window boundary, and closing by one is no verdict): the leg's
        # last window when the recording's end closed it short of the arrival,
        # and a window a reset closed with the ego behind the recording. Only
        # lag the waits do not explain counts.
        incomplete = None
        recording_ran_out = None
        # A window a planned cut closed (2026-11-10) has no arrival: it is judged
        # on the lag the waits do not explain, never on standing at the cut
        # (the recording stood there too).
        cut_closed = leg.get("closes") == "cut"
        end_pose = leg.get("arrival") or leg.get("close_pose")
        if not goal_missed and end_pose and ego_leg and wins:
            for k, (wa, wb, closed_by) in enumerate(wins):
                final = k == len(wins) - 1
                if final:
                    if cut_closed:
                        if leg.get("ended_by") != "cut":
                            continue
                    elif reached or leg.get("ended_by") != "window_end":
                        continue
                elif closed_by != "diverged:slow":
                    continue
                in_w = [e for e in ego_leg if wa <= e.stamp_sec <= wb]
                if not in_w:
                    continue
                last = in_w[-1]
                lag_end = None
                if ref_track is not None and scene_time is not None:
                    try:
                        lag_end = recorded_lag_sec(ref_track, last.x, last.y, last.yaw_rad, scene_time(last.stamp_sec), driving=True)
                    except Exception:  # noqa: BLE001
                        lag_end = None
                credit = sum(w["credit_sec"] for w in waits if wa <= w["t0"] <= wb)
                unexplained = None if lag_end is None else max(0.0, lag_end - credit)
                # standing where the window closed; a reset closes on a driving
                # ego (window_eval resets only a moving one)
                standing = final and not cut_closed and last.speed_mps < INCOMPLETE_SPEED_MPS
                ended = _recording_end_wall(in_w, scene_time, rec_end) if standing else None
                if standing and ended is not None:
                    # The recording ran out while the ego stood: past its end the
                    # scene keeps only what stood still, a stand there tells
                    # nothing. Judged on the live recording alone (operator,
                    # 2026-11-10: H78 caught a 57 s red the bus had passed on
                    # green, which turned green 4.8 s before the recording ended).
                    # The stand the window closed in: begun before the end (the
                    # live part of it alone is judged), or after it.
                    moving = [e for e in in_w if e.speed_mps >= INCOMPLETE_SPEED_MPS]
                    began = moving[-1].stamp_sec if moving else in_w[0].stamp_sec
                    if began < ended:
                        live = [e for e in in_w if e.stamp_sec <= ended]
                        a, e_end, stood = began, live[-1], "before_end"
                    else:
                        # it came to rest after the end, among what stood still —
                        # still real (a parked car is kept): that stand is judged
                        a, e_end, stood = began, last, "after_end"
                    idle = explain_tail(a, e_end.stamp_sec, e_end.x, e_end.y, e_end.yaw_rad)
                    standing = idle >= END_LIVE_UNJUSTIFIED_SEC
                    if not standing:
                        recording_ran_out = {"stood": stood, "idle_sec": round(idle, 1),
                                             "after_end_sec": round(last.stamp_sec - ended, 1)}
                elif standing:
                    tail = explain(max(wa, last.stamp_sec - 2.0), last.stamp_sec, last.x, last.y, last.yaw_rad)
                    standing = not (tail["sec"] > 0 and tail["justified_sec"] >= 0.5 * tail["sec"])
                if standing or (unexplained is not None and unexplained >= INCOMPLETE_LAG_SEC):
                    goal_missed = "did_not_complete"
                    incomplete = {
                        "window": k + 1,
                        "closed_by": closed_by,
                        "ego_v": round(last.speed_mps, 2),
                        "lag_sec": None if lag_end is None else round(lag_end, 1),
                        "credit_sec": round(credit, 1),
                        "unexplained_lag_sec": None if unexplained is None else round(unexplained, 1),
                        "dist_m": round(math.hypot(last.x - float(end_pose[0]), last.y - float(end_pose[1])), 1),
                    }
                    recording_ran_out = None
                    break
        # A contact the scene jump made (collision: scene_jump): from there on
        # the leg is the jump's, not the model's — the ego stands in the
        # object it was given. What follows is not judged; the leg is inconclusive.
        jump_contact = None
        if scene_jump_contacts and wins and ego_slice:
            hit = next((c for c in scene_jump_contacts if any(a <= float(c["time_sec"]) <= b for a, b, _ in wins)), None)
            if hit is not None:
                base = ego_slice[0].stamp_sec
                t_hit = float(hit["time_sec"]) - base
                jump_contact = {"t_rel": round(t_hit, 1), "object_id": hit["object_id"]}
                stuck = [st for st in stuck if float(st[0]) + float(st[1]) < t_hit]
                mutual_waits = [mw for mw in mutual_waits if float(mw["t_rel"]) + float(mw["sec"]) < t_hit]
                if goal_missed in ("did_not_complete", "late_grace_ran_out"):
                    goal_missed, incomplete = None, None
                recording_ran_out = None
                inconclusive = True
        # A window a recorded object drove into, or that the ego left the
        # recorded track in, cannot be judged by the recording: inconclusive.
        closed = [d["kind"] for d in diverged]
        reset_inconclusive = any(k in ("through", "lateral") for k in closed)
        passed = (
            not stalled
            and not stuck
            and not mutual_waits
            and not goal_missed
            and leg.get("ended_by") in ("window_end", "arrived", "cut")
        )
        inconclusive = inconclusive or (passed and (reset_inconclusive or recording_ran_out is not None))
        # Goal-stop error of the leg: where the ego came to rest nearest its
        # arrival pose, in the goal frame. A leg that stalled or never got there
        # has no stop to score (excluded_stuck), as goal_stop_precision does.
        goal_stop: dict[str, Any] = {"status": "excluded_stuck"}
        if cut_closed:
            goal_stop = {"status": "not_required_cut"}
        elif leg.get("drive_through"):
            goal_stop = {"status": "not_required_drive_through"}
        elif reached and not stalled and leg.get("arrival") and ego_leg:
            gx, gy, gyaw = (float(v) for v in leg["arrival"][:3])
            rest = [e for e in ego_leg if e.speed_mps < 0.2 and math.hypot(e.x - gx, e.y - gy) <= 12.0]
            if rest:
                best = min(rest, key=lambda e: math.hypot(e.x - gx, e.y - gy))
                lat, lon, pos, head = goal_frame_errors(best.x, best.y, best.yaw_rad, gx, gy, gyaw)
                goal_stop = {
                    "status": "ok", "lateral_m": round(lat, 2), "longitudinal_m": round(lon, 2),
                    "position_m": round(pos, 2), "heading_deg": round(head, 1),
                }
                # Stopping more than the goal tolerance off the leg's goal fails
                # the leg (operator, 2026-10-28), as the whole-bag goal check
                # did before legs existed — at a bus stop. A leg goal where the
                # recording only waited in traffic has no stopping point to hit:
                # "if it must go and cannot, it fails; if it stops there, it
                # stops" (operator, 2026-10-28; H81 leg 2 came to rest 9.5 m
                # behind where the recorded bus had waited, 70 m from any
                # stop). There the error is reported. Without a stop list every
                # goal is judged.
                goal_stop["flagged"] = bool(
                    pos > thr.goal_tolerance_m
                    or abs(lat) > thr.goal_lateral_tolerance_m
                    or abs(lon) > thr.goal_longitudinal_tolerance_m
                )
                stop = _bus_stop_at(bus_stops, gx, gy)
                goal_stop["bus_stop"] = stop
                goal_stop["gating"] = bus_stops is None or stop is not None
                if goal_stop["gating"]:
                    passed = passed and not goal_stop["flagged"]
            else:
                goal_stop = {"status": "no_stop_near_goal"}
        # Pace of the leg: the ego's drive from set-off to arrival against the
        # recording's. reference_speed's time_ratio spans the whole output bag
        # (start-up, holds, the final grace) and swung 0.96x-1.29x between two
        # near-identical drives (2026-09-15); this ratio excludes the dwells.
        drive = None
        if ego_leg and leg.get("arrival") and leg.get("arrive_t") is not None and first is not None:
            ax, ay = float(leg["arrival"][0]), float(leg["arrival"][1])
            move = next((e.stamp_sec for e in ego_leg if e.speed_mps >= 0.5), None)
            arrive_w = (
                next((e.stamp_sec for e in ego_leg if e.stamp_sec > move and math.hypot(e.x - ax, e.y - ay) <= 5.0), None)
                if move is not None else None
            )
            rec_start = float(leg["depart_t"]) if leg.get("depart_t") is not None else float(first["t_handover"])
            rec_drive = float(leg["arrive_t"]) - rec_start
            if move is not None and arrive_w is not None and rec_drive > 1.0 and arrive_w > move:
                drive = {"ego_sec": round(arrive_w - move, 1), "rec_sec": round(rec_drive, 1), "ratio": round((arrive_w - move) / rec_drive, 2)}
        out["legs"].append(
            {
                "index": leg.get("index"),
                "parent": leg.get("parent") or leg.get("index"),
                "opens": leg.get("opens") or "goal",
                "closes": leg.get("closes") or "goal",
                "placed": any(sg_.get("placed") for sg_ in segs),
                "seamless": any(sg_.get("seamless") and not sg_.get("scene_jumped") for sg_ in segs),
                "scene_jumped": next((sg_.get("skipped_sec") for sg_ in segs if sg_.get("scene_jumped")), None),
                "cut_pass": _cut_pass(leg, ego, light_red),
                "scene_jump_contact": jump_contact,
                "goal_stop": goal_stop,
                "drive": drive,
                "kind": leg.get("kind"),
                "t_start_rel": (round(float(leg["t_start"]) - base_rec, 1) if base_rec is not None else None),
                "t_end_rel": (round(float(leg["t_end"]) - base_rec, 1) if base_rec is not None else None),
                "ended_by": leg.get("ended_by"),
                "reached": reached,
                "arrival_delay_sec": arrival_delay,
                "end_dist_m": leg.get("end_dist_m"),
                "gaps_sec": [round(g1 - g0, 1) for g0, g1 in gaps],
                "segments": len(segs),
                "windows": [
                    {"t_rel": (round(a - base_wall, 1) if base_wall is not None else None),
                     "sec": round(b - a, 1), "closed_by": why}
                    for a, b, why in wins
                ],
                "diverged": diverged,
                "ahead": leg.get("ahead"),
                "stall": leg.get("stall"),
                "stalled": stalled,
                "stuck": stuck,
                "stuck_justified": justified_stuck,
                "mutual_waits": mutual_waits,
                "waits": [
                    {k: w[k] for k in ("t_rel", "sec", "justified_sec", "credit_sec", "reasons")}
                    for w in waits if w["sec"] >= STOP_GATE_SEC
                ],
                "goal_missed": goal_missed,
                "incomplete": incomplete,
                "recording_ran_out": recording_ran_out,
                "scene_limited": scene_limited,
                "inconclusive": inconclusive,
                "did_not_depart": [
                    {"t_rel": (round(float(d["t"]) - base_rec, 1) if base_rec is not None else None), "at": d.get("at"), "delay_sec": d.get("delay_sec")}
                    for d in late
                ],
                "departure_delay_sec": delay,
                "departures": [
                    {"t_rel": (round(float(d["t"]) - base_rec, 1) if base_rec is not None else None), "at": d.get("at"), "delay_sec": d.get("delay_sec")}
                    for d in (leg.get("departures") or [])
                ],
                "stop_go": ({k: sg["run"][k] for k in ("unplanned_stops", "unplanned_stop_sec", "start_brake_cycles")} | {"flagged": sg.get("flagged"), "long_stops": long_stops, "hiccups": hiccups, "red_light_waits": red_waits, "red_light_brakes": red_brakes}) if sg else None,
                "passed": bool(passed),
                "wall_start_rel": (round(float(leg["wall_start"]) - base_wall, 1) if (leg.get("wall_start") is not None and base_wall is not None) else None),
            }
        )
    out["legs_total"] = len(out["legs"])
    out["legs_passed"] = sum(1 for l in out["legs"] if l["passed"])
    out["legs_inconclusive"] = sum(1 for l in out["legs"] if l.get("inconclusive"))
    out["cut_placements"] = sum(1 for l in out["legs"] if l.get("opens") == "cut" and l.get("placed"))
    out["cuts_seamless"] = sum(1 for l in out["legs"] if l.get("opens") == "cut" and l.get("seamless"))
    out["cut_scene_jumps"] = sum(1 for l in out["legs"] if l.get("scene_jumped") is not None)
    out["cut_red_passes"] = sum(1 for l in out["legs"] if (l.get("cut_pass") or {}).get("light") == "red")
    drives = [l["drive"] for l in out["legs"] if l.get("drive")]
    rec_total = sum(d["rec_sec"] for d in drives)
    out["drive_ratio"] = round(sum(d["ego_sec"] for d in drives) / rec_total, 2) if rec_total > 0 else None
    out["flagged"] = out["legs_total"] > 0 and out["legs_passed"] < out["legs_total"]
    return out


def metric_reference_speed(
    series: BagSeries,
    reference: BagSeries | None,
    thr: Thresholds,
) -> dict[str, Any]:
    """Compare the replay's speed against the recording it replays, place by place.

    Every other metric asks whether the run broke a rule. This one asks whether it
    drove the road the way the road was driven — which is the question a replay is
    uniquely able to answer, and the one that was going unasked. A run measured here
    covered its route in 131 s where the recording took 240 s, and the only sign of
    it anywhere in the results was a collision flag: an NPC cut in front of a bus
    that, in the recording, was crawling at 1.75 m/s and had room to spare.

    Samples pair by position rather than by time. The replay arrives at a place at
    its own moment, so pairing on the clock would compare unrelated stretches of road.
    """
    out: dict[str, Any] = {
        "flagged": False,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["reference_speed"],
    }
    if reference is None or not reference.ego:
        out["status"] = "no_reference"
        return out
    if not series.ego:
        out["status"] = "no_ego"
        return out

    ref = reference.ego
    diffs: list[float] = []
    matched = 0
    # Nearest recorded sample by position. The plain double loop was 22 s per
    # run (21k replay x 12k recorded samples); the tree answers in under a second.
    from scipy.spatial import cKDTree

    tree = cKDTree([(r.x, r.y) for r in ref])
    dists, idxs = tree.query([(e.x, e.y) for e in series.ego], distance_upper_bound=thr.ref_speed_match_radius_m)
    for e, d, i in zip(series.ego, dists, idxs):
        if not d <= thr.ref_speed_match_radius_m:
            continue
        best = ref[int(i)]
        matched += 1
        if e.speed_mps < thr.ref_speed_min_speed_mps and best.speed_mps < thr.ref_speed_min_speed_mps:
            continue
        diffs.append(e.speed_mps - best.speed_mps)

    out["matched_samples"] = matched
    out["compared_samples"] = len(diffs)
    if not diffs:
        out["status"] = "no_overlap"
        return out

    ordered = sorted(diffs)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    mean = sum(diffs) / len(diffs)

    def _pct(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = q * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    p90 = _pct(0.90)
    p10 = _pct(0.10)

    def _travel(samples: list[Any]) -> tuple[float, float]:
        dist = sum(
            math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(samples, samples[1:])
        )
        span = samples[-1].stamp_sec - samples[0].stamp_sec
        return dist, span

    ego_dist, ego_span = _travel(series.ego)
    ref_dist, ref_span = _travel(ref)

    out.update(
        {
            "median_diff_mps": median,
            "mean_diff_mps": mean,
            "p90_diff_mps": p90,
            "p10_diff_mps": p10,
            "max_diff_mps": max(diffs, key=abs),
            "ego_mean_speed_mps": (ego_dist / ego_span) if ego_span > 0 else 0.0,
            "reference_mean_speed_mps": (ref_dist / ref_span) if ref_span > 0 else 0.0,
            "ego_duration_sec": ego_span,
            "reference_duration_sec": ref_span,
            "ego_distance_m": ego_dist,
            "reference_distance_m": ref_dist,
            "time_ratio": (ego_span / ref_span) if ref_span > 0 else None,
            "flagged": (
                abs(median) > thr.ref_speed_median_tolerance_mps
                or max(abs(p90), abs(p10)) > thr.ref_speed_p90_tolerance_mps
            ),
        }
    )
    return out


def metric_stack_intervention(
    bag_path: Path,
    topics: TopicSet,
    thr: Thresholds,
    map_path: Path | None = None,
) -> dict[str, Any]:
    """How much of the run the planning stack drove instead of the model.

    The model's plan and the trajectory the controller follows are not the same
    thing. ``autoware_trajectory_modifier`` sits between them, and its
    StopPointFixer replaces the plan with a bare stop point at the ego whenever
    the ego is stopped and the plan does not clear 0.25 m/s inside its first
    second — the profile of any vehicle pulling away from rest. The replacement
    is three points spanning a metre, which the mode transition manager cannot
    locate the ego on ("closest point not found"), so a run that stops can be
    refused permission to start again.

    Nothing in the results showed this. The run failed as "stuck", which names
    the symptom and points at the model, while the recorded bag already held the
    evidence: the trajectories the controller received are in it, and a stop
    point is three points where a plan is eighty.

    Reported, never gating — the stack's behaviour is not the model's fault, and
    a run that legitimately waits at a light also collects stop points. Read it
    when a run flags as stuck: it separates "the model stopped" from "the model
    was stopped".

    Thresholds come from the observed split rather than a guess. Across the
    Hiratsuka bags a completed run spent 7.7% of its cycles on stop points with
    the longest unbroken stretch at 7.4 s, while the two runs that stalled sat at
    69.8% / 79.2 s and 60%+ / 66.4 s. Anything past a quarter of the run, or one
    unbroken stretch beyond 20 s, is the latched case rather than traffic.
    """
    out: dict[str, Any] = {
        "flagged": False,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["stack_intervention"],
    }
    try:
        from dp_multi_eval.bag_reader import open_bag_reader, _type_map
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from rosbag2_py import StorageFilter
    except Exception:  # noqa: BLE001
        out["status"] = "unavailable"
        out["detail"] = "rosbag2 not importable"
        return out

    traj_topic = topics.trajectory
    ego_topic = topics.ego_pose
    try:
        reader = open_bag_reader(bag_path)
        types = _type_map(reader)
        wanted = [t for t in (traj_topic, ego_topic) if t in types]
        if traj_topic not in types:
            out["status"] = "unavailable"
            out["detail"] = f"{traj_topic} was not recorded"
            return out
        traj: list[tuple[float, int]] = []          # (t, point count)
        speeds: list[tuple[float, float]] = []      # (t, speed)
        # 5 Hz is plenty: a hold is a run of stubs with gaps under stub_gap_sec
        # (3 s), and the ratio is invariant under uniform thinning. Reading
        # every 10 Hz trajectory cost 13 s of deserialization per run.
        from dp_multi_eval.bag_reader import iter_topic_messages

        for topic, t, msg in iter_topic_messages(bag_path, wanted, {traj_topic: 0.2, ego_topic: 0.2}):
            if topic == traj_topic:
                traj.append((t, len(getattr(msg, "points", []))))
            else:
                tw = msg.twist.twist.linear
                speeds.append((t, math.hypot(tw.x, tw.y)))
    except Exception as exc:  # noqa: BLE001
        out["status"] = "unavailable"
        out["detail"] = f"could not read {bag_path}: {exc}"
        return out

    if not traj:
        out["status"] = "unavailable"
        out["detail"] = "no trajectory messages in the bag"
        return out

    stub_times = [t for t, n in traj if n <= thr.stub_max_points]
    stub_ratio = len(stub_times) / len(traj)

    # Unbroken stretches of stop points. A gap wider than the publish period
    # means the stack handed the controller a real plan in between.
    longest = 0.0
    held: list[tuple[float, float]] = []
    if stub_times:
        start = prev = stub_times[0]
        for t in stub_times[1:]:
            if t - prev > thr.stub_gap_sec:
                held.append((start, prev))
                start = t
            prev = t
        held.append((start, prev))
        longest = max(b - a for a, b in held)
    windows = len(held)

    out.update(
        {
            "trajectory_messages": len(traj),
            "stop_point_messages": len(stub_times),
            "stop_point_ratio": round(stub_ratio, 4),
            "longest_forced_stop_sec": round(longest, 1),
            "forced_stop_windows": windows,
            # Spans where the controller was being handed stop points. Other
            # metrics need them: what happens to an ego that is not being driven
            # is not the model's doing.
            "held_windows": [[round(a, 3), round(b, 3)] for a, b in held],
            "flagged": bool(
                stub_ratio > thr.stub_ratio_flag
                or longest > thr.forced_stop_flag_sec
            ),
        }
    )
    if speeds:
        moving = [s for _, s in speeds if s > thr.stub_moving_speed_mps]
        out["moving_sample_ratio"] = round(len(moving) / len(speeds), 4)

    # Which light the ego was held at, when the replayed signal state was
    # recorded. A stationary ego at a signalized turn is two different verdicts:
    # waiting at red is correct driving, and sitting through green while the
    # controller receives stop stubs is the departure hold. Measured on one run
    # before this existed: a 31.6 s wait of which 31 s were green and 28 s of
    # those green seconds were stubbed — evidence that had to be dug out by
    # hand each time someone asked why the bus would not turn.
    try:
        green = _green_latched_seconds(
            bag_path, topics, thr, out.get("held_windows") or [], map_path
        )
        if green is not None:
            out["green_latched_sec"] = round(green, 1)
    except Exception:  # noqa: BLE001
        pass

    # Exact attribution, when the modifier's input was recorded. The count above
    # is the symptom seen from the output side; this recomputes the plugin's own
    # decision per cycle and says which of its two rules each replacement rests
    # on — including how many rest on one rule alone, which is what changing a
    # flag would actually change. Best-effort: a bag without the input topic, or
    # a message layout that moved, must not cost a run its metrics.
    # Only when there is something to attribute: the replay deserializes every
    # candidate-trajectory message (19 s per run) to explain stubs, and a run
    # without stubs has none.
    if not stub_times:
        out["attribution"] = {"status": "no_stubs"}
        return out
    try:
        from dp_multi_eval.fixer_replay import replay_fixer

        attribution = replay_fixer(
            bag_path, topics.modifier_input_trajectories, ego_topic
        )
        if attribution.get("status") == "ok":
            out["attribution"] = attribution
    except Exception as exc:  # noqa: BLE001
        out["attribution"] = {"status": "unavailable", "detail": str(exc)}
    return out


def _green_latched_seconds(
    bag_path: Path,
    topics: TopicSet,
    thr: Thresholds,
    held_windows: list,
    map_path: Path | None,
) -> float | None:
    """Seconds of hold spent under a green light, or None when unmeasurable."""
    if not held_windows or map_path is None:
        return None
    import bisect

    from dp_multi_eval.bag_reader import _type_map, open_bag_reader
    from rclpy.serialization import deserialize_message
    from rosbag2_py import StorageFilter
    from rosidl_runtime_py.utilities import get_message

    sig_topic = topics.traffic_signals
    reader = open_bag_reader(bag_path)
    types = _type_map(reader)
    if sig_topic not in types:
        return None
    reader.set_filter(StorageFilter(topics=[sig_topic, topics.ego_pose]))
    sig: list[tuple[float, dict[int, int]]] = []
    ego: list[tuple[float, float, float]] = []
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        t = stamp_ns * 1e-9
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == sig_topic:
            states = {
                g.traffic_light_group_id: g.elements[0].color
                for g in msg.traffic_light_groups
                if g.elements
            }
            if states:
                sig.append((t, states))
        else:
            pos = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            ego.append((t, pos.x, pos.y, yaw))
    if not sig or not ego:
        return None

    # The light that governs each hold: the traffic_light regulatory element of
    # the lanelet nearest where the ego stood.
    lmap = _load_lanelet_map(map_path)
    if lmap is None:
        return None

    GREEN = 3  # autoware_perception_msgs/TrafficLightElement.GREEN
    try:
        from autoware_perception_msgs.msg import TrafficLightElement

        GREEN = int(TrafficLightElement.GREEN)
    except Exception:  # noqa: BLE001
        pass

    ego_t = [e[0] for e in ego]
    sig_t = [x[0] for x in sig]
    total = 0.0
    for start, end in held_windows:
        i = min(bisect.bisect_left(ego_t, float(start)), len(ego) - 1)
        _, ex, ey, eyaw = ego[i]
        group = _light_ahead(lmap, ex, ey, eyaw)
        if group is None:
            continue
        t = float(start)
        while t < float(end):
            j = min(bisect.bisect_left(sig_t, t), len(sig) - 1)
            if sig[j][1].get(group) == GREEN:
                total += 0.5
            t += 0.5
    return total


def metric_goal_stop(
    series: BagSeries,
    goal_x: float,
    goal_y: float,
    goal_yaw: float,
    thr: Thresholds,
    *,
    stuck: dict[str, Any] | None = None,
    required: bool = True,
) -> dict[str, Any]:
    """Average stop precision (position / lateral / longitudinal / heading) near goal.

    Scenes flagged as stuck (away from goal) are excluded from goal precision.
    ``required=False`` when the recording was cut while moving and the run
    ends drive-through at the cut pose: there is no stop to measure.
    """
    empty = {
        "flagged": True,
        "position_error_m": float("nan"),
        "lateral_m": float("nan"),
        "longitudinal_m": float("nan"),
        "abs_lateral_m": float("nan"),
        "abs_longitudinal_m": float("nan"),
        "heading_error_deg": float("nan"),
        "stop_time_sec": float("nan"),
        "stop_speed_mps": float("nan"),
        "sample_count": 0,
        "duration_at_goal_sec": 0.0,
        "status": "no_ego_data",
        "description": METRIC_DESCRIPTIONS["goal_stop_precision"],
    }
    if not series.ego:
        return empty

    if not required:
        return {**empty, "flagged": False, "status": "not_required_cut_recording"}
    if stuck and stuck.get("flagged"):
        return {
            **empty,
            "flagged": False,
            "status": "excluded_stuck",
        }

    at_goal: list[EgoSample] = []
    for ego in series.ego:
        if ego.speed_mps > thr.goal_stop_speed_mps:
            continue
        _lat, _lon, pos, _heading = goal_frame_errors(
            ego.x, ego.y, ego.yaw_rad, goal_x, goal_y, goal_yaw
        )
        if pos <= thr.goal_tolerance_m:
            at_goal.append(ego)

    if not at_goal:
        return {
            **empty,
            "status": "goal_not_reached",
        }

    laterals: list[float] = []
    longitudinals: list[float] = []
    positions: list[float] = []
    headings: list[float] = []
    speeds: list[float] = []
    for ego in at_goal:
        lat, lon, pos, heading = goal_frame_errors(
            ego.x, ego.y, ego.yaw_rad, goal_x, goal_y, goal_yaw
        )
        laterals.append(lat)
        longitudinals.append(lon)
        positions.append(pos)
        headings.append(heading)
        speeds.append(ego.speed_mps)

    n = len(at_goal)
    avg_pos = sum(positions) / n
    avg_lat = sum(laterals) / n
    avg_lon = sum(longitudinals) / n
    avg_abs_lat = sum(abs(v) for v in laterals) / n
    avg_abs_lon = sum(abs(v) for v in longitudinals) / n
    avg_heading = sum(headings) / n
    avg_speed = sum(speeds) / n
    duration_at_goal = at_goal[-1].stamp_sec - at_goal[0].stamp_sec

    flagged = (
        avg_pos > thr.goal_tolerance_m
        or avg_abs_lat > thr.goal_lateral_tolerance_m
        or avg_abs_lon > thr.goal_longitudinal_tolerance_m
    )

    return {
        "flagged": flagged,
        "position_error_m": avg_pos,
        "lateral_m": avg_lat,
        "longitudinal_m": avg_lon,
        "abs_lateral_m": avg_abs_lat,
        "abs_longitudinal_m": avg_abs_lon,
        "heading_error_deg": avg_heading,
        "max_position_error_m": max(positions),
        "max_abs_lateral_m": max(abs(v) for v in laterals),
        "max_abs_longitudinal_m": max(abs(v) for v in longitudinals),
        "max_heading_error_deg": max(abs(h) for h in headings),
        "stop_time_sec": at_goal[-1].stamp_sec,
        "stop_speed_mps": avg_speed,
        "sample_count": n,
        "duration_at_goal_sec": duration_at_goal,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["goal_stop_precision"],
    }


def _envelope_lookup(reference: BagSeries | None, thr: Thresholds):
    """Nearest recorded pose finder, or None when there is no recording to compare to."""
    if reference is None or not reference.ego:
        return None
    # Vectorised: this is called once per ego sample and once per contact, and a
    # Python loop over both made a single run's metrics take minutes.
    import numpy as np

    rx = np.fromiter((r.x for r in reference.ego), dtype=float, count=len(reference.ego))
    ry = np.fromiter((r.y for r in reference.ego), dtype=float, count=len(reference.ego))
    rv = np.fromiter(
        (r.speed_mps for r in reference.ego), dtype=float, count=len(reference.ego)
    )

    def nearest(x: float, y: float) -> tuple[float, float]:
        d2 = (rx - x) ** 2 + (ry - y) ** 2
        i = int(np.argmin(d2))
        return float(math.sqrt(d2[i])), float(rv[i])

    return nearest


def _in_envelope(ego: Any, nearest, thr: Thresholds) -> tuple[bool, float, float]:
    """Whether the replay is a fair test at this sample.

    Position only, deliberately. The replay is position-synced: the NPCs shown
    while the ego is at X are the ones the recorded ego met at X, so the
    configuration around the ego stays self-consistent however fast it got there.
    What breaks that is the ego being somewhere the recording never went, where
    the nearest recorded frame is a poor stand-in and its traffic belongs to a
    different piece of road.

    An earlier version also excluded samples where the ego's speed differed from
    the recorded speed, which was wrong: driving the road differently is the
    behaviour under test, not a defect in the test. It would have excused the
    real finding in 2026-04-28, where the ego noses out of a stop at 0.34 m/s
    into a car crossing 9 degrees off its bow -- the recording waited for that
    car and pulled out behind it. The pace difference is reference_speed's to
    report; it is not grounds for ignoring what the ego did.
    """
    if nearest is None:
        return True, float("nan"), float("nan")
    offset, ref_speed = nearest(ego.x, ego.y)
    return offset <= thr.envelope_offset_m, offset, ref_speed


def metric_replay_validity(
    series: BagSeries,
    reference: BagSeries | None,
    thr: Thresholds,
) -> dict[str, Any]:
    """How much of this run the replay was in a position to judge.

    Every other metric assumes the scene around the ego is the scene the
    recording had there. That assumption fails silently once the ego leaves the
    recorded corridor or runs at a different pace, and its failure looks exactly
    like a planner defect: replayed vehicles and the ego pass through each other,
    because neither side can react. Coverage makes the assumption visible.
    """
    out: dict[str, Any] = {
        "flagged": False,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["replay_validity"],
    }
    nearest = _envelope_lookup(reference, thr)
    if nearest is None:
        out["status"] = "no_reference"
        return out
    ego_ds = downsample_ego(series.ego, thr.sample_dt)
    if not ego_ds:
        out["status"] = "no_ego"
        return out
    inside = 0
    offsets: list[float] = []
    for ego in ego_ds:
        ok, offset, _ = _in_envelope(ego, nearest, thr)
        inside += 1 if ok else 0
        if offset == offset:  # not NaN
            offsets.append(offset)
    coverage = inside / len(ego_ds)
    offsets.sort()
    out.update(
        {
            "coverage": round(coverage, 4),
            "samples": len(ego_ds),
            "median_path_offset_m": round(offsets[len(offsets) // 2], 2) if offsets else None,
            "max_path_offset_m": round(offsets[-1], 2) if offsets else None,
            "flagged": coverage < thr.envelope_min_coverage,
        }
    )
    return out


def _collision_class(
    ego: Any,
    obj: Any,
    thr: Thresholds,
    held_windows: list[tuple[float, float]],
    nearest=None,
    obj_speed: float | None = None,
) -> str:
    """Who is answerable for this contact.

    Counting contacts treats three different things as one. Measured on the
    Hiratsuka bags, 49 of one run's 50 collision frames happened while the ego
    was standing still with the stack handing the controller stop points: the
    scene replay advances on wall time whenever the ego is stopped, so a
    recorded vehicle simply drove through a car that had no way to move. The
    model was not driving, and scoring it for that says nothing about the model.

    The remaining case is the one worth gating on: an ego under its own control
    closing on something in front of it.
    """
    for start, end in held_windows:
        if start <= ego.stamp_sec <= end and ego.speed_mps <= thr.stub_moving_speed_mps:
            return "not_in_control"
    if ego.speed_mps <= thr.stub_moving_speed_mps:
        # Stopped for its own reasons; something else arrived at it.
        return "struck_while_stopped"
    if obj_speed is not None and obj_speed < STATIONARY_OBJ_MPS:
        # A moving ego meeting a standing object drove into it, wherever the
        # ego was and from whichever side: the object had nothing to react to.
        # Off the recorded path (a parked car) or alongside (a scrape) this was
        # out_of_envelope / cut_in and failed nothing (operator, 2026-10-28).
        return "ego_caused"
    ok, _, _ = _in_envelope(ego, nearest, thr)
    if not ok:
        # The ego is not where the recorded traffic was arranged for. Replayed
        # vehicles cannot react and the ego is meeting cars that belonged to a
        # different moment, so the overlap says nothing about the planner.
        return "out_of_envelope"
    bearing = math.degrees(
        abs(_wrap_pi(math.atan2(obj.y - ego.y, obj.x - ego.x) - ego.yaw_rad))
    )
    return "ego_caused" if bearing <= thr.collision_ahead_cone_deg else "cut_in"


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


THROUGH_EGO_MATCH_M = 15.0
STATIONARY_OBJ_MPS = 0.5  # an object slower than this cannot have run into a moving ego
STOP_GATE_SEC = 3.0  # an unplanned stop this long fails a leg; shorter ones are hiccups
WAITED_LIKE_M = 12.0  # a run stop this close to a recorded wait, overlapping it in scene time, is that wait


def _wait_credit(why: dict, start, end, ref_track, scene_time) -> float:
    """Seconds of lag a stop may take off: the lag the recording gained on the
    standing ego over the stop (its driving time meanwhile; a wait the
    recording made too adds none), in the stop's justified share. Without
    the recording's timeline, the justified seconds. ``start``/``end`` are the
    stop's first and last ego samples."""
    sec = float(why.get("sec") or 0.0)
    just = float(why.get("justified_sec") or 0.0)
    if just <= 0.0 or sec <= 0.0:
        return 0.0
    if ref_track is None or scene_time is None:
        return round(just, 1)
    from dp_multi_eval.lag import recorded_lag_sec

    try:
        lag_a = recorded_lag_sec(ref_track, start.x, start.y, start.yaw_rad, scene_time(start.stamp_sec), driving=True)
        lag_b = recorded_lag_sec(ref_track, end.x, end.y, end.yaw_rad, scene_time(end.stamp_sec), driving=True)
    except Exception:  # noqa: BLE001
        return round(just, 1)
    if lag_a is None or lag_b is None:
        return round(just, 1)
    gained = max(0.0, float(lag_b) - max(float(lag_a), 0.0))
    return round(min(gained, sec) * just / sec, 1)


def _waited_like_recording(run_stops, rec_waits, scene_time) -> list[bool]:
    """For each run stop ``(t0_wall, t1_wall, x, y)``: did the recording stand
    within WAITED_LIKE_M of it at the scene time of the stop? ``rec_waits`` are
    ``(t0_rec, t1_rec, x, y)``. A red light waited out in three creeping pieces
    (H82, 2026-09-15: 5.6 + 2.0 + 16.9 s at 0.5-4 m from where the
    recording stood 6-39.5 s) is one planned wait, not two unplanned stops —
    stop_go's one-stop-per-zone rule exists for a stall after a departure and
    must not fail a leg for queueing at a light."""
    out = []
    for t0, t1, x, y in run_stops:
        s0, s1 = (scene_time(t0), scene_time(t1)) if scene_time is not None else (None, None)
        out.append(
            s0 is not None and s1 is not None and any(
                s0 <= rb and s1 >= ra and math.hypot(x - rx, y - ry) <= WAITED_LIKE_M
                for ra, rb, rx, ry in rec_waits
            )
        )
    return out


GOAL_ZONE_M = 12.0  # standing this close to the leg goal is the goal stop's to judge, not stuck
INCOMPLETE_SPEED_MPS = 0.5  # the window closed on an ego slower than this short of the leg's end: not completed
INCOMPLETE_LAG_SEC = 15.0   # ...or on one this far behind the recording
END_LIVE_UNJUSTIFIED_SEC = 5.0  # ...unless the recording ran out first: then only a stand this long without
                                # a reason while the recording still played fails (a start on green takes 2-4 s)


def _recording_end_wall(samples, scene_time, rec_end) -> float | None:
    """Wall time the scene reached the recording's end within ``samples``, or
    None when it never did (or the run has no scene map)."""
    if scene_time is None or rec_end is None:
        return None
    for e in samples:
        try:
            if scene_time(e.stamp_sec) >= rec_end - 0.1:
                return e.stamp_sec
        except Exception:  # noqa: BLE001
            return None
    return None


ARRIVE_M = 5.0  # window_eval.ARRIVE_M: stopping this close to the leg goal is arriving


SHIELD_LOG = "shield_log.jsonl"  # scene_driver.ShieldLog
SCENE_LIMIT_FRAC = 0.5  # a stop is the scene's limit when its cause stood this much of it
WAY_AHEAD_M = 27.5  # the ego's way: its recorded path from its rear axle, front overhang (7.56 m) + 20 m
EGO_HALF_W = 1.45


def metric_scene_audit(bag_path: Path, series: BagSeries) -> dict[str, Any]:
    """The scene audit's findings (shield_log.jsonl), on the run's clock."""
    from dp_multi_eval.scene_audit import summarize

    path = Path(bag_path).parent / SHIELD_LOG
    out = {"status": "ok", "description": METRIC_DESCRIPTIONS["scene_audit"], "flagged": False,
           "counts": {}, "defects": 0, "events": []}
    if not path.is_file():
        return out | {"status": "no_log"}
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if "audit" in rec:
            records.append(rec)
    t0 = series.ego[0].stamp_sec if series.ego else 0.0
    return out | summarize(records, t0)


def _shield_log(path: Path) -> tuple[list[dict], list[tuple[float, float]]]:
    """NPCs the shield held for the ego (or its plan), by time, and the stretches
    the scene stood at the recording's end — both in the bag's clock."""
    holds, ends, open_end = [], [], None
    if not path.is_file():
        return holds, ends
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if rec.get("event") == "recording_end":
            open_end = float(rec["t"])
        elif rec.get("event") == "recording_resume":
            if open_end is not None:
                ends.append((open_end, float(rec["t"])))
            open_end = None
        elif rec.get("reason") in ("ego", "plan", "deadlock"):
            holds.append(rec)
    if open_end is not None:
        ends.append((open_end, float("inf")))
    holds.sort(key=lambda r: r["t"])
    return holds, ends


def _scene_limited(t0: float, t1: float, ego, ref, holds, ends) -> dict | None:
    """Why a stop from t0 to t1 is the scene's limit, or None.

    after_recording: the scene stood at the recording's end for most of it.
    held_npc: for most of it an NPC the shield could not resolve stood on the
    ego's way (its recorded path ahead; shield reason "deadlock": no detour
    round the ego fitted). A real cyclist rides round a bus that stands in its
    way; the replay could not (H78, 2026-11-09: 19.5 s mid-turn, the cyclist
    held 2 m off the turning bus). An NPC merely waiting for the ego (reasons
    "ego", "plan") is no limit: a real one waits for a bus in its way too, and
    the bus has to go — the mutual wait is the model's (operator, 2026-11-10)."""
    dur = t1 - t0
    if dur <= 0.0:
        return None
    after = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in ends)
    if after >= SCENE_LIMIT_FRAC * dur:
        return {"cause": "after_recording"}
    if not holds or not ref or not ego:
        return None
    hold_t = [h["t"] for h in holds]
    ego_t = [e.stamp_sec for e in ego]
    probes, blocked, by = 0, 0, {}
    t = t0
    while t <= t1:
        probes += 1
        e = ego[min(bisect.bisect_left(ego_t, t), len(ego) - 1)]
        k = min(range(len(ref)), key=lambda i: (ref[i].x - e.x) ** 2 + (ref[i].y - e.y) ** 2)
        way, run = [(e.x, e.y)], 0.0
        for r in ref[k:]:
            run += math.hypot(r.x - way[-1][0], r.y - way[-1][1])
            way.append((r.x, r.y))
            if run >= WAY_AHEAD_M:
                break
        lo, hi = bisect.bisect_left(hold_t, t - 0.6), bisect.bisect_right(hold_t, t + 0.6)
        hit = None
        for h in holds[lo:hi]:
            if h.get("reason") != "deadlock":
                continue
            reach = EGO_HALF_W + float(h.get("w") or 1.0) / 2.0 + 0.5
            if any(_point_segment(h["x"], h["y"], a, b) <= reach for a, b in zip(way, way[1:])):
                hit = h["id"]
                break
        if hit is not None:
            blocked += 1
            by[hit] = by.get(hit, 0) + 1
        t += 0.5
    if probes and blocked >= SCENE_LIMIT_FRAC * probes:
        return {"cause": "held_npc", "object_id": max(by, key=by.get)}
    return None


def _point_segment(px: float, py: float, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    f = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / L2))
    return math.hypot(px - (a[0] + f * dx), py - (a[1] + f * dy))


def _leg_stuck(ego, ref, arrival, thr, scene_time, light_states, light_group) -> list[list[float]]:
    """Stretches of the leg the ego stood still (speed <= stuck_speed_mps) for
    stuck_duration_sec or more, away from the leg goal — the whole-bag "stuck"
    rule from before legs, per leg. A red light waited out, or a wait the
    recording made there too, is not stuck. Returns [t_rel, duration] from the
    leg's first sample."""
    runs, i = [], 0
    while i < len(ego):
        if ego[i].speed_mps > thr.stuck_speed_mps:
            i += 1
            continue
        j = i
        while j < len(ego) and ego[j].speed_mps <= thr.stuck_speed_mps:
            j += 1
        a, b = ego[i], ego[j - 1]
        near_goal = arrival is not None and math.hypot(a.x - arrival[0], a.y - arrival[1]) <= GOAL_ZONE_M
        if b.stamp_sec - a.stamp_sec >= thr.stuck_duration_sec and not near_goal:
            runs.append((a.stamp_sec, b.stamp_sec, a.x, a.y, a.yaw_rad))
        i = j
    if not runs:
        return []
    from dp_multi_eval.stop_go import WAIT_MIN_S, _stops

    rec_waits = [(r0.stamp_sec, r1.stamp_sec, r0.x, r0.y) for r0, r1 in _stops(ref, min_s=WAIT_MIN_S)] if ref else []
    like = _waited_like_recording([r[:4] for r in runs], rec_waits, scene_time)
    red = _at_red_light(runs, light_states(), light_group)
    return [
        [round(r[0] - ego[0].stamp_sec, 1), round(r[1] - r[0], 1)]
        for r, lk, rd in zip(runs, like, red)
        if not (lk or rd)
    ]


RED_WAIT_MIN_FRAC = 0.5  # a stop is a red-light wait when the light ahead was red/amber this much of it
LIGHT_AHEAD_M = 10.0  # the bus's rear axle stands this far behind the stop line its light governs


def _load_lanelet_map(map_path):
    """The lanelet2 map under ``map_path`` (a directory or an .osm), or None."""
    try:
        import lanelet2
        from autoware_lanelet2_extension_python.projection import MGRSProjector
        from lanelet2.io import Origin
    except Exception:  # noqa: BLE001
        return None
    if map_path is None:
        return None
    map_path = Path(map_path)
    osm = [map_path] if map_path.suffix == ".osm" else sorted(map_path.glob("*.osm"))
    if not osm:
        return None
    return lanelet2.io.load(str(osm[0]), MGRSProjector(Origin(0.0, 0.0)))


def _lane_yaw(ll, x: float, y: float) -> float:
    c = ll.centerline
    best = (float("inf"), 0.0)
    for k in range(len(c) - 1):
        mx, my = (c[k].x + c[k + 1].x) / 2.0, (c[k].y + c[k + 1].y) / 2.0
        d2 = (mx - x) ** 2 + (my - y) ** 2
        if d2 < best[0]:
            best = (d2, math.atan2(c[k + 1].y - c[k].y, c[k + 1].x - c[k].x))
    return best[1]


def _light_ahead(lmap, x: float, y: float, yaw: float):
    """The traffic_light regulatory element governing an ego at (x, y) facing
    yaw, or None. The governing light is the one on the lane the ego is driving,
    and "nearest lanelet with a light" is direction-blind: at a crosswalk the
    opposing approach sat 5.3 m away and its light — red exactly when ours was
    green — was the one picked, undercounting 56 s of green hold as zero.
    Candidates must face the way the ego faces. The lane under the ego comes
    first; failing that, the lane LIGHT_AHEAD_M ahead — a bus stopped at a
    stop line has its rear axle 9 m behind it, on the lanelet before the
    junction (H95: rear axle on 117, the light on 193)."""
    import lanelet2
    from lanelet2.core import BasicPoint2d

    for probe in (0.0, LIGHT_AHEAD_M):
        px, py = x + probe * math.cos(yaw), y + probe * math.sin(yaw)
        for dist, ll in lanelet2.geometry.findNearest(lmap.laneletLayer, BasicPoint2d(px, py), 10):
            if dist > 8.0:
                break
            diff = abs((_lane_yaw(ll, px, py) - yaw + math.pi) % (2.0 * math.pi) - math.pi)
            if diff > math.pi / 3.0:
                continue
            for reg in ll.regulatoryElements:
                sub = reg.attributes["subtype"] if "subtype" in reg.attributes else ""
                if sub == "traffic_light":
                    return reg.id
    return None


def _light_states(bag_path: Path, topics: TopicSet) -> list[tuple[float, dict[int, int]]]:
    """The replayed signal timeline in the run bag: ``[(t_wall, {group_id: color})]``."""
    try:
        from dp_multi_eval.bag_reader import iter_topic_messages

        out = []
        for _, t, msg in iter_topic_messages(bag_path, [topics.traffic_signals], {topics.traffic_signals: 0.2}):
            states = {g.traffic_light_group_id: g.elements[0].color for g in msg.traffic_light_groups if g.elements}
            if states:
                out.append((t, states))
        return out
    except Exception:  # noqa: BLE001
        return []


CUT_PASS_LOOK_SEC = 6.0  # the light the ego drove past a cut's stand under is looked for this far back


def _cut_pass(leg: dict, ego, light_red) -> dict | None:
    """A window the ego closed by driving on past a cut's stand (window_eval
    passed_cut): how much of the recorded wait it skipped, and the light it
    drove under — the last light ahead of it before its stop line, red or
    amber at that moment or not. The scene then jumps to the recorded
    departure; what the ego did not wait for must not be lost with it.
    Reported, not gating."""
    cp = leg.get("cut_passed")
    if not cp:
        return None
    out = {"skipped_sec": cp.get("skipped_sec"), "ego_v": cp.get("ego_v"), "light": None}
    wall = cp.get("wall")
    if wall is None or not ego:
        return out
    for e in reversed([e for e in ego if float(wall) - CUT_PASS_LOOK_SEC <= e.stamp_sec <= float(wall)]):
        try:
            state = light_red(e.stamp_sec, e.x, e.y, e.yaw_rad, known=True)
        except Exception:  # noqa: BLE001
            state = None
        if state is not None:
            out["light"] = "red" if state else "not_red"
            break
    return out


STOP_LINE_PAST_M = 2.0  # the ego's front this far past its light's stop line: the light is behind it
MUTUAL_WAIT_SEC = 10.0  # standing this long with no reason, an NPC held for the ego near: a mutual wait (fails)
MUTUAL_NEAR_M = 15.0    # ...that near the ego's centre
MUTUAL_BEHIND_FRONT_M = 1.0  # ...and no further back than this behind its front (one behind it queues)
EGO_CENTER_OFFSET_M = 2.12  # base_link to the body centre (lv828l)


def _past_stop_line(stop_lines, group: int, x: float, y: float, yaw: float, front_m: float) -> bool:
    """Is the ego's front (``front_m`` ahead of base_link) more than
    STOP_LINE_PAST_M past the nearest stop line of light ``group``, along the
    traffic that line stops? False when the map has no line for it."""
    fx, fy = x + front_m * math.cos(yaw), y + front_m * math.sin(yaw)
    best = None
    for sl in stop_lines or []:
        if sl.group != group:
            continue
        mx, my = (sl.x0 + sl.x1) / 2.0, (sl.y0 + sl.y1) / 2.0
        d = math.hypot(mx - x, my - y)
        if best is None or d < best[0]:
            best = (d, (fx - mx) * math.cos(sl.yaw) + (fy - my) * math.sin(sl.yaw))
    return best is not None and best[1] > STOP_LINE_PAST_M


def _at_red_light(run_stops, states, group_of) -> list[bool]:
    """For each run stop ``(t0_wall, t1_wall, x, y, yaw)``: was the light ahead
    of the ego red or amber for RED_WAIT_MIN_FRAC of it? ``states`` is the
    replayed signal timeline ``[(t_wall, {group_id: color})]``, ``group_of(x,
    y, yaw)`` names the governing light or None. Waiting at red is correct
    driving whatever the recording did there: on H95 (2026-09-16) the ego set
    off 9 s before the recording, reached the junction on red and stood 3.7 s
    until the green; the recording arrived 2 s after that green and rolled
    through at 1.2 m/s — "a stop the recording did not make" failed the leg in
    every run of every model."""
    import bisect

    RED, AMBER = 1, 2  # autoware_perception_msgs/TrafficLightElement
    times = [t for t, _ in states]
    out = []
    for t0, t1, x, y, yaw in run_stops:
        group = group_of(x, y, yaw) if states and t1 > t0 else None
        if group is None:
            out.append(False)
            continue
        red = 0.0
        t = t0
        while t < t1:
            j = bisect.bisect_right(times, t) - 1  # the state in force at t
            if j >= 0 and states[j][1].get(group) in (RED, AMBER):
                red += 0.5
            t += 0.5
        out.append(red >= RED_WAIT_MIN_FRAC * (t1 - t0))
    return out


JUMP_MPS = 20.0  # an NPC that moved faster than this between two samples was teleported by the replay
# An object first seen this close to the ego this short a time before touching
# it popped up there: objects appear where the RECORDED bus first detected
# them, and an ego ahead of the recording meets them at their birth. H78
# (2026-11-09): a cyclist born 1.6 m ahead of a 2.85 m/s ego, contact 0.2 s
# later, billed ego_caused. Reported as replay_jump, not gating.
POP_IN_SEC = 2.0
POP_IN_M = 15.0
SCENE_JUMP_SEC = 4.0   # a contact beginning this soon after a cut's scene jump is the jump's (3 s of history + 1)


def scene_jump_walls(windows_json: Path) -> list[float]:
    """Wall times at which the scene jumped to an ego that had driven on
    past a cut's stand (windows.json segments with scene_jumped)."""
    try:
        payload = json.loads(Path(windows_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted(
        float(seg["wall_handover"])
        for leg in payload.get("legs") or []
        for seg in leg.get("segments") or []
        if seg.get("scene_jumped") and seg.get("wall_handover") is not None
    )


def through_ego_object(object_id: str, ego_x: float, ego_y: float, through_events: list[dict[str, Any]]) -> bool:
    """Was this object one the reproducer withheld from the model for driving
    through the ego's body near here? Its recorded path ran through the ego
    (open-loop replay: it never saw the ego), so a contact at the edge of
    that episode says nothing about the planner. ``through_events`` are
    metric_objects_through_ego's events: object (8-hex id prefix) and the
    ego position at which the suppression began."""
    prefix = str(object_id)[:8]
    return any(
        ev.get("object") == prefix
        and math.hypot(float(ev.get("x", 0.0)) - ego_x, float(ev.get("y", 0.0)) - ego_y) <= THROUGH_EGO_MATCH_M
        for ev in through_events
    )


def _group_near_misses(raw: list[dict[str, Any]], gap_sec: float = 2.0, limit: int = 20) -> list[dict[str, Any]]:
    """One entry per approach: the closest sample of each run of samples the
    same object spent inside the near-miss distance. A 0.4 m brush past lasts
    several frames and would otherwise fill the list with one row per frame."""
    out: list[dict[str, Any]] = []
    by_obj: dict[str, list[dict[str, Any]]] = {}
    for r in raw:
        by_obj.setdefault(str(r["object_id"]), []).append(r)
    for rows in by_obj.values():
        rows.sort(key=lambda r: r["t_rel"])
        run = [rows[0]]
        for prev, cur in zip(rows, rows[1:]):
            if cur["t_rel"] - prev["t_rel"] > gap_sec:
                out.append(min(run, key=lambda r: r["distance_m"]))
                run = []
            run.append(cur)
        if run:
            out.append(min(run, key=lambda r: r["distance_m"]))
    out.sort(key=lambda r: r["distance_m"])
    return out[:limit]



def metric_collision(
    series: BagSeries,
    thr: Thresholds,
    footprint_local: list[tuple[float, float]],
    held_windows: list[tuple[float, float]] | None = None,
    reference: BagSeries | None = None,
    through_events: list[dict[str, Any]] | None = None,
    lag_at=None,
    scene_jumps: list[float] | None = None,
) -> dict[str, Any]:
    """``lag_at(stamp) -> seconds behind the recording`` (leg-window runs): a
    withheld object is a replay artefact only when the ego was BEHIND the
    recording; an ego ahead of it that drives into the car the recorded bus
    was following (2026-08-24, 7 m/s, 8.7 s ahead) caused that contact."""
    ego_ds = downsample_ego(series.ego, thr.sample_dt)
    if not ego_ds:
        return {
            "flagged": False,
            "rate": 0.0,
            "events": [],
            "event_count": 0,
            "frame_count": 0,
            "min_distance_m": float("nan"),
            "status": "no_ego",
            "description": METRIC_DESCRIPTIONS["collision_rate"],
        }

    # Index objects by rounded time for fast lookup
    objs = series.objects
    oi = 0
    held = [tuple(w) for w in (held_windows or [])]
    nearest = _envelope_lookup(reference, thr)
    by_class: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    min_dist = float("inf")
    first_t = None
    colliding_frames = 0
    # proximity while moving, withheld (through-ego) objects excluded
    prox_min = float("inf")
    prox_closest: dict[str, Any] | None = None
    prox_close_frames = 0
    near_raw: list[dict[str, Any]] = []

    # Each object's previous sample. A contact by an object that jumped to the
    # ego (2025-12-11: a truck 7 m behind in one frame, at the ego's rear corner
    # the next, gone again 20 ms later, while the ego ran 2x the recording) is
    # the replay's doing, not the planner's: replay_jump, reported, not gating.
    prev_by_idx: dict[int, tuple[float, float, float]] = {}
    last_seen: dict[str, tuple[float, float, float]] = {}
    first_seen: dict[str, tuple[float, float, float]] = {}
    for k, o in enumerate(objs):
        if o.object_id in last_seen:
            prev_by_idx[k] = last_seen[o.object_id]
        last_seen[o.object_id] = (o.stamp_sec, o.x, o.y)
        first_seen.setdefault(o.object_id, (o.stamp_sec, o.x, o.y))
    ego_t = [e.stamp_sec for e in ego_ds]

    # A contact that began with a pop-up stays the replay's for as long as it
    # lasts: the ego cannot leave an object it was born touching on one frame.
    popin_until: dict[str, float] = {}

    # The scene jumped under a moving ego (window_eval: it drove on past a
    # cut's stand, the scene came to the recorded departure): every object is
    # new to the planner, its 3 s histories empty. A contact that begins
    # within SCENE_JUMP_SEC with an object whose body was within POP_IN_M of
    # the ego's then is the jump's (H81, 2026-11-11: a car crawling 8 m ahead
    # of the ego's front when the scene arrived, hit 2.1 s later). scene_jump:
    # reported, not gating.
    jumps = sorted(scene_jumps or [])
    by_id: dict[str, tuple[list[float], list[Any]]] = {}
    if jumps:
        for o in objs:
            ts_, ps_ = by_id.setdefault(o.object_id, ([], []))
            ts_.append(o.stamp_sec)
            ps_.append(o)
    jump_until: dict[str, float] = {}
    jump_contacts: dict[str, float] = {}

    def after_scene_jump(obj, now: float) -> bool:
        if not jumps:
            return False
        if now - jump_until.get(obj.object_id, -1e9) <= 1.0:
            jump_until[obj.object_id] = now
            return True
        k = bisect.bisect_right(jumps, now) - 1
        if k < 0 or now - jumps[k] > SCENE_JUMP_SEC:
            return False
        ts_, ps_ = by_id.get(obj.object_id, ([], []))
        i = bisect.bisect_left(ts_, jumps[k])
        if i >= len(ts_) or ts_[i] > now:
            return False
        e = ego_ds[min(bisect.bisect_left(ego_t, ts_[i]), len(ego_t) - 1)]
        o0 = ps_[i]
        gap = polygon_distance(
            transform_local_polygon(footprint_local, e.x, e.y, e.yaw_rad),
            transform_local_polygon(box_local(o0.length_m, o0.width_m), o0.x, o0.y, o0.yaw_rad),
        )
        if gap > POP_IN_M:
            return False
        jump_until[obj.object_id] = now
        jump_contacts.setdefault(obj.object_id, now)
        return True

    def popped_up_at_ego(obj, now: float) -> bool:
        if now - popin_until.get(obj.object_id, -1e9) <= 1.0:
            popin_until[obj.object_id] = now
            return True
        t0, x0, y0 = first_seen.get(obj.object_id, (None, 0.0, 0.0))
        if t0 is None or now - t0 > POP_IN_SEC:
            return False
        i = min(bisect.bisect_left(ego_t, t0), len(ego_t) - 1)
        if math.hypot(x0 - ego_ds[i].x, y0 - ego_ds[i].y) > POP_IN_M:
            return False
        popin_until[obj.object_id] = now
        return True

    for ego in ego_ds:
        # advance object cursor to nearby stamps
        while oi < len(objs) and objs[oi].stamp_sec < ego.stamp_sec - thr.sample_dt:
            oi += 1
        j = oi
        ego_poly = transform_local_polygon(footprint_local, ego.x, ego.y, ego.yaw_rad)
        frame_hit = False
        while j < len(objs) and objs[j].stamp_sec <= ego.stamp_sec + thr.sample_dt:
            obj = objs[j]
            j += 1
            if obj.label not in NPC_COLLISION_LABELS:
                continue
            if obj.length_m <= 0.0 or obj.width_m <= 0.0:
                continue
            npc_poly = transform_local_polygon(
                box_local(obj.length_m, obj.width_m), obj.x, obj.y, obj.yaw_rad
            )
            dist = polygon_distance(ego_poly, npc_poly)
            min_dist = min(min_dist, dist)
            if ego.speed_mps > 0.5 and not (
                through_events and through_ego_object(obj.object_id, ego.x, ego.y, through_events)
            ):
                if dist < prox_min:
                    prox_min = dist
                    prox_closest = {
                        "t_rel": round(ego.stamp_sec - ego_ds[0].stamp_sec, 1),
                        "object_id": obj.object_id, "label": obj.label,
                        "distance_m": round(dist, 2), "ego_speed_mps": round(ego.speed_mps, 2),
                    }
                if dist < thr.proximity_close_m:
                    prox_close_frames += 1
                if dist < thr.proximity_near_miss_m:
                    near_raw.append({
                        "t_rel": round(ego.stamp_sec - ego_ds[0].stamp_sec, 1),
                        "object_id": obj.object_id, "label": obj.label,
                        "distance_m": round(dist, 2),
                        "ego_speed_mps": round(ego.speed_mps, 2),
                        "at": [round(ego.x, 1), round(ego.y, 1)],
                    })
            if dist <= thr.collision_distance_m:
                if first_t is None:
                    first_t = ego.stamp_sec
                if not frame_hit:
                    colliding_frames += 1
                    frame_hit = True
                    prev = prev_by_idx.get(j - 1)
                    obj_speed = (
                        math.hypot(obj.x - prev[1], obj.y - prev[2]) / max(obj.stamp_sec - prev[0], 0.05)
                        if prev is not None and 0.0 < obj.stamp_sec - prev[0] <= 2.0
                        else None
                    )
                    kind = _collision_class(ego, obj, thr, held, nearest, obj_speed)
                    if through_events and through_ego_object(obj.object_id, ego.x, ego.y, through_events):
                        # 2026-04-28: a recorded car whose path ran through a
                        # slower ego was scored ego_caused at the edge of the
                        # frames the reproducer withheld it for.
                        lag = lag_at(ego.stamp_sec) if lag_at is not None else None
                        if lag is None or lag > 2.0:
                            kind = "through_ego"
                    if prev is None or prev[0] < obj.stamp_sec - 2.0 or popped_up_at_ego(obj, ego.stamp_sec):
                        kind = "replay_jump"  # appeared at the ego out of nowhere
                    elif math.hypot(obj.x - prev[1], obj.y - prev[2]) / max(obj.stamp_sec - prev[0], 0.05) > JUMP_MPS:
                        kind = "replay_jump"
                    elif after_scene_jump(obj, ego.stamp_sec):
                        kind = "scene_jump"
                    by_class[kind] = by_class.get(kind, 0) + 1
                    events.append(
                        {
                            "time_sec": ego.stamp_sec,
                            "object_id": obj.object_id,
                            "label": obj.label,
                            "distance_m": dist,
                            "ego_speed_mps": round(ego.speed_mps, 2),
                            "attribution": kind,
                        }
                    )
                break  # one collision event per ego sample
    if min_dist == float("inf"):
        min_dist = float("nan")
    n_frames = len(ego_ds)
    rate = colliding_frames / n_frames if n_frames else 0.0
    # Only contacts the ego drove into gate the run. The rest are reported with
    # their attribution: they say something about the replay, not the planner,
    # and failing a model for them buries the results it is meant to produce.
    ego_caused = by_class.get("ego_caused", 0)
    # The list is truncated, so it must not be truncated by arrival order: on one
    # run 64 of 81 contacts were not_in_control and filled the cap, leaving 1 of
    # the 9 ego_caused frames — the only class that fails the run — visible in
    # the report. Gating events lead.
    rank = {"ego_caused": 0, "cut_in": 1, "out_of_envelope": 2,
            "struck_while_stopped": 3, "not_in_control": 4, "through_ego": 5, "scene_jump": 6}
    events.sort(key=lambda e: (rank.get(e.get("attribution"), 4), e["time_sec"]))
    return {
        "flagged": ego_caused > 0,
        "rate": rate,
        "ego_caused_rate": (ego_caused / n_frames) if n_frames else 0.0,
        "by_attribution": by_class,
        "events": events[:50],
        "scene_jump_contacts": [{"time_sec": t, "object_id": k} for k, t in sorted(jump_contacts.items(), key=lambda kv: kv[1])],
        "event_count": colliding_frames,
        "frame_count": n_frames,
        "min_distance_m": min_dist,
        "first_time_sec": first_t,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["collision_rate"],
        "proximity": {
            "min_distance_moving_m": (round(prox_min, 2) if prox_min != float("inf") else None),
            "closest": prox_closest,
            "close_sec": round(prox_close_frames * thr.sample_dt, 1),
            "near_miss_m": thr.proximity_near_miss_m,
            "near_misses": _group_near_misses(near_raw),
            "status": "ok",
            "description": METRIC_DESCRIPTIONS["proximity"],
        },
    }


def _linestring_type(linestring: Any) -> str:
    attrs = getattr(linestring, "attributes", None)
    if attrs is None or "type" not in attrs:
        return ""
    return str(attrs["type"])


def _orientation(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> bool:
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)
    if (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0):
        return True
    if abs(o1) < 1e-9 and _on_segment(p1, q1, p2):
        return True
    if abs(o2) < 1e-9 and _on_segment(p1, q2, p2):
        return True
    if abs(o3) < 1e-9 and _on_segment(q1, p1, q2):
        return True
    if abs(o4) < 1e-9 and _on_segment(q1, p2, q2):
        return True
    return False


def _point_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _footprint_boundary_hit(
    footprint: list[tuple[float, float]],
    segments: list[tuple[float, float, float, float, str]],
    *,
    ego_x: float,
    ego_y: float,
    search_radius_m: float,
    margin_m: float,
) -> tuple[bool, float, str | None]:
    """Return (hit, min_distance_m, boundary_type)."""
    min_dist = float("inf")
    hit_type: str | None = None
    crossed = False
    for x1, y1, x2, y2, btype in segments:
        if (
            math.hypot(x1 - ego_x, y1 - ego_y) > search_radius_m
            and math.hypot(x2 - ego_x, y2 - ego_y) > search_radius_m
        ):
            continue
        for i in range(len(footprint)):
            p1 = footprint[i]
            p2 = footprint[(i + 1) % len(footprint)]
            if _segments_intersect(p1, p2, (x1, y1), (x2, y2)):
                crossed = True
                hit_type = btype
                min_dist = 0.0
        for px, py in footprint:
            d = _point_segment_distance(px, py, x1, y1, x2, y2)
            if d < min_dist:
                min_dist = d
                if not crossed:
                    hit_type = btype
    if min_dist == float("inf"):
        return False, float("nan"), None
    hit = crossed or (margin_m > 0.0 and min_dist < margin_m)
    return hit, min_dist, hit_type


def _extract_boundary_segments(
    lanelet_map: Any, boundary_types: set[str]
) -> list[tuple[float, float, float, float, str]]:
    segments: list[tuple[float, float, float, float, str]] = []
    for linestring in lanelet_map.lineStringLayer:
        btype = _linestring_type(linestring)
        if btype not in boundary_types:
            continue
        points = [(float(p.x), float(p.y)) for p in linestring]
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            segments.append((x1, y1, x2, y2, btype))
    return segments


OOB_JUMP_M = 2.0  # an ego pose step this long between two samples is the harness re-placing it


def handover_windows(windows_json: Path) -> list[tuple[float, float]]:
    """Wall-time spans in which the harness, not the planner, moves the ego:
    LEAD_SEC (+0.5 s) before each hand-over to 1 s after (windows.json)."""
    try:
        payload = json.loads(Path(windows_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    lead = float((payload.get("params") or {}).get("lead_sec", 3.5))
    return [
        (float(seg["wall_handover"]) - lead - 0.5, float(seg["wall_handover"]) + 1.0)
        for leg in payload.get("legs") or []
        for seg in leg.get("segments") or []
        # a seamless planned cut moved nothing: the planner drove through it
        if seg.get("wall_handover") is not None and not seg.get("seamless")
    ]


def oob_by_cause(episodes: list[list[Any]], ego: list[Any], windows_json: Path) -> dict[str, Any]:
    """Split out-of-boundary episodes into the ones the ego drove into and the
    ones the harness put it in.

    A leg-window run places the ego on the recorded pose at each segment (a bus
    in its bay touched the road border for one frame) and holds it until the
    hand-over. An episode starting before the first hand-over, inside a
    hand-over window, or within 1 s of a pose jump of OOB_JUMP_M is the
    placement's; any other is the ego's own.
    """
    spans = handover_windows(windows_json)
    first_handover = min((hi - 1.0 for _, hi in spans), default=None)
    try:
        legs = json.loads(Path(windows_json).read_text(encoding="utf-8")).get("legs") or []
    except (OSError, json.JSONDecodeError):
        legs = []
    # From a leg's start to its first hand-over the ego stands where the harness
    # put it (7 s in H35 leg 2, longer than lead_sec).
    spans += [
        (float(lg["wall_start"]) - 0.5, float(lg["segments"][0]["wall_handover"]) + 1.0)
        for lg in legs
        if lg.get("wall_start") is not None and (lg.get("segments") or [{}])[0].get("wall_handover") is not None
    ]
    spans += [
        (a.stamp_sec, b.stamp_sec + 1.0)
        for a, b in zip(ego, ego[1:])
        if b.stamp_sec - a.stamp_sec <= 0.5 and math.hypot(b.x - a.x, b.y - a.y) >= OOB_JUMP_M
    ]
    ego_caused, placement = [], []
    for ep in episodes:
        t0 = float(ep[0])
        placed = (first_handover is not None and t0 < first_handover) or any(lo <= t0 <= hi for lo, hi in spans)
        (placement if placed else ego_caused).append(ep)
    return {"ego_caused_episodes": ego_caused, "placement_episodes": placement}


OOB_RECORDED_RADIUS_M = 3.0   # an ego crossing this close to one the recording made is the map's


def split_recorded_too(
    episodes: list[list[Any]],
    ego_frames: list[Any],
    recorded_hits: list[tuple[float, float]],
    radius_m: float = OOB_RECORDED_RADIUS_M,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """(the ego's own episodes, episodes the recording made at the same place).

    With the bus's true 10.88 x 2.87 m body the recorded bus itself crosses the
    mapped road border on 15-19 % of its frames (H39, H45: the map's curb lines
    sit 0.1-0.2 m inside where buses really drive, at stops above all). An ego
    crossing where the recording crossed says that about the map, not the
    model. An episode is the recording's when at least half of its frames lie
    within ``radius_m`` of a recorded crossing. How deep each crossed is not
    compared: a model that crossed further at such a spot is not caught here.
    """
    if not recorded_hits:
        return list(episodes), []
    cell = max(radius_m, 1.0)
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for hx, hy in recorded_hits:
        grid.setdefault((int(hx // cell), int(hy // cell)), []).append((hx, hy))

    def near(x: float, y: float) -> bool:
        cx, cy = int(x // cell), int(y // cell)
        return any(
            math.hypot(hx - x, hy - y) <= radius_m
            for gx in (cx - 1, cx, cx + 1) for gy in (cy - 1, cy, cy + 1)
            for hx, hy in grid.get((gx, gy), ())
        )

    own, recorded = [], []
    for ep in episodes:
        frames = [e for e in ego_frames if float(ep[0]) - 1e-6 <= e.stamp_sec <= float(ep[1]) + 1e-6]
        shared = sum(1 for e in frames if near(e.x, e.y))
        (recorded if frames and shared * 2 >= len(frames) else own).append(ep)
    return own, recorded


def metric_out_of_boundary(
    series: BagSeries,
    map_path: Path | None,
    thr: Thresholds,
    footprint_local: list[tuple[float, float]],
    reference: BagSeries | None = None,
) -> dict[str, Any]:
    """Ego footprint vs road_border/curbstone LineStrings (batch-eval style).
    With ``reference`` (the recording) the same footprint is run along the
    recorded track too, and ego episodes where the recording crossed as well
    are listed apart (``recorded_too_episodes``)."""
    desc = METRIC_DESCRIPTIONS["out_of_boundary"]
    boundary_types = tuple(thr.oob_boundary_types) or ("road_border", "curbstone")
    ego_ds = downsample_ego(series.ego, thr.sample_dt)
    empty = {
        "flagged": False,
        "rate": 0.0,
        "min_distance_m": float("nan"),
        "frame_count": len(ego_ds),
        "oob_frame_count": 0,
        "boundary_types": list(boundary_types),
        "boundary_segment_count": 0,
        "description": desc,
    }
    if not ego_ds:
        return {**empty, "frame_count": 0, "status": "no_ego"}
    if map_path is None:
        return {**empty, "status": "map_not_provided"}

    try:
        from autoware_lanelet2_extension_python.projection import MGRSProjector
        from lanelet2.io import Origin, load
    except ImportError as exc:
        return {**empty, "status": f"lanelet2_unavailable: {exc}"}

    map_path = map_path.expanduser().resolve()
    osm = map_path if map_path.suffix == ".osm" else map_path / "lanelet2_map.osm"
    if not osm.is_file():
        return {**empty, "status": f"map_missing: {osm}"}

    try:
        projector = MGRSProjector(Origin(0.0, 0.0))
        # Prefer map_projector_info when present (same folder as osm / map dir).
        map_dir = map_path if map_path.is_dir() else map_path.parent
        projector_info = map_dir / "map_projector_info.yaml"
        if projector_info.is_file():
            info = yaml.safe_load(projector_info.read_text(encoding="utf-8")) or {}
            ptype = str(info.get("projector_type", "MGRS"))
            if ptype in ("TransverseMercator", "LocalCartesianUTM"):
                from autoware_lanelet2_extension_python._autoware_lanelet2_extension_python_boost_python_projection import (  # noqa: E501
                    TransverseMercatorProjector,
                )

                origin = info["map_origin"]
                projector = TransverseMercatorProjector(
                    Origin(float(origin["latitude"]), float(origin["longitude"]))
                )
        lanelet_map = load(str(osm), projector)
    except Exception as exc:  # noqa: BLE001
        return {**empty, "status": f"map_load_failed: {exc}"}

    try:
        segments = _extract_boundary_segments(lanelet_map, set(boundary_types))
    except Exception as exc:  # noqa: BLE001
        return {**empty, "status": f"boundary_extract_failed: {exc}"}

    if not segments:
        return {
            **empty,
            "status": "no_boundary_linestrings",
            "boundary_segment_count": 0,
        }

    min_dist_overall = float("inf")
    worst_t = None
    violation_t = None
    hit_type_first: str | None = None
    oob_frames = 0
    episodes: list[list[Any]] = []  # [t_first_hit, t_last_hit, boundary_type]
    prev_hit = False
    try:
        for ego in ego_ds:
            poly = transform_local_polygon(footprint_local, ego.x, ego.y, ego.yaw_rad)
            hit, dist, btype = _footprint_boundary_hit(
                poly,
                segments,
                ego_x=ego.x,
                ego_y=ego.y,
                search_radius_m=thr.oob_search_radius_m,
                margin_m=thr.oob_margin_m,
            )
            if isinstance(dist, float) and not math.isnan(dist) and dist < min_dist_overall:
                min_dist_overall = dist
                worst_t = ego.stamp_sec
            if hit:
                oob_frames += 1
                if violation_t is None:
                    violation_t = ego.stamp_sec
                    hit_type_first = btype
                if prev_hit:
                    episodes[-1][1] = ego.stamp_sec
                else:
                    episodes.append([ego.stamp_sec, ego.stamp_sec, btype])
            prev_hit = bool(hit)
    except Exception as exc:  # noqa: BLE001
        return {
            **empty,
            "boundary_segment_count": len(segments),
            "status": f"oob_compute_failed: {exc}",
        }

    if min_dist_overall == float("inf"):
        min_dist_overall = float("nan")
    n_frames = len(ego_ds)
    rate = oob_frames / n_frames if n_frames else 0.0
    recorded_too: list[list[Any]] = []
    recorded_rate = None
    if reference is not None and reference.ego and episodes:
        ref_ds = downsample_ego(reference.ego, thr.sample_dt)
        hits: list[tuple[float, float]] = []
        try:
            for rec in ref_ds:
                poly = transform_local_polygon(footprint_local, rec.x, rec.y, rec.yaw_rad)
                hit, _, _ = _footprint_boundary_hit(
                    poly, segments, ego_x=rec.x, ego_y=rec.y,
                    search_radius_m=thr.oob_search_radius_m, margin_m=thr.oob_margin_m,
                )
                if hit:
                    hits.append((rec.x, rec.y))
        except Exception:  # noqa: BLE001  (the ego's own result stands)
            hits = []
        recorded_rate = len(hits) / len(ref_ds) if ref_ds else None
        _, recorded_too = split_recorded_too(episodes, ego_ds, hits)
    return {
        "recorded_too_episodes": recorded_too,
        "recorded_rate": recorded_rate,
        "flagged": oob_frames > 0,
        "rate": rate,
        "min_distance_m": min_dist_overall,
        "worst_time_sec": worst_t,
        "first_violation_sec": violation_t,
        "first_boundary_type": hit_type_first,
        "frame_count": n_frames,
        "oob_frame_count": oob_frames,
        "episodes": episodes,
        "boundary_types": list(boundary_types),
        "boundary_segment_count": len(segments),
        "status": "ok",
        "description": desc,
    }


def _comfort_profile(samples, thr: Thresholds, exclude: list[tuple[float, float]] | None = None) -> dict[str, Any]:
    """Harsh-brake / harsh-jerk events of one speed series (EgoSample list)."""
    out: dict[str, Any] = {
        "max_decel_mps2": 0.0, "harsh_brake_count": 0, "harsh_brake_sec": 0.0, "harsh_brakes": [],
        "max_abs_jerk_mps3": 0.0, "jerk_p95_mps3": 0.0, "harsh_jerk_count": 0, "samples": 0,
    }
    if len(samples) < 10:
        return out
    dt = 0.1
    t0 = samples[0].stamp_sec
    stamps = [e.stamp_sec for e in samples]
    n = int((stamps[-1] - t0) / dt) + 1
    import bisect as _b

    grid_t = [t0 + i * dt for i in range(n)]
    v = []
    for t in grid_t:
        k = min(_b.bisect_left(stamps, t), len(samples) - 1)
        v.append(float(samples[k].speed_mps))

    def smooth(x, w=3):
        return [sum(x[max(0, i - w):i + w + 1]) / len(x[max(0, i - w):i + w + 1]) for i in range(len(x))]

    vs = smooth(v)
    a = smooth([(vs[i + 1] - vs[i - 1]) / (2 * dt) if 0 < i < n - 1 else 0.0 for i in range(n)])
    j = smooth([(a[i + 1] - a[i - 1]) / (2 * dt) if 0 < i < n - 1 else 0.0 for i in range(n)])
    keep = [
        not any(w0 <= t <= w1 for w0, w1 in (exclude or []))
        for t in grid_t
    ]

    def events(flags, min_len):
        found = []
        i = 0
        while i < n:
            if flags[i] and keep[i]:
                k = i
                while k < n and flags[k] and keep[k]:
                    k += 1
                if (k - i) * dt >= min_len:
                    found.append((i, k))
                i = k
            else:
                i += 1
        return found

    brakes = events([ai <= -thr.harsh_decel_mps2 for ai in a], thr.comfort_event_min_sec)
    jerks = events([abs(ji) >= thr.harsh_jerk_mps3 for ji in j], 0.2)
    kept_a = [a[i] for i in range(n) if keep[i]]
    kept_j = sorted(abs(j[i]) for i in range(n) if keep[i])
    out.update(
        {
            "max_decel_mps2": round(-min(kept_a), 2) if kept_a else 0.0,
            "harsh_brake_count": len(brakes),
            "harsh_brake_sec": round(sum((k - i) * dt for i, k in brakes), 1),
            "harsh_brakes": [
                (round(grid_t[i] - t0, 1), round((k - i) * dt, 1), round(-min(a[i:k]), 2)) for i, k in brakes
            ][:12],
            "max_abs_jerk_mps3": round(kept_j[-1], 2) if kept_j else 0.0,
            "jerk_p95_mps3": round(kept_j[int(0.95 * (len(kept_j) - 1))], 2) if kept_j else 0.0,
            "harsh_jerk_count": len(jerks),
            "samples": len(kept_a),
        }
    )
    return out


def metric_comfort(bag_path: Path, series: BagSeries, reference: BagSeries | None, thr: Thresholds) -> dict[str, Any]:
    """Jerk / harsh brake of the run against the recording. The harness's
    hand-overs (windows.json: LEAD_SEC before each hand-over to 1 s after)
    are excluded from the run; they are the tool moving the ego."""
    out: dict[str, Any] = {"status": "ok", "description": METRIC_DESCRIPTIONS["comfort"], "flagged": False}
    if not series.ego:
        out["status"] = "no_ego"
        return out
    exclude = handover_windows(Path(bag_path).parent / "windows.json")
    run = _comfort_profile(series.ego, thr, exclude)
    rec = None
    if reference and reference.ego:
        # A hole in the recording is a speed jump, not a brake (2026-08-24:
        # 10 m/s^2 across 60 s of missing samples).
        holes = [
            (a.stamp_sec - 1.0, b.stamp_sec + 1.0)
            for a, b in zip(reference.ego, reference.ego[1:])
            if b.stamp_sec - a.stamp_sec > 2.0
        ]
        rec = _comfort_profile(reference.ego, thr, holes)
    out["run"] = run
    out["recording"] = rec
    out["excluded_windows"] = len(exclude)
    if rec is None:
        out["status"] = "no_reference"
    else:
        out["flagged"] = run["harsh_brake_count"] > rec["harsh_brake_count"] + 1
    return out


def compute_metrics(
    bag_path: Path,
    *,
    goal_x: float,
    goal_y: float,
    goal_yaw: float = 0.0,
    map_path: Path | None = None,
    thresholds: Thresholds | None = None,
    topics: TopicSet | None = None,
    vehicle_info_yaml: Path | None = None,
    output_json: Path | None = None,
    series: BagSeries | None = None,
    reference_bag: Path | None = None,
    goal_stop_required: bool = True,
) -> dict[str, Any]:
    thr = thresholds or Thresholds()
    topics = topics or TopicSet()
    if series is None:
        series = load_bag_series(
            bag_path,
            ego_topic=topics.ego_pose,
            objects_topic=topics.predicted_objects,
            velocity_topic=topics.vehicle_status_velocity,
            trajectory_topic=None,
            object_sample_dt=thr.sample_dt,
            skip_zero_size_objects=True,
        )
    footprint = load_vehicle_footprint(vehicle_info_yaml, margin=0.0)

    # The source recording, when it is still on disk, is the reference the replay is
    # measured against. Only ego poses are needed, so objects are skipped.
    reference: BagSeries | None = None
    if reference_bag is not None:
        try:
            reference = BagSeries(
                ego=load_reference_ego(
                    reference_bag,
                    ego_topic=topics.ego_pose,
                    velocity_topic=topics.vehicle_status_velocity,
                )
            )
        except Exception:  # noqa: BLE001
            reference = None

    stuck = metric_stuck(series, goal_x, goal_y, thr, reference=reference)
    goal = metric_goal_stop(
        series, goal_x, goal_y, goal_yaw, thr, stuck=stuck, required=goal_stop_required
    )
    # Runs before collision: contacts that happen while the stack is holding the
    # ego are not the planner's, and the windows say when that was.
    intervention = metric_stack_intervention(bag_path, topics, thr, map_path=map_path)
    validity = metric_replay_validity(series, reference, thr)
    # Before collision: an object the reproducer withheld for driving through
    # the ego is not one the ego collided with.
    through = metric_objects_through_ego(bag_path, series)
    lag_at = None
    scene = None
    if reference is not None and reference.ego:
        from dp_multi_eval.lag import recorded_lag_sec, reference_track, scene_time_map

        scene = scene_time_map(Path(bag_path).parent / "windows.json")
        track = reference_track(reference.ego)
        if scene is not None and track is not None:
            ego_by_t = series.ego
            stamps = [e.stamp_sec for e in ego_by_t]

            def lag_at(stamp: float):
                import bisect as _b

                k = min(_b.bisect_left(stamps, stamp), len(ego_by_t) - 1)
                e = ego_by_t[k]
                return recorded_lag_sec(track, e.x, e.y, e.yaw_rad, scene(stamp))
    collision = metric_collision(
        series,
        thr,
        footprint,
        held_windows=intervention.get("held_windows"),
        reference=reference,
        through_events=list(through.get("events") or []),
        lag_at=lag_at,
        scene_jumps=scene_jump_walls(Path(bag_path).parent / "windows.json"),
    )
    oob = metric_out_of_boundary(series, map_path, thr, footprint, reference=reference)
    ref_speed = metric_reference_speed(series, reference, thr)
    bus_service = metric_bus_stop_service(series, reference, map_path, thr)
    stop_go_m = metric_stop_go(bag_path, series, reference, goal_x, goal_y)
    resync = metric_resync(bag_path, series)
    windows = metric_windows(
        bag_path, series, reference, scene_time=scene if reference is not None else None, topics=topics, map_path=map_path,
        thr=thr, footprint=footprint,
        scene_jump_contacts=list(collision.get("scene_jump_contacts") or []),
    )
    comfort = metric_comfort(bag_path, series, reference, thr)

    overall_fail = bool(stuck.get("flagged") or collision.get("flagged"))
    if stop_go_m.get("status") == "ok" and stop_go_m.get("flagged"):
        overall_fail = True
    if resync.get("flagged"):
        overall_fail = True
    if bus_service.get("status") == "ok" and bus_service.get("flagged"):
        overall_fail = True
    if goal.get("status") == "ok" and goal.get("flagged"):
        overall_fail = True
    if oob.get("status") == "ok" and oob.get("flagged"):
        overall_fail = True
    if windows.get("status") == "ok" and windows.get("legs_total"):
        # A leg-window run is judged by its legs, the contacts the ego caused,
        # the bus stops it served and the road borders it drove over. The
        # whole-bag stop / goal figures are reported, not gating; a border
        # touch the harness caused (a teleport onto the recorded bus in its bay
        # touched the road border for one frame) is reported, not gating.
        if oob.get("status") == "ok":
            # An episode the recording made at the same place is the map's:
            # reported (recorded_too_episodes), never the ego's.
            shared = {tuple(ep) for ep in oob.get("recorded_too_episodes") or []}
            own = [ep for ep in oob.get("episodes") or [] if tuple(ep) not in shared]
            oob.update(oob_by_cause(own, series.ego, Path(bag_path).parent / "windows.json"))
        overall_fail = bool(
            windows.get("flagged")
            or collision.get("flagged")
            or (bus_service.get("status") == "ok" and bus_service.get("flagged"))
            or (comfort.get("status") == "ok" and comfort.get("flagged"))
            or bool(oob.get("ego_caused_episodes"))
        )
    elif comfort.get("status") == "ok" and comfort.get("flagged"):
        overall_fail = True
    # reference_speed is reported, never gating: a replay legitimately differs from
    # its recording (different traffic phase, harness interventions), and promoting
    # that to a hard failure without a calibrated tolerance would relabel runs that
    # drive correctly. It is the number to read first when another metric flags.

    result = {
        "pass": not overall_fail,
        # a leg the scene stopped (scene_limited stuck): its pass is not the model's
        "inconclusive": bool(windows.get("legs_inconclusive")),
        "stuck_rate": stuck,
        "out_of_boundary": oob,
        "collision_rate": collision,
        "goal_stop_precision": goal,
        "reference_speed": ref_speed,
        "bus_stop_service": bus_service,
        "objects_through_ego": through,
        "scene_audit": metric_scene_audit(bag_path, series),
        "stop_go": stop_go_m,
        "resync": resync,
        "windows": windows,
        "comfort": comfort,
        "proximity": collision.pop("proximity", {}),
        "success_rate": {
            "passed": not overall_fail,
            "legs_passed": windows.get("legs_passed"),
            "legs_total": windows.get("legs_total"),
            "status": "ok",
            "description": METRIC_DESCRIPTIONS["success_rate"],
        },
        "stack_intervention": intervention,
        "replay_validity": validity,
        "metric_descriptions": dict(METRIC_DESCRIPTIONS),
        "meta": {
            "bag_path": str(bag_path),
            "ego_samples": len(series.ego),
            "object_samples": len(series.objects),
            "duration_sec": series.duration_sec,
            "goal": {"x": goal_x, "y": goal_y, "yaw": goal_yaw},
            "thresholds": asdict(thr),
        },
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 2: compute metrics from output.bag")
    p.add_argument("--bag_path", type=Path, required=True)
    p.add_argument("--map_path", type=Path, default=None)
    p.add_argument("--goal_pose", nargs=3, type=float, metavar=("X", "Y", "YAW"), required=True)
    p.add_argument("--output", type=Path, default=None, help="metrics.json path")
    p.add_argument("--thresholds", type=Path, default=None)
    p.add_argument("--topics-yaml", type=Path, default=None)
    p.add_argument("--vehicle_info_yaml", type=Path, default=None)
    p.add_argument(
        "--reference_bag",
        type=Path,
        default=None,
        help="the source recording this run replays; enables reference_speed "
        "and the replay validity envelope",
    )
    args = p.parse_args(argv)

    out = args.output
    if out is None:
        out = args.bag_path.expanduser().resolve()
        if out.is_file():
            out = out.parent
        out = out.parent / "metrics.json" if out.name == "output" else out / "metrics.json"

    result = compute_metrics(
        args.bag_path,
        goal_x=args.goal_pose[0],
        goal_y=args.goal_pose[1],
        goal_yaw=args.goal_pose[2],
        map_path=args.map_path,
        thresholds=load_thresholds(args.thresholds),
        topics=load_topics(args.topics_yaml),
        vehicle_info_yaml=args.vehicle_info_yaml,
        reference_bag=args.reference_bag,
        output_json=out,
    )
    print(f"[done] pass={result['pass']} wrote {out}")
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
