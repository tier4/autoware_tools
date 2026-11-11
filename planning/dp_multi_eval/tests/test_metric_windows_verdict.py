"""metric_windows leg verdict end to end on a synthetic leg: an unplanned 4 s
stop is reported (long_stops) but does not fail the leg; the same stop under a
red light ahead is a red_light_wait; a brake cycle under red is a
red_light_brake; stop-and-go under green is reported, not failing; a goal stop off by more
than 2 m fails at a bus stop and is reported at a traffic wait; a missed departure (>= 15 s) is reported, not failing; a
stop more than 2 m off the goal fails the leg, one 1 m off does not; standing
45 s+ away from the goal is stuck and fails it, unless under a red light; a
drive past the goal without arriving fails it, and so does a late grace that ran
out (end_dist_m); a leg merely late at the window end passes. A reset closes a
window without a verdict; a window closed 20 s behind fails (did_not_complete)
unless a justified wait — pedestrians crossing the ego's way — explains the lag.
A passed leg with a vehicle shown inside another object near the ego is inconclusive.
Run: python3 tests/test_metric_windows_verdict.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dp_multi_eval.compute_metrics as cm  # noqa: E402
from dp_multi_eval.bag_reader import BagSeries, EgoSample, ObjectSample  # noqa: E402

RED, GREEN = 1, 3
WALL0 = 1000.0  # wall = recording time + 1000


def ego(t, x, v, wall=False):
    return EgoSample(stamp_sec=(WALL0 + t) if wall else t, x=x, y=0.0, z=0.0, yaw_rad=0.0, vx=v, vy=0.0, speed_mps=v)


def profile(stops, end=60.0):
    """Drive +x at 5 m/s for ``end`` s; ``stops`` = [(t_start, duration)] full stops.
    Returns run samples (wall stamps) at 10 Hz."""
    out, x, t = [], 0.0, 0.0
    while t < end:
        v = 0.0 if any(s <= t < s + d for s, d in stops) else 5.0
        x += v * 0.1
        out.append(ego(round(t, 1), x, v, wall=True))
        t += 0.1
    return out


def leg_payload(arrival_x, departures=(), drive_through=True, arrived=True, end_dist_m=None):
    return {
        "params": {"lead_sec": 3.5},
        "legs": [{
            "index": 1, "kind": "start", "goal": [arrival_x, 0.0, 0.0], "arrival": [arrival_x, 0.0, 0.0],
            "t_start": 0.0, "t_end": 60.0, "depart_t": None, "arrive_t": 60.0, "drive_through": drive_through,
            "arrived_t": (WALL0 + 58.0) if arrived else None, "end_dist_m": end_dist_m, "ended_by": "window_end", "wall_start": WALL0, "wall_end": WALL0 + 60.0,
            "gaps": [], "jumps": [], "departures": list(departures),
            "segments": [{"t_open": 0.0, "t_handover": 3.5, "wall_handover": WALL0 + 3.5, "x": 17.5, "y": 0.0,
                          "v": 5.0, "t_end": 60.0, "ended_by": "window_end"}],
        }],
    }


def run_metric(run, payload, light=None, stops=None, objects=(), clamp=False, shield_log=None, rec=None):
    """light: None = no signals; a colour the (fake) light ahead shows; or
    [(t, colour)] changes. clamp: the scene stops at the recording's end.
    stops: [(name, x, y)] for the map's stop_points.csv; None = no stop list."""
    rec = rec or [ego(i * 0.1, i * 0.5, 5.0) for i in range(600)]  # the recording never stops
    saved = (cm._light_states, cm._load_lanelet_map, cm._light_ahead)
    if isinstance(light, list):
        timeline = [(WALL0 + t, {7: c}) for t, c in light]
        cm._light_states = lambda bag, topics: timeline
    else:
        cm._light_states = (lambda bag, topics: [(WALL0, {7: light})]) if light is not None else (lambda bag, topics: [])
    cm._load_lanelet_map = lambda p: object()
    cm._light_ahead = lambda lmap, x, y, yaw: 7
    try:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "windows.json").write_text(json.dumps(payload))
            if shield_log is not None:
                (Path(d) / "shield_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in shield_log))
            map_path = Path("/nonexistent")
            if stops is not None:
                map_path = Path(d) / "map"
                map_path.mkdir()
                (map_path / "stop_points.csv").write_text(
                    "name,x,y,yaw\n" + "".join(f"{n},{x},{y},0.0\n" for n, x, y in stops))
            return cm.metric_windows(Path(d) / "output", BagSeries(ego=run, objects=list(objects)), BagSeries(ego=rec),
                                     scene_time=(lambda w: min(w - WALL0, 59.9)) if clamp else (lambda w: w - WALL0),
                                     map_path=map_path)
    finally:
        cm._light_states, cm._load_lanelet_map, cm._light_ahead = saved


def main() -> int:
    run = profile([(30.0, 4.0)])  # one 4 s stop at x≈150, the recording drove through
    arrival_x = run[-1].x

    w = run_metric(run, leg_payload(arrival_x))
    leg = w["legs"][0]
    assert w["legs_total"] == 1 and w["legs_passed"] == 1 and leg["passed"], leg
    assert len(leg["stop_go"]["long_stops"]) == 1 and abs(leg["stop_go"]["long_stops"][0][1] - 4.0) < 0.3, leg["stop_go"]
    assert leg["stop_go"]["red_light_waits"] == [] and leg["reached"], leg
    assert leg["drive"] is not None and abs(leg["drive"]["ratio"] - 1.0) < 0.1, leg["drive"]  # same 60 s span
    print("PASS: a 4 s stop the recording did not make is reported; the leg passes")

    w = run_metric(run, leg_payload(arrival_x), light=RED)
    leg = w["legs"][0]
    assert leg["passed"] and w["legs_passed"] == 1, leg
    assert leg["stop_go"]["long_stops"] == [] and len(leg["stop_go"]["red_light_waits"]) == 1, leg["stop_go"]
    print("PASS: the same stop under a red light ahead is a red_light_wait; the leg passes")

    w = run_metric(run, leg_payload(arrival_x), light=GREEN)
    sg = w["legs"][0]["stop_go"]
    assert w["legs"][0]["passed"] and sg["red_light_waits"] == [] and len(sg["long_stops"]) == 1, sg
    print("PASS: under a green light it stays an unplanned stop, reported")

    # stop at red, creep off, stop again 2 s later while still red: one start-brake cycle under red
    run2 = profile([(30.0, 4.0), (37.0, 2.0)])
    w = run_metric(run2, leg_payload(run2[-1].x), light=RED)
    leg = w["legs"][0]
    assert leg["passed"], leg
    assert leg["stop_go"]["red_light_brakes"] >= 1 and leg["stop_go"]["start_brake_cycles"] == 0, leg["stop_go"]
    print("PASS: a brake cycle ending under red is a red_light_brake and not counted")

    # stop and go three times under green (giving way, a crossing): reported, the leg passes
    run3 = profile([(20.0, 2.0), (25.0, 2.0), (30.0, 2.0)])
    w = run_metric(run3, leg_payload(run3[-1].x), light=GREEN)
    leg = w["legs"][0]
    assert leg["passed"] and leg["stop_go"]["start_brake_cycles"] >= 2, leg["stop_go"]
    print("PASS: start-brake cycles are reported; the leg passes")

    late = [{"t": 20.0, "at": [100.0, 0.0], "delay_sec": 17.1}]
    w = run_metric(run, leg_payload(arrival_x, departures=late), light=RED)
    leg = w["legs"][0]
    assert leg["passed"] and len(leg["did_not_depart"]) == 1 and leg["did_not_depart"][0]["delay_sec"] == 17.1
    print("PASS: a departure missed by 17 s is reported; the leg passes")

    # goal stop: the run ends standing at the arrival pose shifted by `off` metres
    for off, ok in ((3.0, False), (1.0, True)):
        end = profile([(56.0, 4.0)])  # stops for good at t=56 s
        w = run_metric(end, leg_payload(end[-1].x + off, drive_through=False))
        leg = w["legs"][0]
        gs = leg["goal_stop"]
        assert gs["status"] == "ok" and abs(gs["position_m"] - off) < 0.3, gs
        assert leg["passed"] is ok and gs["flagged"] is (not ok), (off, leg)
    print("PASS: a stop 3 m off the goal fails the leg; 1 m off passes")

    # ... at a bus stop. Where the recording only waited in traffic (no map stop
    # within 12 m), coming to rest 3 m off is reported and the leg passes.
    end = profile([(56.0, 4.0)])
    goal_x = end[-1].x + 3.0
    w = run_metric(end, leg_payload(goal_x, drive_through=False), stops=[("stop_a", goal_x + 2.0, 0.0)])
    gs = w["legs"][0]["goal_stop"]
    assert not w["legs"][0]["passed"] and gs["flagged"] and gs["gating"] and gs["bus_stop"] == "stop_a", gs
    w = run_metric(end, leg_payload(goal_x, drive_through=False), stops=[("stop_a", goal_x + 70.0, 0.0)])
    gs = w["legs"][0]["goal_stop"]
    assert w["legs"][0]["passed"] and gs["flagged"] and not gs["gating"] and gs["bus_stop"] is None, gs
    print("PASS: 3 m off a bus-stop goal fails; 3 m off a traffic wait is reported")

    # stuck: 50 s standing still 25 m from the goal while the recording drove
    stuck_run = profile([(5.0, 50.0)])
    w = run_metric(stuck_run, leg_payload(stuck_run[-1].x))
    leg = w["legs"][0]
    assert not leg["passed"] and len(leg["stuck"]) == 1 and leg["stuck"][0][1] >= 45.0, leg
    w = run_metric(stuck_run, leg_payload(stuck_run[-1].x), light=RED)
    assert w["legs"][0]["passed"] and w["legs"][0]["stuck"] == [], w["legs"][0]
    print("PASS: 50 s standing away from the goal is stuck and fails; at a red light it is not")

    # drove past: the ego went through the goal (0 m) and never arrived
    w = run_metric(run, leg_payload(arrival_x, drive_through=False, arrived=False))
    leg = w["legs"][0]
    assert not leg["passed"] and leg["goal_missed"] == "drove_past", leg
    # late: the goal is 40 m beyond where the ego got to when the window closed
    w = run_metric(run, leg_payload(arrival_x + 40.0, drive_through=False, arrived=False))
    leg = w["legs"][0]
    assert leg["passed"] and leg["goal_missed"] is None and not leg["reached"], leg
    # ... unless an intermediate leg's late grace ran out on it
    w = run_metric(run, leg_payload(arrival_x + 40.0, drive_through=False, arrived=False, end_dist_m=40.0))
    assert w["legs"][0]["goal_missed"] == "late_grace_ran_out" and not w["legs"][0]["passed"], w["legs"][0]
    print("PASS: a drive past the goal or a run-out grace fails; late at the window end passes")

    # H82, 2026-11-11: the window closed on an ego rolling in to its stop, 0.7 m
    # short at 0.4 m/s — late, no drive past. One that went 9 m on and stood
    # there (H39) did drive past.
    def ending(tail):
        out = profile([], end=54.0)
        x = out[-1].x
        for k, v in enumerate(tail):
            x += v * 0.1
            out.append(ego(round(54.0 + 0.1 * k, 1), x, v, wall=True))
        return out
    rolling = ending([0.4] * 60)
    w = run_metric(rolling, leg_payload(rolling[-1].x + 0.7, drive_through=False, arrived=False))
    assert w["legs"][0]["goal_missed"] != "drove_past" and not w["legs"][0]["reached"], w["legs"][0]
    beyond = ending([5.0] * 20 + [0.0] * 40)
    w = run_metric(beyond, leg_payload(beyond[-1].x - 9.0, drive_through=False, arrived=False))
    assert w["legs"][0]["goal_missed"] == "drove_past" and not w["legs"][0]["passed"], w["legs"][0]
    print("PASS: rolling in when the window closes is late, not a drive past; 9 m on from the stop is one")

    # H82, 2026-11-11: the lag of the last window is the one at the recording's
    # end. An ego that stood 20 s with no reason is 20 s behind when the 60 s
    # recording ends; the window runs on 20 s and by the stopped clock it has
    # caught up when it closes — not completed all the same.
    behind = profile([(10.0, 20.0)], end=80.0)
    pl = leg_payload(10_000.0, drive_through=False, arrived=False)
    pl["legs"][0]["wall_end"] = WALL0 + 80.0
    leg = run_metric(behind, pl, clamp=True)["legs"][0]
    assert leg["goal_missed"] == "did_not_complete" and 19.0 <= leg["incomplete"]["lag_sec"] <= 21.0, leg
    print("PASS: the last window's lag is measured at the recording's end, not after the grace against a stopped clock")

    # A reset is a window boundary (2026-11-10). Segment 1 ends at 30 s, the ego is
    # set back onto the recording (standing 30-34 s while placed), segment 2 runs on.
    def two_segments(kind):
        pl = leg_payload(arrival_x)
        seg1 = dict(pl["legs"][0]["segments"][0], t_end=30.0, ended_by=f"diverged:{kind}", wall_end=WALL0 + 30.0)
        seg2 = dict(pl["legs"][0]["segments"][0], t_handover=34.0, wall_handover=WALL0 + 34.0,
                    ended_by="window_end", wall_end=WALL0 + 60.0)
        pl["legs"][0]["segments"] = [seg1, seg2]
        return pl

    for kind, passed, inconclusive, missed in (("slow", True, False, None),
                                               ("through", True, True, None), ("lateral", True, True, None)):
        leg = run_metric(run, two_segments(kind))["legs"][0]
        assert leg["stop_go"]["long_stops"] == [], ("the reset's own standstill was scored", leg["stop_go"])
        assert [w["closed_by"] for w in leg["windows"]] == [f"diverged:{kind}", "window_end"], leg["windows"]
        assert leg["passed"] is passed and leg["inconclusive"] is inconclusive and leg["goal_missed"] == missed, (kind, leg)
    # runs from before segment wall_end existed: the window ends lead + 2 s before the next hand-over
    pl = two_segments("through")
    del pl["legs"][0]["segments"][0]["wall_end"]
    leg = run_metric(run, pl)["legs"][0]
    assert leg["windows"][0]["sec"] == round(34.0 - 3.5 - 2.0 - 3.5, 1), leg["windows"]
    print("PASS: a reset is a window boundary — the reset is never scored; closing by one is no verdict, "
          "closed by the scene is inconclusive")

    # A window a reset closed 20 s behind: the ego stood 10-30 s at x=50 while
    # the recording drove on, and was reset at 35 s.
    behind = profile([(10.0, 20.0)])
    pl = leg_payload(behind[-1].x)
    seg1 = dict(pl["legs"][0]["segments"][0], t_end=35.0, ended_by="diverged:slow", wall_end=WALL0 + 35.0)
    seg2 = dict(pl["legs"][0]["segments"][0], t_handover=39.0, wall_handover=WALL0 + 39.0,
                ended_by="window_end", wall_end=WALL0 + 60.0)
    pl["legs"][0]["segments"] = [seg1, seg2]
    leg = run_metric(behind, pl)["legs"][0]
    assert not leg["passed"] and leg["goal_missed"] == "did_not_complete", leg
    assert leg["incomplete"]["window"] == 1 and leg["incomplete"]["unexplained_lag_sec"] >= 15.0, leg["incomplete"]
    # ... the same wait with a pedestrian crossing 15 m ahead of the base_link
    # (6.6 m ahead of the front) at 0.35 m/s from 10 to 30 s: justified.
    walker = [
        ObjectSample(stamp_sec=WALL0 + t / 5.0, object_id="p1", label="PEDESTRIAN", x=65.0,
                     y=-3.5 + 0.35 * (t / 5.0 - 10.0), yaw_rad=1.5708, length_m=0.5, width_m=0.5)
        for t in range(50, 160)
    ]
    leg = run_metric(behind, pl, objects=walker)["legs"][0]
    assert leg["passed"] and leg["goal_missed"] is None, leg
    w = leg["waits"][0]
    assert w["reasons"].get("vru", 0) >= 15.0 and w["credit_sec"] >= 15.0, w
    print("PASS: a window closed 20 s behind fails; the same lag spent waiting for a crossing pedestrian does not")


    # The recording runs out (59.9 s) while the ego stands at a light short of
    # the leg's end; the window runs 10 s past it (2026-11-10, H78).
    def end_payload():
        pl = leg_payload(10_000.0, drive_through=False, arrived=False)
        pl["legs"][0]["wall_end"] = WALL0 + 70.0
        pl["legs"][0]["segments"][0]["wall_end"] = WALL0 + 70.0
        return pl

    waiting = profile([(50.0, 30.0)], end=70.0)          # stands from 50 s to the end
    leg = run_metric(waiting, end_payload(), light=[(0.0, RED), (57.0, GREEN)], clamp=True)["legs"][0]
    assert leg["passed"] and leg["inconclusive"] and leg["goal_missed"] is None, leg
    assert leg["recording_ran_out"]["stood"] == "before_end" and leg["recording_ran_out"]["idle_sec"] < 5.0, leg
    leg = run_metric(waiting, end_payload(), light=GREEN, clamp=True)["legs"][0]
    assert not leg["passed"] and leg["goal_missed"] == "did_not_complete", leg
    late_stop = profile([(66.0, 30.0)], end=70.0)        # stops 4 s after the recording ended
    leg = run_metric(late_stop, end_payload(), light=GREEN, clamp=True)["legs"][0]
    assert leg["passed"] and leg["inconclusive"] and leg["recording_ran_out"]["stood"] == "after_end", leg
    early_stop = profile([(62.0, 30.0)], end=70.0)       # stands 8 s for nothing after it ended (H63)
    leg = run_metric(early_stop, end_payload(), light=GREEN, clamp=True)["legs"][0]
    assert not leg["passed"] and leg["goal_missed"] == "did_not_complete", leg
    # standing (justified, red) when it ran out, then moved and stopped 8 s for nothing (H63)
    again = profile([(50.0, 9.0), (62.0, 30.0)], end=70.0)
    leg = run_metric(again, end_payload(), light=[(0.0, RED), (59.0, GREEN)], clamp=True)["legs"][0]
    assert not leg["passed"] and leg["goal_missed"] == "did_not_complete", leg
    print("PASS: a stand when the recording runs out is judged on the live recording — a light that "
          "turned green 3 s before the end is inconclusive, 10 s on green fails; a stop begun after "
          "the end fails after 5 s without a reason")

    # Planned cuts (2026-11-10): leg 1 closes at a recorded departure at 30 s,
    # leg 2 opens there (placed on the recording).
    def cut_payload(run_):
        pl = leg_payload(run_[-1].x)
        l1 = dict(pl["legs"][0], t_end=30.0, arrival=None, arrive_t=None, drive_through=False,
                  closes="cut", close_pose=[150.0, 0.0, 0.0], ended_by="cut", arrived_t=None,
                  wall_end=WALL0 + 30.0, parent=1)
        l1["segments"] = [dict(pl["legs"][0]["segments"][0], t_end=30.0, ended_by="cut", wall_end=WALL0 + 30.0)]
        l2 = dict(pl["legs"][0], index=2, t_start=30.0, depart_t=30.0, opens="cut", cut_pose=[150.0, 0.0, 0.0],
                  objects_from_start=True, wall_start=WALL0 + 30.0, parent=1)
        l2["segments"] = [dict(pl["legs"][0]["segments"][0], t_handover=30.0, wall_handover=WALL0 + 34.0,
                               placed=True, ended_by="window_end", wall_end=WALL0 + 60.0)]
        pl["legs"] = [l1, l2]
        return pl

    standing_at_cut = profile([(25.0, 9.0)])            # stops 5 s before the cut, recording drives on
    w = run_metric(standing_at_cut, cut_payload(standing_at_cut))
    a, b = w["legs"]
    assert a["passed"] and a["goal_stop"]["status"] == "not_required_cut" and a["goal_missed"] is None, a
    assert (a["opens"], a["closes"], b["opens"], b["closes"]) == ("goal", "cut", "cut", "goal") and b["placed"], b
    assert w["cut_placements"] == 1, w
    late_at_cut = profile([(5.0, 20.0)])                # 20 s behind when the cut closes, no reason
    a = run_metric(late_at_cut, cut_payload(late_at_cut))["legs"][0]
    assert not a["passed"] and a["goal_missed"] == "did_not_complete" and a["incomplete"]["closed_by"] == "cut", a
    print("PASS: a window a planned cut closes has no arrival or goal stop, standing there is no failure, "
          "20 s of unexplained lag is")

    # The ego drove on past the cut's stand at 12 s (the recording left it at
    # 30 s): leg 1 closes there, leg 2 opens with the scene jumped 18 s.
    through = profile([])
    pl = cut_payload(through)
    pl["legs"][0].update(wall_end=WALL0 + 12.0, cut_passed={"t": 12.0, "wall": WALL0 + 12.0, "skipped_sec": 18.0, "ego_v": 5.0})
    pl["legs"][0]["segments"][0].update(wall_end=WALL0 + 12.0, passed_cut=True)
    pl["legs"][1]["segments"] = [dict(pl["legs"][1]["segments"][0], wall_handover=WALL0 + 12.0, t_handover=30.0,
                                      placed=None, seamless=True, scene_jumped=True, skipped_sec=18.0)]
    for colour, light in ((RED, "red"), (GREEN, "not_red")):
        w = run_metric(through, pl, light=colour)
        a, b = w["legs"]
        assert a["cut_pass"] == {"skipped_sec": 18.0, "ego_v": 5.0, "light": light}, a["cut_pass"]
        assert b["scene_jumped"] == 18.0 and not b["placed"] and not b["seamless"], b
        assert w["cut_scene_jumps"] == 1 and w["cut_red_passes"] == (1 if light == "red" else 0), w
    assert run_metric(through, pl)["legs"][0]["cut_pass"]["light"] is None        # no signals: unknown
    print("PASS: driving on past a cut's stand — the skipped wait and the light it drove under are reported; "
          "the next window opens on a scene jump")

    # A contact the scene jump made at 14 s (collision: scene_jump): the ego then
    # stands in it for good — not judged from there, the leg is inconclusive.
    hit_run = profile([(14.0, 60.0)])
    pl = leg_payload(10_000.0, drive_through=False, arrived=False)
    saved_mw = cm.metric_windows
    with_contact = lambda *a, **k: saved_mw(*a, **k, scene_jump_contacts=[{"time_sec": WALL0 + 14.0, "object_id": "car"}])  # noqa: E731
    cm.metric_windows = with_contact
    try:
        leg = run_metric(hit_run, pl)["legs"][0]
    finally:
        cm.metric_windows = saved_mw
    assert leg["passed"] and leg["inconclusive"] and leg["scene_jump_contact"]["object_id"] == "car", leg
    assert leg["stuck"] == [] and leg["goal_missed"] is None, leg
    leg = run_metric(hit_run, pl)["legs"][0]                    # without the contact: stuck and not completed
    assert not leg["passed"] and (leg["stuck"] or leg["goal_missed"]), leg
    print("PASS: after a contact the scene jump made the leg is not judged: inconclusive")

    # A mutual wait (2026-11-10): the ego stands 20 s (10-30 s, x=50) with no
    # reason while a car the shield held for it stands 8 m off its centre.
    held = [{"t": WALL0 + 10.0 + 0.5 * k, "id": "c" * 32, "label": 1, "reason": "ego", "x": 60.0, "y": 6.0,
             "l": 4.5, "w": 1.8} for k in range(41)]
    stand = profile([(10.0, 20.0)])
    leg = run_metric(stand, leg_payload(stand[-1].x), shield_log=held)["legs"][0]
    assert not leg["passed"] and leg["mutual_waits"] and leg["mutual_waits"][0]["mutual_sec"] >= 10.0, leg
    assert leg["mutual_waits"][0]["no_way_round"] == [], leg["mutual_waits"]
    # ... the shield found no way round the ego for it (through a parked truck, off the road): named
    boxed = held + [{"t": WALL0 + 15.0, "event": "detour", "id": "c" * 8, "how": "blocked", "shift_m": 0.0}]
    leg = run_metric(stand, leg_payload(stand[-1].x), shield_log=boxed)["legs"][0]
    assert not leg["passed"] and leg["mutual_waits"][0]["no_way_round"] == ["c" * 8], leg["mutual_waits"]
    deadlocked = [dict(h, reason="deadlock") for h in held]
    leg = run_metric(stand, leg_payload(stand[-1].x), shield_log=deadlocked)["legs"][0]
    assert leg["mutual_waits"] == [], leg["mutual_waits"]
    leg = run_metric(stand, leg_payload(stand[-1].x), light=RED, shield_log=held)["legs"][0]
    assert leg["mutual_waits"] == [] and leg["passed"], leg     # it stood for a red light, not for the car
    # ... but not one queueing behind the ego (two-model run, 2026-11-10: a car
    # behind the bus at its stop), nor at the leg's goal, nor where the
    # recording waited too.
    behind_ego = [dict(h, x=41.0, y=2.0) for h in held]
    leg = run_metric(stand, leg_payload(stand[-1].x), shield_log=behind_ego)["legs"][0]
    assert leg["mutual_waits"] == [] and leg["passed"], leg["mutual_waits"]
    at_goal = profile([(10.0, 50.0)])                      # arrives at x=50 and stands for good
    leg = run_metric(at_goal, leg_payload(52.0, drive_through=False), shield_log=held)["legs"][0]
    assert leg["mutual_waits"] == [], leg["mutual_waits"]
    rec_waits = [ego(r.stamp_sec - WALL0, r.x, r.speed_mps) for r in profile([(10.0, 20.0)])]   # the recording stood there too
    leg = run_metric(stand, leg_payload(stand[-1].x), shield_log=held, rec=rec_waits)["legs"][0]
    assert leg["mutual_waits"] == [] and leg["passed"], leg["mutual_waits"]
    print("PASS: standing 20 s with no reason while a car held for the ego stands ahead is a mutual wait and fails; "
          "not when the shield could not resolve it, at a red light, with the car behind the ego, at the leg's goal, "
          "or where the recording waited too")

    # A scene that was not traffic (2026-08-25): the audit found a car inside a parked truck near the ego.
    through = [{"t": WALL0 + 30.0, "audit": "through", "id": "e" * 32, "label": 1, "x": 150.0, "y": 3.0, "d": 3.4,
                "other": "1b" * 16, "other_label": 2, "depth_m": 1.8, "why": "scene/scene"}]
    w = run_metric(profile([]), leg_payload(profile([])[-1].x), shield_log=through)
    leg = w["legs"][0]
    assert leg["passed"] and leg["inconclusive"] and w["legs_inconclusive"] == 1, leg   # t_rel: from the hand-over (3.5 s)
    assert leg["scene_defects"] == [{"kind": "through", "t_rel": 26.5, "object_ids": ["e" * 8, "1b" * 4], "depth_m": 1.8,
                                     "ego_dist_m": 3.4, "why": "scene/scene"}], leg["scene_defects"]
    outside = [dict(through[0], t=WALL0 + 90.0)]
    leg = run_metric(profile([]), leg_payload(profile([])[-1].x), shield_log=outside)["legs"][0]
    assert leg["passed"] and not leg["inconclusive"] and leg["scene_defects"] == [], leg
    leg = run_metric(stand, leg_payload(stand[-1].x), shield_log=held + through)["legs"][0]
    assert not leg["passed"] and leg["mutual_waits"], leg       # a failed leg stays failed
    print("PASS: a passed leg in which a vehicle was shown inside another object near the ego is inconclusive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
