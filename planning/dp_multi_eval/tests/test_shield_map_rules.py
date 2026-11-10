"""The shield's map and light rules (2026-11-09 review) and the scene-limited
stops they feed:

* a delayed NPC stops at the red it now meets, and goes on the green;
* a vehicle held for the ego waits before a junction, not inside it — and so
  does every other class (2026-11-10);
* the plan hold leaves pedestrians alone, and holds vehicles only while the
  ego is inside a junction (with a map);
* the shield log, and a stuck stretch facing a held NPC or past the
  recording's end is scene-limited.

    python3 tests/test_shield_map_rules.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dp_multi_eval.reactive_shield import ReactiveShield, ShieldParams  # noqa: E402
from dp_multi_eval.shield_map import (  # noqa: E402
    AMBER, CIRCLE, GREEN, LEFT_ARROW, RED, ShieldMap, SignalTimeline, StopLine, permits, segment_cross,
)
from test_reactive_shield import BASE, NS, U, ego_at, mkobj  # noqa: E402


def obj(x, y, v, label=1, yaw=0.0, length=4.5):
    o = mkobj(U, x, y, yaw, v)
    o.classification = [NS(label=label)]
    o.shape.dimensions.x = length
    return o


def stamp(t):
    return int(t * 1e9 + 2e18)


def lights(*changes):
    """[(t, color)] for group 7, a circle each."""
    return [(stamp(t), NS(traffic_light_groups=[NS(traffic_light_group_id=7, elements=[NS(color=c, shape=CIRCLE, status=2)])]))
            for t, c in changes]


def run(bag_fn, n, ego_fn, sh_setup, label=1):
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, vru_real_body=True))
    sh.build_tracks([(stamp(k * 0.1), NS(objects=[bag_fn(k)])) for k in range(n)])
    sh_setup(sh)
    xs = []
    for k in range(n):
        msg = NS(objects=[bag_fn(k)])
        sh.apply(msg, ego_fn(k), BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        xs.append(msg.objects[0].kinematics.pose_with_covariance.pose.position.x if msg.objects else None)
    return xs, sh


def test_light_helpers():
    tl = SignalTimeline(lights((0, GREEN), (5, GREEN), (20, RED)))
    assert tl.state(7, 2e9 + 10) == frozenset({(GREEN, CIRCLE)}) and tl.state(7, 2e9 + 25) == frozenset({(RED, CIRCLE)})
    assert tl.state(8, 2e9 + 10) is None
    red, green, amber = frozenset({(RED, CIRCLE)}), frozenset({(GREEN, CIRCLE)}), frozenset({(AMBER, CIRCLE)})
    arrow = frozenset({(RED, CIRCLE), (GREEN, LEFT_ARROW)})
    assert permits(green, green, "straight") and not permits(red, green, "straight")
    assert permits(arrow, green, "left") and not permits(arrow, green, "right")
    assert permits(amber, amber, "straight"), "the state it crossed under in the recording"
    assert permits(None, green, "straight") and permits(red, None, "straight"), "unknown lights stop no one"
    sl = StopLine(50, -3, 50, 3, 7, 0.0)
    assert abs(segment_cross(49, 0, 51, 0, sl) - 0.5) < 1e-9 and segment_cross(51, 0, 53, 0, sl) is None
    print("PASS: lights — timeline, green / arrow / recorded-state permission, stop-line crossing")


def test_delayed_car_stops_at_the_red_it_now_meets():
    """Held 18 s behind a standing bus, a car recorded crossing on green meets
    the line on red: it waits at the line and goes on the green. Without the
    lights it drove through the red."""
    car = lambda k: obj(5.0 * k * 0.1, 0.0, 5.0)  # noqa: E731
    ego = lambda k: ego_at(15.0, -2.0, math.pi / 2) if k < 180 else ego_at(15.0, 60.0)  # noqa: E731
    shield_map = ShieldMap(stop_lines=[StopLine(50, -3, 50, 3, 7, 0.0)])

    def with_lights(sh):
        sh.set_map(shield_map)
        sh.set_signals(lights((0, GREEN), (20, RED), (40, GREEN)))

    xs, sh = run(car, 600, ego, with_lights)
    assert max(xs[:180]) < 15.0, "scenario changed: the car is not held behind the bus"
    t_line = next(k for k, x in enumerate(xs) if x >= 47.0) * 0.1
    assert 20.0 <= t_line <= 39.0, f"scenario changed: reached the line at {t_line:.1f} s, not in the red"
    waiting = [x for k, x in enumerate(xs) if 290 <= k <= 395]
    assert max(waiting) <= 47.4, f"crossed the red: front at {max(waiting) + 2.25:.1f} m (line at 50)"
    assert xs[-1] > 60.0, "never went on the green"

    xs_free, _ = run(car, 600, ego, lambda sh: sh.set_map(shield_map))
    assert max(x for k, x in enumerate(xs_free) if k <= 390) > 50.0, "scenario changed: no red to run without lights"
    print("PASS: a delayed car stops at the red it now meets and goes on the green")


def test_car_waits_before_a_junction_not_inside():
    """A bus stands across the road just past a junction: the car held for it
    stops at the junction's entry, not inside it 3 m short of the bus."""
    car = lambda k: obj(6.0 * k * 0.1, 0.0, 6.0)  # noqa: E731
    ego = lambda k: ego_at(50.0, -2.0, math.pi / 2)  # noqa: E731
    junction = ShieldMap(junctions=[[(30.0, -6.0), (45.0, -6.0), (45.0, 6.0), (30.0, 6.0)]])
    xs_free, _ = run(car, 300, ego, lambda sh: None)
    assert 38.0 < max(xs_free) < 47.0, f"scenario changed: without the map it stood at {max(xs_free):.1f}"
    xs, _ = run(car, 300, ego, lambda sh: sh.set_map(junction))
    assert max(xs) <= 27.4, f"stood inside the junction: centre {max(xs):.1f} (entry for the front at 30)"
    assert max(xs) >= 25.0, f"stopped needlessly early: {max(xs):.1f}"
    # A car behind a bus waiting (to turn) inside the junction queues behind it
    # in the junction, as its own recording did (H95).
    wide = ShieldMap(junctions=[[(30.0, -6.0), (80.0, -6.0), (80.0, 6.0), (30.0, 6.0)]])
    behind = lambda k: ego_at(60.0, 0.0, 0.0)  # noqa: E731
    xs_q, _ = run(car, 300, behind, lambda sh: sh.set_map(wide))
    assert 50.0 < max(xs_q) < 54.5, f"a follower did not queue behind the bus in the junction: {max(xs_q):.1f}"
    print("PASS: a car held for the bus waits before the junction, not inside it; a follower queues behind it")


def test_every_class_waits_where_a_road_user_may_wait():
    """H78, 2026-11-10: a bus standing mid-junction across the cross street. A
    cyclist released by its green rode in and stood 1.2 m from the bus, in the
    bus's way out; a car met the bus's flank, counted as "following" it (by
    their headings where they stood) and queued inside the junction, three
    more behind it. Cyclists, pedestrians and cars alike wait before the
    junction or crosswalk they cannot get through."""
    junction = ShieldMap(junctions=[[(30.0, -8.0), (60.0, -8.0), (60.0, 8.0), (30.0, 8.0)]])
    across = lambda k: ego_at(45.0, 6.0, -math.pi / 2)  # noqa: E731  the bus's body lies across y=0
    for label, v, length, name in ((6, 4.0, 1.8, "cyclist"), (1, 6.0, 4.5, "car"), (0, 3.0, 2.0, "unknown")):
        user = lambda k, v=v, label=label, length=length: obj(v * k * 0.1, 0.0, v, label=label, length=length)  # noqa: E731
        xs_free, _ = run(user, 300, across, lambda sh: None)
        assert max(xs_free) > 36.0, f"scenario changed: the {name} without a map stood at {max(xs_free):.1f}"
        xs, sh = run(user, 300, across, lambda sh: sh.set_map(junction))
        front = max(xs) + max(length, 1.0) / 2.0
        assert front <= 30.0, f"the {name} went into the junction it could not cross: front at {front:.1f}"
        assert front >= 27.5, f"the {name} stopped needlessly early: front at {front:.1f}"

    # A pedestrian whose crosswalk the bus stands on waits at the kerb.
    crosswalk = ShieldMap(crosswalks=[[(8.0, -2.0), (22.0, -2.0), (22.0, 2.0), (8.0, 2.0)]])
    walker = lambda k: obj(1.4 * k * 0.1, 0.0, 1.4, label=7, length=0.6)  # noqa: E731
    on_it = lambda k: ego_at(15.0, 6.0, -math.pi / 2)  # noqa: E731
    xs_free, _ = run(walker, 300, on_it, lambda sh: None)
    assert 11.0 < max(xs_free) < 13.5, f"scenario changed: without the map the pedestrian stood at {max(xs_free):.1f}"
    xs, _ = run(walker, 300, on_it, lambda sh: sh.set_map(crosswalk))
    assert max(xs) <= 7.6, f"the pedestrian stood on the crosswalk, {max(xs):.1f} (kerb at 8)"
    # ... and walks when the bus is clear of it.
    clear = lambda k: ego_at(15.0, 30.0, -math.pi / 2)  # noqa: E731
    xs, _ = run(walker, 300, clear, lambda sh: sh.set_map(crosswalk))
    far = max(x for x in xs if x is not None)
    assert far > 35.0, f"the pedestrian did not cross a free crosswalk: {far:.1f}"

    # The flank is not the tail: a car that meets the side of a bus turning
    # inside the junction, heading within 45 deg of it, waits outside too.
    turning = lambda k: ego_at(47.0, 5.0, -math.radians(40.0))  # noqa: E731
    car = lambda k: obj(6.0 * k * 0.1, 0.0, 6.0)  # noqa: E731
    xs, _ = run(car, 300, turning, lambda sh: sh.set_map(junction))
    assert max(xs) + 2.25 <= 30.0, f"a car meeting the bus's flank queued inside the junction: front {max(xs) + 2.25:.1f}"
    print("PASS: cyclist, car, unknown and pedestrian wait before the junction or crosswalk they cannot get through; "
          "the flank is not the tail")


def _plan_case(label, junction_poly):
    """A road user standing at x=2 on y=0 (sets off +x at k=30); the bus at (10, 15)
    facing south, its plan crossing that way."""
    v = 1.5 if label == 7 else 3.0
    user = lambda k: obj(2.0 + max(0, k - 30) * 0.1 * v, 0.0, v if k >= 30 else 0.0, label=label,  # noqa: E731
                         length=0.6 if label == 7 else 4.5)
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True,
                                     vru_real_body=True, ego_plan_sec=5.0))
    sh.build_tracks([(stamp(k * 0.1), NS(objects=[user(k)])) for k in range(200)])
    if junction_poly is not None:
        sh.set_map(ShieldMap(junctions=[junction_poly]))
    plan = [(i * 0.25, 10.0, 15.0 - i * 1.25, -math.pi / 2) for i in range(21)]
    ego = ego_at(10.0, 15.0, -math.pi / 2)
    xs = []
    for k in range(200):
        sh.set_ego_plan(plan)
        msg = NS(objects=[user(k)])
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        xs.append(msg.objects[0].kinematics.pose_with_covariance.pose.position.x if msg.objects else None)
    return xs


def test_plan_hold_scope():
    """The bus gives way to pedestrians, not they to its plan; a vehicle is held
    by the plan only once the bus is inside a junction."""
    around_bus = [(0.0, 0.0), (20.0, 0.0), (20.0, 30.0), (0.0, 30.0)]
    elsewhere = [(100.0, 100.0), (120.0, 100.0), (120.0, 120.0), (100.0, 120.0)]
    assert max(_plan_case(1, around_bus)[:150]) <= 2.6, "a car set off into the plan of a bus inside the junction"
    assert max(_plan_case(1, None)[:150]) <= 2.6, "without a map the plan holds vehicles anywhere"
    assert _plan_case(1, elsewhere)[100] > 5.0, "held by the plan of a bus outside any junction"
    assert _plan_case(7, around_bus)[100] > 3.0, "a pedestrian waited for the bus's plan"
    print("PASS: plan hold — vehicles only, and with a map only while the bus is in a junction")


def test_no_hold_reported_where_the_recording_stands():
    """A car standing in the recording, the ego touching its rear: the shield
    changes nothing, so it reports no hold (H39: a parked car was logged held
    for 38 s and the ego's wait behind it called the scene's)."""
    parked = lambda k: obj(20.0, 0.0, 0.0)  # noqa: E731
    ego = lambda k: ego_at(20.0 - 2.25 - 7.56 - 0.05, 0.0)  # noqa: E731  front bumper at its rear
    _, sh = run(parked, 100, ego, lambda sh: None)
    assert not sh.held_reason, sh.held_reason
    held_back = lambda k: obj(20.0 + max(0, k - 30) * 0.3, 0.0, 3.0 if k >= 30 else 0.0)  # noqa: E731
    xs, sh2 = run(held_back, 100, lambda k: ego_at(35.0, -2.0, math.pi / 2), lambda sh: None)
    assert sh2.held_reason.get(U) in ("ego", "plan"), sh2.held_reason
    print("PASS: a hold is reported only where the shield puts the NPC behind its recording")


def test_shield_log_and_scene_limited_stops():
    from dp_multi_eval.bag_reader import EgoSample
    from dp_multi_eval.compute_metrics import _scene_limited, _shield_log
    from dp_multi_eval.scene_driver import ShieldLog

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "shield_log.jsonl"
        log = ShieldLog(str(path))
        bike = obj(8.0, 0.5, 0.0, label=6, length=1.8)
        bike.shape.dimensions.y = 0.6
        side = mkobj(bytes([9] * 16), 8.0, 6.0, 0.0, 0.0)
        side.classification = [NS(label=1)]
        for k in range(41):   # 100-120 s
            t = 100.0 + k * 0.5
            log.write(t, [bike, side], {U: "ego", bytes([9] * 16): "ego"} if k < 40 else {}, at_end=(t >= 112.0))
        log.close()
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        assert [x["event"] for x in lines if "event" in x] == ["recording_end"], lines[:3]
        holds, ends = _shield_log(path)
        assert len(holds) == 80 and ends == [(112.0, float("inf"))], (len(holds), ends)

        ego = [EgoSample(stamp_sec=100.0 + k * 0.1, x=0.0, y=0.0, z=0.0, yaw_rad=0.0, vx=0.0, vy=0.0, speed_mps=0.0)
               for k in range(201)]
        ref = [EgoSample(stamp_sec=float(k), x=-50.0 + k, y=0.0, z=0.0, yaw_rad=0.0, vx=1.0, vy=0.0, speed_mps=1.0)
               for k in range(150)]
        # held for the ego ("ego"): it waited for the bus — no limit of the
        # scene, the mutual wait is the model's (operator, 2026-11-10)
        assert _scene_limited(100.0, 111.0, ego, ref, holds, []) is None
        # one the shield could not lead round the bus ("deadlock") is
        stuck = [dict(h, reason="deadlock") for h in holds]
        held = _scene_limited(100.0, 111.0, ego, ref, stuck, [])
        assert held == {"cause": "held_npc", "object_id": U.hex()}, held
        only_side = [h for h in stuck if h["id"] != U.hex()]
        assert _scene_limited(100.0, 111.0, ego, ref, only_side, []) is None, "a car beside the way is no deadlock"
        assert _scene_limited(112.0, 120.0, ego, ref, [], ends) == {"cause": "after_recording"}
        assert _scene_limited(100.0, 111.0, ego, ref, [], ends) is None
    print("PASS: shield log — held NPCs and the recording's end; stuck facing an NPC the shield could not "
          "lead round (deadlock), or past the end, is scene-limited; one merely waiting for the ego is not")


def main() -> int:
    test_light_helpers()
    test_delayed_car_stops_at_the_red_it_now_meets()
    test_car_waits_before_a_junction_not_inside()
    test_every_class_waits_where_a_road_user_may_wait()
    test_plan_hold_scope()
    test_no_hold_reported_where_the_recording_stands()
    test_shield_log_and_scene_limited_stops()
    return 0


if __name__ == "__main__":
    sys.exit(main())
