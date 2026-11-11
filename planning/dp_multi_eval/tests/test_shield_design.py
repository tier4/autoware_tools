"""The shield's general mechanisms (2026-11-10), each on a synthetic case, no
recording or scene in it:

* space-time: a moving ego is judged where it will be when the NPC gets there;
* a vehicle that gives way waits before the ego's planned way, not on it;
* who goes first: a vehicle ahead on the ego's way is never held for it;
  a vehicle at its line does not enter the junction across a moving ego's
  plan that leaves the recorded way;
  pedestrians and cyclists keep to the ego's body only, never to its plan;
* a delayed NPC lives on to its own track's end after the recording lets it go;
* deadlock: an NPC held on the way of a standing ego is led round it (a lateral
  detour), or reported "deadlock" when it cannot be.

    python3 tests/test_shield_design.py
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.reactive_shield import ReactiveShield, ShieldParams  # noqa: E402
from dp_multi_eval.scene_driver import ObjectGate  # noqa: E402

BASE_NS = 2 * 10**18
T0 = BASE_NS * 1e-9
DT = 0.1


def obj(n, x, y, yaw, v, label=1, length=4.5, width=1.8):
    q = NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    return NS(
        object_id=NS(uuid=bytes([n] * 16)),
        classification=[NS(label=label)],
        shape=NS(dimensions=NS(x=length, y=width, z=1.5)),
        kinematics=NS(
            pose_with_covariance=NS(pose=NS(position=NS(x=x, y=y, z=0.0), orientation=q)),
            twist_with_covariance=NS(twist=NS(linear=NS(x=v, y=0.0, z=0.0))),
        ),
    )


def ego(x, y=0.0, yaw=0.0):
    return NS(position=NS(x=x, y=y, z=0.0), orientation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))


def straight(x0, y0, yaw, v):
    """A road user going straight at v from (x0, y0): its pose at time t."""
    return lambda t: (x0 + v * t * math.cos(yaw), y0 + v * t * math.sin(yaw))


def play(movers, ego_fn, n, *, plan_fn=None, params=None, gate=False, present=None, route=None, shield_map=None):
    """Run n ticks. movers: {id: (pose_fn(t), yaw, v, label, length, width)}.
    ego_fn(t) -> (x, y, yaw, speed); plan_fn(t) -> [(s, x, y, yaw)].
    present(id, t): whether the recording has it then (default always).
    Returns ({id: [(t, x, y, reported speed, heading) or None]}, shield)."""
    p = params or ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, vru_real_body=True,
                               npc_following=True, ego_plan_sec=5.0, deadlock_sec=3.0)
    sh = ReactiveShield(p)

    def frame(k):
        t = k * DT
        out = []
        for n_, (fn, yaw, v, label, length, width) in movers.items():
            if present is None or present(n_, t):
                x, y = fn(t)
                out.append(obj(n_, x, y, yaw, v(t) if callable(v) else v, label, length, width))
        return out

    sh.build_tracks([(BASE_NS + int(k * DT * 1e9), NS(objects=frame(k))) for k in range(n)])
    if shield_map is not None:
        sh.set_map(shield_map)
    if route is not None:   # the recorded vehicle: ego_fn over the recording
        ts = [T0 + k * DT for k in range(n)]
        poses = [route(k * DT) for k in range(n)]
        sh.set_route([p[0] for p in poses], [p[1] for p in poses], ts, [p[2] for p in poses])
    g = ObjectGate(shield=sh) if gate else None
    seen = {n_: [] for n_ in movers}
    for k in range(n):
        t = k * DT
        ex, ey, eyaw, ev = ego_fn(t)
        if plan_fn is not None:
            sh.set_ego_plan(plan_fn(t))
        msg = NS(objects=frame(k))
        if g is not None:
            out = g.apply(msg, ego_pose=ego(ex, ey, eyaw), ego_speed=ev, ts_ns=BASE_NS + int(t * 1e9),
                          at_end=False, objects_on=True, now_ns=BASE_NS + int(t * 1e9)).objects
        else:
            sh.apply(msg, ego(ex, ey, eyaw), T0 + t, DT, ev)
            out = msg.objects
        got = {o.object_id.uuid[0]: o for o in out}
        for n_ in movers:
            o = got.get(n_)
            if o is None:
                seen[n_].append(None)
            else:
                pp = o.kinematics.pose_with_covariance.pose.position
                q = o.kinematics.pose_with_covariance.pose.orientation
                lin = o.kinematics.twist_with_covariance.twist.linear
                seen[n_].append((t, pp.x, pp.y, math.hypot(lin.x, lin.y),
                                 math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))))
    return seen, sh


def driving_ego(x0, v):
    return lambda t: (x0 + v * t, 0.0, 0.0, v)


def driving_plan(x0, v):
    return lambda t: [(s * 0.1, x0 + v * (t + s * 0.1), 0.0, 0.0) for s in range(51)]


def standing_plan(x, y=0.0, yaw=0.0):
    return lambda t: [(0.0, x, y, yaw)]


def test_spacetime_moving_ego():
    """A car crossing where the bus is now, arriving when the bus has gone
    (at x=3, in 4 s; the bus leaves at 5 m/s): not held. The same car meeting
    a bus that stands there is held."""
    car = {1: (straight(3.0, -20.0, math.pi / 2, 5.0), math.pi / 2, 5.0, 1, 4.5, 1.8)}
    seen, _ = play(car, driving_ego(0.0, 5.0), 80, plan_fn=driving_plan(0.0, 5.0))
    y_at_4 = seen[1][40][2]
    assert y_at_4 > -1.0, f"held for where the bus was, not where it is: y {y_at_4:.1f} at 4 s (recorded 0)"
    seen, _ = play(car, lambda t: (0.0, 0.0, 0.0, 0.0), 80, plan_fn=standing_plan(0.0))
    assert max(s[2] for s in seen[1] if s) < -1.5, "a car drove into a standing bus"
    print("PASS: space-time — a car crossing where the bus was is not held; where it stands, it is")


def test_gives_way_before_the_ego_way():
    """The bus will be at x 0..12 (its plan) when the car crossing at x=10
    arrives: the car waits before the bus's planned way (front short of
    y=-1.4), not on it, and crosses after."""
    car = {1: (straight(10.0, -30.0, math.pi / 2, 5.0), math.pi / 2, 5.0, 1, 4.5, 1.8)}
    seen, sh = play(car, driving_ego(-30.0, 5.0), 160, plan_fn=driving_plan(-30.0, 5.0))
    ys = [s[2] for s in seen[1] if s]
    pts = seen[1]
    # standing: unmoved for 0.5 s (a sample shown for two ticks is no wait)
    waits = [pts[k] for k in range(len(pts) - 5)
             if all(pts[k + j] and abs(pts[k + j][2] - pts[k][2]) < 0.01 for j in range(1, 6))]
    assert waits, "scenario changed: the car never gave way"
    assert max(w[2] for w in waits) + 2.25 <= -1.4, f"waited on the bus's way: front at {max(w[2] for w in waits) + 2.25:.1f}"
    assert ys[-1] > 10.0, "never crossed after the bus"
    print("PASS: a car that gives way waits before the bus's planned way, then crosses")


def test_who_goes_first():
    """A slower car ahead in the bus's lane is not held for the bus closing
    in behind it; a pedestrian is not held for the bus's plan."""
    car = {1: (straight(30.0, 0.0, 0.0, 3.0), 0.0, 3.0, 1, 4.5, 1.8)}
    seen, _ = play(car, driving_ego(0.0, 8.0), 60, plan_fn=driving_plan(0.0, 8.0))
    xs = [s[1] for s in seen[1][:30]]
    assert xs[-1] - xs[0] > 7.5, f"the car ahead was held for the bus behind it: moved {xs[-1] - xs[0]:.1f} m in 3 s"
    # pedestrian crossing at x=12, 1.4 m/s from y=-5; the bus (5 m/s from x=-8)
    # covers x=12 from 2.4 s: before that the pedestrian walks as recorded
    ped = {2: (straight(12.0, -5.0, math.pi / 2, 1.4), math.pi / 2, 1.4, 7, 0.6, 0.6)}
    seen, _ = play(ped, driving_ego(-8.0, 5.0), 40, plan_fn=driving_plan(-8.0, 5.0))
    y_at_2 = seen[2][20][2]
    assert y_at_2 > -2.5, f"a pedestrian was held for the bus's plan: y {y_at_2:.1f} at 2 s (recorded -2.2)"
    print("PASS: who goes first — the car ahead goes, the pedestrian is not held for the plan")


def test_not_held_for_a_bus_that_will_have_gone():
    """A pedestrian waiting at the kerb (x=2, y=-4.5) sets off north at 1 s; a
    bus at 3 m/s is over x=2 then but past it when the pedestrian gets there.
    Not held (the static rule held it until the bus's rear had passed). A bus
    standing over x=2 still stops it short of its body."""
    walk = lambda t: (2.0, -4.5 + 1.4 * max(0.0, t - 1.0))  # noqa: E731
    ped = {2: (walk, math.pi / 2, lambda t: 0.0 if t < 1.0 else 1.4, 7, 0.6, 0.6)}
    seen, _ = play(ped, driving_ego(0.0, 3.0), 50, plan_fn=driving_plan(0.0, 3.0))
    y_at_2 = seen[2][20][2]
    assert y_at_2 > -3.6, f"held for a bus that will have gone: y {y_at_2:.1f} at 2 s (recorded -3.1)"
    seen, sh = play(ped, lambda t: (0.0, 0.0, 0.0, 0.0), 80, plan_fn=standing_plan(0.0))
    assert max(s[2] for s in seen[2] if s) + 0.3 < -1.435, "walked into a standing bus"
    print("PASS: a pedestrian is not held for a moving bus that will have gone; a standing one still stops it")


def test_following_and_crossing_behind_are_not_held():
    """A car following the bus 17 m behind at its speed, and one crossing
    the bus's way 1.5 s after the bus has passed there: neither is slowed —
    the recording itself had them so."""
    follow = {4: (straight(-20.0, 0.0, 0.0, 8.0), 0.0, 8.0, 1, 4.5, 1.8)}
    seen, _ = play(follow, driving_ego(0.0, 8.0), 100, plan_fn=driving_plan(0.0, 8.0))
    x_end = seen[4][95][1]
    assert x_end > -20.0 + 8.0 * 9.5 - 1.0, f"a follower was slowed: {x_end:.1f} m (recorded {-20 + 76.0:.1f})"
    # the bus's rear passes x=21 at t=(21+3.32)/5=4.9 s; the car's front reaches the bus's lane at 5.5 s
    cross = {5: (straight(20.0, -17.6, math.pi / 2, 2.5), math.pi / 2, 2.5, 1, 4.5, 1.8)}
    seen, _ = play(cross, driving_ego(0.0, 5.0), 100, plan_fn=driving_plan(0.0, 5.0))
    y_at_8 = seen[5][80][2]
    assert y_at_8 > 2.4 - 0.5, f"held although it crossed behind the bus: y {y_at_8:.1f} at 8 s (recorded 2.4)"
    print("PASS: a follower and a car crossing behind the bus are not slowed")


def test_the_recording_is_the_reference():
    """The recorded bus (5 m/s from x=-30) slowed to let a car cross x=10 just
    ahead of it (front clear by 0.6 s, under the 1 s margin). Driving that very
    recording, the car is not held: the recording shows the crossing is
    acceptable. Without the recorded vehicle to compare with, the margin holds it."""
    bus = lambda t: (-30.0 + 5.0 * t, 0.0, 0.0)  # noqa: E731
    car = {6: (straight(10.0, -12.0, math.pi / 2, 2.5), math.pi / 2, 2.5, 1, 4.5, 1.8)}
    ego_fn = lambda t: (*bus(t)[:2], 0.0, 5.0)  # noqa: E731
    seen, _ = play(car, ego_fn, 80, plan_fn=driving_plan(-30.0, 5.0), route=bus)
    y_free = seen[6][60][2]
    seen, _ = play(car, ego_fn, 80, plan_fn=driving_plan(-30.0, 5.0))
    y_held = seen[6][60][2]
    assert y_free > 2.0, f"held although the recording itself crossed so: y {y_free:.1f} at 6 s (recorded 3.0)"
    assert y_held < y_free - 1.0, f"scenario changed: the margin alone does not hold it (y {y_held:.1f})"
    print(f"PASS: the recording is the reference — its own crossing stands (y {y_free:.1f} vs held {y_held:.1f} at 6 s)")


def test_inside_a_junction_clears_first():
    """The same crossing as above, but the car is already inside the junction
    (x 0..20, y -10..10) when the bus comes: it does not stop there for the
    bus's plan — it clears the junction."""
    from dp_multi_eval.shield_map import ShieldMap
    car = {1: (straight(10.0, -8.0, math.pi / 2, 2.5), math.pi / 2, 2.5, 1, 4.5, 1.8)}
    p = ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, vru_real_body=True,
                     npc_following=True, ego_plan_sec=5.0, deadlock_sec=3.0)
    seen, sh = None, None
    sh_map = ShieldMap(junctions=[[(0.0, -10.0), (20.0, -10.0), (20.0, 10.0), (0.0, 10.0)]])
    orig = ReactiveShield.build_tracks

    def with_map(self, data):
        orig(self, data)
        self.set_map(sh_map)
    ReactiveShield.build_tracks = with_map
    try:
        seen, sh = play(car, driving_ego(-14.0, 5.0), 60, plan_fn=driving_plan(-14.0, 5.0), params=p)
    finally:
        ReactiveShield.build_tracks = orig
    ys = [s[2] for s in seen[1] if s]
    assert ys[30] > -3.0, f"held inside the junction for the bus's plan: y {ys[30]:.1f} at 3 s (recorded -0.5; held: -6.7)"
    print("PASS: a car already inside the junction clears it, not held there for the bus's plan")


def test_off_route_plan_holds_at_the_line():
    """A slow ego inside a junction (x 0..30, y -10..10) plans to cut right,
    off the recorded straight line (y=0), across the way of a car standing at
    its line outside (x=24, y=-14) that sets off at 0.5 s; where the ego is,
    and where it will be in the next second, are clear of that way. The car waits at its
    line — the recording never had the bus there — and goes once the plan is
    back on the recorded way. With the plan on the recorded way it is not held.
    H78 planned cuts (2026-11-10): a car on green drove into the junction at a
    bus turning 3.7 m inside the recorded line and stood 3 m off its side."""
    from dp_multi_eval.shield_map import ShieldMap
    car = {7: (lambda t: (24.0, -14.0 + 6.0 * max(0.0, t - 0.5)), math.pi / 2,
               lambda t: 6.0 if t >= 0.5 else 0.0, 1, 4.5, 1.8)}
    bus = lambda t: (-10.0 + 5.0 * t, 0.0, 0.0)  # noqa: E731
    ego_fn = lambda t: (5.0 + 1.0 * t, 0.0, 0.0, 1.0)  # noqa: E731
    yaw_cut = math.atan2(-1.2, 2.4)
    cut = lambda t: [(k * 0.1, 5.0 + t + 2.4 * k * 0.1, -1.2 * k * 0.1, yaw_cut) for k in range(51)] if t < 3.0 \
        else [(k * 0.1, 5.0 + t + 1.0 * k * 0.1, 0.0, 0.0) for k in range(51)]  # noqa: E731
    on_route = lambda t: [(k * 0.1, 5.0 + t + 1.0 * k * 0.1, 0.0, 0.0) for k in range(51)]  # noqa: E731
    sh_map = ShieldMap(junctions=[[(0.0, -10.0), (30.0, -10.0), (30.0, 10.0), (0.0, 10.0)]])
    orig = ReactiveShield.build_tracks

    def with_map(self, data):
        orig(self, data)
        self.set_map(sh_map)

    def run(plan_fn, hold=True):
        p = ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, vru_real_body=True,
                         npc_following=True, ego_plan_sec=5.0, deadlock_sec=3.0, plan_off_route_hold=hold)
        ReactiveShield.build_tracks = with_map
        try:
            seen, _ = play(car, ego_fn, 80, plan_fn=plan_fn, params=p, route=bus)
        finally:
            ReactiveShield.build_tracks = orig
        return [s[2] for s in seen[7] if s]

    held = run(cut)
    free = run(cut, hold=False)
    straight_on = run(on_route)
    assert max(held[:25]) < -12.5, f"left its line across the ego's off-route plan: y {max(held[:25]):.1f} by 2.5 s"
    assert free[25] > -10.0, f"scenario changed: without the rule the car goes (y {free[25]:.1f} at 2.5 s)"
    assert straight_on[25] > -10.0, f"held for a plan on the recorded way: y {straight_on[25]:.1f} at 2.5 s"
    assert held[-1] > -10.0, f"never let go after the plan came back on the way: y {held[-1]:.1f}"
    print("PASS: an off-route plan holds a car at its line; on the recorded way it goes")


def test_carried_past_the_recording():
    """A car held 8 s for a bus standing across its way; its recording ends
    at 10 s. It lives on and drives off when the bus leaves (at 12 s),
    instead of vanishing where it waited."""
    car = {1: (straight(0.0, -40.0, math.pi / 2, 5.0), math.pi / 2, 5.0, 1, 4.5, 1.8)}
    ego_fn = lambda t: (0.0, 0.0, math.pi / 2 + math.pi / 2, 0.0) if t < 12.0 else (0.0, 60.0, 0.0, 0.0)  # noqa: E731
    seen, _ = play(car, ego_fn, 250, gate=True, present=lambda n, t: t < 10.0)
    after = [s for s in seen[1][100:120]]
    assert all(s is not None for s in after), "it vanished when its recording ended"
    later = [s for s in seen[1][120:] if s]
    assert later and later[-1][2] > 5.0, "it never drove on after the bus left"
    print("PASS: a delayed car lives on past its recording's end and drives off")


def _deadlock_case(params=None):
    """A bus standing at the origin facing +x (nose at x=7.56), its way ahead;
    a cyclist coming the other way in its lane (y=0.3), whose path runs
    through the bus: it waits at the bus's nose, and the bus cannot go."""
    fn = lambda t: (30.0 - 4.0 * t, 0.3)  # noqa: E731
    bike = {3: (fn, math.pi, 4.0, 6, 1.8, 0.7)}
    plan = lambda t: [(0.0, 0.0, 0.0, 0.0)]  # noqa: E731
    return play(bike, lambda t: (0.0, 0.0, 0.0, 0.0), 200, plan_fn=plan, params=params)


def test_deadlock_detour():
    seen, sh = _deadlock_case()
    pts = [s for s in seen[3] if s]
    gaps = [sh._rect_clearance(obj(3, x, y, h, 0.0, 6, 1.8, 0.7), ego(0.0)) for t, x, y, _, h in pts]
    assert min(gaps) >= 0.15, f"the cyclist touched the bus: {min(gaps):.2f}"
    assert [d["how"] for d in sh.detours] == ["shift"], sh.detours
    assert 1.5 <= abs(sh.detours[0]["shift_m"]) <= 4.0, sh.detours
    assert pts[-1][1] < -5.0, f"never got past the bus: last x {pts[-1][1]:.1f}"
    still = sum(1 for a, b in zip(pts, pts[1:]) if math.hypot(b[1] - a[1], b[2] - a[2]) < 0.01) * DT
    assert still < 8.0, f"stood {still:.1f} s"
    steps = [math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(pts, pts[1:])]
    assert max(steps) < 1.0, f"the detour jumped {max(steps):.1f} m"
    print(f"PASS: deadlock — the cyclist is led round the standing bus ({sh.detours[0]['shift_m']:+.2f} m), "
          "never touching it, without a jump")


def test_not_a_deadlock_when_the_bus_can_go():
    """A cyclist waiting beside the bus's front corner, across whose path the
    bus would drive away: no deadlock, no detour."""
    d = (-0.5, -0.866)
    fn = lambda t: (11.8 + 4.0 * t * d[0], 10.0 + 4.0 * t * d[1])  # noqa: E731
    bike = {3: (fn, math.atan2(d[1], d[0]), 4.0, 6, 1.8, 0.7)}
    seen, sh = play(bike, lambda t: (0.0, 0.0, 0.0, 0.0), 120, plan_fn=lambda t: [(0.0, 0.0, 0.0, 0.0)])
    assert sh.detours == [], sh.detours
    print("PASS: one waiting beside the bus that the bus could drive away from is no deadlock")


def test_deadlock_reported_when_no_detour():
    p = ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, vru_real_body=True,
                     npc_following=True, ego_plan_sec=5.0, deadlock_sec=3.0, detour_max_vru_m=0.25)
    seen, sh = _deadlock_case(p)
    assert [d["how"] for d in sh.detours] == ["stuck"], sh.detours
    assert sh.held_reason.get(bytes([3] * 16)) == "deadlock", sh.held_reason
    print("PASS: a deadlock no detour can solve is reported as such")


def _oncoming_car(extra=None, n=300, **kw):
    """The bus standing at the origin (body x -3.3..7.6, y -1.4..1.4) and a
    car coming the other way along y=0.3, whose way runs through the bus."""
    movers = {3: (lambda t: (40.0 - 5.0 * t, 0.3), math.pi, 5.0, 1, 4.5, 1.8)} | (extra or {})
    return play(movers, lambda t: (0.0, 0.0, 0.0, 0.0), n, plan_fn=lambda t: [(0.0, 0.0, 0.0, 0.0)], **kw)


def _overlap(a, b):
    from dp_multi_eval.geometry import box_overlap_m
    return box_overlap_m((a[1], a[2], a[4], a[5], a[6]), (b[1], b[2], b[4], b[5], b[6]))


def test_no_detour_through_another_object():
    """2026-08-25: a truck parked on either side of the bus. Every way round
    the bus runs through one of them: the car waits for the bus (held "ego" —
    the mutual wait is the bus's to answer for), it is not led through."""
    free, sh0 = _oncoming_car()
    assert [d["how"] for d in sh0.detours] == ["shift"], sh0.detours
    assert [s for s in free[3] if s][-1][1] < -5.0, "scenario changed: a free road lets the car round the bus"
    trucks = {4: (lambda t: (3.0, 3.7), 0.0, 0.0, 2, 8.0, 2.5), 5: (lambda t: (3.0, -3.7), 0.0, 0.0, 2, 8.0, 2.5)}
    seen, sh = _oncoming_car(trucks)
    assert sh.detours and {d["how"] for d in sh.detours} == {"blocked"}, sh.detours
    assert sh.held_reason.get(bytes([3] * 16)) == "ego", sh.held_reason
    car = [s for s in seen[3] if s]
    assert car[-1][1] > 7.6 + 2.25, f"the car got past the bus's nose: x {car[-1][1]:.1f}"
    for n_ in (4, 5):
        deep = max(_overlap(a + (2.25, 0.9), b + (4.0, 1.25)) for a, b in zip(seen[3], seen[n_]) if a and b)
        assert deep == 0.0, f"the car was {deep:.1f} m inside truck {n_}"
    print(f"PASS: no way round the bus through a parked truck — the car waits for the bus ({len(sh.detours)} asks, all blocked)")


def test_vehicle_detour_stays_on_the_road():
    """The way round may not leave the carriageway (a vehicle's; a cyclist
    rides where there is room)."""
    from dp_multi_eval.shield_map import ShieldMap
    narrow = ShieldMap(roads=[[(-60.0, -2.2), (60.0, -2.2), (60.0, 2.6), (-60.0, 2.6)]])
    wide = ShieldMap(roads=[[(-60.0, -2.2), (60.0, -2.2), (60.0, 6.0), (-60.0, 6.0)]])
    seen, sh = _oncoming_car(shield_map=narrow)
    assert sh.detours and {d["how"] for d in sh.detours} == {"blocked"}, sh.detours
    assert [s for s in seen[3] if s][-1][1] > 7.6 + 2.25
    seen, sh = _oncoming_car(shield_map=wide)
    assert [d["how"] for d in sh.detours] == ["shift"] and sh.detours[0]["shift_m"] < 0.0, sh.detours   # heading -x: its right is +y
    assert max(s[2] for s in seen[3] if s) > 2.5 and [s for s in seen[3] if s][-1][1] < -5.0
    bike = {3: (lambda t: (30.0 - 4.0 * t, 0.3), math.pi, 4.0, 6, 1.8, 0.7)}
    seen, sh = play(bike, lambda t: (0.0, 0.0, 0.0, 0.0), 200, plan_fn=lambda t: [(0.0, 0.0, 0.0, 0.0)], shield_map=narrow)
    assert [d["how"] for d in sh.detours] == ["shift"], sh.detours
    print("PASS: a car is led round the bus only on the carriageway (the lane beside it), a cyclist wherever it fits")


def test_detoured_car_stops_for_what_comes_into_its_way():
    """On its way round the bus the car is where its recording never was: a
    car that pulls up beside the bus after the detour was given is in its
    way. It stops for it; it does not drive through."""
    seen, sh = _oncoming_car()
    t_go = next(s[0] for a, s in zip(seen[3], seen[3][1:]) if a and s and abs(s[2] - 0.3) > 0.05)   # leaves its line
    y_round = max((s[2] for s in seen[3] if s), key=lambda y: abs(y - 0.3))                        # where it passes the bus
    late = {4: (lambda t: (-4.0, y_round), 0.0, 0.0, 1, 4.5, 1.8)}
    thru = play({3: (lambda t: (40.0 - 5.0 * t, 0.3), math.pi, 5.0, 1, 4.5, 1.8)} | late, lambda t: (0.0, 0.0, 0.0, 0.0), 300,
                plan_fn=lambda t: [(0.0, 0.0, 0.0, 0.0)], present=lambda n_, t: n_ != 4 or t >= t_go + 1.0,
                params=ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, vru_real_body=True,
                                    npc_following=False, ego_plan_sec=5.0, deadlock_sec=3.0))[0]
    assert max(_overlap(a + (2.25, 0.9), b + (2.25, 0.9)) for a, b in zip(thru[3], thru[4]) if a and b) > 1.0, \
        "scenario changed: without following, the detour runs through the car that pulled up"
    seen, sh = _oncoming_car(late, present=lambda n_, t: n_ != 4 or t >= t_go + 1.0)
    assert [d["how"] for d in sh.detours][:1] == ["shift"], sh.detours
    deep = max(_overlap(a + (2.25, 0.9), b + (2.25, 0.9)) for a, b in zip(seen[3], seen[4]) if a and b)
    assert deep == 0.0, f"the car on its detour drove {deep:.1f} m into the one that pulled up"
    last = [s for s in seen[3] if s][-1]
    assert last[1] > -4.0 + 4.5, f"it passed the car in its way: x {last[1]:.1f}"
    print(f"PASS: a car on its way round the bus stops for one that pulls up in it (stands at x {last[1]:.1f}, gap "
          f"{last[1] - 2.25 - (-4.0 + 2.25):.1f} m)")


def main() -> int:
    test_spacetime_moving_ego()
    test_gives_way_before_the_ego_way()
    test_who_goes_first()
    test_inside_a_junction_clears_first()
    test_not_held_for_a_bus_that_will_have_gone()
    test_following_and_crossing_behind_are_not_held()
    test_the_recording_is_the_reference()
    test_off_route_plan_holds_at_the_line()
    test_carried_past_the_recording()
    test_deadlock_detour()
    test_not_a_deadlock_when_the_bus_can_go()
    test_deadlock_reported_when_no_detour()
    test_no_detour_through_another_object()
    test_vehicle_detour_stays_on_the_road()
    test_detoured_car_stops_for_what_comes_into_its_way()
    return 0


if __name__ == "__main__":
    sys.exit(main())
