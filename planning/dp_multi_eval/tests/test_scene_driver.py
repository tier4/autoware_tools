"""Scene driver logic, no ROS. The fed sample is the clock's instant (the first
at or after it). The window clock is held until the ego exists, jumps while
held, runs at 1x; once following, a standing ego sees the recording play on in
real time and a moving one the frame of its position. A harness jump lands
whether it arrives while following or just before following is switched off.
The object gate empties a frame with objects off, keeps only stationary
objects past the recording's end, hands the frame to the shield at the scene
instant and hides nothing. The indicator replay plays active recorded signals
and the vehicle's own report otherwise. Run: python3 tests/test_scene_driver.py"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.scene_driver import (  # noqa: E402
    IndicatorReplay,
    ObjectGate,
    SceneTimeline,
    frame_index,
)

S = 1_000_000_000
TICK = S // 10


def straight_track(n: int = 600, v: float = 5.0):
    """The recorded vehicle drives +x at v m/s without stopping, 10 Hz: x = v * t."""
    t = [i * TICK for i in range(n)]
    return t, [v * i * 0.1 for i in range(n)], [0.0] * n, [v] * n, [0.0] * n


def run(tl: SceneTimeline, now: int, sec: float, ego) -> int:
    """Tick every 0.1 s for ``sec``; returns (last instant, now)."""
    ts = None
    for _ in range(int(round(sec * 10))):
        now += TICK
        ts = tl.tick(now, ego)
    return ts, now


def obj(n: int, x: float, y: float, v: float = 0.0):
    return NS(
        object_id=NS(uuid=bytes([n] * 16)),
        kinematics=NS(
            pose_with_covariance=NS(pose=NS(position=NS(x=x, y=y))),
            twist_with_covariance=NS(twist=NS(linear=NS(x=v, y=0.0))),
        ),
        shape=NS(dimensions=NS(x=4.5, y=1.8)),
    )


EGO_POSE = NS(position=NS(x=0.0, y=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))


def gate_apply(gate, objs, **kw):
    msg = NS(objects=list(objs))
    args = dict(ego_pose=EGO_POSE, ego_speed=0.0, ts_ns=5 * S, at_end=False, objects_on=True, now_ns=0)
    args.update(kw)
    return gate.apply(msg, **args).objects


def main() -> int:
    t = [0, 20, 40, 60]
    assert [frame_index(t, v) for v in (0, 20, 21, 59, 999)] == [0, 1, 2, 3, 3]
    print("PASS: the fed sample is the first at or after the scene instant")

    # -- the window clock, then following the ego
    tl = SceneTimeline(*straight_track(), window=True)
    now = 1000 * S
    assert tl.tick(now, None) == 0                       # no ego yet: the clock waits
    standing_at_100 = (100.0, 0.0, 0.0, 0.0)            # the recording passed x=100 at t=20 s
    tl.jump(20.0)
    tl.hold = True
    assert tl.tick(now, standing_at_100) == 20 * S       # a jump lands while held
    ts, now = run(tl, now, 2.0, standing_at_100)
    assert ts == 20 * S, ts                              # held
    tl.hold = False
    ts, now = run(tl, now, 2.0, standing_at_100)
    assert abs(ts - 22 * S) < TICK, ts                   # 1x
    print("PASS: window clock — held until the ego exists, jumps while held, 1x after")

    assert tl.set_follow(True)
    seed, now = run(tl, now, 0.1, standing_at_100)
    ts, now = run(tl, now, 3.0, standing_at_100)
    assert abs(ts - (seed + 3 * S)) < 2 * TICK, (ts, seed)
    print("PASS: following, a standing ego sees the recording play on in real time")

    # the ego drives off from a spot the recording reached later: the scene
    # runs on at real time from where it is, no jump or fast-forward to the
    # ego's position (2026-10-29: catch-up faster than 1x fast-forwarded traffic)
    x, before = 130.0, ts
    for _ in range(10):
        now += TICK
        x += 0.2
        ts = tl.tick(now, (x, 0.0, 2.0, 0.0))
    assert abs((ts - before) - S) <= TICK, ((ts - before) / S, x)
    print("PASS: following, an ego ahead of the scene sees it run on at real time (no fast-forward)")

    assert tl.set_follow(False)
    ts_off, now = run(tl, now, 0.1, (x, 0.0, 2.0, 0.0))
    assert abs(ts_off - ts) <= 2 * TICK, (ts_off, ts)    # 1x again from where following left it
    print("PASS: following off, the window clock resumes from the followed instant")

    # a leg opens back in recorded time while the ego stands where the recording
    # was much later (H81 leg 3, 2026-10-28: the jump was lost, the leg opened 24 s late)
    driving_at_200 = (200.0, 0.0, 5.0, 0.0)              # the recording passed x=200 at t=40 s
    for order in ("follow_off_first", "jump_first"):     # the harness's order, and one cycle the other way
        tl.hold = False
        tl.set_follow(True)
        _, now = run(tl, now, 0.5, driving_at_200)
        tl.hold = True
        if order == "follow_off_first":
            tl.set_follow(False)
            tl.jump(10.0)
        else:
            tl.jump(10.0)
            tl.set_follow(False)
        ts, now = run(tl, now, 0.5, driving_at_200)
        assert ts == 10 * S, (order, ts)
    # the departure reveal: following on and the jump to the recorded departure, in either order
    standing_at_50 = (50.0, 0.0, 0.0, 0.0)               # the recording passed x=50 at t=10 s
    tl.hold = False
    tl.set_follow(True)                                   # taken before the jump this time
    _, now = run(tl, now, 0.1, standing_at_50)
    tl.jump(30.0)
    ts, now = run(tl, now, 0.1, standing_at_50)
    assert abs(ts - 30 * S) <= 2 * TICK, ts
    print("PASS: a harness jump lands at a leg's opening and at a departure reveal, either order")

    # the ego stands 15 m short of a long recorded wait and then creeps off: the
    # scene goes on from where it stands, it does not jump to the recorded
    # departure (H81, 2026-10-28: a 40 s jump put a bicycle on the ego's nose)
    n = 1200
    stamps = [i * TICK for i in range(n)]
    xs, vs = [], []
    for i in range(n):
        tt = i * 0.1
        x_ = 5.0 * min(tt, 20.0) + 5.0 * max(0.0, tt - 70.0)   # drives to x=100, waits 50 s, drives on
        xs.append(x_)
        vs.append(0.0 if 20.0 <= tt < 70.0 else 5.0)
    w = SceneTimeline(stamps, xs, [0.0] * n, vs, [0.0] * n, window=True)
    now2 = 5000 * S
    w.tick(now2, (50.0, 0.0, 5.0, 0.0))
    w.jump(10.0)
    w.set_follow(True)
    for i in range(70):                                   # drive to x=85 at the recording's pace
        now2 += TICK
        w.tick(now2, (50.0 + 0.5 * i, 0.0, 5.0, 0.0))
    stood, now2 = run(w, now2, 8.0, (85.0, 0.0, 0.0, 0.0))
    crept, now2 = run(w, now2, 2.0, (85.5, 0.0, 0.5, 0.0))
    assert abs((crept - stood) / S - 2.0) < 0.3 and crept < 40 * S, (stood / S, crept / S)
    print("PASS: creeping off 15 m short of a recorded wait: no jump to its departure, and the scene"
          " keeps running at 1x while the ego is behind it (it does not freeze under a creeping ego)")

    # an ego twice the recording's pace: the scene runs at real time, no fast-forward
    fast = SceneTimeline(*straight_track(), window=True)   # recording at 5 m/s
    now3 = 9000 * S
    fast.tick(now3, (50.0, 0.0, 10.0, 0.0))
    fast.jump(10.0)
    fast.set_follow(True)
    start, now3 = run(fast, now3, 0.1, (50.0, 0.0, 10.0, 0.0))
    x = 50.0
    for _ in range(40):                                   # 4 s at 10 m/s: the recording covers 8 s
        now3 += TICK
        x += 1.0
        ts = fast.tick(now3, (x, 0.0, 10.0, 0.0))
    assert abs((ts - start) / S - 4.0) < 0.3, (ts - start) / S
    print("PASS: following a faster ego, recorded time runs at real time (no fast-forward)")

    # ... and when that ego comes to rest the scene does not jump to its spot
    # (H81, 2026-10-29: 18.8 s skipped on stopping, a crossing pedestrian vanished)
    x_before = x
    stop_ts, now3 = run(fast, now3, 0.1, (x, 0.0, 0.0, 0.0))
    assert stop_ts - ts <= TICK + 1, ((stop_ts - ts) / S, "jumped on coming to rest")
    held, now3 = run(fast, now3, 3.0, (x_before, 0.0, 0.0, 0.0))
    assert abs((held - stop_ts) / S - 3.0) < 0.3, (held - stop_ts) / S
    print("PASS: a faster ego coming to rest: the scene runs on from where it was, no jump to the ego's spot")

    # -- the object gate
    inside, far = obj(1, 2.12, 0.0), obj(2, 30.0, 0.0, v=5.0)
    calls = []
    shield = NS(apply=lambda msg, pose, t_sec, dt, v: calls.append((t_sec, v)))
    gate = ObjectGate(shield=shield)
    assert gate_apply(gate, [inside, far], ts_ns=7 * S, ego_speed=1.5) == [inside, far]
    assert calls == [(7.0, 1.5)], calls
    assert gate_apply(gate, [inside, far], objects_on=False) == [] and len(calls) == 1
    assert gate_apply(gate, [obj(3, 50.0, 0.0), far], at_end=True)[0].object_id.uuid == bytes([3] * 16)
    assert len(gate_apply(gate, [obj(3, 50.0, 0.0), far], at_end=True)) == 1
    print("PASS: gate — objects off, stationary only past the end, the shield at the scene instant, nothing hidden")

    # H39, 2026-11-09: an UNKNOWN car driving at 9 m/s reports 0 (tracker); at
    # the recording's end it was kept, frozen in the oncoming lane.
    from dp_multi_eval.reactive_shield import ReactiveShield, ShieldParams
    def frame(k):
        driving = obj(4, 100.0 - 0.9 * k, 3.5)          # 9 m/s, reported 0
        driving.kinematics.pose_with_covariance.pose.orientation = NS(x=0.0, y=0.0, z=0.0, w=1.0)
        parked = obj(5, 40.0, -3.0)
        parked.kinematics.pose_with_covariance.pose.orientation = NS(x=0.0, y=0.0, z=0.0, w=1.0)
        return [driving, parked]
    real = ReactiveShield(ShieldParams())
    real.build_tracks([(2 * 10**18 + k * TICK, NS(objects=frame(k))) for k in range(50)])
    assert ReactiveShield.end_speed(real.tracks[bytes([4] * 16)]) > 8.0
    end = gate_apply(ObjectGate(shield=real), frame(49), at_end=True, ts_ns=49 * TICK)
    assert [o.object_id.uuid for o in end] == [bytes([5] * 16)], [o.object_id.uuid for o in end]
    print("PASS: past the end, an object still driving by its poses is dropped though it reports 0 m/s")

    # H95, 2026-11-10: a car born 0.2 s before the end (3 frames, 0.8 m, then
    # reporting 0.4 m/s) has no evidence that it stands: not frozen in the lane.
    def frame2(k):
        out = frame(k)
        if k >= 47:
            late = obj(6, 20.0 + (0.8 if k > 47 else 0.0), 0.0, v=0.4)
            late.kinematics.pose_with_covariance.pose.orientation = NS(x=0.0, y=0.0, z=0.0, w=1.0)
            out.append(late)
        return out
    real2 = ReactiveShield(ShieldParams())
    real2.build_tracks([(2 * 10**18 + k * TICK, NS(objects=frame2(k))) for k in range(50)])
    end = gate_apply(ObjectGate(shield=real2), frame2(49), at_end=True, ts_ns=49 * TICK)
    assert [o.object_id.uuid for o in end] == [bytes([5] * 16)], [o.object_id.uuid for o in end]
    print("PASS: past the end, a track born in its last second is dropped (no evidence it stands)")

    # H95, 2026-11-11: ... and it is not shown before the end either — in front
    # of a late ego it flashed up in the next lane and was gone 0.3 s later.
    # A short track the recording did not cut (tracker noise mid-drive) stays.
    def frame3(k):
        out = frame2(k)
        if 20 <= k < 23:
            noise = obj(7, 60.0, 0.0)
            noise.kinematics.pose_with_covariance.pose.orientation = NS(x=0.0, y=0.0, z=0.0, w=1.0)
            out.append(noise)
        return out
    real3 = ReactiveShield(ShieldParams())
    real3.build_tracks([(2 * 10**18 + k * TICK, NS(objects=frame3(k))) for k in range(50)])
    gate3 = ObjectGate(shield=NS(tracks=real3.tracks, bag_end=real3.bag_end, apply=lambda *a: None))
    shown = lambda k: {o.object_id.uuid[0] for o in gate_apply(gate3, frame3(k), ts_ns=k * TICK)}
    assert shown(48) == {4, 5} and shown(21) == {4, 5, 7}, (shown(48), shown(21))
    print("PASS: a track born in the recording's last second is never shown; a short track mid-recording is")

    # -- the indicator follows the ego's place (H78, 2026-11-09): the bus
    # turned right at x=100 (t 40-45 s, RIGHT), stood at a stop at x=300
    # (t 80-100 s) and signalled LEFT pulling in, RIGHT pulling out.
    from dp_multi_eval.scene_driver import IndicatorClock
    ts_, xs_ = [], []
    for k in range(1300):          # 130 s at 10 Hz
        t = k * 0.1
        ts_.append(k * TICK)
    xs_ = [min(2.5 * k * 0.1, 200.0) if k * 0.1 < 80 else (200.0 if k * 0.1 < 100 else 200.0 + 5.0 * (k * 0.1 - 100)) for k in range(1300)]
    clock = IndicatorClock(ts_, xs_, [0.0] * 1300, [0.0] * 1300)
    def state(t):   # the recorded report in force at t (s)
        return 3 if 35 <= t < 50 else 2 if 75 <= t < 95 else 3 if 95 <= t < 105 else 1
    ind = IndicatorReplay([(k * TICK, state(k * 0.1)) for k in range(1300)], merge=True)
    # standing at x=100 (passed at 40 s, RIGHT) while the scene is at 85 s (LEFT): RIGHT
    assert ind.report_at(clock.at(85 * S, (100.0, 0.0, 0.0, 0.0))) == 3
    assert ind.report_at(85 * S) == 2, "scenario changed: the scene's instant is not LEFT"
    # standing at the stop (x=200, 80-100 s): the scene's instant decides within the dwell
    assert ind.report_at(clock.at(90 * S, (200.0, 0.0, 0.0, 0.0))) == 2
    assert ind.report_at(clock.at(97 * S, (200.0, 0.0, 0.0, 0.0))) == 3
    # ...and a scene run past the dwell gives the departure's RIGHT, not what came after
    assert ind.report_at(clock.at(120 * S, (200.0, 0.0, 0.0, 0.0))) == 3
    # off the recorded track (or the opposite way): the scene's instant
    assert clock.at(85 * S, (100.0, 30.0, 0.0, 0.0)) == 85 * S
    assert IndicatorClock(ts_, xs_, [0.0] * 1300, [0.0] * 1300).at(85 * S, (100.0, 0.0, 0.0, math.pi)) == 85 * S
    print("PASS: indicator by the ego's place — a lagging ego gets the bus's signal there, a dwell keeps the scene's timing")

    # -- turn indicators (1 DISABLE, 2 LEFT, 3 RIGHT, 0 NO_COMMAND)
    ind = IndicatorReplay([(0, 1), (10, 3), (20, 1)], merge=True)
    assert ind.enabled and ind.report_at(10) == 3 and ind.report_at(15) == 1
    ind.on_vehicle_report(0)
    assert ind.vehicle == 1
    ind.on_vehicle_report(2)
    assert ind.report_at(20) == 2 and ind.report_at(5) == 3
    assert IndicatorReplay([(0, 1), (10, 1)], merge=False).report_at(10) is None
    assert not IndicatorReplay([(0, 3)], merge=True).enabled
    print("PASS: indicators — recorded signal when active, else the vehicle's own report (merge only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
