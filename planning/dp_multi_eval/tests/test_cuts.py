"""Planned cuts (2026-11-10): find_cuts and plan_legs(cuts=...). A leg is also
cut where the recorded vehicle stood 3 s or more, at its departure — not where
the recording opens or ends standing, nor at a leg goal, nor at a hole; a
creeping queue is one cut; no piece shorter than 10 s. Only the first piece
routes, only the last arrives; without cuts plan_legs is unchanged.
Run: python3 tests/test_cuts.py"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval import window_eval as we  # noqa: E402
from dp_multi_eval.window_eval import find_cuts, plan_legs  # noqa: E402


@dataclass
class S:
    stamp_sec: float
    x: float
    y: float
    z: float = 0.0
    yaw_rad: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    speed_mps: float = 0.0


def profile(segments, dt=0.1):
    """segments: [(duration, speed)]; driving +x. Returns samples."""
    out, t, x = [], 0.0, 0.0
    for dur, v in segments:
        for _ in range(int(round(dur / dt))):
            out.append(S(round(t, 2), x, 0.0, speed_mps=v))
            x += v * dt
            t += dt
    return out


def h78():
    """Like H78: stands 0-8 s (start), 22-25 s (junction), 52-66 s (in the
    junction), 107-120 s (the stop, to the end)."""
    return profile([(8, 0), (14, 5), (3, 0), (27, 5), (14, 0), (41, 4), (13, 0)])


def main() -> int:
    ref = h78()
    stop = (ref[-1].x, 0.0, 0.0)
    cuts = find_cuts(ref, [stop])
    assert [round(c.t_depart, 1) for c in cuts] == [25.0, 66.0], cuts
    print("PASS: the H78 profile cuts at the junction (25 s) and in the junction (66 s); not at the start or the stop")

    # a stand shorter than 3 s is no cut; one at a leg goal is a boundary already
    short = profile([(8, 0), (14, 5), (2.5, 0), (40, 5), (5, 0)])
    assert find_cuts(short, [(short[-1].x, 0, 0)]) == []
    goal_mid = (ref[260].x, 0.0, 0.0)                    # the junction stand is a leg goal
    assert [round(c.t_depart, 1) for c in find_cuts(ref, [goal_mid, stop])] == [66.0]
    # a creeping queue (stand, creep 3 m, stand) is one cut at the last departure
    creep = profile([(8, 0), (14, 5), (5, 0), (2, 1.5), (6, 0), (30, 5), (5, 0)])
    got = find_cuts(creep, [(creep[-1].x, 0, 0)])
    assert len(got) == 1 and round(got[0].t_depart, 1) == 35.0 and round(got[0].t_arrive, 1) == 22.0, got
    # a hole in the recording touching the stand
    holed = [s for s in ref if not (24.0 < s.stamp_sec < 27.0)]
    assert [round(c.t_depart, 1) for c in find_cuts(holed, [stop])] == [66.0]
    print("PASS: short stands, stands at a goal and stands at a hole are no cuts; a creeping queue is one")

    legs0 = plan_legs(ref, [stop])
    legs = plan_legs(ref, [stop], cuts=cuts)
    assert len(legs0) == 1 and len(legs) == 3, legs
    assert [(l.index, l.parent, l.opens, l.closes) for l in legs] == [
        (1, 1, "goal", "cut"), (2, 1, "cut", "cut"), (3, 1, "cut", "goal")]
    a, b, c = legs
    assert a.goal == stop and b.goal is None and c.goal is None
    assert a.arrival is None and b.arrival is None and c.arrival == stop and c.arrive_t == legs0[0].arrive_t
    assert a.t_start == legs0[0].t_start and c.t_end == legs0[0].t_end
    assert a.t_end == b.t_start == b.depart_t == 25.0 and b.t_end == c.t_start == c.depart_t == 66.0
    assert a.depart_t == legs0[0].depart_t and not a.objects_from_start and b.objects_from_start and c.objects_from_start
    assert b.cut_pose == (cuts[0].x, 0.0, 0.0) and a.close_pose == b.cut_pose and c.close_pose is None
    assert all(l.kind == legs0[0].kind for l in legs)
    print("PASS: plan_legs splits at the cuts — first piece routes, last arrives, the pieces join")

    # no piece shorter than CUT_MIN_LEG_SEC: a cut 5 s after the window opens is dropped
    early = profile([(1, 0), (4, 5), (3, 0), (60, 5), (13, 0)])    # a stand 20 m on, left at 8 s
    ec = find_cuts(early, [(early[-1].x, 0, 0)])
    assert len(ec) == 1 and round(ec[0].t_depart, 1) == 8.0, ec
    el = plan_legs(early, [(early[-1].x, 0, 0)], cuts=ec)
    assert len(el) == 1, [(l.t_start, l.t_end) for l in el]
    print("PASS: no piece shorter than CUT_MIN_LEG_SEC")

    # passed_cut: the ego drove on past a cut's stand without waiting. The
    # recorded stand is at x=100 heading +x, left at t=40.
    pose, d = (100.0, 0.0, 0.0), 40.0
    P = lambda x, v=3.0, ts=12.0, y=0.0, yaw=0.0: we.passed_cut(x, y, yaw, v, pose, ts, d, 4.0)  # noqa: E731
    assert P(104.0)                                  # 4 m past, rolling, 28 s before the departure
    assert not P(101.5)                              # stopping a little further on than the recording
    assert not P(96.0)                               # H81 (2026-10-28): 15 m short and creeping is no pass
    assert not P(104.0, v=0.2)                       # standing there
    assert not P(104.0, ts=38.5)                     # the recording leaves within CUT_SYNC_SEC anyway
    assert not P(104.0, y=6.0)                       # another lane, off the track
    assert not P(104.0, yaw=math.pi)                 # the return trip on the same street
    assert not we.passed_cut(104.0, 0.0, 0.0, 3.0, None, 12.0, d, 4.0)
    print("PASS: passed_cut — rolling 3 m+ past the stand before its departure; not short of it, standing, "
          "about to be in step, off the track or going the other way")
    return 0


if __name__ == "__main__":
    sys.exit(main())
