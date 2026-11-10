"""plan_legs: one window per route leg, opening PREROLL_SEC before the recorded
departure. Run: python3 tests/test_plan_legs.py"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.window_eval import PREROLL_SEC, plan_legs  # noqa: E402


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


def build():
    """Stopped 0-10 s at x=0; drive at 5 m/s; stop 40-52 s at x=150 (bay);
    drive on; end moving at x=400 (the recording was cut)."""
    ref = []
    for i in range(0, 100):
        ref.append(S(i * 0.1, 0.0, 0.0))
    x = 0.0
    for i in range(100, 400):
        x += 0.5; ref.append(S(i * 0.1, x, 0.0, speed_mps=5.0))
    for i in range(400, 520):
        ref.append(S(i * 0.1, x, 0.0))
    for i in range(520, 1020):
        x += 0.5; ref.append(S(i * 0.1, x, 0.0, speed_mps=5.0))
    return ref


def main() -> int:
    ref = build()
    bay = (150.0, 0.0, 0.0)
    cut = (ref[-1].x, 0.0, 0.0)
    legs = plan_legs(ref, [bay, cut], cut_final=True, cut_end_goal=(500.0, 0.0, 0.0))
    assert [l.kind for l in legs] == ["start", "final"]
    a, b = legs
    assert a.t_start == 0.0 and abs(a.t_end - (51.9 - PREROLL_SEC)) < 0.2, (a.t_start, a.t_end)
    assert a.arrival == bay and a.goal == bay and abs(a.depart_t - 10.0) < 0.2, a.depart_t
    assert abs(b.t_start - (51.9 - PREROLL_SEC)) < 0.2 and abs(b.depart_t - 51.9) < 0.2
    assert b.goal == (500.0, 0.0, 0.0) and b.arrival == cut and b.drive_through
    assert abs(b.t_end - ref[-1].stamp_sec) < 1e-9
    assert abs(a.arrive_t - 40.0) < 0.2 and abs(b.arrive_t - ref[-1].stamp_sec) < 1e-9, (a.arrive_t, b.arrive_t)
    # a goal the recording drove past (no stop within the zone): the pass instant
    mid = (75.0, 0.0, 0.0)
    legs3 = plan_legs(ref, [mid, bay, cut])
    assert abs(legs3[0].t_end - (25.0 - PREROLL_SEC)) < 0.2, legs3[0].t_end
    assert legs3[1].kind == "departure" and abs(legs3[1].depart_t - 25.0) < 0.2
    # a hole in the recording: 20 s and 100 m missing inside the final leg is
    # marked on the leg, not split out of it
    gap = [x for x in ref if not (60.0 < x.stamp_sec < 80.0)]
    legs4 = plan_legs(gap, [bay, cut], cut_final=True, cut_end_goal=(500.0, 0.0, 0.0))
    assert [(l.index, l.kind) for l in legs4] == [(1, "start"), (2, "final")]
    assert legs4[0].gaps == () and len(legs4[1].gaps) == 1
    g0, g1 = legs4[1].gaps[0]
    assert abs(g0 - 60.0) < 0.2 and abs(g1 - 80.0) < 0.2 and legs4[1].arrival == cut
    print("PASS: plan_legs — start / departure / final windows, cut pose, drive-past goal, recording gap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
