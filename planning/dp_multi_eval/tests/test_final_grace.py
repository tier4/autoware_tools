"""window_eval.final_grace_over: the last window closes FINAL_GRACE_SEC after
the recording's end — unless the ego is coming to rest at its stop, which is
let stand ARRIVAL_DWELL_SEC (the stop's service is measured), at most
ARRIVAL_EXTRA_SEC longer. Run: python3 tests/test_final_grace.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval import window_eval as we  # noqa: E402


def over(t, dist=None, arrived=None, drive_through=False):
    return we.final_grace_over(100.0 + t, 100.0, arrival_dist_m=dist, arrived_wall=None if arrived is None else 100.0 + arrived,
                               drive_through=drive_through)


def main() -> int:
    g = we.FINAL_GRACE_SEC
    assert not over(g - 0.1) and over(g)                       # no arrival pose: the plain grace
    assert over(g, dist=40.0)                                  # far from the stop
    # H78 (2026-11-10): came to rest 0.9 m off the stop 1 s before the grace ran out
    assert not over(g, dist=0.9, arrived=g - 1.0)
    assert over(g - 1.0 + we.ARRIVAL_DWELL_SEC, dist=0.9, arrived=g - 1.0)
    assert not over(g + 3.0, dist=8.0)                         # still rolling in
    assert over(g + we.ARRIVAL_EXTRA_SEC, dist=8.0)             # ... but not forever
    assert over(g, dist=0.5, drive_through=True)               # a cut pose: nothing to dwell for
    assert over(g, dist=0.9, arrived=0.0)                      # arrived long ago: dwell already measured
    print("PASS: the last window waits for a late ego's stop dwell, within bounds")
    # H95 (2026-11-11): a recording with one leg — its "start" leg is the last
    assert we.is_last_leg("start", 60.0, 60.0) and we.is_last_leg("final", 50.0, 120.0)
    assert not we.is_last_leg("start", 40.0, 120.0) and not we.is_last_leg("departure", 80.0, 120.0)
    print("PASS: the leg the recording ends in is the last one, whatever its kind")
    # H82 (2026-11-11): a fed sample of the scene before a jump is not the scene's time after it
    assert we.before_jump(48.7, target=66.1, came_from=48.6) and not we.before_jump(66.2, target=66.1, came_from=48.6)
    assert not we.before_jump(72.0, target=66.1, came_from=48.6)       # the position clock runs on past the target
    assert we.before_jump(40.0, target=10.0, came_from=40.5) and not we.before_jump(10.0, target=10.0, came_from=40.5)
    print("PASS: a sample of the scene before a jump is told from one after it, forward and back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
