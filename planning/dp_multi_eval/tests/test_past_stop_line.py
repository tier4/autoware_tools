"""compute_metrics._past_stop_line: a light is the ego's to wait for only while
its front has not passed the light's stop line (by more than 2 m). Standing
inside the junction after the line, a red behind it is no reason (H78,
2026-11-10). Run: python3 tests/test_past_stop_line.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.compute_metrics import _past_stop_line  # noqa: E402
from dp_multi_eval.shield_map import StopLine  # noqa: E402


def main() -> int:
    # a stop line across x=50 for traffic going +x (group 7); another group's line at x=20
    lines = [StopLine(50.0, -3.0, 50.0, 3.0, 7, 0.0), StopLine(20.0, -3.0, 20.0, 3.0, 9, 0.0)]
    front = 8.4
    assert not _past_stop_line(lines, 7, 50.0 - front - 1.0, 0.0, 0.0, front)    # waiting 1 m short
    assert not _past_stop_line(lines, 7, 50.0 - front + 1.5, 0.0, 0.0, front)    # a nose 1.5 m over
    assert _past_stop_line(lines, 7, 70.0, 0.0, 0.0, front)                      # inside the junction
    assert _past_stop_line(lines, 7, 60.0, -8.0, -math.pi / 2, front) is True    # turned, well past
    assert not _past_stop_line(lines, 5, 70.0, 0.0, 0.0, front)                  # no line for the group
    print("PASS: a light is behind the ego once its front is 2 m past the stop line")
    return 0


if __name__ == "__main__":
    sys.exit(main())
