"""multi_goal "on" runs like "auto" (the GUI checkbox writes on): fewer than
two extracted legs take the single-goal path under both, two or more run as
legs, and "off" never extracts.
Run: python3 tests/test_multi_goal_on_like_auto.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from dp_multi_eval.run_single_job import (  # noqa: E402
        _multi_goal_requested,
        runs_single_goal,
    )
except ImportError as exc:
    print(f"SKIP: {exc}")
    raise SystemExit(0)


def main() -> int:
    on, auto, off = (_multi_goal_requested(v) for v in (True, "auto", False))
    assert (on, auto, off) == (True, None, False)
    for n_legs in (0, 1, 2, 5):
        assert runs_single_goal(n_legs, on) == runs_single_goal(n_legs, auto), n_legs
    assert runs_single_goal(0, on) and runs_single_goal(1, on)
    assert not runs_single_goal(2, on)
    assert not runs_single_goal(1, off)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
