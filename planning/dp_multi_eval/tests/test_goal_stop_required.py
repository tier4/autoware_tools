"""resolve_goal_stop_required: a run whose last leg is the cut pose of a recording
cut while moving ends drive-through; the goal-stop metric must not demand a park.
Run: python3 tests/test_goal_stop_required.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.run_orchestrator import resolve_goal_stop_required  # noqa: E402


def _job(payload) -> dict:
    d = Path(tempfile.mkdtemp())
    (d / "route_goals.json").write_text(json.dumps(payload))
    return {"output_dir": str(d)}


def main() -> int:
    stop = {"x": 50244.2, "y": 10174.9, "yaw": -1.4, "source": "stop_points:商工会議所前（往路）"}
    cut = {"x": 50295.8, "y": 9849.2, "yaw": -1.4, "source": "bag_end:cut"}
    assert not resolve_goal_stop_required(_job({"goals": [stop, cut], "source": "auto"})), "multi-goal, last leg = cut pose"
    assert resolve_goal_stop_required(_job({"goals": [stop, stop], "source": "auto"})), "last leg = a bus stop"
    assert not resolve_goal_stop_required(_job({"goals": [cut], "arrival": "drive_through_at_cut_pose"}))
    assert resolve_goal_stop_required({"output_dir": tempfile.mkdtemp()}), "no file: a stop is required"
    print("PASS: resolve_goal_stop_required — cut-pose last leg is drive-through")
    return 0


if __name__ == "__main__":
    sys.exit(main())
