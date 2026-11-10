"""Out-of-boundary attribution in leg-window runs: an episode before the first
hand-over, inside a hand-over window (lead_sec + 0.5 before to 1 s after) or
within 1 s of a pose jump is the harness's placement; one that starts while the
ego drives on its own is the ego's and fails the run.
Run: python3 tests/test_oob_by_cause.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.bag_reader import EgoSample  # noqa: E402
from dp_multi_eval.compute_metrics import oob_by_cause  # noqa: E402


def ego(t, x):
    return EgoSample(stamp_sec=t, x=x, y=0.0, z=0.0, yaw_rad=0.0, vx=0.0, vy=0.0, speed_mps=0.0)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        wj = Path(d) / "windows.json"
        wj.write_text(json.dumps({"params": {"lead_sec": 3.5}, "legs": [
            {"segments": [{"wall_handover": 100.0}, {"wall_handover": 200.0}]}]}))
        # 10 Hz drive from t=90 to 210 s at 1 m/s, with a 30 m re-placement at t=150
        run = [ego(90 + i * 0.1, i * 0.1) for i in range(1200)]
        run = [e if e.stamp_sec < 150 else ego(e.stamp_sec, e.x + 30.0) for e in run]
        episodes = [
            [95.0, 96.0, "road_border"],    # before the first hand-over: placement
            [100.5, 101.0, "road_border"],  # inside the hand-over window (to +1 s): placement
            [150.5, 150.8, "curbstone"],    # 0.5 s after a 30 m pose jump: placement
            [196.5, 197.0, "road_border"],  # lead of the second hand-over (196.0-201.0): placement
            [120.0, 121.0, "road_border"],  # mid-drive: the ego's
            [190.0, 190.4, "curbstone"],    # mid-drive, before the second lead: the ego's
        ]
        out = oob_by_cause(episodes, run, wj)
        ego_t = sorted(e[0] for e in out["ego_caused_episodes"])
        placed_t = sorted(e[0] for e in out["placement_episodes"])
        assert ego_t == [120.0, 190.0], out
        assert placed_t == [95.0, 100.5, 150.5, 196.5], out
        # no windows.json: only pose jumps excuse
        out = oob_by_cause(episodes, run, Path(d) / "missing.json")
        assert sorted(e[0] for e in out["placement_episodes"]) == [150.5], out
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
