"""A long stop the recording also made at the same place is a wait, not a stall.
Run: python3 tests/test_metrics_recorded_wait.py (ROS env sourced)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.bag_reader import EgoSample  # noqa: E402
from dp_multi_eval.bag_reader import BagSeries  # noqa: E402
from dp_multi_eval.compute_metrics import Thresholds, find_stuck_events, metric_stuck  # noqa: E402


def ego(t, x, v):
    return EgoSample(stamp_sec=t, x=x, y=0.0, z=0.0, yaw_rad=0.0, vx=v, vy=0.0, speed_mps=v)


def main() -> int:
    thr = Thresholds()
    # ego: drive, 57 s stop at x=100, drive on (goal far away at x=1000)
    samples = [ego(i * 0.1, i * 0.5, 5.0) for i in range(200)]
    samples += [ego(20 + i * 0.1, 100.0, 0.0) for i in range(570)]
    samples += [ego(77 + i * 0.1, 100 + i * 0.5, 5.0) for i in range(200)]
    # recording: stood 56 s at x=101
    rec = [ego(i * 0.1, i * 0.5, 5.0) for i in range(202)]
    rec += [ego(20.2 + i * 0.1, 101.0, 0.0) for i in range(560)]
    rec += [ego(76.2 + i * 0.1, 101 + i * 0.5, 5.0) for i in range(200)]
    ev = find_stuck_events(samples, 1000.0, 0.0, thr, reference_ego=rec)
    assert len(ev) == 1 and ev[0]["waited_like_recording"], ev
    assert abs(ev[0]["recorded_wait_sec"] - 56.0) < 1.0, ev[0]["recorded_wait_sec"]
    print("PASS: 57 s stop where the recording waited 56 s -> recorded wait, not a stall")
    # recording rolled through here -> a stall
    rec2 = [ego(i * 0.1, i * 0.5, 5.0) for i in range(1000)]
    ev2 = find_stuck_events(samples, 1000.0, 0.0, thr, reference_ego=rec2)
    assert len(ev2) == 1 and not ev2[0]["waited_like_recording"], ev2
    print("PASS: same stop where the recording rolled through -> stall")
    # no reference -> behaves as before (stall)
    ev3 = find_stuck_events(samples, 1000.0, 0.0, thr)
    assert len(ev3) == 1 and not ev3[0]["waited_like_recording"]
    print("PASS: without a reference nothing is excused")
    st = metric_stuck(BagSeries(ego=samples), 1000.0, 0.0, thr, reference=BagSeries(ego=rec))
    assert st["flagged"] is False and len(st["recorded_waits"]) == 1 and abs(st["recorded_wait_sec"] - 57.0) < 0.5, st
    print("PASS: metric_stuck reports the recorded wait and does not flag")
    return 0


if __name__ == "__main__":
    sys.exit(main())
