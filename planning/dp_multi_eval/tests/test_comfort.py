"""_comfort_profile: harsh brakes and jerk from a speed series, with the recording
scored the same way. Run: python3 tests/test_comfort.py"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.compute_metrics import Thresholds, _comfort_profile  # noqa: E402


@dataclass
class S:
    stamp_sec: float
    speed_mps: float
    x: float = 0.0
    y: float = 0.0
    yaw_rad: float = 0.0


def series(profile):
    """profile: list of (duration_s, accel_mps2); 50 Hz samples."""
    out, t, v = [], 0.0, 0.0
    for dur, acc in profile:
        n = int(dur * 50)
        for _ in range(n):
            out.append(S(t, max(0.0, v)))
            v += acc * 0.02
            t += 0.02
    return out


def main() -> int:
    thr = Thresholds()
    gentle = series([(5, 1.0), (10, 0.0), (5, -1.0), (3, 0.0)])          # 1 m/s^2 either way
    harsh = series([(5, 1.0), (10, 0.0), (2, -3.0), (3, 0.0)])           # one 3 m/s^2 brake, 2 s
    g = _comfort_profile(gentle, thr)
    h = _comfort_profile(harsh, thr)
    assert g["harsh_brake_count"] == 0 and g["max_decel_mps2"] < 1.5, g
    assert h["harsh_brake_count"] == 1 and h["max_decel_mps2"] > 2.5, h
    assert h["harsh_brakes"][0][1] >= 1.0, h["harsh_brakes"]  # 5 m/s at 3 m/s^2 is ~1.7 s of braking
    # the hand-over window is excluded: the same harsh brake inside it does not count
    t_brake = 15.0
    e = _comfort_profile(harsh, thr, exclude=[(t_brake - 1.0, t_brake + 3.0)])
    assert e["harsh_brake_count"] == 0, e
    print("PASS: comfort — gentle vs harsh brake, hand-over excluded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
