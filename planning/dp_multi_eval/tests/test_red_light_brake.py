"""stop_go reports each start-brake cycle's braking phase (cycles_at), and a
cycle braked under a red light ahead is excused by _at_red_light like a stop.
Run: python3 tests/test_red_light_brake.py"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.compute_metrics import _at_red_light  # noqa: E402
from dp_multi_eval.stop_go import stop_go  # noqa: E402


@dataclass
class S:
    stamp_sec: float
    x: float
    y: float
    speed_mps: float
    yaw_rad: float = 0.0


def drive(profile):
    """profile: [(until_t, v)] piecewise-constant speed from t=0 at 20 Hz along +x."""
    out, t, x = [], 0.0, 0.0
    for until, v in profile:
        while t < until:
            x += v * 0.05
            out.append(S(round(t, 2), round(x, 3), 0.0, v))
            t += 0.05
    return out

# stand 5 s at the spawn, drive off, brake to a standstill 10-14 s, drive on 60 m later
run = drive([(5.0, 0.0), (8.0, 3.0), (10.0, 3.0), (14.0, 0.0), (30.0, 4.0)])
res = stop_go(run, [], [])["run"]
assert res["start_brake_cycles"] == 1, res
(t_peak, t_end, x, y, yaw), = res["cycles_at"]
assert 5.0 <= t_peak <= 10.0 and 9.5 <= t_end <= 11.0, res["cycles_at"]
assert abs(x - run[int(t_end * 20)].x) < 0.5 and yaw == 0.0

# the light ahead of where it braked: red while braking -> excused; green -> not
RED, GREEN = 1, 3
base = 1000.0
cyc = [(base + t_peak, base + t_end, x, y, yaw)]
assert _at_red_light(cyc, [(base, {7: RED})], lambda *_: 7) == [True]
assert _at_red_light(cyc, [(base, {7: GREEN})], lambda *_: 7) == [False]
print("PASS: start-brake cycles carry their braking phase; braking under a red light ahead is excused")
