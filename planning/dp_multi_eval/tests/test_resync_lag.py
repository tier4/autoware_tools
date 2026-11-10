"""recorded_lag_sec: the ego's lag behind the recording, in seconds.
Run: python3 tests/test_resync_lag.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.job_waiters import recorded_lag_sec  # noqa: E402


def build():
    """20 s at 5 m/s along +x, 13 s stop at x=100 (departure at 33 s), drive on;
    then, 200 s later, the return trip along -x at y=3.5."""
    t, x, y, yaw = [], [], [], []
    for i in range(0, 200):
        t.append(i * 0.1); x.append(i * 0.5); y.append(0.0); yaw.append(0.0)
    for i in range(200, 330):
        t.append(i * 0.1); x.append(100.0); y.append(0.0); yaw.append(0.0)
    for i in range(330, 500):
        t.append(i * 0.1); x.append(x[-1] + 0.5); y.append(0.0); yaw.append(0.0)
    for i in range(0, 200):
        t.append(250.0 + i * 0.1); x.append(100.0 - i * 0.5); y.append(3.5); yaw.append(np.pi)
    dwells = np.asarray([[100.0, 0.0, 20.0, 32.9]], dtype=float)
    return tuple(np.asarray(v, dtype=float) for v in (t, x, y, yaw)) + (dwells,)


def main() -> int:
    tr = build()
    # on pace: ego at x=50 (recorded there at 10 s) while the scene shows 10 s
    assert abs(recorded_lag_sec(tr, 50.0, 0.0, 0.0, 10.0)) < 0.2
    # stalled at x=50 while the scene ran on to 18 s: 8 s behind
    assert abs(recorded_lag_sec(tr, 50.0, 0.0, 0.0, 18.0) - 8.0) < 0.2
    # standing at the recorded stop (20..33 s) while the recording still stands: no lag
    assert recorded_lag_sec(tr, 100.0, 0.0, 0.0, 25.0) < 0.0, "latest sample at the spot = departure"
    # still standing 5 s after the recorded departure: 5 s behind
    assert abs(recorded_lag_sec(tr, 100.0, 0.0, 0.0, 38.0) - 5.0) < 0.2
    # the return trip passes the same x on the opposite heading: heading check keeps
    # the outbound samples (a lateral offset widens the spot along the track: +-0.3 s)
    assert abs(recorded_lag_sec(tr, 50.0, 1.5, 0.0, 12.0) - 2.0) < 0.4
    # on the return trip itself (yaw pi, scene at 260 s) the lag is against the return samples
    assert abs(recorded_lag_sec(tr, 50.0, 3.5, np.pi, 262.0) - 2.0) < 0.2
    # a later pass over the same spot (same heading) does not count while the scene is on the earlier one
    t2 = np.concatenate([tr[0], tr[0] + 600.0]); x2 = np.concatenate([tr[1], tr[1]])
    y2 = np.concatenate([tr[2], tr[2]]); yaw2 = np.concatenate([tr[3], tr[3]])
    assert abs(recorded_lag_sec((t2, x2, y2, yaw2, tr[4]), 50.0, 0.0, 0.0, 18.0) - 8.0) < 0.2
    # stopped 4.5 m short of the recorded stop while the recording still stands there
    # (2026-04-28 red light): waiting with the recording, not behind it
    assert recorded_lag_sec(tr, 95.5, 0.0, 0.0, 25.0) < 0.0
    # ... and 5 s after the recorded departure: 5 s behind, counted from the departure
    assert abs(recorded_lag_sec(tr, 95.5, 0.0, 0.0, 38.0) - 5.0) < 0.2
    # a recorded stop 20 m ahead is not this ego's wait
    assert recorded_lag_sec(tr, 80.0, 0.0, 0.0, 25.0) > 8.0
    # without the dwell rule the same spot reads its own pass: braking 8 m short of a
    # recorded 13 s stop, the ego is on pace, not "ahead" by the whole wait
    assert abs(recorded_lag_sec(tr, 92.0, 0.0, 0.0, 18.5, dwell_rule=False) - 0.1) < 0.3
    assert recorded_lag_sec(tr, 92.0, 0.0, 0.0, 18.5) < -10.0
    # at the recorded stop itself while the recording still stands there, the ego is
    # not ahead: against the FIRST pass it is behind (arrived later), against the
    # last it would read as ahead by the rest of the dwell
    assert abs(recorded_lag_sec(tr, 100.0, 0.0, 0.0, 25.0, dwell_rule=False, earliest=True) - 5.0) < 0.3
    assert recorded_lag_sec(tr, 100.0, 0.0, 0.0, 25.0, dwell_rule=False) < -7.0
    # ahead for real: at x=80 (recorded there at 16 s) while the scene shows 10 s
    assert abs(recorded_lag_sec(tr, 80.0, 0.0, 0.0, 10.0, dwell_rule=False, earliest=True) + 6.0) < 0.3
    # ahead of the recording: negative, never a re-sync
    assert recorded_lag_sec(tr, 80.0, 0.0, 0.0, 10.0) < 0.0
    # H78, 2026-11-09: standing 30 m short of where the recording then waited
    # (x=70, passed at 14 s; the recording stands at x=100 from 20 s): 16 s by
    # the clock at 30 s, but the recording drove only 6 s of it
    assert abs(recorded_lag_sec(tr, 70.0, 0.0, 0.0, 30.0) - 16.0) < 0.2
    assert abs(recorded_lag_sec(tr, 70.0, 0.0, 0.0, 30.0, driving=True) - 6.0) < 0.3
    # after the recorded departure (32.9 s) its driving counts again
    assert abs(recorded_lag_sec(tr, 70.0, 0.0, 0.0, 40.0, driving=True) - (26.0 - 12.9)) < 0.3
    # no wait in between: the same as the clock
    assert abs(recorded_lag_sec(tr, 50.0, 0.0, 0.0, 18.0, driving=True) - 8.0) < 0.2
    from dp_multi_eval.lag import recorded_gap_m
    # the recording 25 m on at 15 s while the ego is at x=50: 25 m ahead
    assert abs(recorded_gap_m(tr, 50.0, 0.0, 0.0, 15.0) - 25.0) < 0.6
    # standing at the stop (x=100) while the ego is at x=70: 30 m, however long it stands
    assert abs(recorded_gap_m(tr, 70.0, 0.0, 0.0, 30.0) - 30.0) < 0.6
    print("PASS: recorded_lag_sec — pace, stall, dwell, heading, later pass, driving lag, gap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
