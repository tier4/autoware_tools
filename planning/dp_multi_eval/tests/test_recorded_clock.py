# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RecordedClock against a synthetic recording: 20 s drive at 5 m/s, 13 s stop,
drive on. Run: python3 tests/test_recorded_clock.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.scene_clock import RecordedClock  # noqa: E402

S = int(1e9)


def build(**kw):
    t, x, y, v = [], [], [], []
    ts = 0
    # drive 0..20 s at 5 m/s along +x
    for i in range(0, 200):
        t.append(i * S // 10); x.append(i * 0.5); y.append(0.0); v.append(5.0)
    # stop 20..33 s at x=100
    for i in range(200, 330):
        t.append(i * S // 10); x.append(100.0); y.append(0.0); v.append(0.0)
    # depart 33 s: 2 m/s, then 5 m/s
    for i in range(330, 500):
        spd = 2.0 if i < 350 else 5.0
        t.append(i * S // 10); x.append(x[-1] + spd * 0.1); y.append(0.0); v.append(spd)
    return RecordedClock(t, x, y, v, **kw)


def main() -> int:
    c = build()
    # Rule 1: moving -> position lock
    assert abs(c.tick(50.0, 0.0, 5.0, 0) - 10 * S) < S // 5, "position lock at x=50 -> 10 s"
    # arrive at the stop spot (recording stopped here 20..33 s, departed ~33-35 s)
    wall = 100 * S
    ts0 = c.tick(100.0, 0.0, 0.2, wall)
    # earliest pose within 0.5 m of the stop spot: the last rolling sample (19.9 s) counts as the arrival
    assert 19.5 * S <= ts0 <= 21 * S, ts0
    # Rule 2: real-time flow while stopped
    ts5 = c.tick(100.0, 0.0, 0.0, wall + 5 * S)
    assert abs((ts5 - ts0) - 5 * S) < S // 10, "dwell flows in real time"
    # Rule 3a: waiting longer than the recording -> hold at departure, bounded
    dep = c._dwell.depart_ns
    assert abs(dep - 33 * S) <= S // 5, f"departure = first motion after the stop: {dep/S:.1f}s"
    ts_long = c.tick(100.0, 0.0, 0.0, wall + 20 * S)   # recording would be at 40 s
    assert ts_long == dep, "held at the departure instant"
    ts_longer = c.tick(100.0, 0.0, 0.0, wall + 20 * S + RecordedClock.HOLD_MAX_NS + 2 * S)
    assert ts_longer > dep, "bounded hold: the recording flows on"
    # planned cuts (2026-11-10): hold_max_ns=0 — no freeze at the departure
    c0 = build(hold_max_ns=0)
    w0 = c0.tick(100.0, 0.0, 0.2, wall)
    c0.tick(100.0, 0.0, 0.0, wall + 5 * S)
    t_a = c0.tick(100.0, 0.0, 0.0, wall + 20 * S)
    t_b = c0.tick(100.0, 0.0, 0.0, wall + 22 * S)
    assert t_a >= c0._dwell.depart_ns and abs((t_b - t_a) - 2 * S) < S // 10, (w0, t_a, t_b)   # flows on, no 6 s freeze
    # Rule 3b: leaving early -> jump to the departure scene
    c2 = build()
    c2.tick(100.0, 0.0, 0.2, wall)
    c2.tick(100.0, 0.0, 0.0, wall + 3 * S)
    ts_go = c2.tick(100.5, 0.0, 0.6, wall + 3 * S + S // 10)
    assert ts_go >= c2._floor[0] and ts_go >= 33 * S, f"departure scene on leaving early: {ts_go/S:.1f}s"
    # monotonic within the episode, release after 25 m
    ts_a = c2.tick(101.0, 0.0, 1.0, wall + 4 * S)
    ts_b = c2.tick(102.0, 0.0, 2.0, wall + 5 * S)
    assert ts_b >= ts_a
    # a stop where the recording never stopped: no cap, real-time flow
    c3 = build()
    t1 = c3.tick(50.0, 0.0, 0.0, wall)
    t2 = c3.tick(50.0, 0.0, 0.0, wall + 10 * S)
    assert c3._dwell.depart_ns is None and abs((t2 - t1) - 10 * S) < S // 10, "drive-through spot: recording flows on"
    # heading check: a return trip on the same street must not capture the lock
    t2, x2, y2, v2, yw = [], [], [], [], []
    for i in range(0, 200):  # outbound along +x at y=0
        t2.append(i * S // 10); x2.append(i * 0.5); y2.append(0.0); v2.append(5.0); yw.append(0.0)
    for i in range(200, 400):  # return along -x at y=3.5, 20 s later
        t2.append(i * S // 10); x2.append(100 - (i - 200) * 0.5); y2.append(3.5); v2.append(5.0); yw.append(3.14159)
    c4 = RecordedClock(t2, x2, y2, v2, yaws=yw)
    ts_out = c4.tick(50.0, 1.5, 5.0, 0, yaw=0.0)      # ego outbound, drifted 1.5 m toward the return lane
    assert ts_out < 20 * S, f"heading check keeps the outbound lock: {ts_out/S:.1f}s"
    c5 = RecordedClock(t2, x2, y2, v2)                 # without yaws the nearest pose may be the return trip
    # at the recorded dwell spot the position lock is the ARRIVAL instant
    c6 = build()
    assert abs(c6.tick(100.0, 0.0, 5.0, 0) - 20 * S) <= S // 5, "dwell spot -> arrival (20 s), not a random dwell frame"
    # harness hold: no route / not engaged -> the departure scene waits, unbounded;
    # the 6 s hesitation budget starts once the ego is free to go
    c7 = build()
    c7.tick(100.0, 0.0, 0.2, wall)
    dep7 = c7._dwell.depart_ns
    held = c7.tick(100.0, 0.0, 0.0, wall + 60 * S, harness_hold=True)
    assert held == dep7, f"held at departure through 60 s of harness time: {held/S:.1f}s"
    free_a = c7.tick(100.0, 0.0, 0.0, wall + 63 * S, harness_hold=False)
    assert free_a == dep7, "budget not yet spent after release"
    free_b = c7.tick(100.0, 0.0, 0.0, wall + 60 * S + RecordedClock.HOLD_MAX_NS + 2 * S, harness_hold=False)
    assert free_b > dep7, "budget runs from the release, then the recording flows on"
    # Rule 4: a slower ego does not slow the world. Recorded 5 m/s along +x;
    # the ego crawls at 1 m/s from x=0: after 10 s it stands at x=10 (recorded
    # 2 s) but the scene is at 10 s. A faster ego is still placed by position.
    c4 = build(); wall = 5000 * S
    c4.tick(0.0, 0.0, 5.0, wall)
    assert not c4.at_end
    assert abs(c4.tick(10.0, 0.0, 1.0, wall + 10 * S) - 10 * S) < S // 5, "slow ego: real time"
    assert abs(c4.tick(90.0, 0.0, 5.0, wall + 11 * S) - 18 * S) < S // 5, "fast ego: position"
    prev = 0
    for k in range(1, 40):
        ts = c4.tick(90.0 + k * 0.1, 0.0, 5.0, wall + (11 + k) * S)
        assert ts >= prev; prev = ts
    assert c4.tick(95.0, 0.0, 5.0, wall + 200 * S) == c4.t[-1] and c4.at_end, "clamped at the recording's end"
    print("PASS: recorded clock — position lock, real-time dwell, departure alignment (hold + jump), free flow where the recording did not stop")

    # Position-mode clock (1x, as the scene driver builds it): an ego twice the
    # recording's pace reaches the stop at x=100 while the scene trails 50 m
    # back. Its dwell starts from the scene's instant, but the departure it
    # holds at is the recording's departure from where the EGO stands — the
    # capped anchor searched from 50 m back found none (2026-11-05 review).
    t, x, y, v = c.t, c.x, c.y, c.v
    c5 = RecordedClock(t, x, y, v, real_time_floor=False, departure_jump=False,
                       max_rate=1.0, flow_when_behind=True)
    wall = 0
    for k in range(0, 101):                        # 10 s at 10 m/s: x 0 -> 100
        ts = c5.tick(k * 1.0, 0.0, 10.0, wall + k * S // 10)
    assert abs(ts - 10 * S) < S // 5, ts / S      # the scene ran at 1x: 10 s, the recording at x=50
    stop_wall = wall + 10 * S
    for k in range(1, 321):                        # stands 32 s at the recorded stop
        ts = c5.tick(100.0, 0.0, 0.0, stop_wall + k * S // 10)
        if k == 260:                               # 26 s in: past the departure (33 s) by 3 s
            held = ts
    assert abs(held - 33 * S) < S // 2, f"no hold at the recorded departure: {held / S:.1f}s"
    assert ts - held > 2 * S, "the hold is bounded (6 s)"
    print("PASS: recorded clock (1x, trailing the ego) — the departure hold is the one at the ego's stop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
