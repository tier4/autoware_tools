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

"""WindowClock: 1x from the harness's jump, paused while held, nothing else.
Run: python3 tests/test_window_clock.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.scene_clock import WindowClock  # noqa: E402

S = int(1e9)
DT = S // 10  # the reproducer ticks at the recorded odom rate (~20-50 Hz)


def run(c, start_ns, end_ns, hold=False):
    ts = None
    now = start_ns
    while now <= end_ns:
        ts = c.tick(now, hold=hold)
        now += DT
    return ts


def main() -> int:
    c = WindowClock([i * S for i in range(0, 100)])
    # held until the first jump, whatever the wall clock does
    assert run(c, 0, 50 * S) == 0 and not c.at_end
    c.jump(20 * S)
    assert c.tick(51 * S) == 20 * S, "jump lands on the requested recorded instant"
    assert abs(run(c, 51 * S + DT, 53 * S) - 22 * S) <= DT, "then 1x"
    held = run(c, 53 * S + DT, 60 * S, hold=True)
    assert abs(held - 22 * S) <= DT, "held"
    assert abs(run(c, 60 * S + DT, 61 * S) - 23 * S) <= 2 * DT, "released: the held time did not count"
    c.jump(10 * S)
    assert c.tick(62 * S) == 10 * S, "a jump may go backwards (the next window)"
    c.jump(500 * S)
    assert c.tick(63 * S) == 99 * S and c.at_end, "clamped at the recording's end"
    c.jump(10 * S)
    c.tick(64 * S)
    assert c.tick(124 * S) == 10 * S + WindowClock.MAX_STEP_NS, "a starved tick advances at most MAX_STEP"
    print("PASS: window clock — held, jump, 1x, hold, clamp, starved tick")
    return 0


if __name__ == "__main__":
    sys.exit(main())
