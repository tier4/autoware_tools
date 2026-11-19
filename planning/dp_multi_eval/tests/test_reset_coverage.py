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

"""window_eval: a slow ego is reset only once no shown object is left around it.
Behind by time (lag) and by distance (the recorded vehicle 40 m+ ahead) is not
enough while objects are within COVER_M; a scene driver that does not report
the nearest object leaves the time rules alone in charge.
Run: python3 tests/test_reset_coverage.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval import window_eval as we  # noqa: E402
from dp_multi_eval.scene_driver import nearest_object_m  # noqa: E402


def main() -> int:
    now = 100.0
    covered = we.scene_uncovered(12.0, covered_wall=now - 0.5, now=now)
    lost = we.scene_uncovered(80.0, covered_wall=now - we.COVER_LOST_SEC - 0.1, now=now)
    blink = we.scene_uncovered(80.0, covered_wall=now - 1.0, now=now)
    assert not covered and lost and not blink
    assert we.scene_uncovered(None, 0.0, now)   # no report: time rules decide
    print("PASS: covered while an object is within COVER_M or was within COVER_LOST_SEC")

    # H78 (2026-11-10): 22 s behind, the bus 100 m ahead, pedestrians around the ego
    assert not we.falling_behind(22.0, 20.7, 100.0, uncovered=False)
    assert we.falling_behind(22.0, 20.7, 100.0, uncovered=True)
    # the time and distance rules still apply
    assert not we.falling_behind(22.0, 20.7, 30.0, uncovered=True)    # bus within 40 m
    assert not we.falling_behind(10.0, 5.0, 100.0, uncovered=True)    # lag grew 5 s, under 20 s
    assert we.falling_behind(14.0, 5.0, 100.0, uncovered=True)        # grew 9 s
    assert not we.falling_behind(None, 0.0, 100.0, uncovered=True)
    print("PASS: a slow ego is reset only when behind by time and distance AND out of the scene")

    class P:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class O:
        def __init__(self, x, y):
            self.kinematics = type("K", (), {"pose_with_covariance": type("C", (), {"pose": type("Q", (), {"position": P(x, y)})})})

    assert nearest_object_m([O(3.0, 4.0), O(30.0, 0.0)], 0.0, 0.0) == 5.0
    assert nearest_object_m([], 0.0, 0.0) >= 1e8
    print("PASS: the scene driver reports the nearest shown object")
    return 0


if __name__ == "__main__":
    sys.exit(main())
