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

"""_waited_like_recording: a run stop overlapping a recorded wait in scene time
and place is that wait; elsewhere or at another time it is not.
Run: python3 tests/test_waited_like_recording.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.compute_metrics import _waited_like_recording  # noqa: E402

scene = lambda w: w - 1000.0  # wall -> recording time
rec_waits = [(6.0, 39.5, 100.0, 50.0)]  # the recording stood 6-39.5 s at (100, 50)
run_stops = [
    (1004.5, 1010.1, 100.4, 49.6),   # 4.5-10.1 s, 0.5 m away: the same wait
    (1019.9, 1036.8, 99.5, 54.0),    # 19.9-36.8 s, 4 m away: the same wait, crept forward
    (1050.0, 1055.0, 100.0, 50.0),   # after the recording left: not the wait
    (1020.0, 1025.0, 130.0, 50.0),   # 30 m away while it stood: not the wait
]
assert _waited_like_recording(run_stops, rec_waits, scene) == [True, True, False, False]
assert _waited_like_recording(run_stops, rec_waits, None) == [False] * 4
assert _waited_like_recording(run_stops, [], scene) == [False] * 4
print("PASS: waited like the recording — same place and scene time excuses; elsewhere or later does not")
