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

"""stop_go._stops: a short sub-0.5 m/s blip inside a standstill does not split
it; a real move does. Run: python3 tests/test_stop_go_blip.py"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.stop_go import _stops, stop_go  # noqa: E402


@dataclass
class S:
    stamp_sec: float
    x: float
    y: float
    speed_mps: float


def dwell(blip_v, blip_s, move_after=True):
    """Stand 0-14 s at x=0 with a blip at 5 s, then drive off at 5 m/s."""
    out = []
    t = 0.0
    while t < 14.0:
        v = blip_v if 5.0 <= t < 5.0 + blip_s else 0.0
        out.append(S(round(t, 2), 0.0, 0.0, v))
        t += 0.05
    if move_after:
        x = 0.0
        while t < 30.0:
            x += 5.0 * 0.05
            out.append(S(round(t, 2), x, 0.0, 5.0))
            t += 0.05
    return out


ref = dwell(0.12, 0.15)
assert len(_stops(ref)) == 1, _stops(ref)
assert len(_stops(dwell(0.12, 1.5))) == 2, "a blip longer than BLIP_S still splits"
assert len(_stops(dwell(1.5, 0.15))) == 2, "a move above STOP_GO_V still splits"
res = stop_go(ref, ref, [])
assert res["recording"]["unplanned_stops"] == 0, res["recording"]
print("PASS: stop_go blip — one dwell, not two; long or fast blips still split")
