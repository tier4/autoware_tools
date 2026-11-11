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

"""_light_ahead on the real Hiratsuka map: the light governing a bus stopped at
the H95 junction stop line is regulatory element 1061 (rear axle on lanelet
117, the light on 193); facing the other way, or 10 m further back, finds none.
Skips without lanelet2 (sourced workspace) or the map (DP_EVAL_MAP or the
default path). Run: python3 tests/test_light_ahead_map.py"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.compute_metrics import _light_ahead, _load_lanelet_map  # noqa: E402

MAP = Path(os.environ.get("DP_EVAL_MAP", "/home/kaan/autoware_map/hiratsuka-map"))


def main() -> int:
    if not (MAP / "lanelet2_map.osm").is_file():
        print(f"SKIP: map not found at {MAP} (set DP_EVAL_MAP)")
        return 0
    try:
        lmap = _load_lanelet_map(MAP)
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: lanelet2 unavailable ({exc})")
        return 0
    if lmap is None:
        print("SKIP: lanelet2 python bindings not importable (source the workspace)")
        return 0

    west = math.atan2(-1.9, -6.7)  # the bus's heading into the junction (H95, 2026-09-16)
    assert _light_ahead(lmap, 49796.6, 9757.4, west) == 1061, "stop line ahead: light 1061"
    assert _light_ahead(lmap, 49806.0, 9759.6, west) == 1061, "queued 10 m further back: still that light"
    assert _light_ahead(lmap, 49812.0, 9761.0, west) is None, "25 m back: the junction is out of reach"
    assert _light_ahead(lmap, 49796.6, 9757.4, west + math.pi) != 1061, "facing away: not our light"
    print("PASS: light ahead on the real map — 1061 at the H95 stop line and 10 m behind it, none 25 m back or facing away")
    return 0


if __name__ == "__main__":
    sys.exit(main())
