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

"""Contact attribution: a moving ego meeting a standing object is ego_caused
from any side and off the recorded path too; a moving object alongside stays
cut_in, off the path out_of_envelope; a standing ego is struck_while_stopped.
PEDESTRIAN contacts count, UNKNOWN do not.
Run: python3 tests/test_collision_class.py"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.compute_metrics import (  # noqa: E402
    NPC_COLLISION_LABELS,
    Thresholds,
    _collision_class,
)


def main() -> int:
    thr = Thresholds()
    ego = NS(stamp_sec=10.0, x=0.0, y=0.0, yaw_rad=0.0, speed_mps=5.0)
    beside = NS(x=0.0, y=2.0)  # 90 degrees off the bow
    ahead = NS(x=5.0, y=0.0)
    on_path = lambda x, y: (0.5, 5.0)  # noqa: E731 - nearest recorded pose 0.5 m away
    off_path = lambda x, y: (12.0, 5.0)  # noqa: E731

    assert _collision_class(ego, beside, thr, [], on_path, obj_speed=0.0) == "ego_caused"
    assert _collision_class(ego, beside, thr, [], on_path, obj_speed=4.0) == "cut_in"
    assert _collision_class(ego, ahead, thr, [], on_path, obj_speed=4.0) == "ego_caused"
    assert _collision_class(ego, ahead, thr, [], off_path, obj_speed=0.0) == "ego_caused"
    assert _collision_class(ego, ahead, thr, [], off_path, obj_speed=4.0) == "out_of_envelope"
    assert _collision_class(ego, beside, thr, [], on_path, obj_speed=None) == "cut_in"  # speed unknown: as before
    stopped = NS(stamp_sec=10.0, x=0.0, y=0.0, yaw_rad=0.0, speed_mps=0.0)
    assert _collision_class(stopped, ahead, thr, [], on_path, obj_speed=0.0) == "struck_while_stopped"

    assert "PEDESTRIAN" in NPC_COLLISION_LABELS and "UNKNOWN" not in NPC_COLLISION_LABELS
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
