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

"""A contact with an object that popped up at the ego is the replay's, not the
model's: objects appear where the recorded bus first detected them, and an ego
ahead of the recording meets them at their birth (H78, 2026-11-09: a cyclist
born 1.6 m ahead of a 2.85 m/s ego, contact 0.2 s later). The same object seen
long before is the ego's to avoid.

    python3 tests/test_collision_pop_in.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.bag_reader import BagSeries, EgoSample, ObjectSample  # noqa: E402
from dp_multi_eval.compute_metrics import Thresholds, metric_collision  # noqa: E402
from dp_multi_eval.geometry import ego_local_footprint  # noqa: E402

FOOT = ego_local_footprint(wheel_base=4.99, front_overhang=2.57, rear_overhang=3.32, left=1.45, right=1.43)


def run(first_seen_t: float, scene_jumps=None) -> dict:
    ego, objs = [], []
    for k in range(120):                       # 12 s, 2.85 m/s along +x
        t = k * 0.1
        ego.append(EgoSample(stamp_sec=t, x=2.85 * t, y=0.0, z=0.0, yaw_rad=0.0, vx=2.85, vy=0.0, speed_mps=2.85))
    stop_x = 2.85 * 6.0 + 7.56 + 1.6 + 0.9    # 1.6 m ahead of the bumper at t=6 s
    for k in range(int(first_seen_t * 10), 120):
        t = k * 0.1
        objs.append(ObjectSample(stamp_sec=t, object_id="bike", label="BICYCLE", x=stop_x, y=0.0,
                                 yaw_rad=0.0, length_m=1.8, width_m=0.6))
    return metric_collision(BagSeries(ego=ego, objects=objs), Thresholds(), FOOT, scene_jumps=scene_jumps)


def main() -> int:
    popped = run(first_seen_t=6.0)            # born 1.6 m ahead, touched ~0.6 s later
    assert popped["event_count"] > 0, "scenario changed: no contact"
    assert popped["by_attribution"].get("ego_caused", 0) == 0, popped["by_attribution"]
    assert popped["by_attribution"].get("replay_jump", 0) > 0 and not popped["flagged"], popped["by_attribution"]
    print("PASS: an object born at the ego just before the contact: replay_jump, not gating")

    seen = run(first_seen_t=0.0)               # visible 30 m ahead from the start
    assert seen["by_attribution"].get("ego_caused", 0) > 0 and seen["flagged"], seen["by_attribution"]
    print("PASS: the same object seen long before: ego_caused, gating")

    # The scene jumped to the ego 3 s before the contact (a cut it drove past,
    # 2026-11-11): the object, 10 m ahead of the bumper then, is the jump's.
    jumped = run(first_seen_t=0.0, scene_jumps=[3.5])
    assert jumped["by_attribution"].get("scene_jump", 0) > 0 and not jumped["flagged"], jumped["by_attribution"]
    assert jumped["by_attribution"].get("ego_caused", 0) == 0 and jumped["scene_jump_contacts"][0]["object_id"] == "bike", jumped
    early = run(first_seen_t=0.0, scene_jumps=[1.0])          # the jump was 5.5 s before: the ego's to avoid
    assert early["by_attribution"].get("ego_caused", 0) > 0 and early["flagged"] and early["scene_jump_contacts"] == [], early["by_attribution"]
    print("PASS: a contact beginning within 4 s of a scene jump is the jump's; later it is the ego's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
