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

"""through_ego_object: a contact with an object the reproducer withheld for driving
through the ego is a replay artefact, not an ego-caused collision.
Run: python3 tests/test_through_ego_attribution.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.compute_metrics import through_ego_object  # noqa: E402


def main() -> int:
    # 2026-04-28: SUPPRESS #11 object 65530000 at (50289.6, 9893.9); the contact
    # frame followed at (50290.0, 9888.8) with the full 32-hex id.
    events = [{"object": "65530000", "x": 50289.6, "y": 9893.9}, {"object": "c40f493e", "x": 50219.9, "y": 10313.1}]
    assert through_ego_object("65530000000000000000000000000000", 50290.0, 9888.8, events)
    assert not through_ego_object("65530000000000000000000000000000", 50220.0, 10300.0, events), "same object, another place"
    assert not through_ego_object("deadbeef00000000", 50290.0, 9888.8, events), "an object never withheld"
    assert not through_ego_object("65530000000000000000000000000000", 50290.0, 9888.8, [])
    print("PASS: through_ego_object — id prefix + place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
