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

"""multi_goal "on" runs like "auto" (the GUI checkbox writes on): fewer than
two extracted legs take the single-goal path under both, two or more run as
legs, and "off" never extracts.
Run: python3 tests/test_multi_goal_on_like_auto.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from dp_multi_eval.run_single_job import (  # noqa: E402
        _multi_goal_requested,
        runs_single_goal,
    )
except ImportError as exc:
    print(f"SKIP: {exc}")
    raise SystemExit(0)


def main() -> int:
    on, auto, off = (_multi_goal_requested(v) for v in (True, "auto", False))
    assert (on, auto, off) == (True, None, False)
    for n_legs in (0, 1, 2, 5):
        assert runs_single_goal(n_legs, on) == runs_single_goal(n_legs, auto), n_legs
    assert runs_single_goal(0, on) and runs_single_goal(1, on)
    assert not runs_single_goal(2, on)
    assert not runs_single_goal(1, off)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
