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

"""plan_says_go: the fixer-veto test of a candidate plan. Run: python3 tests/test_plan_go.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.job_waiters import plan_says_go  # noqa: E402


def main() -> int:
    veto = [[(0.0, 0.05), (1.0, 0.08), (2.0, 0.11), (4.0, 1.71), (8.0, 4.0)]]   # 2026-04-28 y=9904
    hold = [[(0.0, 0.02), (1.0, 0.01), (2.0, 0.01), (4.0, 0.02), (8.0, 0.30)]]  # the bay-exit bicycle stop
    late = [[(0.0, 0.0), (4.0, 0.2), (6.0, 3.0)]]                               # go, but only after the horizon
    assert plan_says_go(veto)
    assert not plan_says_go(hold)
    assert not plan_says_go(late)
    assert not plan_says_go([])
    assert plan_says_go(hold + veto), "any candidate that means to move counts"
    print("PASS: plan_says_go — veto vs hold vs late")
    return 0


if __name__ == "__main__":
    sys.exit(main())
