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

"""leg_cuts knob: on by default with leg windows; off, a YAML bare off (False),
no, 0 turn it off; never without leg windows; the scene driver gets
--no-departure-hold only with cuts. Run: python3 tests/test_leg_cuts_knob.py"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.job_config import PIPELINE_KNOBS, JobConfig  # noqa: E402
from dp_multi_eval.run_single_job import build_reproducer_command, leg_cuts_on  # noqa: E402


def main() -> int:
    assert JobConfig.leg_cuts == "auto" and "leg_cuts" in PIPELINE_KNOBS
    assert leg_cuts_on(NS(window_eval=True, leg_cuts="auto"))
    for off in ("off", "OFF", False, "no", 0):
        assert not leg_cuts_on(NS(window_eval=True, leg_cuts=off)), off
    assert not leg_cuts_on(NS(window_eval=False, leg_cuts="auto"))
    cmd = build_reproducer_command(Path("/b"), window_clock=True, departure_hold=False)
    assert "--no-departure-hold" in cmd
    assert "--no-departure-hold" not in build_reproducer_command(Path("/b"), window_clock=True)
    assert "--no-departure-hold" not in build_reproducer_command(Path("/b"), window_clock=False, departure_hold=False)
    print("PASS: leg_cuts — auto by default, off / False / no / 0 off, never without windows; departure hold flag")
    return 0


if __name__ == "__main__":
    sys.exit(main())
