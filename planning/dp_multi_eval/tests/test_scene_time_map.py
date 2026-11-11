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

"""_scene_time_map: wall time -> recorded time of a leg-window run (segments held at
their lead-in, departure alignments as jumps). Run: python3 tests/test_scene_time_map.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.render_video import _scene_time_map  # noqa: E402


def main() -> int:
    d = Path(tempfile.mkdtemp())
    payload = {
        "params": {"lead_sec": 3.5},
        "legs": [
            {"segments": [{"wall_handover": 1000.0, "t_handover": 103.5}],
             "jumps": [{"wall": 1020.0, "from": 123.5, "to": 115.0}]},
            {"segments": [{"wall_handover": 1100.0, "t_handover": 153.5}], "jumps": []},
        ],
    }
    (d / "windows.json").write_text(json.dumps(payload))
    f = _scene_time_map(d / "windows.json")
    assert f is not None
    assert abs(f(996.5) - 100.0) < 1e-6, "lead-in start"
    assert abs(f(1000.0) - 103.5) < 1e-6, "hand-over"
    assert abs(f(1010.0) - 113.5) < 1e-6, "1x"
    assert abs(f(1020.0) - 115.0) < 1e-6, "aligned to the recorded departure"
    assert abs(f(1030.0) - 125.0) < 1e-6, "1x after the jump"
    assert abs(f(1090.0) - 150.0) < 1e-6, "held at the next lead-in until it starts"
    assert abs(f(1101.0) - 154.5) < 1e-6
    assert _scene_time_map(d / "missing.json") is None
    print("PASS: scene time map — lead-in, 1x, jump, hold")

    # Clock samples (runs from 2026-11-10): a 6 s freeze at a recorded
    # departure and the recording's end are followed, not run through at 1x.
    clock = [[1000.0 + k * 0.5, 103.5 + k * 0.5] for k in range(21)]            # 1x to 113.5
    clock += [[1010.5 + k * 0.5, 113.5] for k in range(12)]                     # frozen 6 s
    clock += [[1016.5 + k * 0.5, 113.5 + k * 0.5] for k in range(20)]           # 1x again
    payload = {"params": {"lead_sec": 3.5},
               "legs": [{"segments": [{"wall_handover": 1000.0, "t_handover": 103.5}], "jumps": [],
                         "clock": clock, "t_end": 120.0}]}
    (d / "windows.json").write_text(json.dumps(payload))
    f = _scene_time_map(d / "windows.json")
    assert abs(f(1013.0) - 113.5) < 1e-6, ("frozen", f(1013.0))
    assert abs(f(1020.0) - 117.0) < 0.51, ("after the freeze", f(1020.0))
    assert abs(f(1040.0) - 120.0) < 1e-6, ("clamped at the recording's end", f(1040.0))
    print("PASS: scene time map — clock samples follow a freeze; never past the recording's end")

    # A seamless planned cut (2026-11-10): the next window's segment starts where
    # the scene was, with no lead-in held before it.
    payload = {"params": {"lead_sec": 3.5},
               "legs": [{"segments": [{"wall_handover": 1000.0, "t_handover": 103.5}], "jumps": []},
                        {"segments": [{"wall_handover": 1020.0, "t_handover": 123.5, "seamless": True}], "jumps": []}]}
    (d / "windows.json").write_text(json.dumps(payload))
    f = _scene_time_map(d / "windows.json")
    assert abs(f(1018.0) - 121.5) < 1e-6, ("no hold before a seamless cut", f(1018.0))
    assert abs(f(1025.0) - 128.5) < 1e-6, f(1025.0)
    print("PASS: scene time map — a seamless cut holds nothing")

    # The ego drove on past a cut's stand: the scene jumped 30 s to the recorded
    # departure at wall 1012, the ego untouched.
    payload = {"params": {"lead_sec": 3.5},
               "legs": [{"segments": [{"wall_handover": 1000.0, "t_handover": 103.5}], "jumps": []},
                        {"segments": [{"wall_handover": 1012.0, "t_handover": 145.5, "seamless": True, "scene_jumped": True}],
                         "jumps": [{"wall": 1012.0, "from": 115.5, "to": 145.5}]}]}
    (d / "windows.json").write_text(json.dumps(payload))
    f = _scene_time_map(d / "windows.json")
    assert abs(f(1011.0) - 114.5) < 1e-6 and abs(f(1012.0) - 145.5) < 1e-6 and abs(f(1015.0) - 148.5) < 1e-6, (f(1011.0), f(1012.0), f(1015.0))
    print("PASS: scene time map — a scene jump at a cut the ego drove past")
    return 0


if __name__ == "__main__":
    sys.exit(main())
