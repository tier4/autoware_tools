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

"""Window runs start the scene driver (the original perception_reproducer on
dp_multi_eval's clock) with --window-clock; a dashboard column is the model.
Run: python3 tests/test_scene_mode.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.build_dashboard import _model_key  # noqa: E402
from dp_multi_eval.run_single_job import build_reproducer_command  # noqa: E402

bag = Path("/tmp/bag")
window = build_reproducer_command(bag, window_clock=True)
assert window[:4] == ["ros2", "run", "dp_multi_eval", "scene_driver.py"], window
assert window[-1] == "--window-clock", window
assert "--window-clock" not in build_reproducer_command(bag, window_clock=False)
assert build_reproducer_command(bag, window_clock=True, objects=False)[-1] == "--no-objects"
assert _model_key({"model_name": "v4"}) == "v4"
print("PASS: scene command — the scene driver, --window-clock for window runs")
