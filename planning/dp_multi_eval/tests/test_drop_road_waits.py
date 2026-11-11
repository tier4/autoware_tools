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

"""Leg goals: a recorded wait on the road (no map stop within 12 m) is not a leg
goal; a wait at a bus stop and the recording's end stay; without a stop list
nothing is dropped. Run: python3 tests/test_drop_road_waits.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.multi_goal import RouteGoal  # noqa: E402
from dp_multi_eval.run_single_job import drop_road_waits  # noqa: E402

stop = RouteGoal(x=100.0, y=0.0, yaw=0.0, source="stop_segment")
road = RouteGoal(x=170.0, y=0.0, yaw=0.0, source="stop_segment")
end = RouteGoal(x=400.0, y=0.0, yaw=0.0, source="bag_end:cut")
with tempfile.TemporaryDirectory() as d:
    (Path(d) / "stop_points.csv").write_text("name,x,y,yaw\nstop_a,103.0,0.0,0.0\n")
    assert drop_road_waits([stop, road, end], Path(d)) == [stop, end]
    assert drop_road_waits([road, end], Path(d)) == [end]
assert drop_road_waits([stop, road, end], Path("/nonexistent")) == [stop, road, end]
print("PASS: road waits are not leg goals; bus-stop waits and the end stay")
