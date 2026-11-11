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

"""A recording cut while moving routes to the next map stop ahead, not to the
cut pose. Pure geometry; no ROS needed.

Run directly: python3 tests/test_cut_end_goal.py
"""
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.multi_goal import next_stop_ahead  # noqa: E402

CSV = (
    "name,x,y,z,yaw,lane_id,stop_point_id,allow_goal_modification\n"
    "behind,0.0,100.0,0.0,-1.57,1,1,0\n"      # passed already (behind the heading)
    "ahead_far,0.0,-300.0,0.0,-1.57,2,2,0\n"  # ahead but beyond max_m
    "ahead_near,30.0,-60.0,0.0,-2.0,3,3,0\n"  # ahead, 67 m: the destination
    "ahead_nearer,0.0,-5.0,0.0,-1.57,4,4,0\n"  # ahead but inside min_m (the cut spot itself)
)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "stop_points.csv"
        csv.write_text(CSV, encoding="utf-8")
        # heading -pi/2: the bus drives toward -y
        end = (0.0, 0.0, -math.pi / 2)
        g = next_stop_ahead(end, map_path=Path(d), stop_points_csv="stop_points.csv")
        assert g is not None, "a stop 67 m ahead must be chosen"
        assert g.source == "cut_end_next_stop:ahead_near", g.source
        assert abs(g.x - 30.0) < 1e-9 and abs(g.y + 60.0) < 1e-9
        print("PASS: next stop ahead chosen; behind / too far / too close rejected")

        # heading +pi/2: now 'behind' is the one ahead, at 100 m
        g2 = next_stop_ahead((0.0, 0.0, math.pi / 2), map_path=Path(d), stop_points_csv="stop_points.csv")
        assert g2 is not None and g2.source.endswith(":behind"), g2
        print("PASS: heading decides which side counts as ahead")

        # nothing ahead within range
        g3 = next_stop_ahead((1000.0, 1000.0, 0.0), map_path=Path(d), stop_points_csv="stop_points.csv")
        assert g3 is None
        print("PASS: no candidate -> None (caller keeps the raw cut pose)")

    # the real map (v206) with the fleet's stop order: the recording passed
    # 平塚駅南口, 商工会議所前（往路）, 教会前（往路） and was cut before 松風公園入口 —
    # the next stop in order, present in the map, so it is the goal (whether
    # the planner routes there is the planner's verdict at run time).
    real = Path("/home/kaan/autoware_map/hiratsuka-map")
    if (real / "stop_points.csv").is_file():
        import csv
        pts = {r["name"]: (float(r["x"]), float(r["y"])) for r in csv.DictReader(open(real / "stop_points.csv", encoding="utf-8"))}
        order = ["平塚駅南口", "商工会議所前（往路）", "教会前（往路）", "松風公園入口", "花水小学校前", "すみれ平", "なぎさプロムナード"]
        path = [pts[n] for n in order[:3] if n in pts] + [(50295.8, 9849.2)]
        g4 = next_stop_ahead((50295.8, 9849.2, -1.40), map_path=real, stop_order=order, recorded_path_xy=path)
        assert g4 is not None and "松風公園入口" in g4.source, g4
        print(f"PASS: real map + order -> {g4.source} at ({g4.x:.1f}, {g4.y:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
