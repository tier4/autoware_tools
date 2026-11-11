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

"""The line's stop sequence comes from the vehicle's own recorded routes, not
from a written list (2026-11-11): the destination of a recording cut mid-drive
is its own route's goal when it carries one, else the stop that follows the
last one it left behind in the routes of the other recordings; a written order
and the nearest stop ahead only after those. The ego's body comes from the
vehicle's vehicle_info. Run: python3 tests/test_stop_sequence.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dp_multi_eval.multi_goal as mg  # noqa: E402
from dp_multi_eval.geometry import body_of, load_vehicle_body  # noqa: E402
from dp_multi_eval.multi_goal import RouteGoal  # noqa: E402

# A westbound road: A (x=300), B (x=200), C (x=100), D (x=0) — and E off to the side.
STOPS = {"A": (300.0, 0.0), "B": (200.0, 0.0), "C": (100.0, 0.0), "D": (0.0, 0.0), "E": (150.0, 80.0)}


def goal(name: str, t: float) -> RouteGoal:
    x, y = STOPS[name]
    return RouteGoal(x=x, y=y, yaw=3.14, stamp_sec=t, source="recorded_route")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "map").mkdir()
        (root / "map" / "stop_points.csv").write_text(
            "name,x,y,z,yaw,lane_id,stop_point_id,allow_goal_modification\n"
            + "".join(f"{n},{x},{y},0.0,3.14,,{i},1\n" for i, (n, (x, y)) in enumerate(STOPS.items())),
            encoding="utf-8")
        for bag in ("full", "short", "other"):
            (root / "bags" / bag).mkdir(parents=True)
        # what the vehicle recorded: one drive A -> B -> C -> D, and B -> C once more
        routes = {"full": [goal("A", 0), goal("B", 10), goal("C", 20), goal("D", 30)],
                  "short": [],
                  "other": [goal("B", 0), goal("C", 5)]}
        saved = mg._recorded_route_goals
        mg._recorded_route_goals = lambda bag: routes.get(Path(bag).name, [])
        try:
            succ = mg.recorded_stop_successors(root / "bags", root / "map")
            assert succ == {"A": "B", "B": "C", "C": "D"}, succ
            print("PASS: the stop sequence is read from the routes the recordings carry")

            # cut at x=170, heading west, having left B behind; no route of its own
            end = (170.0, 0.0, 3.14159)
            path = [(float(x), 0.0) for x in range(230, 169, -5)]
            got = mg.next_stop_ahead(end, map_path=root / "map", recorded_path_xy=path, successors=succ)
            assert got is not None and (got.x, got.y) == STOPS["C"], got
            # a written order that says otherwise does not win over the vehicle's own sequence
            got = mg.next_stop_ahead(end, map_path=root / "map", recorded_path_xy=path, successors=succ,
                                     stop_order=["A", "B", "D", "C"])
            assert (got.x, got.y) == STOPS["C"], got
            # ... and without either, the nearest stop ahead
            got = mg.next_stop_ahead(end, map_path=root / "map", recorded_path_xy=path)
            assert (got.x, got.y) == STOPS["C"], got
            print("PASS: a cut recording goes to the stop after the last one it left, by the recorded sequence")

            # 25 m short of C: C is still ahead, not left behind — the stop after B, not after C
            end2 = (125.0, 0.0, 3.14159)
            path2 = [(float(x), 0.0) for x in range(230, 124, -5)]
            got = mg.next_stop_ahead(end2, map_path=root / "map", recorded_path_xy=path2, successors=succ)
            assert (got.x, got.y) == STOPS["C"], got
            print("PASS: a stop still ahead of the cut is not one the recording left behind")

            # its own route says where it was going: that, whatever the sequence says
            routes["short"] = [goal("B", 0), goal("E", 8)]
            own = mg.recorded_destination(root / "bags" / "short", end[:2])
            assert own is not None and (own.x, own.y) == STOPS["E"] and own.source == "cut_end_recorded_route", own
            routes["short"] = [goal("B", 0)]            # 30 m behind the cut... its last goal, already passed
            assert mg.recorded_destination(root / "bags" / "short", (200.0, 5.0)) is None, "a goal at the end is no destination beyond it"
            routes["short"] = []
            assert mg.recorded_destination(root / "bags" / "short", end[:2]) is None
            print("PASS: a recording's own last route goal beyond the cut is its destination")
        finally:
            mg._recorded_route_goals = saved

        # the ego's body from a vehicle_info file
        info = root / "vehicle_info.param.yaml"
        info.write_text(
            "/**:\n  ros__parameters:\n    wheel_base: 5.0\n    front_overhang: 2.0\n    rear_overhang: 3.0\n"
            "    wheel_tread: 2.0\n    left_overhang: 0.3\n    right_overhang: 0.3\n", encoding="utf-8")
        body = load_vehicle_body(info)
        assert abs(body.length_m - 10.0) < 1e-6 and abs(body.width_m - 2.6) < 1e-6, body
        assert abs(body.center_offset_m - 2.0) < 1e-6 and abs(body.front_m - 7.0) < 1e-6, body
        assert load_vehicle_body(root / "missing.yaml") is None and load_vehicle_body(None) is None
        assert body_of([(7.0, 1.3), (7.0, -1.3), (-3.0, -1.3), (-3.0, 1.3)]) == body
        from dp_multi_eval.scene_driver import BOX_EXTRA_LENGTH_M, BOX_EXTRA_WIDTH_M, build_scene, shield_body_params

        sp = shield_body_params(body)
        assert sp["real_length_m"] == body.length_m and sp["ego_length_m"] == body.length_m + BOX_EXTRA_LENGTH_M
        assert sp["ego_width_m"] == body.width_m + BOX_EXTRA_WIDTH_M and sp["real_center_offset_m"] == 2.0
        from types import SimpleNamespace as NS

        odom = [(int(2e18 + k * 1e8), NS(pose=NS(pose=NS(position=NS(x=float(k), y=0.0, z=0.0),
                                                           orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
                                         twist=NS(twist=NS(linear=NS(x=10.0, y=0.0, z=0.0)))))
                for k in range(30)]
        scene = build_scene([], odom, [], None, body=body)
        assert scene.shield.p.real_length_m == 10.0 and scene.gate.audit.body is body
        print("PASS: the ego's body is read from the vehicle's vehicle_info and reaches the shield and the audit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
