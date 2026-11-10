"""Overlay disables BOTH stop-stub rules sim-only; vehicle config untouched.

Run directly: python3 tests/test_ament_overlay.py  (env sourced)
"""
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.ament_overlay import (  # noqa: E402
    DP_LAUNCH_REL,
    TRAJ_MOD_REL,
    VEHICLE_TURN_INDICATOR_TOPIC,
    make_model_overlay,
    resolve_autoware_launch_share,
)

VEHICLE_CFG = Path(
    "/home/kaan/projects/pilot-auto.x2.isuzu.erga2/src/autoware/launcher/autoware_launch/"
    "config/planning/neural_net_planner/trajectory_modifier.param.yaml"
)


def main() -> int:
    try:
        resolve_autoware_launch_share()
    except FileNotFoundError:
        print("SKIP: needs a sourced Autoware workspace (ros2 pkg prefix autoware_launch)")
        return 0
    with tempfile.TemporaryDirectory() as d:
        mc = Path(d) / "model.param.yaml"
        mc.write_text("/**:\n  ros__parameters: {}\n")

        root = Path(d) / "ov"
        make_model_overlay(mc, root, disable_long_stop=True)
        data = yaml.safe_load((root / "share/autoware_launch" / TRAJ_MOD_REL).read_text())
        sp = data["/**"]["ros__parameters"]["stop_point_fixer"]
        assert sp["force_stop_long_stopped_trajectories"] is False
        assert sp["force_stop_close_stopped_trajectories"] is False
        assert abs(sp["velocity_threshold"] - 0.25) < 1e-9
        ob = data["/**"]["ros__parameters"]["obstacle_stop"]
        assert ob["enable_stop_for_objects"] is True, "obstacle_stop must stay armed"
        print("PASS: overlay disables both stub rules; obstacle_stop stays armed")

        root2 = Path(d) / "ov2"
        make_model_overlay(mc, root2)
        d2 = yaml.safe_load((root2 / "share/autoware_launch" / TRAJ_MOD_REL).read_text())
        sp2 = d2["/**"]["ros__parameters"]["stop_point_fixer"]
        assert sp2["force_stop_long_stopped_trajectories"] is True
        assert sp2["force_stop_close_stopped_trajectories"] is True
        print("PASS: without the flag the overlay leaves the rules alone")

        merged = "/dp_multi_eval/turn_indicators_status"
        root3 = Path(d) / "ov3"
        make_model_overlay(mc, root3, dp_turn_indicator_topic=merged)
        xml3 = (root3 / "share/autoware_launch" / DP_LAUNCH_REL).read_text()
        assert f'<arg name="input_turn_indicators" value="{merged}"/>' in xml3
        assert f'value="{VEHICLE_TURN_INDICATOR_TOPIC}"' not in xml3
        xml2 = (root2 / "share/autoware_launch" / DP_LAUNCH_REL).read_text()
        assert (
            f'<arg name="input_turn_indicators" value="{VEHICLE_TURN_INDICATOR_TOPIC}"/>'
            in xml2
        )
        print("PASS: overlay rewires the planner's indicator input only when asked")

    real_xml = (resolve_autoware_launch_share() / DP_LAUNCH_REL).read_text()
    assert (
        f'<arg name="input_turn_indicators" value="{VEHICLE_TURN_INDICATOR_TOPIC}"/>'
        in real_xml
    )
    print("PASS: installed launch file untouched")

    if VEHICLE_CFG.is_file():  # this workspace's source tree; absent elsewhere
        orig = yaml.safe_load(VEHICLE_CFG.read_text())
        so = orig["/**"]["ros__parameters"]["stop_point_fixer"]
        assert so["force_stop_long_stopped_trajectories"] is True
        assert so["force_stop_close_stopped_trajectories"] is True
    print("PASS: vehicle config untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
