#!/usr/bin/env python3

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

"""Convert Autoware topics into MarkerArray for Foxglove Image overlay.

Inputs:
  - /planning/trajectory
  - /perception/object_recognition/objects
  - /planning/planning_factors/modifier_obstacle_stop
  - /planning/planning_factors/diffusion_planner
  - /vehicle/status/control_mode
  - /api/operation_mode/state

Outputs (visualization_msgs/MarkerArray):
  - /foxglove/overlay/trajectory
  - /foxglove/overlay/predicted_objects
  - /foxglove/overlay/planning_factors/modifier_obstacle_stop
  - /foxglove/overlay/planning_factors/diffusion_planner
  - /foxglove/overlay/control_mode   (AUTO / MANUAL HUD text in base_link)
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

import rclpy
from autoware_adapi_v1_msgs.msg import OperationModeState
from autoware_internal_planning_msgs.msg import PlanningFactor, PlanningFactorArray
from autoware_perception_msgs.msg import PredictedObjects
from autoware_planning_msgs.msg import Trajectory
from autoware_vehicle_msgs.msg import ControlModeReport
from foxglove_msgs.msg import ImageAnnotations, TextAnnotation
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

_BEHAVIOR = {
    PlanningFactor.UNKNOWN: "UNKNOWN",
    PlanningFactor.NONE: "NONE",
    PlanningFactor.SLOW_DOWN: "SLOW_DOWN",
    PlanningFactor.STOP: "STOP",
    PlanningFactor.SHIFT_LEFT: "SHIFT_LEFT",
    PlanningFactor.SHIFT_RIGHT: "SHIFT_RIGHT",
    PlanningFactor.TURN_LEFT: "TURN_LEFT",
    PlanningFactor.TURN_RIGHT: "TURN_RIGHT",
}

_CONTROL_MODE = {
    ControlModeReport.NO_COMMAND: "NO_COMMAND",
    ControlModeReport.AUTONOMOUS: "AUTO",
    ControlModeReport.AUTONOMOUS_STEER_ONLY: "AUTO_STEER",
    ControlModeReport.AUTONOMOUS_VELOCITY_ONLY: "AUTO_VEL",
    ControlModeReport.MANUAL: "MANUAL",
    ControlModeReport.DISENGAGED: "DISENGAGED",
    ControlModeReport.NOT_READY: "NOT_READY",
}

_OP_MODE = {
    OperationModeState.UNKNOWN: "OP_UNKNOWN",
    OperationModeState.STOP: "OP_STOP",
    OperationModeState.AUTONOMOUS: "OP_AUTO",
    OperationModeState.LOCAL: "OP_LOCAL",
    OperationModeState.REMOTE: "OP_REMOTE",
}

_GLYPHS = {
    "A": [((0.0, 1.0), (0.5, 0.0)), ((0.5, 0.0), (1.0, 1.0)), ((0.2, 0.6), (0.8, 0.6))],
    "D": [((0.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (0.7, 0.0)), ((0.7, 0.0), (1.0, 0.3)), ((1.0, 0.3), (1.0, 0.7)), ((1.0, 0.7), (0.7, 1.0)), ((0.7, 1.0), (0.0, 1.0))],
    "E": [((1.0, 0.0), (0.0, 0.0)), ((0.0, 0.0), (0.0, 1.0)), ((0.0, 0.5), (0.8, 0.5)), ((0.0, 1.0), (1.0, 1.0))],
    "G": [((1.0, 0.2), (0.8, 0.0)), ((0.8, 0.0), (0.2, 0.0)), ((0.2, 0.0), (0.0, 0.2)), ((0.0, 0.2), (0.0, 0.8)), ((0.0, 0.8), (0.2, 1.0)), ((0.2, 1.0), (1.0, 1.0)), ((1.0, 1.0), (1.0, 0.55)), ((1.0, 0.55), (0.55, 0.55))],
    "I": [((0.0, 0.0), (1.0, 0.0)), ((0.5, 0.0), (0.5, 1.0)), ((0.0, 1.0), (1.0, 1.0))],
    "L": [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (1.0, 1.0))],
    "M": [((0.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (0.5, 0.55)), ((0.5, 0.55), (1.0, 0.0)), ((1.0, 0.0), (1.0, 1.0))],
    "N": [((0.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (1.0, 1.0)), ((1.0, 1.0), (1.0, 0.0))],
    "O": [((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, 1.0)), ((1.0, 1.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0))],
    "R": [((0.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (0.8, 0.0)), ((0.8, 0.0), (1.0, 0.25)), ((1.0, 0.25), (0.8, 0.5)), ((0.8, 0.5), (0.0, 0.5)), ((0.5, 0.5), (1.0, 1.0))],
    "T": [((0.0, 0.0), (1.0, 0.0)), ((0.5, 0.0), (0.5, 1.0))],
    "U": [((0.0, 0.0), (0.0, 0.8)), ((0.0, 0.8), (0.2, 1.0)), ((0.2, 1.0), (0.8, 1.0)), ((0.8, 1.0), (1.0, 0.8)), ((1.0, 0.8), (1.0, 0.0))],
    "V": [((0.0, 0.0), (0.5, 1.0)), ((0.5, 1.0), (1.0, 0.0))],
    "?": [((0.0, 0.2), (0.2, 0.0)), ((0.2, 0.0), (0.8, 0.0)), ((0.8, 0.0), (1.0, 0.2)), ((1.0, 0.2), (0.5, 0.55)), ((0.5, 0.8), (0.5, 1.0))],
}


def _best_effort_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _delete_all(header: Header, ns: str) -> Marker:
    m = Marker()
    m.header = header
    m.ns = ns
    m.id = 0
    m.action = Marker.DELETEALL
    return m


def _behavior_color(behavior: int) -> tuple[float, float, float, float]:
    if behavior == PlanningFactor.STOP:
        return (1.0, 0.15, 0.15, 0.85)
    if behavior == PlanningFactor.SLOW_DOWN:
        return (1.0, 0.75, 0.1, 0.85)
    return (0.4, 0.7, 1.0, 0.75)


class FoxgloveOverlayMarkers(Node):
    def __init__(self) -> None:
        super().__init__("foxglove_overlay_markers")

        self._traj_pub = self.create_publisher(MarkerArray, "/foxglove/overlay/trajectory", 1)
        self._obj_pub = self.create_publisher(
            MarkerArray, "/foxglove/overlay/predicted_objects", 1
        )
        self._mode_pub = self.create_publisher(
            ImageAnnotations, "/foxglove/overlay/control_mode_2d", 5
        )
        self._factor_pubs: dict[str, Any] = {
            "modifier_obstacle_stop": self.create_publisher(
                MarkerArray,
                "/foxglove/overlay/planning_factors/modifier_obstacle_stop",
                1,
            ),
            "diffusion_planner": self.create_publisher(
                MarkerArray,
                "/foxglove/overlay/planning_factors/diffusion_planner",
                1,
            ),
        }

        self._control_mode_label = "MODE?"
        self._operation_mode_label = ""
        self._autoware_control_enabled: bool | None = None

        self.create_subscription(
            Trajectory, "/planning/trajectory", self._on_trajectory, _best_effort_qos()
        )
        self.create_subscription(
            PredictedObjects,
            "/perception/object_recognition/objects",
            self._on_objects,
            _best_effort_qos(),
        )
        self.create_subscription(
            PlanningFactorArray,
            "/planning/planning_factors/modifier_obstacle_stop",
            self._make_factor_cb("modifier_obstacle_stop"),
            _best_effort_qos(),
        )
        self.create_subscription(
            PlanningFactorArray,
            "/planning/planning_factors/diffusion_planner",
            self._make_factor_cb("diffusion_planner"),
            _best_effort_qos(),
        )
        self.create_subscription(
            ControlModeReport,
            "/vehicle/status/control_mode",
            self._on_control_mode,
            _best_effort_qos(),
        )
        self.create_subscription(
            OperationModeState,
            "/api/operation_mode/state",
            self._on_operation_mode,
            _best_effort_qos(),
        )

        # Publish the control-mode HUD on a steady timer using the last-known
        # label. Foxglove's image-annotation layer renders the latest message,
        # so a continuous stream guarantees the text is always visible even
        # when control_mode / operation_mode updates are sparse.
        self._mode_timer = self.create_timer(0.2, self._on_mode_timer)

        self.get_logger().info(
            "Foxglove overlays: trajectory, objects, planning_factors, control_mode"
        )

    def _on_trajectory(self, msg: Trajectory) -> None:
        markers = MarkerArray()
        line = Marker()
        line.header = msg.header
        line.ns = "foxglove_trajectory"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 2.0
        line.color.r = 0.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.color.a = 0.85
        line.pose.orientation.w = 1.0
        for pt in msg.points:
            p = Point()
            p.x = pt.pose.position.x
            p.y = pt.pose.position.y
            p.z = pt.pose.position.z
            line.points.append(p)
        markers.markers.append(line)
        self._traj_pub.publish(markers)

    def _on_objects(self, msg: PredictedObjects) -> None:
        markers = MarkerArray()
        markers.markers.append(_delete_all(msg.header, "foxglove_predicted_objects"))
        for i, obj in enumerate(msg.objects):
            box = Marker()
            box.header = msg.header
            box.ns = "foxglove_predicted_objects"
            box.id = i + 1
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose = obj.kinematics.initial_pose_with_covariance.pose
            dims = obj.shape.dimensions
            box.scale.x = max(float(dims.x), 0.2)
            box.scale.y = max(float(dims.y), 0.2)
            box.scale.z = max(float(dims.z), 0.2)
            box.color.r = 1.0
            box.color.g = 0.55
            box.color.b = 0.1
            box.color.a = 0.55
            markers.markers.append(box)
        self._obj_pub.publish(markers)

    def _make_factor_cb(self, source: str) -> Callable[[PlanningFactorArray], None]:
        def _cb(msg: PlanningFactorArray) -> None:
            self._on_planning_factors(msg, source)

        return _cb

    def _on_planning_factors(self, msg: PlanningFactorArray, source: str) -> None:
        pub = self._factor_pubs[source]
        markers = MarkerArray()
        ns = f"foxglove_pf_{source}"
        markers.markers.append(_delete_all(msg.header, ns))

        mid = 1
        for factor in msg.factors:
            behavior = int(factor.behavior)
            label = _BEHAVIOR.get(behavior, str(behavior))
            detail = (factor.detail or "").strip()
            module = (factor.module or source).strip() or source
            text = f"{module}:{label}" + (f" ({detail})" if detail else "")
            r, g, b, a = _behavior_color(behavior)

            for cp in factor.control_points:
                # Virtual-wall plate (approx Autoware stop wall).
                wall = Marker()
                wall.header = msg.header
                wall.ns = ns
                wall.id = mid
                mid += 1
                wall.type = Marker.CUBE
                wall.action = Marker.ADD
                wall.pose = cp.pose
                wall.scale.x = 0.2
                wall.scale.y = 3.0
                wall.scale.z = 2.0
                wall.color.r = r
                wall.color.g = g
                wall.color.b = b
                wall.color.a = a
                markers.markers.append(wall)

                txt = Marker()
                txt.header = msg.header
                txt.ns = ns
                txt.id = mid
                mid += 1
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose = cp.pose
                txt.pose.position.z = float(cp.pose.position.z) + 1.5
                txt.scale.z = 0.8
                txt.color.r = 1.0
                txt.color.g = 1.0
                txt.color.b = 1.0
                txt.color.a = 1.0
                txt.text = text
                markers.markers.append(txt)

        pub.publish(markers)

    def _on_control_mode(self, msg: ControlModeReport) -> None:
        self._control_mode_label = _CONTROL_MODE.get(int(msg.mode), f"MODE_{msg.mode}")

    def _on_operation_mode(self, msg: OperationModeState) -> None:
        self._operation_mode_label = _OP_MODE.get(int(msg.mode), f"OP_{msg.mode}")
        self._autoware_control_enabled = bool(msg.is_autoware_control_enabled)

    def _on_mode_timer(self) -> None:
        self._publish_mode_hud(self.get_clock().now().to_msg())

    def _publish_mode_hud(self, stamp) -> None:
        # Native Foxglove annotation is rendered after rectification, so the
        # label remains straight and fixed in the upper-left corner.
        text = self._control_mode_label
        if text not in ("AUTO", "MANUAL"):
            text = "MANUAL" if text in (
                "DISENGAGED", "NOT_READY", "NO_COMMAND", "MODE?"
            ) else text
        is_manual = text != "AUTO"
        if is_manual:
            color = (1.0, 0.15, 0.15, 1.0)
        else:
            color = (0.1, 1.0, 0.3, 1.0)

        annotation = TextAnnotation()
        annotation.timestamp = stamp
        annotation.position.x = 24.0
        annotation.position.y = 54.0
        annotation.text = text
        annotation.font_size = 34.0
        annotation.text_color.r = color[0]
        annotation.text_color.g = color[1]
        annotation.text_color.b = color[2]
        annotation.text_color.a = color[3]
        annotation.background_color.r = 0.02
        annotation.background_color.g = 0.02
        annotation.background_color.b = 0.02
        annotation.background_color.a = 0.78

        msg = ImageAnnotations()
        msg.timestamp = stamp
        msg.texts.append(annotation)
        self._mode_pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = FoxgloveOverlayMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
