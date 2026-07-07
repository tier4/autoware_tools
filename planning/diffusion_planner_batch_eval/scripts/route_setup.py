#!/usr/bin/env python3

# Copyright 2025 TIER IV, Inc.
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

"""Initialize localization and set a route goal for planning simulator batch evaluation."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np
import rclpy
from autoware_adapi_v1_msgs.msg import LocalizationInitializationState
from autoware_adapi_v1_msgs.msg import OperationModeState
from autoware_adapi_v1_msgs.msg import RouteState as AdapiRouteState
from autoware_adapi_v1_msgs.srv import ChangeOperationMode
from autoware_adapi_v1_msgs.srv import ClearRoute as AdapiClearRoute
from autoware_adapi_v1_msgs.srv import SetRoutePoints
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader
from tier4_localization_msgs.srv import InitializeLocalization
from tier4_planning_msgs.msg import RouteState as MissionRouteState
from tier4_planning_msgs.srv import ClearRoute as MissionClearRoute
from tier4_planning_msgs.srv import SetWaypointRoute

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rosbag_utils import open_bag_reader
from stop_points import DEFAULT_HIRATSUKA_ROUTE_ORDER
from stop_points import MapStopPoint
from stop_points import find_nearest_stop_point
from stop_points import load_stop_points
from stop_points import resolve_routing_goal
from stop_points import resolve_stop_points_path
from stop_points import route_segment_stop_names
from stop_points import stop_point_to_pose
from stop_points import stops_by_name


class RouteBackend(str, Enum):
    ADAPI = "adapi"
    MISSION_PLANNER = "mission_planner"


ADAPI_CLEAR_ROUTE = "/api/routing/clear_route"
ADAPI_SET_ROUTE = "/api/routing/set_route_points"
ADAPI_ROUTE_STATE = "/api/routing/state"

MISSION_CLEAR_ROUTE = "/planning/mission_planning/route_selector/main/clear_route"
MISSION_SET_ROUTE = "/planning/mission_planning/route_selector/main/set_waypoint_route"
MISSION_ROUTE_STATE = "/planning/mission_planning/route_selector/main/state"


def _distance_xy(p0: Pose, p1: Pose) -> float:
    return math.hypot(p0.position.x - p1.position.x, p0.position.y - p1.position.y)


def _yaw_from_quaternion(orientation) -> float:
    q = orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def offset_pose_longitudinal(pose: Pose, offset_m: float) -> Pose:
    """Shift pose along its heading. Positive = forward, negative = backward."""
    yaw = _yaw_from_quaternion(pose.orientation)
    shifted = Pose()
    shifted.position.x = pose.position.x + offset_m * math.cos(yaw)
    shifted.position.y = pose.position.y + offset_m * math.sin(yaw)
    shifted.position.z = pose.position.z
    shifted.orientation = pose.orientation
    return shifted


def _poses_from_kinematic_state(reader: SequentialReader, type_map: dict[str, str], min_move_m: float) -> list[Pose]:
    poses: list[Pose] = []
    prev_pose: Pose | None = None

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != "/localization/kinematic_state":
            continue
        msg_type = get_message(type_map[topic])
        odom: Odometry = deserialize_message(data, msg_type)
        pose = odom.pose.pose
        if prev_pose is None:
            poses.append(pose)
            prev_pose = pose
        elif _distance_xy(prev_pose, pose) >= min_move_m:
            poses.append(pose)
            prev_pose = pose

    return poses


def _poses_from_tf(reader: SequentialReader, type_map: dict[str, str], min_move_m: float) -> list[Pose]:
    from geometry_msgs.msg import Point
    from geometry_msgs.msg import Quaternion

    poses: list[Pose] = []
    prev_pose: Pose | None = None

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != "/tf":
            continue
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        for transform in msg.transforms:
            if transform.child_frame_id != "base_link":
                continue
            trans = transform.transform.translation
            rot = transform.transform.rotation
            pose = Pose(
                position=Point(x=trans.x, y=trans.y, z=trans.z),
                orientation=Quaternion(x=rot.x, y=rot.y, z=rot.z, w=rot.w),
            )
            if prev_pose is None:
                poses.append(pose)
                prev_pose = pose
            elif _distance_xy(prev_pose, pose) >= min_move_m:
                poses.append(pose)
                prev_pose = pose

    return poses


def get_poses_from_bag(bag_path: Path, min_move_m: float = 0.1) -> tuple[Pose, Pose]:
    reader = open_bag_reader(bag_path)
    type_map = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    poses: list[Pose] = []
    if "/localization/kinematic_state" in type_map:
        poses = _poses_from_kinematic_state(reader, type_map, min_move_m)

    if not poses and "/tf" in type_map:
        reader = open_bag_reader(bag_path)
        type_map = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
        poses = _poses_from_tf(reader, type_map, min_move_m)

    if not poses:
        raise ValueError(
            f"No poses found in {bag_path}. Need /localization/kinematic_state or /tf (base_link)."
        )

    return poses[0], poses[-1]


def list_ros_services() -> list[str]:
    try:
        output = subprocess.check_output(
            ["ros2", "service", "list"], text=True, stderr=subprocess.DEVNULL, timeout=10.0
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def format_service_diagnosis() -> str:
    services = list_ros_services()
    if not services:
        return (
            "No ROS services visible. Source install/setup.bash, check ROS_DOMAIN_ID, and "
            "launch planning simulator before running route_setup."
        )

    lines = [
        "Routing services not ready. Launch planning simulator first, for example:",
        "  ros2 launch autoware_launch planning_simulator.launch.xml "
        "planning_setting:=diffusion_planner",
        "",
        "Expected one of:",
        f"  - AD API: {ADAPI_CLEAR_ROUTE}, {ADAPI_SET_ROUTE}",
        f"  - Mission planner: {MISSION_CLEAR_ROUTE}, {MISSION_SET_ROUTE}",
    ]

    related = [
        service
        for service in services
        if "routing" in service or "mission_planning" in service or "localization/initialize" in service
    ]
    if related:
        lines.append("")
        lines.append("Visible related services:")
        lines.extend(f"  {service}" for service in sorted(related)[:20])
    else:
        lines.append("")
        lines.append("No routing or mission_planner services are visible in this ROS graph.")

    return "\n".join(lines)


class RouteSetupNode(Node):
    def __init__(self) -> None:
        super().__init__("diffusion_planner_batch_eval_route_setup")
        durable_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.callback_group = ReentrantCallbackGroup()
        self.backend: RouteBackend | None = None
        self.adapi_route_state: int | None = None
        self.mission_route_state: int | None = None
        self.localization_state: int | None = None
        self.operation_mode: int | None = None
        self.autoware_control_enabled = False
        self.autonomous_mode_available = False

        self.create_subscription(
            AdapiRouteState, ADAPI_ROUTE_STATE, self._on_adapi_route_state, durable_qos
        )
        self.create_subscription(
            MissionRouteState, MISSION_ROUTE_STATE, self._on_mission_route_state, durable_qos
        )
        self.create_subscription(
            LocalizationInitializationState,
            "/api/localization/initialization_state",
            self._on_localization_state,
            durable_qos,
        )
        self.create_subscription(
            OperationModeState,
            "/api/operation_mode/state",
            self._on_operation_mode,
            durable_qos,
        )

        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose3d", 1
        )

        self.adapi_clear_route_client = self.create_client(
            AdapiClearRoute, ADAPI_CLEAR_ROUTE, callback_group=self.callback_group
        )
        self.adapi_set_route_client = self.create_client(
            SetRoutePoints, ADAPI_SET_ROUTE, callback_group=self.callback_group
        )
        self.mission_clear_route_client = self.create_client(
            MissionClearRoute, MISSION_CLEAR_ROUTE, callback_group=self.callback_group
        )
        self.mission_set_route_client = self.create_client(
            SetWaypointRoute, MISSION_SET_ROUTE, callback_group=self.callback_group
        )
        self.localization_client = self.create_client(
            InitializeLocalization,
            "/localization/initialize",
            callback_group=self.callback_group,
        )
        self.enable_autoware_control_client = self.create_client(
            ChangeOperationMode,
            "/api/operation_mode/enable_autoware_control",
            callback_group=self.callback_group,
        )
        self.change_to_autonomous_client = self.create_client(
            ChangeOperationMode,
            "/api/operation_mode/change_to_autonomous",
            callback_group=self.callback_group,
        )
        self.change_to_stop_client = self.create_client(
            ChangeOperationMode,
            "/api/operation_mode/change_to_stop",
            callback_group=self.callback_group,
        )
        self.speed_mps = 0.0
        self.ego_pose: Pose | None = None
        sensor_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Odometry,
            "/localization/kinematic_state",
            self._on_kinematic_state,
            sensor_qos,
        )

    def _on_kinematic_state(self, msg: Odometry) -> None:
        twist = msg.twist.twist.linear
        self.speed_mps = math.hypot(twist.x, twist.y)
        self.ego_pose = msg.pose.pose

    def _wait_for_ego_near_pose(
        self, target: Pose, timeout_sec: float, tol_m: float = 2.0
    ) -> tuple[bool, str]:
        """Mission planner routes from /localization/kinematic_state, not /initialpose."""
        deadline = time.time() + timeout_sec
        last_log = 0.0
        while time.time() < deadline and rclpy.ok():
            self._spin_until(time.time() + 0.5)
            if self.ego_pose is None:
                continue
            dist = _distance_xy(self.ego_pose, target)
            if dist <= tol_m:
                return True, f"ego_at_initial_pose (dist={dist:.2f}m)"
            now = time.time()
            if now - last_log >= 5.0:
                self.get_logger().info(
                    f"Waiting for ego pose to match bag start (dist={dist:.2f}m, tol={tol_m}m)..."
                )
                last_log = now

        if self.ego_pose is None:
            return False, "ego_pose_timeout: no /localization/kinematic_state received"
        dist = _distance_xy(self.ego_pose, target)
        return False, f"ego_pose_timeout (dist={dist:.2f}m, tol={tol_m}m)"

    def _format_pose_xy(self, pose: Pose) -> str:
        return f"({pose.position.x:.2f}, {pose.position.y:.2f})"

    def _on_adapi_route_state(self, msg: AdapiRouteState) -> None:
        self.adapi_route_state = msg.state

    def _on_mission_route_state(self, msg: MissionRouteState) -> None:
        self.mission_route_state = msg.state

    def _on_localization_state(self, msg: LocalizationInitializationState) -> None:
        self.localization_state = msg.state

    def _on_operation_mode(self, msg: OperationModeState) -> None:
        self.operation_mode = msg.mode
        self.autoware_control_enabled = msg.is_autoware_control_enabled
        self.autonomous_mode_available = msg.is_autonomous_mode_available

    def _spin_until(self, deadline: float) -> None:
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def _route_is_unset(self) -> bool:
        if self.backend == RouteBackend.ADAPI:
            return self.adapi_route_state in (None, AdapiRouteState.UNSET)
        return self.mission_route_state in (None, MissionRouteState.UNSET)

    def _route_is_set(self) -> bool:
        if self.backend == RouteBackend.ADAPI:
            return self.adapi_route_state == AdapiRouteState.SET
        return self.mission_route_state == MissionRouteState.SET

    def _route_is_verified(self) -> bool:
        """Route is ready when AD API or mission planner reports SET."""
        if self.adapi_route_state == AdapiRouteState.SET:
            return True
        if self.mission_route_state == MissionRouteState.SET:
            return True
        return False

    def _route_state_summary(self) -> str:
        return f"adapi={self.adapi_route_state}, mission={self.mission_route_state}"

    def _disengage_and_wait_stopped(
        self, timeout_sec: float, stop_speed_mps: float = 0.2
    ) -> tuple[bool, str]:
        """Stop Auto mode and wait until ego is nearly stopped (required before clear_route)."""
        deadline = time.time() + timeout_sec
        self._spin_until(time.time() + 1.0)

        if self.operation_mode == OperationModeState.AUTONOMOUS:
            ok, message = self._call_change_mode(
                self.change_to_stop_client, "change_to_stop", timeout_sec=10.0
            )
            if not ok:
                self.get_logger().warn(f"change_to_stop failed: {message}")
            self._spin_until(time.time() + 2.0)

        last_log = 0.0
        while time.time() < deadline and rclpy.ok():
            self._spin_until(time.time() + 0.5)
            stopped = self.speed_mps <= stop_speed_mps
            not_auto = self.operation_mode != OperationModeState.AUTONOMOUS
            if stopped and not_auto:
                return True, f"disengaged (speed={self.speed_mps:.2f} m/s, mode={self.operation_mode})"

            now = time.time()
            if now - last_log >= 5.0:
                self.get_logger().info(
                    "Waiting to disengage before route reset "
                    f"(speed={self.speed_mps:.2f} m/s, mode={self.operation_mode})..."
                )
                last_log = now

        return False, (
            f"disengage_timeout (speed={self.speed_mps:.2f} m/s, mode={self.operation_mode})"
        )

    def _clear_route_once(self, timeout_sec: float) -> tuple[bool, str]:
        if self.backend == RouteBackend.ADAPI:
            client = self.adapi_clear_route_client
            request = AdapiClearRoute.Request()
        else:
            client = self.mission_clear_route_client
            request = MissionClearRoute.Request()

        if not client.wait_for_service(timeout_sec=5.0):
            return False, "clear_route_service_unavailable"

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done() or future.result() is None:
            return False, "clear_route_no_response"
        if not future.result().status.success:
            message = future.result().status.message
            return False, f"clear_route_failed: {message}"

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self._spin_until(time.time() + 0.5)
            if self._route_is_unset():
                return True, "route_cleared"
        return False, "clear_route_timeout"

    def _wait_for_backend(self, timeout_sec: float, preference: str) -> tuple[RouteBackend | None, str]:
        # New nodes need a moment for DDS service discovery between batch bags.
        self._spin_until(time.time() + 3.0)
        deadline = time.time() + timeout_sec
        last_log = 0.0
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            adapi_ready = (
                self.adapi_clear_route_client.wait_for_service(timeout_sec=0.0)
                and self.adapi_set_route_client.wait_for_service(timeout_sec=0.0)
            )
            mission_ready = (
                self.mission_clear_route_client.wait_for_service(timeout_sec=0.0)
                and self.mission_set_route_client.wait_for_service(timeout_sec=0.0)
            )

            if preference in ("auto", "adapi") and adapi_ready:
                return RouteBackend.ADAPI, "adapi services ready"
            if preference in ("auto", "mission_planner") and mission_ready:
                return RouteBackend.MISSION_PLANNER, "mission planner services ready"

            now = time.time()
            if now - last_log >= 10.0:
                self.get_logger().info(
                    "Waiting for routing services "
                    f"(adapi={adapi_ready}, mission_planner={mission_ready})..."
                )
                last_log = now
            time.sleep(0.2)

        return None, format_service_diagnosis()

    def _clear_route_if_needed(self, timeout_sec: float) -> tuple[bool, str]:
        self._spin_until(time.time() + 3.0)
        if self._route_is_unset():
            return True, "route_already_unset"

        if self.ego_pose is None:
            return False, (
                "clear_route_skipped: no /localization/kinematic_state "
                "(initialize localization before clearing route)"
            )

        clear_wait_sec = min(timeout_sec, 15.0)
        disengage_wait_sec = min(timeout_sec, 15.0)
        last_error = "clear_route_unknown"
        for attempt in range(3):
            if attempt > 0:
                self.get_logger().info(f"Retrying route clear (attempt {attempt + 1}/3)...")
                disengage_ok, disengage_msg = self._disengage_and_wait_stopped(disengage_wait_sec)
                if not disengage_ok:
                    last_error = f"disengage_before_clear_failed: {disengage_msg}"
                    continue

            ok, message = self._clear_route_once(clear_wait_sec)
            if ok:
                return True, message
            last_error = message
            if "cannot be cleared while it is in use" in message:
                self._disengage_and_wait_stopped(disengage_wait_sec)
            time.sleep(1.0)

        return False, last_error

    def _make_initial_pose_msg(self, pose: Pose) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        pose_with_cov = PoseWithCovariance()
        pose_with_cov.pose = pose
        pose_with_cov.covariance = (np.identity(6) * 0.01).flatten().tolist()
        msg.pose = pose_with_cov
        return msg

    def _initialize_localization(self, pose: Pose, timeout_sec: float) -> tuple[bool, str]:
        deadline = time.time() + timeout_sec
        initial_pose_msg = self._make_initial_pose_msg(pose)

        if self.localization_client.wait_for_service(timeout_sec=5.0):
            request = InitializeLocalization.Request()
            request.pose_with_covariance = [initial_pose_msg]
            request.method = 1
            future = self.localization_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            if not future.done() or future.result() is None:
                self.get_logger().warn(
                    "localization initialize service had no response; retrying with /initialpose3d"
                )
        else:
            self.get_logger().warn(
                "localization initialize service unavailable; publishing /initialpose3d only"
            )

        last_log = 0.0
        while time.time() < deadline and rclpy.ok():
            self.initial_pose_pub.publish(initial_pose_msg)
            self._spin_until(time.time() + 0.5)
            if self.ego_pose is not None:
                dist = _distance_xy(self.ego_pose, pose)
                if dist <= 2.0:
                    return True, f"initialized_kinematic_state (dist={dist:.2f}m)"
            now = time.time()
            if now - last_log >= 5.0:
                state = (
                    "no_kinematic_state"
                    if self.ego_pose is None
                    else f"dist={_distance_xy(self.ego_pose, pose):.1f}m"
                )
                self.get_logger().info(
                    f"Waiting for planning simulator initialization ({state})..."
                )
                last_log = now

        if self.ego_pose is None:
            return False, (
                "localization_failed: no /localization/kinematic_state "
                "(planning simulator still waiting initialization — increase psim_startup_sec)"
            )
        dist = _distance_xy(self.ego_pose, pose)
        return False, f"localization_initialize_timeout (dist={dist:.2f}m)"

    def _set_route(
        self, goal_pose: Pose, timeout_sec: float, waypoints: list[Pose] | None = None
    ) -> tuple[bool, str]:
        if not self._route_is_unset():
            return False, (
                f"route_not_unset_before_set: "
                f"adapi={self.adapi_route_state}, mission={self.mission_route_state}"
            )

        route_waypoints = waypoints or []

        if self.backend == RouteBackend.ADAPI:
            client = self.adapi_set_route_client
            request = SetRoutePoints.Request()
            request.header.frame_id = "map"
            request.header.stamp = self.get_clock().now().to_msg()
            request.goal = goal_pose
            request.waypoints = route_waypoints
        else:
            client = self.mission_set_route_client
            request = SetWaypointRoute.Request()
            request.header.frame_id = "map"
            request.header.stamp = self.get_clock().now().to_msg()
            request.goal_pose = goal_pose
            request.waypoints = route_waypoints
            request.allow_modification = True

        if not client.wait_for_service(timeout_sec=5.0):
            return False, "set_route_service_unavailable"

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        if not future.done() or future.result() is None:
            return False, "set_route_no_response"
        if not future.result().status.success:
            return False, f"set_route_rejected: {future.result().status.message}"

        set_wait_sec = min(timeout_sec, 20.0)
        deadline = time.time() + set_wait_sec
        while time.time() < deadline:
            self._spin_until(time.time() + 0.5)
            if self._route_is_set():
                waypoint_note = (
                    f" with_{len(route_waypoints)}_waypoints" if route_waypoints else ""
                )
                return True, f"route_set_via_{self.backend.value}{waypoint_note}"
        return False, "route_set_timeout"

    def _set_route_with_retry(
        self,
        initial_pose: Pose,
        goal_pose: Pose,
        timeout_sec: float,
        waypoints: list[Pose] | None = None,
    ) -> tuple[bool, str]:
        pose_wait_sec = min(timeout_sec, 20.0)
        ok, pose_message = self._wait_for_ego_near_pose(initial_pose, pose_wait_sec)
        if not ok:
            self.get_logger().warn(
                f"Ego pose not at bag start before routing ({pose_message}); "
                "mission planner may reject the route."
            )

        last_error = "set_route_unknown"
        for attempt in range(3):
            if attempt > 0:
                self.get_logger().info(f"Retrying set_route (attempt {attempt + 1}/3)...")
                time.sleep(2.0)
                self._wait_for_ego_near_pose(initial_pose, min(timeout_sec, 10.0))

            ok, message = self._set_route(goal_pose, timeout_sec, waypoints=waypoints)
            if ok:
                return True, f"{pose_message}; {message}"
            last_error = message
            if "planned route is empty" not in message:
                break

        pose_hint = (
            f" start={self._format_pose_xy(initial_pose)}"
            f" goal={self._format_pose_xy(goal_pose)}"
        )
        if waypoints:
            pose_hint += f" waypoints={len(waypoints)}"
        if self.ego_pose is not None:
            pose_hint += f" ego={self._format_pose_xy(self.ego_pose)}"
        return False, f"{last_error}{pose_hint}"

    def _try_route_to_goal(
        self,
        spawn_pose: Pose,
        bag_goal_pose: Pose,
        timeout_sec: float,
        stops: list[MapStopPoint] | None,
        stop_point_goal_fallback: bool,
        stop_point_snap_goal_m: float,
        stop_point_goal_min_dist_m: float,
        spawn_stop: MapStopPoint | None,
        route_stop_order: list[str],
        stop_point_waypoint_fallback: bool,
    ) -> tuple[bool, str]:
        routing_goal_pose = bag_goal_pose
        goal_stop: MapStopPoint | None = None
        goal_dist = float("nan")
        goal_snapped = False

        if stops:
            routing_goal_pose, goal_stop, goal_dist, goal_snapped = resolve_routing_goal(
                bag_goal_pose, stops, stop_point_snap_goal_m
            )
            if goal_snapped and goal_stop is not None:
                self.get_logger().info(
                    f"Routing goal snapped to map stop '{goal_stop.name}' ({goal_dist:.1f}m from bag goal)"
                )

        ok, message = self._set_route_with_retry(spawn_pose, routing_goal_pose, timeout_sec)
        if ok:
            goal_note = (
                f"goal_snapped={goal_stop.name}({goal_dist:.1f}m); "
                if goal_snapped and goal_stop is not None
                else "goal_from_bag; "
            )
            return True, f"{goal_note}{message}"

        if (
            stop_point_goal_fallback
            and stops
            and goal_stop is not None
            and not goal_snapped
            and goal_dist >= stop_point_goal_min_dist_m
        ):
            self.get_logger().warn(
                f"Bag goal route failed; retrying with map stop '{goal_stop.name}' "
                f"({goal_dist:.1f}m from bag goal)"
            )
            clear_ok, clear_message = self._clear_route_if_needed(timeout_sec)
            if not clear_ok:
                return False, f"goal_from_bag; {message}; stop_fallback_clear_failed: {clear_message}"
            time.sleep(0.5)
            routing_goal_pose = stop_point_to_pose(goal_stop)
            ok, fallback_message = self._set_route_with_retry(
                spawn_pose, routing_goal_pose, timeout_sec
            )
            if ok:
                return True, (
                    f"goal_fallback={goal_stop.name}({goal_dist:.1f}m); {fallback_message}"
                )
            message = f"goal_from_bag; {message}; stop_fallback_failed: {fallback_message}"
            goal_snapped = True

        if (
            not stop_point_waypoint_fallback
            or stops is None
            or spawn_stop is None
            or goal_stop is None
            or spawn_stop.name == goal_stop.name
        ):
            prefix = "" if goal_snapped else "goal_from_bag; "
            return False, f"{prefix}{message}"

        intermediate_names = route_segment_stop_names(
            spawn_stop.name, goal_stop.name, route_stop_order
        )
        if not intermediate_names:
            prefix = "" if goal_snapped else "goal_from_bag; "
            return False, message if goal_snapped else f"{prefix}{message}"

        stop_lookup = stops_by_name(stops)
        waypoint_poses = [
            stop_point_to_pose(stop_lookup[name])
            for name in intermediate_names
            if name in stop_lookup
        ]
        if not waypoint_poses:
            return False, message

        self.get_logger().warn(
            f"Direct route empty; retrying with waypoints "
            f"{' -> '.join(intermediate_names)} -> {goal_stop.name}"
        )
        clear_ok, clear_message = self._clear_route_if_needed(timeout_sec)
        if not clear_ok:
            return False, f"{message}; waypoint_clear_failed: {clear_message}"
        time.sleep(0.5)

        ok, waypoint_message = self._set_route_with_retry(
            spawn_pose,
            stop_point_to_pose(goal_stop),
            timeout_sec,
            waypoints=waypoint_poses,
        )
        if ok:
            return True, (
                f"waypoint_route={'->'.join(intermediate_names)};"
                f"goal={goal_stop.name}; {waypoint_message}"
            )
        return False, f"{message}; waypoint_route_failed: {waypoint_message}"

    def _try_set_route_with_stop_fallback(
        self,
        initial_pose: Pose,
        bag_goal_pose: Pose,
        timeout_sec: float,
        map_path: Path | None,
        stop_point_goal_fallback: bool,
        stop_points_csv: str,
        stop_point_goal_min_dist_m: float,
        stop_point_snap_goal_m: float,
        route_stop_order: list[str],
        stop_point_waypoint_fallback: bool,
        spawn_stop: MapStopPoint | None = None,
    ) -> tuple[bool, str]:
        stops = self._load_map_stops(map_path, stop_points_csv)
        if stops and spawn_stop is None:
            spawn_stop, spawn_dist = find_nearest_stop_point(initial_pose, stops)
            if spawn_dist <= stop_point_snap_goal_m:
                self.get_logger().info(
                    f"Spawn matches map stop '{spawn_stop.name}' ({spawn_dist:.1f}m)"
                )

        return self._try_route_to_goal(
            initial_pose,
            bag_goal_pose,
            timeout_sec,
            stops,
            stop_point_goal_fallback=stop_point_goal_fallback,
            stop_point_snap_goal_m=stop_point_snap_goal_m,
            stop_point_goal_min_dist_m=stop_point_goal_min_dist_m,
            spawn_stop=spawn_stop,
            route_stop_order=route_stop_order,
            stop_point_waypoint_fallback=stop_point_waypoint_fallback,
        )

    def _load_map_stops(
        self, map_path: Path | None, stop_points_csv: str
    ) -> list[MapStopPoint] | None:
        if map_path is None:
            return None
        stop_points_path = resolve_stop_points_path(map_path, stop_points_csv)
        if not stop_points_path.is_file():
            self.get_logger().warn(f"Stop points file missing: {stop_points_path}")
            return None
        try:
            return load_stop_points(stop_points_path)
        except (OSError, ValueError) as error:
            self.get_logger().warn(f"Could not load stop points ({error})")
            return None

    def _try_start_stop_fallback(
        self,
        bag_start_pose: Pose,
        bag_goal_pose: Pose,
        route_failure_message: str,
        timeout_sec: float,
        map_path: Path | None,
        stop_points_csv: str,
        start_pose_stop_min_dist_m: float,
        stop_point_goal_fallback: bool,
        stop_point_goal_min_dist_m: float,
        stop_point_snap_goal_m: float,
        route_stop_order: list[str],
        stop_point_waypoint_fallback: bool,
    ) -> tuple[bool, str]:
        stops = self._load_map_stops(map_path, stop_points_csv)
        if not stops:
            return False, route_failure_message

        start_stop, start_dist = find_nearest_stop_point(bag_start_pose, stops)
        stop_spawn = stop_point_to_pose(start_stop)
        if _distance_xy(bag_start_pose, stop_spawn) < start_pose_stop_min_dist_m:
            return False, (
                f"{route_failure_message}; "
                f"start_stop_skipped: bag_start_already_at_{start_stop.name}({start_dist:.1f}m)"
            )

        self.get_logger().warn(
            f"Bag start route failed; re-localizing at map stop '{start_stop.name}' "
            f"({start_dist:.1f}m from bag start)"
        )

        clear_ok, clear_message = self._clear_route_if_needed(timeout_sec)
        if not clear_ok:
            return False, (
                f"{route_failure_message}; start_fallback_clear_failed: {clear_message}"
            )
        time.sleep(0.5)

        init_ok, init_message = self._initialize_localization(stop_spawn, timeout_sec)
        if not init_ok:
            return False, (
                f"{route_failure_message}; start_fallback_init_failed: {init_message}"
            )
        time.sleep(1.0)

        ok, route_message = self._try_set_route_with_stop_fallback(
            stop_spawn,
            bag_goal_pose,
            timeout_sec,
            map_path=map_path,
            stop_point_goal_fallback=stop_point_goal_fallback,
            stop_points_csv=stop_points_csv,
            stop_point_goal_min_dist_m=stop_point_goal_min_dist_m,
            stop_point_snap_goal_m=stop_point_snap_goal_m,
            route_stop_order=route_stop_order,
            stop_point_waypoint_fallback=stop_point_waypoint_fallback,
            spawn_stop=start_stop,
        )
        if ok:
            return True, (
                f"start_fallback={start_stop.name}({start_dist:.1f}m); "
                f"spawn={self._format_pose_xy(stop_spawn)}; {route_message}"
            )
        return False, (
            f"{route_failure_message}; start_fallback_route_failed: {route_message}"
        )

    def _call_change_mode(
        self, client, service_name: str, timeout_sec: float
    ) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=5.0):
            return False, f"{service_name}_unavailable"

        future = client.call_async(ChangeOperationMode.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done() or future.result() is None:
            return False, f"{service_name}_no_response"
        if not future.result().status.success:
            return False, f"{service_name}_rejected: {future.result().status.message}"
        return True, service_name

    def _engage_autonomous(self, timeout_sec: float) -> tuple[bool, str]:
        deadline = time.time() + timeout_sec
        self._spin_until(time.time() + 1.0)

        if not self._route_is_verified():
            return False, (
                f"route_not_set_before_engage ({self._route_state_summary()})"
            )

        if self.operation_mode == OperationModeState.AUTONOMOUS:
            return True, "already_autonomous"

        last_failure = "autonomous_mode_not_available_yet"
        last_log = 0.0
        while time.time() < deadline and rclpy.ok():
            self._spin_until(time.time() + 0.5)

            if self.operation_mode == OperationModeState.AUTONOMOUS:
                return True, "autonomous_engaged"

            now = time.time()
            if now - last_log >= 10.0:
                self.get_logger().info(
                    "Waiting to engage Auto "
                    f"(available={self.autonomous_mode_available}, "
                    f"control_enabled={self.autoware_control_enabled}, "
                    f"mode={self.operation_mode})..."
                )
                last_log = now

            if not self.autonomous_mode_available:
                continue

            if not self.autoware_control_enabled:
                ok, message = self._call_change_mode(
                    self.enable_autoware_control_client,
                    "enable_autoware_control",
                    timeout_sec=10.0,
                )
                if not ok:
                    last_failure = message
                    time.sleep(2.0)
                    continue
                self._spin_until(time.time() + 1.0)

            ok, message = self._call_change_mode(
                self.change_to_autonomous_client, "change_to_autonomous", timeout_sec=10.0
            )
            if not ok:
                last_failure = message
                time.sleep(3.0)
                continue

            confirm_deadline = time.time() + 15.0
            held_since: float | None = None
            while time.time() < confirm_deadline:
                self._spin_until(time.time() + 0.2)
                if self.operation_mode == OperationModeState.AUTONOMOUS:
                    if held_since is None:
                        held_since = time.time()
                    elif time.time() - held_since >= 2.0:
                        return True, (
                            f"autonomous_engaged (mode={self.operation_mode}, "
                            f"control={self.autoware_control_enabled})"
                        )
                else:
                    held_since = None
            last_failure = (
                f"autonomous_engage_not_held (mode={self.operation_mode}, "
                f"control={self.autoware_control_enabled})"
            )
            time.sleep(3.0)

        return False, last_failure

    def setup_for_bag(
        self,
        bag_path: Path,
        timeout_sec: float,
        min_move_m: float,
        service_wait_sec: float,
        backend_preference: str,
        auto_engage: bool,
        start_pose_offset_m: float = 0.0,
        map_path: Path | None = None,
        stop_point_goal_fallback: bool = False,
        stop_points_csv: str = "stop_points.csv",
        stop_point_goal_min_dist_m: float = 0.5,
        stop_point_snap_goal_m: float = 2.0,
        stop_point_waypoint_fallback: bool = True,
        route_stop_order: list[str] | None = None,
        start_pose_stop_fallback: bool = False,
        start_pose_stop_min_dist_m: float = 1.0,
    ) -> tuple[bool, str]:
        route_order = route_stop_order or DEFAULT_HIRATSUKA_ROUTE_ORDER
        backend, message = self._wait_for_backend(service_wait_sec, backend_preference)
        if backend is None:
            return False, message
        self.backend = backend
        self.get_logger().info(f"Using route backend: {backend.value}")

        try:
            bag_start_pose, goal_pose = get_poses_from_bag(bag_path, min_move_m=min_move_m)
        except ValueError as error:
            return False, str(error)

        initial_pose = bag_start_pose
        offset_note = ""
        if abs(start_pose_offset_m) > 1e-6:
            initial_pose = offset_pose_longitudinal(bag_start_pose, start_pose_offset_m)
            offset_note = (
                f"start_offset={start_pose_offset_m:+.2f}m "
                f"bag_start={self._format_pose_xy(bag_start_pose)} "
                f"spawn={self._format_pose_xy(initial_pose)}"
            )
            self.get_logger().info(f"Applying start pose offset: {offset_note}")

        disengage_ok, disengage_msg = self._disengage_and_wait_stopped(min(timeout_sec, 25.0))
        if not disengage_ok:
            self.get_logger().warn(
                f"Proceeding with route reset despite disengage warning: {disengage_msg}"
            )

        ok, step_message = self._initialize_localization(initial_pose, timeout_sec)
        if not ok:
            return False, step_message

        time.sleep(1.0)

        ok, step_message = self._clear_route_if_needed(timeout_sec)
        if not ok:
            return False, step_message

        time.sleep(1.0)

        ok, step_message = self._try_set_route_with_stop_fallback(
            initial_pose,
            goal_pose,
            timeout_sec,
            map_path=map_path,
            stop_point_goal_fallback=stop_point_goal_fallback,
            stop_points_csv=stop_points_csv,
            stop_point_goal_min_dist_m=stop_point_goal_min_dist_m,
            stop_point_snap_goal_m=stop_point_snap_goal_m,
            route_stop_order=route_order,
            stop_point_waypoint_fallback=stop_point_waypoint_fallback,
        )
        if not ok and start_pose_stop_fallback:
            ok, step_message = self._try_start_stop_fallback(
                bag_start_pose,
                goal_pose,
                step_message,
                timeout_sec,
                map_path=map_path,
                stop_points_csv=stop_points_csv,
                start_pose_stop_min_dist_m=start_pose_stop_min_dist_m,
                stop_point_goal_fallback=stop_point_goal_fallback,
                stop_point_goal_min_dist_m=stop_point_goal_min_dist_m,
                stop_point_snap_goal_m=stop_point_snap_goal_m,
                route_stop_order=route_order,
                stop_point_waypoint_fallback=stop_point_waypoint_fallback,
            )
        if not ok:
            return False, step_message

        verify_deadline = time.time() + min(timeout_sec, 15.0)
        while time.time() < verify_deadline:
            self._spin_until(time.time() + 0.5)
            if self._route_is_verified():
                step_message = f"{step_message}; route_verified ({self._route_state_summary()})"
                break
        else:
            return False, (
                f"route_verify_failed: goal not confirmed SET "
                f"({self._route_state_summary()})"
            )

        if auto_engage:
            ok, engage_message = self._engage_autonomous(timeout_sec)
            if not ok:
                self.get_logger().warn(
                    f"Auto engage failed ({engage_message}). "
                    "Route is set — click Auto in RViz if the vehicle does not move."
                )
                result = f"{step_message}; engage_warn: {engage_message}"
                return True, f"{offset_note}; {result}" if offset_note else result
            result = f"{step_message}; {engage_message}"
            return True, f"{offset_note}; {result}" if offset_note else result

        return True, f"{offset_note}; {step_message}" if offset_note else step_message


def verify_route_is_set(timeout_sec: float = 10.0) -> tuple[bool, str]:
    """Wait until AD API or mission planner reports route SET."""
    if not rclpy.ok():
        rclpy.init()

    node = RouteSetupNode()
    try:
        deadline = time.time() + timeout_sec
        while time.time() < deadline and rclpy.ok():
            node._spin_until(time.time() + 0.5)
            if node._route_is_verified():
                return True, f"route_verified ({node._route_state_summary()})"
        return False, f"route_not_set ({node._route_state_summary()})"
    finally:
        node.destroy_node()


def disengage_for_next_bag(timeout_sec: float = 25.0) -> tuple[bool, str]:
    """Disengage Auto and wait for stop so the next bag can clear/set route."""
    if not rclpy.ok():
        rclpy.init()

    node = RouteSetupNode()
    try:
        return node._disengage_and_wait_stopped(timeout_sec)
    finally:
        node.destroy_node()


def engage_autonomous_mode(timeout_sec: float = 60.0) -> tuple[bool, str]:
    """Call AD API to switch to Auto. Retry until timeout (modules often need perception first)."""
    if not rclpy.ok():
        rclpy.init()

    node = RouteSetupNode()
    try:
        return node._engage_autonomous(timeout_sec)
    finally:
        node.destroy_node()


def wait_for_autonomous_mode(timeout_sec: float = 60.0) -> tuple[bool, str]:
    """Wait until /api/operation_mode/state reports AUTONOMOUS (e.g. manual RViz engage)."""
    if not rclpy.ok():
        rclpy.init()

    node = RouteSetupNode()
    try:
        deadline = time.time() + timeout_sec
        last_log = 0.0
        while time.time() < deadline and rclpy.ok():
            node._spin_until(time.time() + 0.5)
            if node.operation_mode == OperationModeState.AUTONOMOUS:
                return True, "autonomous_engaged"
            now = time.time()
            if now - last_log >= 10.0:
                node.get_logger().info(
                    "Waiting for Auto mode "
                    f"(mode={node.operation_mode}, timeout in {deadline - now:.0f}s)..."
                )
                last_log = now
        return False, "wait_autonomous_timeout"
    finally:
        node.destroy_node()


def setup_route_for_bag(
    bag_path: Path,
    timeout_sec: float = 60.0,
    min_move_m: float = 0.1,
    service_wait_sec: float = 120.0,
    backend_preference: str = "auto",
    auto_engage: bool = True,
    start_pose_offset_m: float = 0.0,
    map_path: Path | None = None,
    stop_point_goal_fallback: bool = False,
    stop_points_csv: str = "stop_points.csv",
    stop_point_goal_min_dist_m: float = 0.5,
    stop_point_snap_goal_m: float = 2.0,
    stop_point_waypoint_fallback: bool = True,
    route_stop_order: list[str] | None = None,
    start_pose_stop_fallback: bool = False,
    start_pose_stop_min_dist_m: float = 1.0,
) -> tuple[bool, str]:
    if not rclpy.ok():
        rclpy.init()

    node = RouteSetupNode()
    try:
        return node.setup_for_bag(
            bag_path,
            timeout_sec=timeout_sec,
            min_move_m=min_move_m,
            service_wait_sec=service_wait_sec,
            backend_preference=backend_preference,
            auto_engage=auto_engage,
            start_pose_offset_m=start_pose_offset_m,
            map_path=map_path,
            stop_point_goal_fallback=stop_point_goal_fallback,
            stop_points_csv=stop_points_csv,
            stop_point_goal_min_dist_m=stop_point_goal_min_dist_m,
            stop_point_snap_goal_m=stop_point_snap_goal_m,
            stop_point_waypoint_fallback=stop_point_waypoint_fallback,
            route_stop_order=route_stop_order,
            start_pose_stop_fallback=start_pose_stop_fallback,
            start_pose_stop_min_dist_m=start_pose_stop_min_dist_m,
        )
    finally:
        node.destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description="Set initial pose and route goal from a rosbag.")
    parser.add_argument("-b", "--bag", required=True, type=Path, help="Rosbag file or directory")
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout per step in seconds")
    parser.add_argument(
        "--service-wait",
        type=float,
        default=120.0,
        help="How long to wait for routing services to appear",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "adapi", "mission_planner"],
        default="auto",
        help="Routing backend to use",
    )
    parser.add_argument(
        "--min-move-m", type=float, default=0.1, help="Minimum ego motion between sampled poses"
    )
    parser.add_argument(
        "--start-offset-m",
        type=float,
        default=0.0,
        help="Shift spawn pose along bag-start heading (negative = backward, e.g. -2.0)",
    )
    parser.add_argument(
        "--no-auto-engage",
        action="store_true",
        help="Do not call enable_autoware_control / change_to_autonomous after route is set",
    )
    parser.add_argument(
        "--map-path",
        type=Path,
        default=None,
        help="Map directory containing stop_points.csv (enables stop fallback with --stop-point-fallback)",
    )
    parser.add_argument(
        "--stop-point-fallback",
        action="store_true",
        help="If bag goal routing fails, retry with nearest map stop point",
    )
    parser.add_argument(
        "--start-stop-fallback",
        action="store_true",
        help="If routing fails, re-localize at nearest map stop to bag start and retry",
    )
    parser.add_argument(
        "--stop-points-csv",
        default="stop_points.csv",
        help="Stop points filename inside map_path (default: stop_points.csv)",
    )
    args = parser.parse_args()

    ok, message = setup_route_for_bag(
        args.bag,
        timeout_sec=args.timeout,
        min_move_m=args.min_move_m,
        service_wait_sec=args.service_wait,
        backend_preference=args.backend,
        auto_engage=not args.no_auto_engage,
        start_pose_offset_m=args.start_offset_m,
        map_path=args.map_path,
        stop_point_goal_fallback=args.stop_point_fallback,
        stop_points_csv=args.stop_points_csv,
        start_pose_stop_fallback=args.start_stop_fallback,
    )
    if ok:
        print(f"Route setup succeeded: {message}")
        sys.exit(0)
    print(f"Route setup failed:\n{message}")
    sys.exit(1)


if __name__ == "__main__":
    main()
