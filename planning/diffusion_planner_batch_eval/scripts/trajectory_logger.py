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

"""Record ego pose, planned trajectory, and tracked objects to CSV during batch eval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import time
from pathlib import Path

import rclpy
from autoware_control_msgs.msg import Control
from autoware_perception_msgs.msg import TrackedObjects
from autoware_planning_msgs.msg import Trajectory
from geometry_msgs.msg import AccelWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

OBJECT_LABELS = {
    0: "UNKNOWN",
    1: "CAR",
    2: "TRUCK",
    3: "BUS",
    4: "TRAILER",
    5: "MOTORCYCLE",
    6: "BICYCLE",
    7: "PEDESTRIAN",
}


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def uuid_to_str(uuid_msg) -> str:
    return "".join(f"{byte:02x}" for byte in uuid_msg.uuid)


def primary_label(classifications) -> str:
    if not classifications:
        return "UNKNOWN"
    best = max(classifications, key=lambda item: item.probability)
    return OBJECT_LABELS.get(best.label, f"LABEL_{best.label}")


class TrajectoryLoggerNode(Node):
    def __init__(self, output_dir: Path, model_name: str, bag_path: str) -> None:
        super().__init__("diffusion_planner_batch_eval_trajectory_logger")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ego_file = (self.output_dir / "ego_pose.csv").open("w", encoding="utf-8", newline="")
        self.traj_file = (self.output_dir / "planned_trajectory.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self.objects_file = (self.output_dir / "objects.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self.ego_accel_file = (self.output_dir / "ego_accel.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self.control_cmd_file = (self.output_dir / "control_cmd.csv").open(
            "w", encoding="utf-8", newline=""
        )

        self.ego_writer = csv.writer(self.ego_file)
        self.traj_writer = csv.writer(self.traj_file)
        self.objects_writer = csv.writer(self.objects_file)
        self.ego_accel_writer = csv.writer(self.ego_accel_file)
        self.control_cmd_writer = csv.writer(self.control_cmd_file)

        self.ego_writer.writerow(
            ["stamp_sec", "x", "y", "z", "yaw_rad", "vx", "vy", "speed_mps"]
        )
        self.traj_writer.writerow(
            [
                "stamp_sec",
                "point_idx",
                "x",
                "y",
                "z",
                "yaw_rad",
                "longitudinal_velocity_mps",
                "lateral_velocity_mps",
                "acceleration_mps2",
            ]
        )
        self.objects_writer.writerow(
            [
                "stamp_sec",
                "object_id",
                "label",
                "x",
                "y",
                "z",
                "yaw_rad",
                "length_m",
                "width_m",
                "height_m",
                "vx",
                "vy",
                "is_stationary",
            ]
        )
        self.ego_accel_writer.writerow(
            [
                "stamp_sec",
                "ax_mps2",
                "ay_mps2",
                "az_mps2",
                "a_long_mps2",
                "a_lat_mps2",
            ]
        )
        self.control_cmd_writer.writerow(
            [
                "stamp_sec",
                "cmd_velocity_mps",
                "cmd_acceleration_mps2",
                "cmd_jerk_mps3",
                "cmd_steering_tire_angle_rad",
                "cmd_steering_tire_rotation_rate_rps",
                "is_defined_acceleration",
                "is_defined_jerk",
            ]
        )

        metadata = {
            "model_name": model_name,
            "bag_path": bag_path,
            "started_at_unix": time.time(),
            "topics": {
                "ego": "/localization/kinematic_state",
                "ego_accel": "/localization/acceleration",
                "trajectory": "/planning/trajectory",
                "objects": "/perception/object_recognition/tracking/objects",
                "control_cmd": "/control/command/control_cmd",
            },
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        sensor_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, "/localization/kinematic_state", self._on_ego, sensor_qos)
        self.create_subscription(Trajectory, "/planning/trajectory", self._on_trajectory, sensor_qos)
        self.create_subscription(
            TrackedObjects,
            "/perception/object_recognition/tracking/objects",
            self._on_objects,
            sensor_qos,
        )
        self.create_subscription(
            AccelWithCovarianceStamped,
            "/localization/acceleration",
            self._on_ego_accel,
            sensor_qos,
        )
        self.create_subscription(
            Control,
            "/control/command/control_cmd",
            self._on_control_cmd,
            sensor_qos,
        )

        self._counts = {"ego": 0, "trajectory": 0, "objects": 0, "ego_accel": 0, "control_cmd": 0}
        self.get_logger().info(f"Logging trajectories to {self.output_dir}")

    def _on_ego(self, msg: Odometry) -> None:
        stamp = stamp_to_sec(msg.header.stamp)
        pose = msg.pose.pose
        twist = msg.twist.twist.linear
        yaw = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        speed = math.hypot(twist.x, twist.y)
        self.ego_writer.writerow(
            [f"{stamp:.6f}", pose.position.x, pose.position.y, pose.position.z, yaw, twist.x, twist.y, speed]
        )
        self._counts["ego"] += 1

    def _on_trajectory(self, msg: Trajectory) -> None:
        stamp = stamp_to_sec(msg.header.stamp)
        for point_idx, point in enumerate(msg.points):
            pose = point.pose
            yaw = yaw_from_quaternion(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            self.traj_writer.writerow(
                [
                    f"{stamp:.6f}",
                    point_idx,
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    yaw,
                    point.longitudinal_velocity_mps,
                    point.lateral_velocity_mps,
                    point.acceleration_mps2,
                ]
            )
        self._counts["trajectory"] += 1

    def _on_objects(self, msg: TrackedObjects) -> None:
        stamp = stamp_to_sec(msg.header.stamp)
        for obj in msg.objects:
            pose = obj.kinematics.pose_with_covariance.pose
            twist = obj.kinematics.twist_with_covariance.twist
            yaw = yaw_from_quaternion(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            shape = obj.shape
            self.objects_writer.writerow(
                [
                    f"{stamp:.6f}",
                    uuid_to_str(obj.object_id),
                    primary_label(obj.classification),
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    yaw,
                    shape.dimensions.x,
                    shape.dimensions.y,
                    shape.dimensions.z,
                    twist.linear.x,
                    twist.linear.y,
                    int(obj.kinematics.is_stationary),
                ]
            )
        self._counts["objects"] += 1

    def _on_ego_accel(self, msg: AccelWithCovarianceStamped) -> None:
        stamp = stamp_to_sec(msg.header.stamp)
        linear = msg.accel.accel.linear
        # base_link: +X forward, +Y left
        self.ego_accel_writer.writerow(
            [
                f"{stamp:.6f}",
                linear.x,
                linear.y,
                linear.z,
                linear.x,
                linear.y,
            ]
        )
        self._counts["ego_accel"] += 1

    def _on_control_cmd(self, msg: Control) -> None:
        stamp = stamp_to_sec(msg.stamp)
        lon = msg.longitudinal
        lat = msg.lateral
        self.control_cmd_writer.writerow(
            [
                f"{stamp:.6f}",
                lon.velocity,
                lon.acceleration,
                lon.jerk,
                lat.steering_tire_angle,
                lat.steering_tire_rotation_rate,
                int(lon.is_defined_acceleration),
                int(lon.is_defined_jerk),
            ]
        )
        self._counts["control_cmd"] += 1

    def close(self) -> None:
        for handle in (
            self.ego_file,
            self.traj_file,
            self.objects_file,
            self.ego_accel_file,
            self.control_cmd_file,
        ):
            handle.flush()
            handle.close()

        summary = {
            **json.loads((self.output_dir / "metadata.json").read_text(encoding="utf-8")),
            "finished_at_unix": time.time(),
            "message_counts": self._counts,
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        self.get_logger().info(
            "Trajectory log closed: "
            f"ego={self._counts['ego']}, "
            f"ego_accel={self._counts['ego_accel']}, "
            f"control_cmd={self._counts['control_cmd']}, "
            f"trajectory={self._counts['trajectory']}, "
            f"object_frames={self._counts['objects']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Log ego, trajectory, and NPC data to CSV.")
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for ego_pose.csv, planned_trajectory.csv, objects.csv, "
        "ego_accel.csv, control_cmd.csv",
    )
    parser.add_argument("--model-name", default="", help="Model name stored in metadata.json")
    parser.add_argument("--bag-path", default="", help="Source rosbag path stored in metadata.json")
    args = parser.parse_args()

    rclpy.init()
    node = TrajectoryLoggerNode(args.output_dir, args.model_name, args.bag_path)

    def shutdown_handler(*_args: object) -> None:
        node.close()
        node.destroy_node()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.close()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
