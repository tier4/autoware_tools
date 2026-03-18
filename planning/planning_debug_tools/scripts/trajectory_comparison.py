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

"""
Trajectory comparison visualization with BEV, velocity, and acceleration plots.

Subscribes to original (from rosbag) and new (from live nodes) CandidateTrajectories,
ego odometry, and predicted objects.

Usage:
  ros2 run planning_debug_tools trajectory_comparison.py
  ros2 run planning_debug_tools trajectory_comparison.py --view-range 80
"""

import argparse
import math
import sys
import threading

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from autoware_internal_planning_msgs.msg import CandidateTrajectories
from autoware_perception_msgs.msg import PredictedObjects
from geometry_msgs.msg import AccelWithCovarianceStamped
from nav_msgs.msg import Odometry

matplotlib.use("TkAgg")

ORIGINAL_TRAJ_TOPIC = "/trajectory_replayer/original/candidate_trajectories"
NEW_TRAJ_TOPIC = "/planning/generator/candidate_trajectories"
ODOM_TOPIC = "/localization/kinematic_state"
OBJECTS_TOPIC = "/perception/object_recognition/objects"
ACCEL_TOPIC = "/localization/acceleration"

# ObjectClassification label constants
LABEL_UNKNOWN = 0
LABEL_CAR = 1
LABEL_TRUCK = 2
LABEL_BUS = 3
LABEL_TRAILER = 4
LABEL_MOTORCYCLE = 5
LABEL_BICYCLE = 6
LABEL_PEDESTRIAN = 7

OBJECT_COLORS = {
    LABEL_CAR: "#4488ff",
    LABEL_TRUCK: "#2255cc",
    LABEL_BUS: "#2255cc",
    LABEL_TRAILER: "#2255cc",
    LABEL_MOTORCYCLE: "#ff8800",
    LABEL_BICYCLE: "#ff8800",
    LABEL_PEDESTRIAN: "#ff3333",
    LABEL_UNKNOWN: "#888888",
}

OBJECT_LABELS = {
    LABEL_CAR: "Car",
    LABEL_TRUCK: "Truck",
    LABEL_BUS: "Bus",
    LABEL_TRAILER: "Trailer",
    LABEL_MOTORCYCLE: "Motorcycle",
    LABEL_BICYCLE: "Bicycle",
    LABEL_PEDESTRIAN: "Pedestrian",
    LABEL_UNKNOWN: "Unknown",
}


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def extract_time_from_start(points):
    return np.array([
        p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
        for p in points
    ])


def extract_xy(points):
    x = np.array([p.pose.position.x for p in points])
    y = np.array([p.pose.position.y for p in points])
    return x, y


def extract_velocity(points):
    return np.array([p.longitudinal_velocity_mps for p in points])


def extract_acceleration(points):
    return np.array([p.acceleration_mps2 for p in points])


def make_rotated_rect(cx, cy, yaw, length, width):
    """Create corners of a rotated rectangle centered at (cx, cy)."""
    cos_a = math.cos(yaw)
    sin_a = math.sin(yaw)
    hl = length / 2.0
    hw = width / 2.0
    corners = [
        (-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw),
    ]
    return [
        (cx + cos_a * dx - sin_a * dy, cy + sin_a * dx + cos_a * dy)
        for dx, dy in corners
    ]


class TrajectoryComparisonNode(Node):
    def __init__(self, candidate_index=0):
        super().__init__("trajectory_comparison")
        self.candidate_index_ = candidate_index
        self.original_traj_ = None
        self.new_traj_ = None
        self.odom_ = None
        self.accel_ = None
        self.objects_ = None
        self.lock_ = threading.Lock()

        qos = QoSProfile(depth=1)
        self.create_subscription(
            CandidateTrajectories, ORIGINAL_TRAJ_TOPIC, self._on_original, qos
        )
        self.create_subscription(
            CandidateTrajectories, NEW_TRAJ_TOPIC, self._on_new, qos
        )
        self.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, qos)
        self.create_subscription(
            AccelWithCovarianceStamped, ACCEL_TOPIC, self._on_accel, qos
        )
        self.create_subscription(
            PredictedObjects, OBJECTS_TOPIC, self._on_objects, qos
        )
        self.get_logger().info("Trajectory comparison node started")

    def _on_original(self, msg):
        with self.lock_:
            self.original_traj_ = msg

    def _on_new(self, msg):
        with self.lock_:
            self.new_traj_ = msg

    def _on_accel(self, msg):
        with self.lock_:
            self.accel_ = msg

    def _on_odom(self, msg):
        with self.lock_:
            self.odom_ = msg

    def _on_objects(self, msg):
        with self.lock_:
            self.objects_ = msg

    def get_data(self):
        with self.lock_:
            return self.original_traj_, self.new_traj_, self.odom_, self.accel_, self.objects_


def draw_ego(ax, odom):
    """Draw ego vehicle as a green rectangle with heading arrow."""
    pos = odom.pose.pose.position
    yaw = quat_to_yaw(odom.pose.pose.orientation)
    ego_length = 4.9
    ego_width = 1.8

    corners = make_rotated_rect(pos.x, pos.y, yaw, ego_length, ego_width)
    poly = plt.Polygon(corners, closed=True, facecolor="#44cc44", edgecolor="black",
                       linewidth=1.5, alpha=0.9, zorder=10)
    ax.add_patch(poly)

    arrow_len = ego_length * 0.7
    ax.annotate(
        "", xy=(pos.x + arrow_len * math.cos(yaw), pos.y + arrow_len * math.sin(yaw)),
        xytext=(pos.x, pos.y),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2),
        zorder=11,
    )


def draw_objects(ax, objects_msg):
    """Draw predicted objects as colored rectangles/circles."""
    for obj in objects_msg.objects:
        label = LABEL_UNKNOWN
        if obj.classification:
            label = obj.classification[0].label

        pose = obj.kinematics.initial_pose_with_covariance.pose
        ox = pose.position.x
        oy = pose.position.y
        yaw = quat_to_yaw(pose.orientation)

        color = OBJECT_COLORS.get(label, "#888888")
        dim = obj.shape.dimensions

        if label == LABEL_PEDESTRIAN:
            circle = plt.Circle(
                (ox, oy), radius=max(0.3, dim.x / 2.0),
                facecolor=color, edgecolor="black", linewidth=0.8,
                alpha=0.8, zorder=8,
            )
            ax.add_patch(circle)
        else:
            length = max(dim.x, 1.0)
            width = max(dim.y, 0.5)
            corners = make_rotated_rect(ox, oy, yaw, length, width)
            poly = plt.Polygon(
                corners, closed=True, facecolor=color, edgecolor="black",
                linewidth=0.8, alpha=0.7, zorder=8,
            )
            ax.add_patch(poly)


def main():
    parser = argparse.ArgumentParser(description="Trajectory comparison visualization")
    parser.add_argument(
        "--candidate-index", type=int, default=0,
        help="Index of candidate trajectory to compare (default: 0)",
    )
    parser.add_argument(
        "--update-rate", type=float, default=5.0,
        help="Plot update rate in Hz (default: 5.0)",
    )
    parser.add_argument(
        "--view-range", type=float, default=60.0,
        help="BEV half-range in meters around ego (default: 60.0)",
    )
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = TrajectoryComparisonNode(candidate_index=args.candidate_index)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

    ax_bev = fig.add_subplot(gs[:, 0])
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[1, 1], sharex=ax_vel)

    fig.suptitle("Trajectory Comparison: Original (bag) vs New (live)", fontsize=13)

    # velocity plot setup
    (line_vel_orig,) = ax_vel.plot([], [], "b-", label="Original", linewidth=2)
    (line_vel_new,) = ax_vel.plot([], [], "r--", label="New", linewidth=2)
    ax_vel.set_ylabel("Velocity [m/s]")
    ax_vel.legend(loc="upper right", fontsize=8)
    ax_vel.grid(True, alpha=0.3)

    # acceleration plot setup
    (line_acc_orig,) = ax_acc.plot([], [], "b-", label="Original", linewidth=2)
    (line_acc_new,) = ax_acc.plot([], [], "r--", label="New", linewidth=2)
    ax_acc.set_ylabel("Acceleration [m/s^2]")
    ax_acc.set_xlabel("Time [s]")
    ax_acc.legend(loc="upper right", fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    status_text = fig.text(0.02, 0.01, "Waiting for data...", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.ion()
    plt.show()

    idx = args.candidate_index
    view_range = args.view_range
    update_interval = 1.0 / args.update_rate

    try:
        while rclpy.ok() and plt.fignum_exists(fig.number):
            original_traj, new_traj, odom, accel, objects = node.get_data()

            has_original = (
                original_traj is not None
                and len(original_traj.candidate_trajectories) > idx
            )
            has_new = (
                new_traj is not None
                and len(new_traj.candidate_trajectories) > idx
            )
            has_odom = odom is not None
            has_objects = objects is not None

            # -- BEV --
            ax_bev.cla()
            ax_bev.set_aspect("equal")
            ax_bev.grid(True, alpha=0.2)
            ax_bev.set_title("Bird's Eye View", fontsize=11)

            ego_x, ego_y = 0.0, 0.0
            if has_odom:
                ego_x = odom.pose.pose.position.x
                ego_y = odom.pose.pose.position.y

            if has_original:
                pts = original_traj.candidate_trajectories[idx].points
                tx, ty = extract_xy(pts)
                ax_bev.plot(tx, ty, "b-", linewidth=2.5, label="Original", zorder=5)

            if has_new:
                pts = new_traj.candidate_trajectories[idx].points
                tx, ty = extract_xy(pts)
                ax_bev.plot(tx, ty, "r--", linewidth=2.5, label="New", zorder=6)

            if has_objects:
                draw_objects(ax_bev, objects)

            if has_odom:
                draw_ego(ax_bev, odom)
                ax_bev.set_xlim(ego_x - view_range, ego_x + view_range)
                ax_bev.set_ylim(ego_y - view_range, ego_y + view_range)

                ego_speed = math.sqrt(
                    odom.twist.twist.linear.x ** 2
                    + odom.twist.twist.linear.y ** 2
                )
                ego_speed_kmh = ego_speed * 3.6
                info_lines = f"Speed: {ego_speed:.1f} m/s ({ego_speed_kmh:.1f} km/h)"
                if accel is not None:
                    ego_acc = accel.accel.accel.linear.x
                    info_lines += f"\nAccel: {ego_acc:.2f} m/s^2"
                ax_bev.text(
                    0.98, 0.98, info_lines,
                    transform=ax_bev.transAxes, fontsize=10, fontweight="bold",
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
                    zorder=20,
                )

            # BEV legend
            legend_handles = []
            if has_original or has_new:
                legend_handles.append(
                    mpatches.Patch(color="#4488ff", label="Original traj")
                )
                legend_handles.append(
                    mpatches.Patch(color="#ff4444", label="New traj")
                )
            if has_odom:
                legend_handles.append(
                    mpatches.Patch(color="#44cc44", label="Ego")
                )
            if has_objects:
                seen_labels = set()
                for obj in objects.objects:
                    lbl = obj.classification[0].label if obj.classification else LABEL_UNKNOWN
                    if lbl not in seen_labels:
                        seen_labels.add(lbl)
                        legend_handles.append(
                            mpatches.Patch(
                                color=OBJECT_COLORS.get(lbl, "#888888"),
                                label=OBJECT_LABELS.get(lbl, "Unknown"),
                            )
                        )
            if legend_handles:
                ax_bev.legend(handles=legend_handles, loc="upper left", fontsize=7)

            ax_bev.set_xlabel("X [m]")
            ax_bev.set_ylabel("Y [m]")

            # -- Velocity / Acceleration --
            status_parts = []

            if has_original:
                pts = original_traj.candidate_trajectories[idx].points
                s = extract_time_from_start(pts)
                line_vel_orig.set_data(s, extract_velocity(pts))
                line_acc_orig.set_data(s, extract_acceleration(pts))
                status_parts.append(
                    f"Original: {len(pts)} pts, "
                    f"{len(original_traj.candidate_trajectories)} cands"
                )
            else:
                line_vel_orig.set_data([], [])
                line_acc_orig.set_data([], [])

            if has_new:
                pts = new_traj.candidate_trajectories[idx].points
                s = extract_time_from_start(pts)
                line_vel_new.set_data(s, extract_velocity(pts))
                line_acc_new.set_data(s, extract_acceleration(pts))
                status_parts.append(
                    f"New: {len(pts)} pts, "
                    f"{len(new_traj.candidate_trajectories)} cands"
                )
            else:
                line_vel_new.set_data([], [])
                line_acc_new.set_data([], [])

            if has_original or has_new:
                for ax in (ax_vel, ax_acc):
                    ax.relim()
                    ax.autoscale_view()
                status_text.set_text(" | ".join(status_parts))
            else:
                status_text.set_text("Waiting for data...")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(update_interval)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close("all")


if __name__ == "__main__":
    main()
