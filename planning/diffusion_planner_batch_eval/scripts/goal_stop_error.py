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

"""Goal stop error: ego pose at stop vs route goal from the source rosbag."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rosbag_utils import open_bag_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw_rad: float


@dataclass
class GoalStopMetrics:
    goal_stop_lateral_m: float
    goal_stop_longitudinal_m: float
    goal_stop_position_m: float
    goal_stop_heading_deg: float
    goal_stop_speed_mps: float
    goal_stop_time_sec: float
    stop_sample_idx: float

    def as_dict(self) -> dict[str, float]:
        return {
            "goal_stop_lateral_m": self.goal_stop_lateral_m,
            "goal_stop_longitudinal_m": self.goal_stop_longitudinal_m,
            "goal_stop_position_m": self.goal_stop_position_m,
            "goal_stop_heading_deg": self.goal_stop_heading_deg,
            "goal_stop_speed_mps": self.goal_stop_speed_mps,
            "goal_stop_time_sec": self.goal_stop_time_sec,
            "stop_sample_idx": self.stop_sample_idx,
        }


def normalize_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def goal_frame_offset(
    ego_x: float, ego_y: float, ego_yaw: float, goal: Pose2D
) -> tuple[float, float, float, float]:
    """Return lateral [m], longitudinal [m], position [m], heading [deg] in goal frame."""
    dx = ego_x - goal.x
    dy = ego_y - goal.y
    cos_g = math.cos(goal.yaw_rad)
    sin_g = math.sin(goal.yaw_rad)
    longitudinal = dx * cos_g + dy * sin_g
    lateral = -dx * sin_g + dy * cos_g
    position = math.hypot(dx, dy)
    heading_deg = math.degrees(normalize_angle_rad(ego_yaw - goal.yaw_rad))
    return lateral, longitudinal, position, heading_deg


def find_final_stop_index(rows: list[dict[str, str]], speed_threshold: float) -> int:
    """Index of the first sample in the final stopped segment (speed below threshold)."""
    if not rows:
        return 0
    stop_idx = len(rows) - 1
    while stop_idx > 0 and float(rows[stop_idx - 1]["speed_mps"]) <= speed_threshold:
        stop_idx -= 1
    return stop_idx


def load_ego_pose_rows(ego_pose_csv: Path) -> list[dict[str, str]]:
    with ego_pose_csv.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def compute_goal_stop_metrics(
    rows: list[dict[str, str]], goal: Pose2D, speed_threshold: float = 0.2
) -> GoalStopMetrics | None:
    if not rows:
        return None

    stop_idx = find_final_stop_index(rows, speed_threshold)
    row = rows[stop_idx]
    ego_x = float(row["x"])
    ego_y = float(row["y"])
    ego_yaw = float(row["yaw_rad"])
    speed = float(row["speed_mps"])
    start_stamp = float(rows[0]["stamp_sec"])
    stop_time = float(row["stamp_sec"]) - start_stamp

    lateral, longitudinal, position, heading_deg = goal_frame_offset(
        ego_x, ego_y, ego_yaw, goal
    )
    return GoalStopMetrics(
        goal_stop_lateral_m=lateral,
        goal_stop_longitudinal_m=longitudinal,
        goal_stop_position_m=position,
        goal_stop_heading_deg=heading_deg,
        goal_stop_speed_mps=speed,
        goal_stop_time_sec=stop_time,
        stop_sample_idx=float(stop_idx),
    )


def resolve_bag_path(
    trace_dir: Path,
    bag_key: str,
    *,
    rosbag_dir: Path | None = None,
    batch_log_rows: list[dict[str, str]] | None = None,
    model_name: str | None = None,
) -> Path | None:
    metadata_path = trace_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            bag_path = Path(str(metadata.get("bag_path", ""))).expanduser()
            if bag_path.is_file() or bag_path.is_dir():
                return bag_path
        except (json.JSONDecodeError, OSError):
            pass

    if batch_log_rows and model_name:
        for row in batch_log_rows:
            if row.get("model_name") != model_name:
                continue
            trace_path = row.get("trace_path", "")
            if trace_path and (trace_path.endswith(bag_key) or bag_key in trace_path):
                candidate = Path(row.get("bag_path", "")).expanduser()
                if candidate.is_file() or candidate.is_dir():
                    return candidate

    if rosbag_dir is not None:
        candidate = (rosbag_dir.expanduser() / bag_key).resolve()
        if candidate.is_file() or candidate.is_dir():
            return candidate

    return None


def pose_to_pose2d(pose) -> Pose2D:
    orientation = pose.orientation
    siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
    cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return Pose2D(pose.position.x, pose.position.y, yaw)


def get_goal_pose_from_bag_fast(bag_path: Path) -> Pose2D:
    """Read the last localization pose in the bag (matches batch route goal)."""
    reader = open_bag_reader(bag_path)
    type_map = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    last_pose = None

    if "/localization/kinematic_state" in type_map:
        msg_type = get_message(type_map["/localization/kinematic_state"])
        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic != "/localization/kinematic_state":
                continue
            odom = deserialize_message(data, msg_type)
            last_pose = odom.pose.pose

    if last_pose is None and "/tf" in type_map:
        reader = open_bag_reader(bag_path)
        msg_type = get_message(type_map["/tf"])
        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic != "/tf":
                continue
            msg = deserialize_message(data, msg_type)
            for transform in msg.transforms:
                if transform.child_frame_id != "base_link":
                    continue
                from geometry_msgs.msg import Point
                from geometry_msgs.msg import Pose
                from geometry_msgs.msg import Quaternion

                trans = transform.transform.translation
                rot = transform.transform.rotation
                last_pose = Pose(
                    position=Point(x=trans.x, y=trans.y, z=trans.z),
                    orientation=Quaternion(x=rot.x, y=rot.y, z=rot.z, w=rot.w),
                )

    if last_pose is None:
        raise ValueError(f"No goal pose found in {bag_path}")

    return pose_to_pose2d(last_pose)


def load_goal_pose_cache(cache_path: Path) -> dict[str, dict[str, float]]:
    if not cache_path.is_file():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_goal_pose_cache(cache_path: Path, cache: dict[str, dict[str, float]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def load_goal_for_bag(
    bag_path: Path,
    min_move_m: float = 0.1,
    cache_path: Path | None = None,
) -> Pose2D:
    key = str(bag_path.expanduser().resolve())
    cache = load_goal_pose_cache(cache_path) if cache_path else {}
    if key in cache:
        entry = cache[key]
        return Pose2D(entry["x"], entry["y"], entry["yaw_rad"])

    try:
        goal = get_goal_pose_from_bag_fast(bag_path)
    except ValueError:
        from route_setup import get_poses_from_bag

        _initial, goal_pose = get_poses_from_bag(bag_path, min_move_m=min_move_m)
        goal = pose_to_pose2d(goal_pose)

    if cache_path is not None:
        try:
            cache[key] = {"x": goal.x, "y": goal.y, "yaw_rad": goal.yaw_rad}
            save_goal_pose_cache(cache_path, cache)
        except OSError:
            pass
    return goal


@lru_cache(maxsize=128)
def load_goal_from_bag(bag_path: str, min_move_m: float) -> Pose2D:
    return load_goal_for_bag(Path(bag_path), min_move_m=min_move_m, cache_path=None)


def analyze_trace_goal_stop(
    trace_dir: Path,
    bag_key: str,
    *,
    rosbag_dir: Path | None = None,
    batch_log_rows: list[dict[str, str]] | None = None,
    model_name: str | None = None,
    speed_threshold: float = 0.2,
    goal_min_move_m: float = 0.1,
    goal_cache_path: Path | None = None,
) -> tuple[GoalStopMetrics | None, str]:
    ego_csv = trace_dir / "ego_pose.csv"
    if not ego_csv.is_file():
        return None, "missing_ego_pose"

    bag_path = resolve_bag_path(
        trace_dir,
        bag_key,
        rosbag_dir=rosbag_dir,
        batch_log_rows=batch_log_rows,
        model_name=model_name,
    )
    if bag_path is None:
        return None, "bag_path_unresolved"

    try:
        goal = load_goal_for_bag(bag_path, min_move_m=goal_min_move_m, cache_path=goal_cache_path)
    except (ValueError, OSError) as exc:
        return None, f"goal_load_failed: {exc}"

    rows = load_ego_pose_rows(ego_csv)
    metrics = compute_goal_stop_metrics(rows, goal, speed_threshold=speed_threshold)
    if metrics is None:
        return None, "empty_ego_pose"
    return metrics, "ok"


def pairwise_stop_shift(
    metrics_a: GoalStopMetrics,
    metrics_b: GoalStopMetrics,
    goal: Pose2D,
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
) -> dict[str, float]:
    """Shift between two models' stop poses (goal-frame lateral/longitudinal and Euclidean)."""
    idx_a = int(metrics_a.stop_sample_idx)
    idx_b = int(metrics_b.stop_sample_idx)
    row_a = rows_a[idx_a]
    row_b = rows_b[idx_b]

    lat_a, lon_a, _, _ = goal_frame_offset(
        float(row_a["x"]), float(row_a["y"]), float(row_a["yaw_rad"]), goal
    )
    lat_b, lon_b, _, _ = goal_frame_offset(
        float(row_b["x"]), float(row_b["y"]), float(row_b["yaw_rad"]), goal
    )
    dx = float(row_b["x"]) - float(row_a["x"])
    dy = float(row_b["y"]) - float(row_a["y"])
    return {
        "stop_shift_lateral_m": lat_b - lat_a,
        "stop_shift_longitudinal_m": lon_b - lon_a,
        "stop_shift_position_m": math.hypot(dx, dy),
        "stop_shift_heading_deg": metrics_b.goal_stop_heading_deg - metrics_a.goal_stop_heading_deg,
    }
