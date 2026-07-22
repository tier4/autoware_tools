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

"""Load Hiratsuka map bus stop points and match rosbag goals to on-lane poses."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion


@dataclass(frozen=True)
class MapStopPoint:
    name: str
    x: float
    y: float
    z: float
    yaw: float
    lane_id: str
    stop_point_id: str


def load_stop_points(csv_path: Path) -> list[MapStopPoint]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Stop points file not found: {csv_path}")

    stops: list[MapStopPoint] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"name", "x", "y", "z", "yaw", "lane_id", "stop_point_id"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Invalid stop points CSV {csv_path}: expected columns {sorted(required)}"
            )
        for row in reader:
            name = row["name"].strip()
            if not name:
                continue
            stops.append(
                MapStopPoint(
                    name=name,
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    yaw=float(row["yaw"]),
                    lane_id=str(row["lane_id"]),
                    stop_point_id=str(row["stop_point_id"]),
                )
            )

    if not stops:
        raise ValueError(f"No stop points loaded from {csv_path}")
    return stops


def yaw_to_quaternion(yaw: float) -> Quaternion:
    quaternion = Quaternion()
    quaternion.z = math.sin(yaw / 2.0)
    quaternion.w = math.cos(yaw / 2.0)
    return quaternion


def stop_point_to_pose(stop: MapStopPoint) -> Pose:
    pose = Pose()
    pose.position.x = stop.x
    pose.position.y = stop.y
    pose.position.z = stop.z
    pose.orientation = yaw_to_quaternion(stop.yaw)
    return pose


def find_nearest_stop_point(
    pose: Pose, stops: list[MapStopPoint]
) -> tuple[MapStopPoint, float]:
    best_stop: MapStopPoint | None = None
    best_dist = float("inf")
    for stop in stops:
        dist = math.hypot(pose.position.x - stop.x, pose.position.y - stop.y)
        if dist < best_dist:
            best_dist = dist
            best_stop = stop
    if best_stop is None:
        raise ValueError("No stop points provided")
    return best_stop, best_dist


def resolve_stop_points_path(map_path: Path, filename: str = "stop_points.csv") -> Path:
    return map_path / filename


# Hiratsuka loop (往路 / 復路 naming in current stop_points.csv)
DEFAULT_HIRATSUKA_ROUTE_ORDER = [
    "平塚駅南口",
    "商工会議所前（往路）",
    "教会前（往路）",
    "松風公園入口",
    "花水小学校前",
    "すみれ平",
    "松風町",
    "八間通り入口",
    "袖ヶ浜",
    "湘南海岸公園前",
    "なぎさプロムナード",
    "教会前（復路）",
    "商工会議所前（復路）",
]


def stops_by_name(stops: list[MapStopPoint]) -> dict[str, MapStopPoint]:
    return {stop.name: stop for stop in stops}


def route_segment_stop_names(
    start_name: str, goal_name: str, route_order: list[str]
) -> list[str]:
    """Intermediate stop names along the shorter direction on the loop (exclusive of start/goal).

    Duplicate terminal names (e.g. loop ending at the same stop) are collapsed so
    index lookup uses the first occurrence of start and the nearest later goal.
    """
    if start_name == goal_name:
        return []

    # Collapse consecutive duplicate names while preserving order for routing.
    compact: list[str] = []
    for name in route_order:
        if not compact or compact[-1] != name:
            compact.append(name)

    if start_name not in compact or goal_name not in compact:
        return []

    count = len(compact)
    start_index = compact.index(start_name)
    # Prefer the next occurrence of goal after start (forward along the list).
    goal_index = None
    for offset in range(1, count):
        idx = (start_index + offset) % count
        if compact[idx] == goal_name:
            goal_index = idx
            break
    if goal_index is None:
        goal_index = compact.index(goal_name)

    forward: list[str] = []
    index = (start_index + 1) % count
    while index != goal_index:
        forward.append(compact[index])
        index = (index + 1) % count

    backward: list[str] = []
    index = (start_index - 1) % count
    while index != goal_index:
        backward.append(compact[index])
        index = (index - 1) % count

    return forward if len(forward) <= len(backward) else backward


def resolve_routing_goal(
    bag_goal_pose: Pose,
    stops: list[MapStopPoint],
    snap_threshold_m: float,
) -> tuple[Pose, MapStopPoint, float, bool]:
    """Return goal pose for routing; snap to nearest map stop when bag goal is close."""
    nearest_stop, nearest_dist = find_nearest_stop_point(bag_goal_pose, stops)
    if nearest_dist <= snap_threshold_m:
        return stop_point_to_pose(nearest_stop), nearest_stop, nearest_dist, True
    return bag_goal_pose, nearest_stop, nearest_dist, False
