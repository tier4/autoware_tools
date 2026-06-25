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

"""Offline ego vs NPC collision / near-miss checks from batch-eval CSV logs."""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from lanelet_boundary_check import EgoPoseSample
from lanelet_boundary_check import VehicleFootprintSpec
from lanelet_boundary_check import downsample_poses
from lanelet_boundary_check import load_vehicle_footprint_spec
from lanelet_boundary_check import resolve_vehicle_info_yaml


DEFAULT_NPC_LABELS = frozenset(
    {"CAR", "TRUCK", "BUS", "TRAILER", "MOTORCYCLE", "BICYCLE"}
)


@dataclass(frozen=True)
class NpcBox:
    object_id: str
    label: str
    x: float
    y: float
    yaw_rad: float
    length_m: float
    width_m: float

    def world_polygon(self) -> list[tuple[float, float]]:
        half_length = self.length_m * 0.5
        half_width = self.width_m * 0.5
        local = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
        ]
        cos_yaw = math.cos(self.yaw_rad)
        sin_yaw = math.sin(self.yaw_rad)
        return [
            (
                self.x + cos_yaw * lx - sin_yaw * ly,
                self.y + sin_yaw * lx + cos_yaw * ly,
            )
            for lx, ly in local
        ]


@dataclass
class NpcCollisionMetrics:
    npc_check_samples: int
    npc_overlap_frames: int
    npc_collision_rate: float
    npc_near_miss_frames: int
    npc_near_miss_rate: float
    npc_min_distance_m: float
    first_npc_overlap_sec: float
    had_npc_collision: float

    def as_dict(self) -> dict[str, float]:
        return {
            "npc_check_samples": float(self.npc_check_samples),
            "npc_overlap_frames": float(self.npc_overlap_frames),
            "npc_collision_rate": self.npc_collision_rate,
            "npc_near_miss_frames": float(self.npc_near_miss_frames),
            "npc_near_miss_rate": self.npc_near_miss_rate,
            "npc_min_distance_m": self.npc_min_distance_m,
            "first_npc_overlap_sec": self.first_npc_overlap_sec,
            "had_npc_collision": self.had_npc_collision,
        }


def _orientation(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c) -> bool:
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def _segments_intersect(p1, p2, q1, q2) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0):
        return True

    if abs(o1) < 1e-9 and _on_segment(p1, q1, p2):
        return True
    if abs(o2) < 1e-9 and _on_segment(p1, q2, p2):
        return True
    if abs(o3) < 1e-9 and _on_segment(q1, p1, q2):
        return True
    if abs(o4) < 1e-9 and _on_segment(q1, p2, q2):
        return True
    return False


def _point_in_convex_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    sign: bool | None = None
    for idx in range(len(polygon)):
        cross = _orientation(polygon[idx], polygon[(idx + 1) % len(polygon)], point)
        if abs(cross) < 1e-9:
            continue
        current = cross > 0.0
        if sign is None:
            sign = current
        elif current != sign:
            return False
    return sign is not None


def polygons_intersect(
    poly_a: list[tuple[float, float]], poly_b: list[tuple[float, float]]
) -> bool:
    if any(_point_in_convex_polygon(point, poly_b) for point in poly_a):
        return True
    if any(_point_in_convex_polygon(point, poly_a) for point in poly_b):
        return True
    for idx_a in range(len(poly_a)):
        a1 = poly_a[idx_a]
        a2 = poly_a[(idx_a + 1) % len(poly_a)]
        for idx_b in range(len(poly_b)):
            b1 = poly_b[idx_b]
            b2 = poly_b[(idx_b + 1) % len(poly_b)]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _point_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def polygon_min_distance(
    poly_a: list[tuple[float, float]], poly_b: list[tuple[float, float]]
) -> float:
    if polygons_intersect(poly_a, poly_b):
        return 0.0

    min_dist = float("inf")
    for ax, ay in poly_a:
        for idx in range(len(poly_b)):
            x1, y1 = poly_b[idx]
            x2, y2 = poly_b[(idx + 1) % len(poly_b)]
            min_dist = min(min_dist, _point_segment_distance(ax, ay, x1, y1, x2, y2))
    for bx, by in poly_b:
        for idx in range(len(poly_a)):
            x1, y1 = poly_a[idx]
            x2, y2 = poly_a[(idx + 1) % len(poly_a)]
            min_dist = min(min_dist, _point_segment_distance(bx, by, x1, y1, x2, y2))
    return min_dist


def _transform_footprint(
    local_polygon: list[tuple[float, float]], x: float, y: float, yaw_rad: float
) -> list[tuple[float, float]]:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    return [
        (x + cos_yaw * lx - sin_yaw * ly, y + sin_yaw * lx + cos_yaw * ly)
        for lx, ly in local_polygon
    ]


def load_ego_pose_samples(ego_pose_csv: Path) -> list[EgoPoseSample]:
    with ego_pose_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [
        EgoPoseSample(
            stamp_sec=float(row["stamp_sec"]),
            x=float(row["x"]),
            y=float(row["y"]),
            yaw_rad=float(row["yaw_rad"]),
        )
        for row in rows
    ]


def load_object_frames(
    objects_csv: Path, labels: set[str] | None
) -> tuple[list[float], list[list[NpcBox]]]:
    grouped: dict[float, list[NpcBox]] = {}
    with objects_csv.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            label = row.get("label", "UNKNOWN")
            if labels is not None and label not in labels:
                continue
            length_m = float(row["length_m"])
            width_m = float(row["width_m"])
            if length_m <= 0.0 or width_m <= 0.0:
                continue
            stamp = float(row["stamp_sec"])
            grouped.setdefault(stamp, []).append(
                NpcBox(
                    object_id=row.get("object_id", ""),
                    label=label,
                    x=float(row["x"]),
                    y=float(row["y"]),
                    yaw_rad=float(row["yaw_rad"]),
                    length_m=length_m,
                    width_m=width_m,
                )
            )

    stamps = sorted(grouped)
    return stamps, [grouped[stamp] for stamp in stamps]


def nearest_frame_index(stamps: list[float], stamp: float, max_dt: float) -> int | None:
    if not stamps:
        return None
    idx = bisect.bisect_left(stamps, stamp)
    candidates: list[int] = []
    if idx < len(stamps):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best = min(candidates, key=lambda i: abs(stamps[i] - stamp))
    if abs(stamps[best] - stamp) > max_dt:
        return None
    return best


class NpcCollisionChecker:
    def __init__(
        self,
        vehicle_spec: VehicleFootprintSpec,
        near_miss_threshold_m: float = 0.5,
        align_max_dt: float = 0.15,
        labels: set[str] | None = None,
    ) -> None:
        self.local_footprint = vehicle_spec.local_polygon()
        self.near_miss_threshold_m = near_miss_threshold_m
        self.align_max_dt = align_max_dt
        self.labels = set(labels) if labels is not None else set(DEFAULT_NPC_LABELS)

    def analyze_samples(
        self,
        ego_samples: list[EgoPoseSample],
        object_stamps: list[float],
        object_frames: list[list[NpcBox]],
        sample_dt: float = 0.2,
    ) -> NpcCollisionMetrics:
        poses = downsample_poses(ego_samples, sample_dt)
        if not poses or not object_stamps:
            return NpcCollisionMetrics(
                0, 0, 0.0, 0, 0.0, float("nan"), float("nan"), 0.0
            )

        start_stamp = poses[0].stamp_sec
        overlap_frames = 0
        near_miss_frames = 0
        min_distance = float("inf")
        first_overlap_sec = float("nan")
        checked = 0

        for pose in poses:
            frame_idx = nearest_frame_index(object_stamps, pose.stamp_sec, self.align_max_dt)
            if frame_idx is None:
                continue

            ego_poly = _transform_footprint(
                self.local_footprint, pose.x, pose.y, pose.yaw_rad
            )
            frame_min = float("inf")
            frame_overlap = False

            for npc in object_frames[frame_idx]:
                npc_poly = npc.world_polygon()
                dist = polygon_min_distance(ego_poly, npc_poly)
                frame_min = min(frame_min, dist)
                if dist <= 0.0:
                    frame_overlap = True

            checked += 1
            min_distance = min(min_distance, frame_min)

            if frame_overlap:
                overlap_frames += 1
                if math.isnan(first_overlap_sec):
                    first_overlap_sec = pose.stamp_sec - start_stamp
            elif frame_min <= self.near_miss_threshold_m:
                near_miss_frames += 1

        if checked == 0:
            return NpcCollisionMetrics(
                0, 0, 0.0, 0, 0.0, float("nan"), float("nan"), 0.0
            )

        return NpcCollisionMetrics(
            npc_check_samples=checked,
            npc_overlap_frames=overlap_frames,
            npc_collision_rate=overlap_frames / checked,
            npc_near_miss_frames=near_miss_frames,
            npc_near_miss_rate=near_miss_frames / checked,
            npc_min_distance_m=min_distance,
            first_npc_overlap_sec=first_overlap_sec,
            had_npc_collision=1.0 if overlap_frames > 0 else 0.0,
        )

    def analyze_trace_dir(self, trace_dir: Path, sample_dt: float = 0.2) -> tuple[NpcCollisionMetrics | None, str]:
        ego_csv = trace_dir / "ego_pose.csv"
        objects_csv = trace_dir / "objects.csv"
        if not ego_csv.is_file():
            return None, "missing_ego_pose"
        if not objects_csv.is_file():
            return None, "missing_objects"

        ego_samples = load_ego_pose_samples(ego_csv)
        object_stamps, object_frames = load_object_frames(objects_csv, self.labels)
        if not object_stamps:
            return None, "no_npc_objects"

        metrics = self.analyze_samples(ego_samples, object_stamps, object_frames, sample_dt)
        if metrics.npc_check_samples == 0:
            return None, "no_time_aligned_samples"
        return metrics, "ok"
