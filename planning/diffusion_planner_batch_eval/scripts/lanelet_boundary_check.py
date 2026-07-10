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

"""Offline lanelet-based checks for ego footprint vs drivable lanes and road borders."""

from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import lanelet2.geometry
import yaml
from autoware_lanelet2_extension_python.projection import MGRSProjector
from autoware_lanelet2_extension_python.utility import query as lanelet_query
from lanelet2.core import BasicPoint2d
from lanelet2.core import BoundingBox2d
from lanelet2.io import Origin
from lanelet2.io import load


@dataclass(frozen=True)
class VehicleFootprintSpec:
    wheel_base: float
    wheel_tread: float
    front_overhang: float
    rear_overhang: float
    left_overhang: float
    right_overhang: float
    footprint_extra_margin: float = 0.0

    def local_polygon(self) -> list[tuple[float, float]]:
        """Match autoware_vehicle_info_utils::VehicleInfo::createFootprint (base_link frame)."""
        margin = self.footprint_extra_margin
        x_front = self.front_overhang + self.wheel_base + margin
        x_center = 0.0
        x_rear = -(self.rear_overhang + margin)

        y_left_front = self.wheel_tread * 0.5 + self.left_overhang + margin
        y_right_front = -(self.wheel_tread * 0.5 + self.right_overhang + margin)
        y_left_center = self.wheel_tread * 0.5 + self.left_overhang + margin
        y_right_center = -(self.wheel_tread * 0.5 + self.right_overhang + margin)
        y_left_rear = self.wheel_tread * 0.5 + self.left_overhang + margin
        y_right_rear = -(self.wheel_tread * 0.5 + self.right_overhang + margin)

        return [
            (x_front, y_left_front),
            (x_front, y_right_front),
            (x_center, y_right_center),
            (x_rear, y_right_rear),
            (x_rear, y_left_rear),
            (x_center, y_left_center),
        ]


@dataclass
class EgoPoseSample:
    stamp_sec: float
    x: float
    y: float
    yaw_rad: float


@dataclass
class BoundaryCheckMetrics:
    lanelet_samples: int
    out_of_lane_frames: int
    boundary_crossing_frames: int
    out_of_lane_ratio: float
    boundary_crossing_ratio: float
    first_out_of_lane_sec: float
    first_boundary_crossing_sec: float

    def as_dict(self) -> dict[str, float]:
        return {
            "lanelet_samples": float(self.lanelet_samples),
            "out_of_lane_frames": float(self.out_of_lane_frames),
            "boundary_crossing_frames": float(self.boundary_crossing_frames),
            "out_of_lane_ratio": self.out_of_lane_ratio,
            "boundary_crossing_ratio": self.boundary_crossing_ratio,
            "first_out_of_lane_sec": self.first_out_of_lane_sec,
            "first_boundary_crossing_sec": self.first_boundary_crossing_sec,
        }


def _linestring_type(linestring) -> str:
    if "type" not in linestring.attributes:
        return ""
    return str(linestring.attributes["type"])


def _ros_param_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    if "/**" in raw and isinstance(raw["/**"], dict):
        return raw["/**"].get("ros__parameters", {})
    return raw.get("ros__parameters", raw)


def load_vehicle_footprint_spec(vehicle_info_yaml: Path, footprint_extra_margin: float = 0.0) -> VehicleFootprintSpec:
    params = _ros_param_yaml(vehicle_info_yaml)
    return VehicleFootprintSpec(
        wheel_base=float(params["wheel_base"]),
        wheel_tread=float(params["wheel_tread"]),
        front_overhang=float(params["front_overhang"]),
        rear_overhang=float(params["rear_overhang"]),
        left_overhang=float(params["left_overhang"]),
        right_overhang=float(params["right_overhang"]),
        footprint_extra_margin=footprint_extra_margin,
    )


def resolve_vehicle_info_yaml(vehicle_model: str) -> Path:
    prefix = subprocess.check_output(
        ["ros2", "pkg", "prefix", f"{vehicle_model}_description"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return Path(prefix) / "share" / f"{vehicle_model}_description" / "config" / "vehicle_info.param.yaml"


def resolve_lanelet_map_path(map_path: Path, lanelet_map_file: str = "lanelet2_map.osm") -> Path:
    map_path = map_path.expanduser()
    if map_path.is_file():
        return map_path
    candidate = map_path / lanelet_map_file
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Lanelet2 map not found under {map_path}")


def create_map_projector(map_path: Path):
    projector_info_path = map_path / "map_projector_info.yaml"
    if not projector_info_path.is_file():
        return MGRSProjector(Origin(0.0, 0.0))

    info = yaml.safe_load(projector_info_path.read_text(encoding="utf-8"))
    projector_type = str(info.get("projector_type", "MGRS"))

    if projector_type == "MGRS":
        return MGRSProjector(Origin(0.0, 0.0))

    if projector_type in ("TransverseMercator", "LocalCartesianUTM"):
        from autoware_lanelet2_extension_python._autoware_lanelet2_extension_python_boost_python_projection import (  # noqa: E501
            TransverseMercatorProjector,
        )

        origin = info["map_origin"]
        return TransverseMercatorProjector(Origin(float(origin["latitude"]), float(origin["longitude"])))

    if projector_type == "Local":
        return MGRSProjector(Origin(0.0, 0.0))

    raise ValueError(f"Unsupported projector_type in {projector_info_path}: {projector_type}")


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


def downsample_poses(samples: list[EgoPoseSample], sample_dt: float) -> list[EgoPoseSample]:
    if not samples or sample_dt <= 0.0:
        return samples

    selected = [samples[0]]
    next_stamp = samples[0].stamp_sec + sample_dt
    for sample in samples[1:]:
        if sample.stamp_sec >= next_stamp:
            selected.append(sample)
            next_stamp = sample.stamp_sec + sample_dt
    if selected[-1] is not samples[-1]:
        selected.append(samples[-1])
    return selected


def _transform_footprint(
    local_polygon: list[tuple[float, float]], x: float, y: float, yaw_rad: float
) -> list[tuple[float, float]]:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    world: list[tuple[float, float]] = []
    for local_x, local_y in local_polygon:
        world.append(
            (
                x + cos_yaw * local_x - sin_yaw * local_y,
                y + sin_yaw * local_x + cos_yaw * local_y,
            )
        )
    return world


def _polygon_bbox(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _segment_bbox(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


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


def _footprint_intersects_segment(
    footprint: list[tuple[float, float]], x1: float, y1: float, x2: float, y2: float
) -> bool:
    for idx in range(len(footprint)):
        p1 = footprint[idx]
        p2 = footprint[(idx + 1) % len(footprint)]
        if _segments_intersect(p1, p2, (x1, y1), (x2, y2)):
            return True
    return False


@dataclass(frozen=True)
class _BoundarySegment:
    x1: float
    y1: float
    x2: float
    y2: float
    boundary_type: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return _segment_bbox(self.x1, self.y1, self.x2, self.y2)


class LaneletBoundaryChecker:
    def __init__(
        self,
        map_path: Path,
        vehicle_spec: VehicleFootprintSpec,
        boundary_types: list[str] | None = None,
        lanelet_map_file: str = "lanelet2_map.osm",
        boundary_search_radius_m: float = 80.0,
    ) -> None:
        self.vehicle_spec = vehicle_spec
        self.local_footprint = vehicle_spec.local_polygon()
        self.boundary_types = set(boundary_types or ["road_border", "curbstone"])
        self.boundary_search_radius_m = boundary_search_radius_m

        map_dir = map_path if map_path.is_dir() else map_path.parent
        lanelet_map_path = resolve_lanelet_map_path(map_path, lanelet_map_file)
        projector = create_map_projector(map_dir)
        self.lanelet_map = load(str(lanelet_map_path), projector)

        self.road_lanelets = lanelet_query.roadLanelets(self.lanelet_map.laneletLayer)
        self.road_lanelet_ids = {lanelet.id for lanelet in self.road_lanelets}
        self.boundary_segments = self._extract_boundary_segments()

    def _extract_boundary_segments(self) -> list[_BoundarySegment]:
        segments: list[_BoundarySegment] = []
        for linestring in self.lanelet_map.lineStringLayer:
            boundary_type = _linestring_type(linestring)
            if boundary_type not in self.boundary_types:
                continue
            points = [(point.x, point.y) for point in linestring]
            for idx in range(len(points) - 1):
                x1, y1 = points[idx]
                x2, y2 = points[idx + 1]
                segments.append(_BoundarySegment(x1, y1, x2, y2, boundary_type))
        return segments

    def _candidate_road_lanelets(self, footprint: list[tuple[float, float]]):
        min_x, min_y, max_x, max_y = _polygon_bbox(footprint)
        min_pt = BasicPoint2d(min_x, min_y)
        max_pt = BasicPoint2d(max_x, max_y)
        search_box = BoundingBox2d(min_pt, max_pt)
        return [
            lanelet
            for lanelet in self.lanelet_map.laneletLayer.search(search_box)
            if lanelet.id in self.road_lanelet_ids
        ]

    def _is_out_of_lane(self, footprint: list[tuple[float, float]], candidate_lanelets) -> bool:
        if not candidate_lanelets:
            return True
        for x, y in footprint:
            point = BasicPoint2d(x, y)
            if not any(lanelet2.geometry.inside(lanelet, point) for lanelet in candidate_lanelets):
                return True
        return False

    def _crosses_boundary(self, footprint: list[tuple[float, float]], ego_x: float, ego_y: float) -> bool:
        footprint_bbox = _polygon_bbox(footprint)
        search_bbox = (
            footprint_bbox[0] - self.boundary_search_radius_m,
            footprint_bbox[1] - self.boundary_search_radius_m,
            footprint_bbox[2] + self.boundary_search_radius_m,
            footprint_bbox[3] + self.boundary_search_radius_m,
        )

        for segment in self.boundary_segments:
            if not _bbox_overlap(segment.bbox, search_bbox):
                continue
            if math.hypot(segment.x1 - ego_x, segment.y1 - ego_y) > self.boundary_search_radius_m and math.hypot(
                segment.x2 - ego_x, segment.y2 - ego_y
            ) > self.boundary_search_radius_m:
                continue
            if _footprint_intersects_segment(footprint, segment.x1, segment.y1, segment.x2, segment.y2):
                return True
        return False

    def analyze_samples(self, samples: list[EgoPoseSample], sample_dt: float = 0.2) -> BoundaryCheckMetrics:
        poses = downsample_poses(samples, sample_dt)
        if not poses:
            return BoundaryCheckMetrics(0, 0, 0, 0.0, 0.0, float("nan"), float("nan"))

        start_stamp = poses[0].stamp_sec
        out_of_lane_frames = 0
        boundary_crossing_frames = 0
        first_out_of_lane_sec = float("nan")
        first_boundary_crossing_sec = float("nan")

        for pose in poses:
            footprint = _transform_footprint(self.local_footprint, pose.x, pose.y, pose.yaw_rad)
            candidate_lanelets = self._candidate_road_lanelets(footprint)

            if self._is_out_of_lane(footprint, candidate_lanelets):
                out_of_lane_frames += 1
                if math.isnan(first_out_of_lane_sec):
                    first_out_of_lane_sec = pose.stamp_sec - start_stamp

            if self._crosses_boundary(footprint, pose.x, pose.y):
                boundary_crossing_frames += 1
                if math.isnan(first_boundary_crossing_sec):
                    first_boundary_crossing_sec = pose.stamp_sec - start_stamp

        count = len(poses)
        return BoundaryCheckMetrics(
            lanelet_samples=count,
            out_of_lane_frames=out_of_lane_frames,
            boundary_crossing_frames=boundary_crossing_frames,
            out_of_lane_ratio=out_of_lane_frames / count,
            boundary_crossing_ratio=boundary_crossing_frames / count,
            first_out_of_lane_sec=first_out_of_lane_sec,
            first_boundary_crossing_sec=first_boundary_crossing_sec,
        )

    def analyze_ego_pose_csv(self, ego_pose_csv: Path, sample_dt: float = 0.2) -> BoundaryCheckMetrics:
        return self.analyze_samples(load_ego_pose_samples(ego_pose_csv), sample_dt=sample_dt)
