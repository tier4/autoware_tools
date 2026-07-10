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

"""Longitudinal/lateral jerk and harsh deceleration metrics from ego_pose.csv."""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ComfortMetrics:
    comfort_samples: int
    motion_duration_sec: float
    max_longitudinal_accel_mps2: float
    min_longitudinal_accel_mps2: float
    max_lateral_accel_mps2: float
    min_lateral_accel_mps2: float
    max_longitudinal_jerk_mps3: float
    max_lateral_jerk_mps3: float
    rms_longitudinal_jerk_mps3: float
    rms_lateral_jerk_mps3: float
    p95_longitudinal_jerk_mps3: float
    p95_lateral_jerk_mps3: float
    harsh_decel_count: int
    harsh_decel_time_sec: float
    harsh_decel_ratio: float
    max_decel_mps2: float
    plan_max_longitudinal_accel_mps2: float
    plan_max_longitudinal_jerk_mps3: float

    def as_dict(self) -> dict[str, float]:
        return {
            "comfort_samples": float(self.comfort_samples),
            "motion_duration_sec": self.motion_duration_sec,
            "max_longitudinal_accel_mps2": self.max_longitudinal_accel_mps2,
            "min_longitudinal_accel_mps2": self.min_longitudinal_accel_mps2,
            "max_lateral_accel_mps2": self.max_lateral_accel_mps2,
            "min_lateral_accel_mps2": self.min_lateral_accel_mps2,
            "max_longitudinal_jerk_mps3": self.max_longitudinal_jerk_mps3,
            "max_lateral_jerk_mps3": self.max_lateral_jerk_mps3,
            "rms_longitudinal_jerk_mps3": self.rms_longitudinal_jerk_mps3,
            "rms_lateral_jerk_mps3": self.rms_lateral_jerk_mps3,
            "p95_longitudinal_jerk_mps3": self.p95_longitudinal_jerk_mps3,
            "p95_lateral_jerk_mps3": self.p95_lateral_jerk_mps3,
            "harsh_decel_count": float(self.harsh_decel_count),
            "harsh_decel_time_sec": self.harsh_decel_time_sec,
            "harsh_decel_ratio": self.harsh_decel_ratio,
            "max_decel_mps2": self.max_decel_mps2,
            "plan_max_longitudinal_accel_mps2": self.plan_max_longitudinal_accel_mps2,
            "plan_max_longitudinal_jerk_mps3": self.plan_max_longitudinal_jerk_mps3,
        }


@dataclass
class ComfortAnalysisConfig:
    sample_dt: float
    min_derivative_dt: float
    max_derivative_dt: float
    harsh_decel_threshold_mps2: float
    harsh_decel_min_duration_sec: float


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_ego_pose_rows(ego_pose_csv: Path) -> list[dict[str, str]]:
    return load_csv_rows(ego_pose_csv)


def nearest_value(
    stamps: list[float], values: list[float], query_stamp: float
) -> float | None:
    if not stamps:
        return None
    idx = min(range(len(stamps)), key=lambda i: abs(stamps[i] - query_stamp))
    return values[idx]


def derive_jerk(
    stamps: list[float],
    values: list[float],
    config: ComfortAnalysisConfig,
) -> tuple[list[float], list[float]]:
    jerk_stamps: list[float] = []
    jerk_values: list[float] = []
    for i in range(1, len(values)):
        dt = stamps[i] - stamps[i - 1]
        if dt < config.min_derivative_dt or dt > config.max_derivative_dt:
            continue
        jerk_stamps.append(stamps[i])
        jerk_values.append((values[i] - values[i - 1]) / dt)
    return jerk_stamps, jerk_values


def harsh_decel_spans(
    stamps: list[float],
    a_long: list[float],
    threshold_mps2: float,
    min_duration_sec: float,
    time_origin: float = 0.0,
) -> list[tuple[float, float]]:
    if len(stamps) != len(a_long) or not a_long:
        return []

    spans: list[tuple[float, float]] = []
    in_event = False
    event_start = stamps[0]

    for idx, accel in enumerate(a_long):
        stamp = stamps[idx]
        below = accel < threshold_mps2
        if below and not in_event:
            in_event = True
            event_start = stamp
        elif not below and in_event:
            if stamp - event_start >= min_duration_sec:
                spans.append((event_start - time_origin, stamp - time_origin))
            in_event = False

    if in_event and stamps[-1] - event_start >= min_duration_sec:
        spans.append((event_start - time_origin, stamps[-1] - time_origin))

    return spans


def compute_ego_motion_signals(
    ego_pose_rows: list[dict[str, str]],
    config: ComfortAnalysisConfig,
    ego_accel_rows: list[dict[str, str]] | None = None,
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]] | None:
    pose_sampled = downsample_rows(ego_pose_rows, config.sample_dt)
    if len(pose_sampled) < 3:
        return None

    pose_stamps = [float(row["stamp_sec"]) for row in pose_sampled]
    v_long = [
        ego_frame_velocity(float(row["vx"]), float(row["vy"]), float(row["yaw_rad"]))[0]
        for row in pose_sampled
    ]

    if ego_accel_rows:
        accel_sampled = downsample_rows(ego_accel_rows, config.sample_dt)
        if len(accel_sampled) < 2:
            return None
        stamps = [float(row["stamp_sec"]) for row in accel_sampled]
        a_long = [float(row["a_long_mps2"]) for row in accel_sampled]
        a_lat = [float(row["a_lat_mps2"]) for row in accel_sampled]
        v_long = [
            nearest_value(pose_stamps, v_long, stamp) or 0.0 for stamp in stamps
        ]
    else:
        stamps = pose_stamps
        a_long = []
        a_lat = []
        for i in range(1, len(stamps)):
            dt = stamps[i] - stamps[i - 1]
            if dt < config.min_derivative_dt or dt > config.max_derivative_dt:
                continue
            long_v0, lat_v0 = ego_frame_velocity(
                float(pose_sampled[i - 1]["vx"]),
                float(pose_sampled[i - 1]["vy"]),
                float(pose_sampled[i - 1]["yaw_rad"]),
            )
            long_v1, lat_v1 = ego_frame_velocity(
                float(pose_sampled[i]["vx"]),
                float(pose_sampled[i]["vy"]),
                float(pose_sampled[i]["yaw_rad"]),
            )
            a_long.append((long_v1 - long_v0) / dt)
            a_lat.append((lat_v1 - lat_v0) / dt)
        if not a_long:
            return None
        stamps = stamps[1 : 1 + len(a_long)]
        v_long = v_long[1 : 1 + len(a_long)]

    jerk_stamps, j_long = derive_jerk(stamps, a_long, config)
    _, j_lat = derive_jerk(stamps, a_lat, config)
    return stamps, v_long, a_long, a_lat, jerk_stamps, j_long


@dataclass
class ComfortTraceSeries:
    model_label: str
    time_sec: list[float]
    speed_mps: list[float]
    a_long_mps2: list[float]
    a_lat_mps2: list[float]
    jerk_time_sec: list[float]
    jerk_long_mps3: list[float]
    harsh_spans: list[tuple[float, float]]
    control_time_sec: list[float]
    control_velocity_mps: list[float]
    control_accel_mps2: list[float]
    control_jerk_mps3: list[float]
    plan_time_sec: list[float]
    plan_accel_mps2: list[float]
    plan_velocity_mps: list[float]
    uses_measured_accel: bool


def load_plan_profile(planned_trajectory_csv: Path) -> tuple[list[float], list[float], list[float]]:
    rows = load_csv_rows(planned_trajectory_csv)
    if not rows:
        return [], [], []

    by_stamp: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        by_stamp.setdefault(float(row["stamp_sec"]), []).append(row)

    times: list[float] = []
    velocities: list[float] = []
    accels: list[float] = []
    for stamp in sorted(by_stamp):
        points = sorted(by_stamp[stamp], key=lambda row: int(row["point_idx"]))
        if not points:
            continue
        times.append(stamp)
        velocities.append(float(points[0]["longitudinal_velocity_mps"]))
        accels.append(float(points[0]["acceleration_mps2"]))
    return times, velocities, accels


def load_trace_comfort_series(
    trace_dir: Path,
    model_label: str,
    config: ComfortAnalysisConfig,
) -> ComfortTraceSeries | None:
    ego_pose_rows = load_ego_pose_rows(trace_dir / "ego_pose.csv")
    if not ego_pose_rows:
        return None

    ego_accel_rows = load_csv_rows(trace_dir / "ego_accel.csv")
    signals = compute_ego_motion_signals(
        ego_pose_rows,
        config,
        ego_accel_rows=ego_accel_rows if ego_accel_rows else None,
    )
    if signals is None:
        return None

    stamps, v_long, a_long, a_lat, jerk_stamps, j_long = signals
    t0 = stamps[0]
    rel_time = [stamp - t0 for stamp in stamps]
    rel_jerk_time = [stamp - t0 for stamp in jerk_stamps]

    control_rows = load_csv_rows(trace_dir / "control_cmd.csv")
    control_time = [float(row["stamp_sec"]) - t0 for row in control_rows]
    control_velocity = [float(row["cmd_velocity_mps"]) for row in control_rows]
    control_accel = [float(row["cmd_acceleration_mps2"]) for row in control_rows]
    control_jerk = [float(row["cmd_jerk_mps3"]) for row in control_rows]

    plan_times, plan_vel, plan_acc = load_plan_profile(trace_dir / "planned_trajectory.csv")
    plan_time = [stamp - t0 for stamp in plan_times]

    return ComfortTraceSeries(
        model_label=model_label,
        time_sec=rel_time,
        speed_mps=v_long,
        a_long_mps2=a_long,
        a_lat_mps2=a_lat,
        jerk_time_sec=rel_jerk_time,
        jerk_long_mps3=j_long,
        harsh_spans=harsh_decel_spans(
            stamps,
            a_long,
            config.harsh_decel_threshold_mps2,
            config.harsh_decel_min_duration_sec,
            time_origin=t0,
        ),
        control_time_sec=control_time,
        control_velocity_mps=control_velocity,
        control_accel_mps2=control_accel,
        control_jerk_mps3=control_jerk,
        plan_time_sec=plan_time,
        plan_accel_mps2=plan_acc,
        plan_velocity_mps=plan_vel,
        uses_measured_accel=bool(ego_accel_rows),
    )


def downsample_rows(rows: list[dict[str, str]], sample_dt: float) -> list[dict[str, str]]:
    if not rows or sample_dt <= 0.0:
        return rows

    selected = [rows[0]]
    next_stamp = float(rows[0]["stamp_sec"]) + sample_dt
    for row in rows[1:]:
        stamp = float(row["stamp_sec"])
        if stamp >= next_stamp:
            selected.append(row)
            next_stamp = stamp + sample_dt
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    return selected


def ego_frame_velocity(vx: float, vy: float, yaw_rad: float) -> tuple[float, float]:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    v_long = vx * cos_yaw + vy * sin_yaw
    v_lat = -vx * sin_yaw + vy * cos_yaw
    return v_long, v_lat


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def count_harsh_decel_events(
    accel_stamps: list[float],
    a_long: list[float],
    threshold_mps2: float,
    min_duration_sec: float,
) -> tuple[int, float]:
    """Count decel events where longitudinal accel stays below threshold long enough."""
    if len(accel_stamps) != len(a_long) or not a_long:
        return 0, 0.0

    event_count = 0
    harsh_time = 0.0
    in_event = False
    event_start = accel_stamps[0]

    for idx, accel in enumerate(a_long):
        stamp = accel_stamps[idx]
        below = accel < threshold_mps2

        if below and not in_event:
            in_event = True
            event_start = stamp
        elif not below and in_event:
            duration = stamp - event_start
            if duration >= min_duration_sec:
                event_count += 1
                harsh_time += duration
            in_event = False

    if in_event:
        duration = accel_stamps[-1] - event_start
        if duration >= min_duration_sec:
            event_count += 1
            harsh_time += duration

    return event_count, harsh_time


def compute_plan_comfort_metrics(planned_trajectory_csv: Path) -> tuple[float, float]:
    rows = load_csv_rows(planned_trajectory_csv)
    if not rows:
        return float("nan"), float("nan")

    by_stamp: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        stamp = float(row["stamp_sec"])
        by_stamp.setdefault(stamp, []).append(row)

    max_accel = float("-inf")
    max_jerk = float("-inf")
    prev_stamp: float | None = None
    prev_accel: float | None = None

    for stamp in sorted(by_stamp):
        points = sorted(by_stamp[stamp], key=lambda row: int(row["point_idx"]))
        if not points:
            continue
        accel_values = [float(point["acceleration_mps2"]) for point in points]
        traj_accel = max(accel_values, key=abs)
        max_accel = max(max_accel, traj_accel)

        if prev_stamp is not None and prev_accel is not None:
            dt = stamp - prev_stamp
            if dt > 0.0:
                jerk = (traj_accel - prev_accel) / dt
                max_jerk = max(max_jerk, abs(jerk))

        prev_stamp = stamp
        prev_accel = traj_accel

    if max_accel == float("-inf"):
        max_accel = float("nan")
    if max_jerk == float("-inf"):
        max_jerk = float("nan")
    return max_accel, max_jerk


def compute_comfort_metrics(
    rows: list[dict[str, str]],
    config: ComfortAnalysisConfig,
    planned_trajectory_csv: Path | None = None,
    ego_accel_rows: list[dict[str, str]] | None = None,
) -> ComfortMetrics | None:
    signals = compute_ego_motion_signals(rows, config, ego_accel_rows=ego_accel_rows)
    if signals is None:
        return None

    stamps, v_long, a_long, a_lat, jerk_stamps, j_long = signals
    _, j_lat = derive_jerk(stamps, a_lat, config)

    motion_duration = stamps[-1] - stamps[0]
    harsh_count, harsh_time = count_harsh_decel_events(
        stamps,
        a_long,
        config.harsh_decel_threshold_mps2,
        config.harsh_decel_min_duration_sec,
    )

    min_long_accel = min(a_long)
    plan_max_accel, plan_max_jerk = (
        compute_plan_comfort_metrics(planned_trajectory_csv)
        if planned_trajectory_csv is not None and planned_trajectory_csv.is_file()
        else (float("nan"), float("nan"))
    )

    return ComfortMetrics(
        comfort_samples=len(stamps),
        motion_duration_sec=motion_duration,
        max_longitudinal_accel_mps2=max(a_long),
        min_longitudinal_accel_mps2=min_long_accel,
        max_lateral_accel_mps2=max(a_lat),
        min_lateral_accel_mps2=min(a_lat),
        max_longitudinal_jerk_mps3=max((abs(value) for value in j_long), default=float("nan")),
        max_lateral_jerk_mps3=max((abs(value) for value in j_lat), default=float("nan")),
        rms_longitudinal_jerk_mps3=(
            math.sqrt(statistics.fmean(value * value for value in j_long)) if j_long else float("nan")
        ),
        rms_lateral_jerk_mps3=(
            math.sqrt(statistics.fmean(value * value for value in j_lat)) if j_lat else float("nan")
        ),
        p95_longitudinal_jerk_mps3=percentile([abs(value) for value in j_long], 95.0),
        p95_lateral_jerk_mps3=percentile([abs(value) for value in j_lat], 95.0),
        harsh_decel_count=harsh_count,
        harsh_decel_time_sec=harsh_time,
        harsh_decel_ratio=(harsh_time / motion_duration if motion_duration > 0.0 else 0.0),
        max_decel_mps2=abs(min_long_accel) if min_long_accel < 0.0 else 0.0,
        plan_max_longitudinal_accel_mps2=plan_max_accel,
        plan_max_longitudinal_jerk_mps3=plan_max_jerk,
    )


def analyze_trace_comfort(
    trace_dir: Path,
    config: ComfortAnalysisConfig,
) -> tuple[ComfortMetrics | None, str]:
    ego_csv = trace_dir / "ego_pose.csv"
    if not ego_csv.is_file():
        return None, "ego_pose_missing"

    rows = load_ego_pose_rows(ego_csv)
    if not rows:
        return None, "ego_pose_empty"

    required = {"stamp_sec", "vx", "vy", "yaw_rad"}
    if not required.issubset(rows[0].keys()):
        return None, "ego_pose_missing_velocity"

    ego_accel_rows = load_csv_rows(trace_dir / "ego_accel.csv")
    metrics = compute_comfort_metrics(
        rows,
        config,
        planned_trajectory_csv=trace_dir / "planned_trajectory.csv",
        ego_accel_rows=ego_accel_rows if ego_accel_rows else None,
    )
    if metrics is None:
        return None, "insufficient_samples"
    return metrics, "ok"


def pairwise_comfort_diff(
    metrics_a: ComfortMetrics,
    metrics_b: ComfortMetrics,
) -> dict[str, float]:
    return {
        "rms_longitudinal_jerk_diff_mps3": (
            metrics_b.rms_longitudinal_jerk_mps3 - metrics_a.rms_longitudinal_jerk_mps3
        ),
        "rms_lateral_jerk_diff_mps3": (
            metrics_b.rms_lateral_jerk_mps3 - metrics_a.rms_lateral_jerk_mps3
        ),
        "harsh_decel_count_diff": float(metrics_b.harsh_decel_count - metrics_a.harsh_decel_count),
        "harsh_decel_ratio_diff": metrics_b.harsh_decel_ratio - metrics_a.harsh_decel_ratio,
        "max_decel_diff_mps2": metrics_b.max_decel_mps2 - metrics_a.max_decel_mps2,
    }
