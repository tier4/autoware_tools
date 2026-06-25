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

"""Quantitative comparison of diffusion planner batch-eval results across models."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class EgoSeries:
    stamps: list[float]
    xs: list[float]
    ys: list[float]
    speeds: list[float]

    @property
    def duration_sec(self) -> float:
        if len(self.stamps) < 2:
            return 0.0
        return self.stamps[-1] - self.stamps[0]

    @property
    def path_length_m(self) -> float:
        total = 0.0
        for i in range(1, len(self.xs)):
            total += math.hypot(self.xs[i] - self.xs[i - 1], self.ys[i] - self.ys[i - 1])
        return total

    @property
    def mean_speed_mps(self) -> float:
        return statistics.fmean(self.speeds) if self.speeds else 0.0

    @property
    def max_speed_mps(self) -> float:
        return max(self.speeds) if self.speeds else 0.0

    def xy_path(self) -> list[tuple[float, float]]:
        return list(zip(self.xs, self.ys))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def discover_models(output_dir: Path) -> list[str]:
    models = []
    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and child.name.startswith("."):
            continue
        if child.name in ("comparisons", "quantitative_analysis"):
            continue
        if any(child.rglob("ego_pose.csv")):
            models.append(child.name)
    return models


def discover_bag_keys(output_dir: Path, model: str) -> list[str]:
    model_dir = output_dir / model
    keys: set[str] = set()
    for ego_csv in model_dir.rglob("ego_pose.csv"):
        keys.add(str(ego_csv.parent.relative_to(model_dir)))
    return sorted(keys)


def common_bag_keys(output_dir: Path, models: list[str]) -> list[str]:
    if not models:
        return []
    shared = set(discover_bag_keys(output_dir, models[0]))
    for model in models[1:]:
        shared &= set(discover_bag_keys(output_dir, model))
    return sorted(shared)


def load_ego_series(trace_dir: Path) -> EgoSeries | None:
    rows = read_csv_rows(trace_dir / "ego_pose.csv")
    if not rows:
        return None
    return EgoSeries(
        stamps=[float(row["stamp_sec"]) for row in rows],
        xs=[float(row["x"]) for row in rows],
        ys=[float(row["y"]) for row in rows],
        speeds=[float(row["speed_mps"]) for row in rows],
    )


def load_plan_stats(trace_dir: Path) -> dict[str, float]:
    rows = read_csv_rows(trace_dir / "planned_trajectory.csv")
    if not rows:
        return {"plan_messages": 0.0, "mean_plan_points": 0.0}
    stamps = sorted({float(row["stamp_sec"]) for row in rows})
    point_counts = [sum(1 for row in rows if float(row["stamp_sec"]) == stamp) for stamp in stamps]
    return {
        "plan_messages": float(len(stamps)),
        "mean_plan_points": statistics.fmean(point_counts) if point_counts else 0.0,
    }


def directed_hausdorff(path_a: list[tuple[float, float]], path_b: list[tuple[float, float]]) -> float:
    if not path_a or not path_b:
        return float("nan")
    return max(min(math.hypot(ax - bx, ay - by) for bx, by in path_b) for ax, ay in path_a)


def symmetric_hausdorff(
    path_a: list[tuple[float, float]], path_b: list[tuple[float, float]]
) -> float:
    if not path_a or not path_b:
        return float("nan")
    return max(directed_hausdorff(path_a, path_b), directed_hausdorff(path_b, path_a))


def time_aligned_separation(
    series_a: EgoSeries, series_b: EgoSeries, max_dt: float = 0.15
) -> dict[str, float]:
    if not series_a.stamps or not series_b.stamps:
        return {
            "mean_separation_m": float("nan"),
            "max_separation_m": float("nan"),
            "fde_m": float("nan"),
            "aligned_samples": 0.0,
        }

    separations: list[float] = []
    for ta, xa, ya in zip(series_a.stamps, series_a.xs, series_a.ys):
        best_idx = min(range(len(series_b.stamps)), key=lambda i: abs(series_b.stamps[i] - ta))
        if abs(series_b.stamps[best_idx] - ta) > max_dt:
            continue
        xb, yb = series_b.xs[best_idx], series_b.ys[best_idx]
        separations.append(math.hypot(xa - xb, ya - yb))

    fde = math.hypot(series_a.xs[-1] - series_b.xs[-1], series_a.ys[-1] - series_b.ys[-1])

    if not separations:
        return {
            "mean_separation_m": float("nan"),
            "max_separation_m": float("nan"),
            "fde_m": fde,
            "aligned_samples": 0.0,
        }

    return {
        "mean_separation_m": statistics.fmean(separations),
        "max_separation_m": max(separations),
        "fde_m": fde,
        "aligned_samples": float(len(separations)),
    }


def load_batch_log(output_dir: Path) -> list[dict[str, str]]:
    log_path = output_dir / "batch_eval_log.csv"
    return read_csv_rows(log_path)


def load_run_status(output_dir: Path, model: str, bag_key: str) -> str:
    for row in load_batch_log(output_dir):
        if row.get("model_name") != model:
            continue
        trace_path = row.get("trace_path", "")
        if trace_path and (trace_path.endswith(bag_key) or bag_key in trace_path):
            return row.get("status", "")
    return ""


@dataclass
class GoalStopAnalysisConfig:
    rosbag_dir: Path | None
    stop_speed_threshold: float
    goal_min_move_m: float


def load_goal_stop_analysis_config(
    config_path: Path | None,
    *,
    rosbag_dir: str | None,
    stop_speed_threshold: float,
    goal_min_move_m: float,
) -> GoalStopAnalysisConfig:
    raw: dict = {}
    top_level: dict = {}
    if config_path is not None:
        with config_path.expanduser().open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if isinstance(loaded, dict):
            top_level = loaded
            raw = loaded.get("analysis", {})

    resolved_rosbag_dir = rosbag_dir or top_level.get("rosbag_dir") or raw.get("rosbag_dir")
    return GoalStopAnalysisConfig(
        rosbag_dir=Path(resolved_rosbag_dir).expanduser() if resolved_rosbag_dir else None,
        stop_speed_threshold=float(
            raw.get("goal_stop_speed_threshold", stop_speed_threshold)
        ),
        goal_min_move_m=float(raw.get("goal_min_move_m", goal_min_move_m)),
    )


def run_goal_stop_analysis(
    output_dir: Path,
    models: list[str],
    goal_config: GoalStopAnalysisConfig,
) -> list[dict[str, float | str]]:
    from goal_stop_error import analyze_trace_goal_stop

    batch_log_rows = load_batch_log(output_dir)
    goal_cache_path = output_dir / "quantitative_analysis" / ".goal_pose_cache.json"
    rows: list[dict[str, float | str]] = []
    empty_metrics = {
        "goal_stop_lateral_m": float("nan"),
        "goal_stop_longitudinal_m": float("nan"),
        "goal_stop_position_m": float("nan"),
        "goal_stop_heading_deg": float("nan"),
        "goal_stop_speed_mps": float("nan"),
        "goal_stop_time_sec": float("nan"),
        "stop_sample_idx": float("nan"),
    }
    for model in models:
        for bag_key in discover_bag_keys(output_dir, model):
            trace_dir = output_dir / model / bag_key
            metrics, status = analyze_trace_goal_stop(
                trace_dir,
                bag_key,
                rosbag_dir=goal_config.rosbag_dir,
                batch_log_rows=batch_log_rows,
                model_name=model,
                speed_threshold=goal_config.stop_speed_threshold,
                goal_min_move_m=goal_config.goal_min_move_m,
                goal_cache_path=goal_cache_path,
            )
            row: dict[str, float | str] = {
                "model": model,
                "bag_key": bag_key,
                "run_status": load_run_status(output_dir, model, bag_key),
                "goal_stop_status": status,
            }
            if metrics is not None:
                row.update(metrics.as_dict())
            else:
                row.update(empty_metrics)
            rows.append(row)
    return rows


def model_run_summary(output_dir: Path, model: str) -> dict[str, float]:
    rows = [row for row in load_batch_log(output_dir) if row.get("model_name") == model]
    if not rows:
        return {}
    total = len(rows)
    statuses = [row.get("status", "") for row in rows]
    return {
        "runs": float(total),
        "success_rate": sum(status == "success" for status in statuses) / total,
        "stuck_rate": sum(status == "stuck" for status in statuses) / total,
        "timeout_rate": sum(status == "timeout" for status in statuses) / total,
        "error_rate": sum(status not in ("success", "stuck", "timeout") for status in statuses) / total,
        "mean_duration_sec": statistics.fmean(float(row.get("duration_sec", 0.0)) for row in rows),
    }


def analyze_pair_bag(
    output_dir: Path,
    model_a: str,
    model_b: str,
    bag_key: str,
    max_dt: float,
    goal_config: GoalStopAnalysisConfig | None = None,
    comfort_config: ComfortAnalysisConfig | None = None,
) -> dict[str, float | str]:
    dir_a = output_dir / model_a / bag_key
    dir_b = output_dir / model_b / bag_key
    ego_a = load_ego_series(dir_a)
    ego_b = load_ego_series(dir_b)
    if ego_a is None or ego_b is None:
        return {"bag_key": bag_key, "model_a": model_a, "model_b": model_b}

    sep = time_aligned_separation(ego_a, ego_b, max_dt=max_dt)
    plan_a = load_plan_stats(dir_a)
    plan_b = load_plan_stats(dir_b)

    result: dict[str, float | str] = {
        "bag_key": bag_key,
        "model_a": model_a,
        "model_b": model_b,
        "path_length_a_m": ego_a.path_length_m,
        "path_length_b_m": ego_b.path_length_m,
        "path_length_delta_m": ego_b.path_length_m - ego_a.path_length_m,
        "duration_a_sec": ego_a.duration_sec,
        "duration_b_sec": ego_b.duration_sec,
        "mean_speed_a_mps": ego_a.mean_speed_mps,
        "mean_speed_b_mps": ego_b.mean_speed_mps,
        "hausdorff_m": symmetric_hausdorff(ego_a.xy_path(), ego_b.xy_path()),
        "plan_messages_a": plan_a["plan_messages"],
        "plan_messages_b": plan_b["plan_messages"],
        "mean_plan_points_a": plan_a["mean_plan_points"],
        "mean_plan_points_b": plan_b["mean_plan_points"],
        **sep,
    }

    if goal_config is not None:
        from goal_stop_error import analyze_trace_goal_stop
        from goal_stop_error import load_ego_pose_rows
        from goal_stop_error import load_goal_for_bag
        from goal_stop_error import pairwise_stop_shift
        from goal_stop_error import resolve_bag_path

        batch_log_rows = load_batch_log(output_dir)
        goal_cache_path = output_dir / "quantitative_analysis" / ".goal_pose_cache.json"
        metrics_a, status_a = analyze_trace_goal_stop(
            dir_a,
            bag_key,
            rosbag_dir=goal_config.rosbag_dir,
            batch_log_rows=batch_log_rows,
            model_name=model_a,
            speed_threshold=goal_config.stop_speed_threshold,
            goal_min_move_m=goal_config.goal_min_move_m,
            goal_cache_path=goal_cache_path,
        )
        metrics_b, status_b = analyze_trace_goal_stop(
            dir_b,
            bag_key,
            rosbag_dir=goal_config.rosbag_dir,
            batch_log_rows=batch_log_rows,
            model_name=model_b,
            speed_threshold=goal_config.stop_speed_threshold,
            goal_min_move_m=goal_config.goal_min_move_m,
            goal_cache_path=goal_cache_path,
        )
        if metrics_a is not None:
            result["goal_stop_position_a_m"] = metrics_a.goal_stop_position_m
            result["goal_stop_lateral_a_m"] = metrics_a.goal_stop_lateral_m
            result["goal_stop_longitudinal_a_m"] = metrics_a.goal_stop_longitudinal_m
        if metrics_b is not None:
            result["goal_stop_position_b_m"] = metrics_b.goal_stop_position_m
            result["goal_stop_lateral_b_m"] = metrics_b.goal_stop_lateral_m
            result["goal_stop_longitudinal_b_m"] = metrics_b.goal_stop_longitudinal_m
        if metrics_a is not None and metrics_b is not None:
            bag_path = resolve_bag_path(
                dir_a,
                bag_key,
                rosbag_dir=goal_config.rosbag_dir,
                batch_log_rows=batch_log_rows,
                model_name=model_a,
            )
            if bag_path is not None:
                goal = load_goal_for_bag(
                    bag_path, min_move_m=goal_config.goal_min_move_m, cache_path=goal_cache_path
                )
                rows_a = load_ego_pose_rows(dir_a / "ego_pose.csv")
                rows_b = load_ego_pose_rows(dir_b / "ego_pose.csv")
                result.update(
                    pairwise_stop_shift(metrics_a, metrics_b, goal, rows_a, rows_b)
                )
        if status_a != "ok" or status_b != "ok":
            result["goal_stop_status"] = f"a={status_a},b={status_b}"

    if comfort_config is not None:
        from comfort_metrics import analyze_trace_comfort
        from comfort_metrics import pairwise_comfort_diff

        metrics_a, status_a = analyze_trace_comfort(dir_a, comfort_config)
        metrics_b, status_b = analyze_trace_comfort(dir_b, comfort_config)
        if metrics_a is not None:
            result["rms_longitudinal_jerk_a_mps3"] = metrics_a.rms_longitudinal_jerk_mps3
            result["harsh_decel_count_a"] = float(metrics_a.harsh_decel_count)
            result["max_decel_a_mps2"] = metrics_a.max_decel_mps2
        if metrics_b is not None:
            result["rms_longitudinal_jerk_b_mps3"] = metrics_b.rms_longitudinal_jerk_mps3
            result["harsh_decel_count_b"] = float(metrics_b.harsh_decel_count)
            result["max_decel_b_mps2"] = metrics_b.max_decel_mps2
        if metrics_a is not None and metrics_b is not None:
            result.update(pairwise_comfort_diff(metrics_a, metrics_b))
        if status_a != "ok" or status_b != "ok":
            result["comfort_status"] = f"a={status_a},b={status_b}"

    return result


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


@dataclass
class LaneletAnalysisConfig:
    enabled: bool
    map_path: Path
    vehicle_model: str
    vehicle_info_yaml: Path | None
    boundary_types: list[str]
    lanelet_sample_dt: float
    footprint_extra_margin: float
    boundary_search_radius_m: float
    lanelet_map_file: str


def load_lanelet_analysis_config(
    config_path: Path | None,
    *,
    enabled: bool,
    map_path: str | None,
    vehicle_model: str | None,
    vehicle_info_yaml: Path | None,
    boundary_types: list[str] | None,
    lanelet_sample_dt: float,
    footprint_extra_margin: float,
    boundary_search_radius_m: float,
    lanelet_map_file: str,
) -> LaneletAnalysisConfig:
    raw: dict = {}
    if config_path is not None:
        with config_path.expanduser().open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        raw = loaded.get("analysis", {}) if isinstance(loaded, dict) else {}
        if not map_path and isinstance(loaded, dict) and loaded.get("map_path"):
            map_path = str(loaded["map_path"])
        if not vehicle_model and isinstance(loaded, dict) and loaded.get("vehicle_model"):
            vehicle_model = str(loaded["vehicle_model"])

    resolved_map_path = Path(map_path or raw.get("map_path", "/opt/autoware/maps")).expanduser()
    resolved_vehicle_model = vehicle_model or str(raw.get("vehicle_model", "lv828l"))
    resolved_vehicle_info = vehicle_info_yaml
    if resolved_vehicle_info is None and raw.get("vehicle_info_yaml"):
        resolved_vehicle_info = Path(str(raw["vehicle_info_yaml"])).expanduser()

    resolved_boundary_types = boundary_types or list(
        raw.get("boundary_types", ["road_border", "curbstone"])
    )
    return LaneletAnalysisConfig(
        enabled=enabled or bool(raw.get("lanelet_boundary_check", False)),
        map_path=resolved_map_path,
        vehicle_model=resolved_vehicle_model,
        vehicle_info_yaml=resolved_vehicle_info,
        boundary_types=resolved_boundary_types,
        lanelet_sample_dt=float(raw.get("lanelet_sample_dt", lanelet_sample_dt)),
        footprint_extra_margin=float(raw.get("footprint_extra_margin", footprint_extra_margin)),
        boundary_search_radius_m=float(raw.get("boundary_search_radius_m", boundary_search_radius_m)),
        lanelet_map_file=str(raw.get("lanelet_map_file", lanelet_map_file)),
    )


def run_lanelet_boundary_analysis(
    output_dir: Path,
    models: list[str],
    lanelet_config: LaneletAnalysisConfig,
) -> list[dict[str, float | str]]:
    from lanelet_boundary_check import LaneletBoundaryChecker
    from lanelet_boundary_check import load_vehicle_footprint_spec
    from lanelet_boundary_check import resolve_vehicle_info_yaml

    vehicle_info_yaml = lanelet_config.vehicle_info_yaml
    if vehicle_info_yaml is None:
        vehicle_info_yaml = resolve_vehicle_info_yaml(lanelet_config.vehicle_model)

    vehicle_spec = load_vehicle_footprint_spec(
        vehicle_info_yaml, footprint_extra_margin=lanelet_config.footprint_extra_margin
    )
    checker = LaneletBoundaryChecker(
        lanelet_config.map_path,
        vehicle_spec,
        boundary_types=lanelet_config.boundary_types,
        lanelet_map_file=lanelet_config.lanelet_map_file,
        boundary_search_radius_m=lanelet_config.boundary_search_radius_m,
    )

    rows: list[dict[str, float | str]] = []
    for model in models:
        for bag_key in discover_bag_keys(output_dir, model):
            ego_csv = output_dir / model / bag_key / "ego_pose.csv"
            if not ego_csv.is_file():
                continue
            metrics = checker.analyze_ego_pose_csv(
                ego_csv, sample_dt=lanelet_config.lanelet_sample_dt
            )
            rows.append({"model": model, "bag_key": bag_key, **metrics.as_dict()})
    return rows


@dataclass
class NpcCollisionAnalysisConfig:
    vehicle_model: str
    vehicle_info_yaml: Path | None
    sample_dt: float
    align_max_dt: float
    near_miss_threshold_m: float
    footprint_extra_margin: float
    labels: set[str]


def load_npc_collision_analysis_config(
    config_path: Path | None,
    *,
    vehicle_model: str | None,
    vehicle_info_yaml: Path | None,
    sample_dt: float,
    align_max_dt: float,
    near_miss_threshold_m: float,
    footprint_extra_margin: float,
    npc_labels: list[str] | None,
) -> NpcCollisionAnalysisConfig:
    raw: dict = {}
    top_level: dict = {}
    if config_path is not None:
        with config_path.expanduser().open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if isinstance(loaded, dict):
            top_level = loaded
            raw = loaded.get("analysis", {})

    resolved_vehicle_model = (
        vehicle_model or top_level.get("vehicle_model") or raw.get("vehicle_model") or "lv828l"
    )
    resolved_vehicle_info = vehicle_info_yaml
    if resolved_vehicle_info is None and raw.get("vehicle_info_yaml"):
        resolved_vehicle_info = Path(str(raw["vehicle_info_yaml"])).expanduser()

    if npc_labels is not None:
        labels = set(npc_labels)
    else:
        labels = set(raw.get("npc_labels", ["CAR", "TRUCK", "BUS", "TRAILER", "MOTORCYCLE", "BICYCLE"]))

    return NpcCollisionAnalysisConfig(
        vehicle_model=str(resolved_vehicle_model),
        vehicle_info_yaml=resolved_vehicle_info,
        sample_dt=float(raw.get("npc_sample_dt", sample_dt)),
        align_max_dt=float(raw.get("npc_align_max_dt", align_max_dt)),
        near_miss_threshold_m=float(raw.get("npc_near_miss_threshold_m", near_miss_threshold_m)),
        footprint_extra_margin=float(
            raw.get("npc_footprint_extra_margin", footprint_extra_margin)
        ),
        labels=labels,
    )


def run_npc_collision_analysis(
    output_dir: Path,
    models: list[str],
    npc_config: NpcCollisionAnalysisConfig,
) -> list[dict[str, float | str]]:
    from lanelet_boundary_check import load_vehicle_footprint_spec
    from lanelet_boundary_check import resolve_vehicle_info_yaml
    from npc_collision_check import NpcCollisionChecker

    vehicle_info_yaml = npc_config.vehicle_info_yaml
    if vehicle_info_yaml is None:
        vehicle_info_yaml = resolve_vehicle_info_yaml(npc_config.vehicle_model)

    vehicle_spec = load_vehicle_footprint_spec(
        vehicle_info_yaml, footprint_extra_margin=npc_config.footprint_extra_margin
    )
    checker = NpcCollisionChecker(
        vehicle_spec,
        near_miss_threshold_m=npc_config.near_miss_threshold_m,
        align_max_dt=npc_config.align_max_dt,
        labels=npc_config.labels,
    )

    empty_metrics = {
        "npc_check_samples": float("nan"),
        "npc_overlap_frames": float("nan"),
        "npc_collision_rate": float("nan"),
        "npc_near_miss_frames": float("nan"),
        "npc_near_miss_rate": float("nan"),
        "npc_min_distance_m": float("nan"),
        "first_npc_overlap_sec": float("nan"),
        "had_npc_collision": float("nan"),
    }

    rows: list[dict[str, float | str]] = []
    for model in models:
        for bag_key in discover_bag_keys(output_dir, model):
            trace_dir = output_dir / model / bag_key
            metrics, status = checker.analyze_trace_dir(trace_dir, sample_dt=npc_config.sample_dt)
            row: dict[str, float | str] = {
                "model": model,
                "bag_key": bag_key,
                "run_status": load_run_status(output_dir, model, bag_key),
                "npc_collision_status": status,
            }
            if metrics is not None:
                row.update(metrics.as_dict())
            else:
                row.update(empty_metrics)
            rows.append(row)
    return rows


@dataclass
class ComfortAnalysisConfig:
    sample_dt: float
    min_derivative_dt: float
    max_derivative_dt: float
    harsh_decel_threshold_mps2: float
    harsh_decel_min_duration_sec: float


def load_comfort_analysis_config(
    config_path: Path | None,
    *,
    sample_dt: float,
    min_derivative_dt: float,
    max_derivative_dt: float,
    harsh_decel_threshold_mps2: float,
    harsh_decel_min_duration_sec: float,
) -> ComfortAnalysisConfig:
    raw: dict = {}
    if config_path is not None:
        with config_path.expanduser().open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        raw = loaded.get("analysis", {}) if isinstance(loaded, dict) else {}

    return ComfortAnalysisConfig(
        sample_dt=float(raw.get("comfort_sample_dt", sample_dt)),
        min_derivative_dt=float(raw.get("comfort_min_derivative_dt", min_derivative_dt)),
        max_derivative_dt=float(raw.get("comfort_max_derivative_dt", max_derivative_dt)),
        harsh_decel_threshold_mps2=float(
            raw.get("harsh_decel_threshold_mps2", harsh_decel_threshold_mps2)
        ),
        harsh_decel_min_duration_sec=float(
            raw.get("harsh_decel_min_duration_sec", harsh_decel_min_duration_sec)
        ),
    )


def run_comfort_analysis(
    output_dir: Path,
    models: list[str],
    comfort_config: ComfortAnalysisConfig,
) -> list[dict[str, float | str]]:
    from comfort_metrics import ComfortMetrics
    from comfort_metrics import analyze_trace_comfort

    empty_metrics = ComfortMetrics(
        comfort_samples=0,
        motion_duration_sec=float("nan"),
        max_longitudinal_accel_mps2=float("nan"),
        min_longitudinal_accel_mps2=float("nan"),
        max_lateral_accel_mps2=float("nan"),
        min_lateral_accel_mps2=float("nan"),
        max_longitudinal_jerk_mps3=float("nan"),
        max_lateral_jerk_mps3=float("nan"),
        rms_longitudinal_jerk_mps3=float("nan"),
        rms_lateral_jerk_mps3=float("nan"),
        p95_longitudinal_jerk_mps3=float("nan"),
        p95_lateral_jerk_mps3=float("nan"),
        harsh_decel_count=0,
        harsh_decel_time_sec=float("nan"),
        harsh_decel_ratio=float("nan"),
        max_decel_mps2=float("nan"),
        plan_max_longitudinal_accel_mps2=float("nan"),
        plan_max_longitudinal_jerk_mps3=float("nan"),
    ).as_dict()

    rows: list[dict[str, float | str]] = []
    for model in models:
        for bag_key in discover_bag_keys(output_dir, model):
            trace_dir = output_dir / model / bag_key
            metrics, status = analyze_trace_comfort(trace_dir, comfort_config)
            row: dict[str, float | str] = {
                "model": model,
                "bag_key": bag_key,
                "run_status": load_run_status(output_dir, model, bag_key),
                "comfort_status": status,
            }
            if metrics is not None:
                row.update(metrics.as_dict())
            else:
                row.update(empty_metrics)
            rows.append(row)
    return rows


def aggregate_comfort_summary(
    comfort_rows: list[dict[str, float | str]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for model in {str(row["model"]) for row in comfort_rows}:
        ok_rows = [
            row
            for row in comfort_rows
            if row["model"] == model and row.get("comfort_status") == "ok"
        ]
        if not ok_rows:
            continue
        summary[model] = {
            "mean_rms_longitudinal_jerk_mps3": statistics.fmean(
                float(row["rms_longitudinal_jerk_mps3"]) for row in ok_rows
            ),
            "mean_rms_lateral_jerk_mps3": statistics.fmean(
                float(row["rms_lateral_jerk_mps3"]) for row in ok_rows
            ),
            "mean_harsh_decel_count": statistics.fmean(
                float(row["harsh_decel_count"]) for row in ok_rows
            ),
            "mean_harsh_decel_ratio": statistics.fmean(
                float(row["harsh_decel_ratio"]) for row in ok_rows
            ),
            "mean_max_decel_mps2": statistics.fmean(float(row["max_decel_mps2"]) for row in ok_rows),
        }
    return summary


def print_pair_summary(rows: list[dict[str, float | str]]) -> None:
    if not rows:
        print("[warn] No pairwise metrics computed.")
        return
    numeric_keys = [
        key
        for key in rows[0]
        if key not in ("bag_key", "model_a", "model_b") and isinstance(rows[0][key], (int, float))
    ]
    print("\nPairwise summary (mean across common bags):")
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if not math.isnan(float(row[key]))]
        if values:
            print(f"  {key}: mean={statistics.fmean(values):.3f}, max={max(values):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantitative metrics for comparing batch-eval model results."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Batch eval output directory",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated models to compare. Default: all models with traces.",
    )
    parser.add_argument(
        "--align-max-dt",
        type=float,
        default=0.15,
        help="Max timestamp gap [s] when time-aligning ego paths (default: 0.15)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Directory for CSV output (default: <output-dir>/quantitative_analysis)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Batch-eval YAML config (reads analysis.map_path / analysis.lanelet_boundary_check)",
    )
    parser.add_argument(
        "--lanelet-boundary-check",
        action="store_true",
        help="Check ego footprint vs drivable lanes and road_border/curbstone (needs lanelet2 map)",
    )
    parser.add_argument("--map-path", help="Lanelet2 map directory or .osm file (overrides config)")
    parser.add_argument(
        "--vehicle-model",
        help="Vehicle model for footprint dimensions, e.g. lv828l (overrides config)",
    )
    parser.add_argument(
        "--vehicle-info-yaml",
        type=Path,
        help="vehicle_info.param.yaml path (default: from vehicle_model package)",
    )
    parser.add_argument(
        "--boundary-types",
        help="Comma-separated linestring types to treat as uncrossable (default: road_border,curbstone)",
    )
    parser.add_argument(
        "--lanelet-sample-dt",
        type=float,
        default=0.2,
        help="Downsample ego poses to this interval [s] for lanelet checks (default: 0.2)",
    )
    parser.add_argument(
        "--footprint-extra-margin",
        type=float,
        default=0.0,
        help="Extra margin [m] added to ego footprint for lanelet checks (default: 0.0)",
    )
    parser.add_argument(
        "--boundary-search-radius-m",
        type=float,
        default=80.0,
        help="Search radius [m] around ego for boundary segments (default: 80)",
    )
    parser.add_argument(
        "--lanelet-map-file",
        default="lanelet2_map.osm",
        help="Lanelet map filename inside map_path (default: lanelet2_map.osm)",
    )
    parser.add_argument(
        "--rosbag-dir",
        type=Path,
        help="Rosbag root directory (fallback to resolve goal pose from bag_key)",
    )
    parser.add_argument(
        "--goal-stop-speed-threshold",
        type=float,
        default=0.2,
        help="Speed [m/s] below which ego is considered stopped at goal (default: 0.2)",
    )
    parser.add_argument(
        "--goal-min-move-m",
        type=float,
        default=0.1,
        help="Min ego motion when sampling goal pose from rosbag (default: 0.1)",
    )
    parser.add_argument(
        "--skip-goal-stop",
        action="store_true",
        help="Skip goal stop error analysis",
    )
    parser.add_argument(
        "--skip-npc-collision",
        action="store_true",
        help="Skip ego vs NPC collision / near-miss analysis",
    )
    parser.add_argument(
        "--npc-sample-dt",
        type=float,
        default=0.2,
        help="Downsample ego poses to this interval [s] for NPC checks (default: 0.2)",
    )
    parser.add_argument(
        "--npc-near-miss-threshold-m",
        type=float,
        default=0.5,
        help="Distance [m] below which a non-overlap frame counts as near miss (default: 0.5)",
    )
    parser.add_argument(
        "--npc-labels",
        help="Comma-separated NPC labels to include (default: CAR,TRUCK,BUS,TRAILER,MOTORCYCLE,BICYCLE)",
    )
    parser.add_argument(
        "--skip-comfort",
        action="store_true",
        help="Skip longitudinal/lateral jerk and harsh deceleration analysis",
    )
    parser.add_argument(
        "--comfort-sample-dt",
        type=float,
        default=0.1,
        help="Downsample ego poses to this interval [s] for comfort metrics (default: 0.1)",
    )
    parser.add_argument(
        "--harsh-decel-threshold-mps2",
        type=float,
        default=-2.5,
        help="Longitudinal accel [m/s²] below which decel counts as harsh (default: -2.5)",
    )
    parser.add_argument(
        "--harsh-decel-min-duration-sec",
        type=float,
        default=0.3,
        help="Min duration [s] for a harsh decel event (default: 0.3)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_dir():
        print(f"[error] Output directory not found: {output_dir}")
        sys.exit(1)

    models = (
        [name.strip() for name in args.models.split(",") if name.strip()]
        if args.models
        else discover_models(output_dir)
    )
    if len(models) < 2:
        print("[error] Need at least two models for pairwise comparison.")
        sys.exit(1)

    out_dir = args.out_csv or (output_dir / "quantitative_analysis")
    bag_keys = common_bag_keys(output_dir, models)
    print(f"[info] Models: {', '.join(models)}")
    print(f"[info] Common bags with traces: {len(bag_keys)}")

    per_model_rows: list[dict[str, float | str]] = []
    for model in models:
        summary = model_run_summary(output_dir, model)
        if summary:
            per_model_rows.append({"model": model, **summary})

    per_bag_rows: list[dict[str, float | str]] = []
    for model in models:
        for bag_key in discover_bag_keys(output_dir, model):
            trace_dir = output_dir / model / bag_key
            ego = load_ego_series(trace_dir)
            if ego is None:
                continue
            plan = load_plan_stats(trace_dir)
            per_bag_rows.append(
                {
                    "model": model,
                    "bag_key": bag_key,
                    "path_length_m": ego.path_length_m,
                    "duration_sec": ego.duration_sec,
                    "mean_speed_mps": ego.mean_speed_mps,
                    "max_speed_mps": ego.max_speed_mps,
                    "samples": float(len(ego.stamps)),
                    **plan,
                }
            )
    write_csv(out_dir / "per_model_per_bag.csv", per_bag_rows)

    goal_config = load_goal_stop_analysis_config(
        args.config,
        rosbag_dir=str(args.rosbag_dir) if args.rosbag_dir else None,
        stop_speed_threshold=args.goal_stop_speed_threshold,
        goal_min_move_m=args.goal_min_move_m,
    )
    goal_stop_enabled = not args.skip_goal_stop
    npc_collision_enabled = not args.skip_npc_collision
    comfort_enabled = not args.skip_comfort

    comfort_config = load_comfort_analysis_config(
        args.config,
        sample_dt=args.comfort_sample_dt,
        min_derivative_dt=0.05,
        max_derivative_dt=0.5,
        harsh_decel_threshold_mps2=args.harsh_decel_threshold_mps2,
        harsh_decel_min_duration_sec=args.harsh_decel_min_duration_sec,
    )

    npc_labels = (
        [name.strip() for name in args.npc_labels.split(",") if name.strip()]
        if args.npc_labels
        else None
    )
    npc_config = load_npc_collision_analysis_config(
        args.config,
        vehicle_model=args.vehicle_model,
        vehicle_info_yaml=args.vehicle_info_yaml,
        sample_dt=args.npc_sample_dt,
        align_max_dt=args.align_max_dt,
        near_miss_threshold_m=args.npc_near_miss_threshold_m,
        footprint_extra_margin=args.footprint_extra_margin,
        npc_labels=npc_labels,
    )

    pair_rows: list[dict[str, float | str]] = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            for bag_key in bag_keys:
                pair_rows.append(
                    analyze_pair_bag(
                        output_dir,
                        model_a,
                        model_b,
                        bag_key,
                        args.align_max_dt,
                        goal_config if goal_stop_enabled else None,
                        comfort_config if comfort_enabled else None,
                    )
                )
    write_csv(out_dir / "pairwise_per_bag.csv", pair_rows)

    if goal_stop_enabled:
        print(
            f"\n[info] Computing goal stop error "
            f"(stop_speed<={goal_config.stop_speed_threshold} m/s)"
        )
        goal_stop_rows = run_goal_stop_analysis(output_dir, models, goal_config)
        write_csv(out_dir / "per_model_per_bag_goal_stop.csv", goal_stop_rows)
        success_rows = [
            row
            for row in goal_stop_rows
            if row.get("goal_stop_status") == "ok" and row.get("run_status") == "success"
        ]
        if success_rows:
            print("\nGoal stop error summary (successful runs only):")
            for model in models:
                model_rows = [row for row in success_rows if row["model"] == model]
                if not model_rows:
                    continue
                mean_lat = statistics.fmean(float(row["goal_stop_lateral_m"]) for row in model_rows)
                mean_lon = statistics.fmean(
                    float(row["goal_stop_longitudinal_m"]) for row in model_rows
                )
                mean_pos = statistics.fmean(float(row["goal_stop_position_m"]) for row in model_rows)
                print(
                    f"  {model}: mean lateral={mean_lat:.2f} m, "
                    f"longitudinal={mean_lon:.2f} m, position={mean_pos:.2f} m"
                )
        failed = sum(1 for row in goal_stop_rows if row.get("goal_stop_status") != "ok")
        if failed:
            print(f"[warn] Goal stop analysis failed for {failed} trace(s) (see goal_stop_status column)")

    if npc_collision_enabled:
        print(
            f"\n[info] Computing NPC collision / near-miss "
            f"(sample_dt={npc_config.sample_dt}s, near_miss<{npc_config.near_miss_threshold_m}m)"
        )
        npc_rows = run_npc_collision_analysis(output_dir, models, npc_config)
        write_csv(out_dir / "per_model_per_bag_npc_collision.csv", npc_rows)
        ok_rows = [row for row in npc_rows if row.get("npc_collision_status") == "ok"]
        if ok_rows:
            print("\nNPC collision summary (per model):")
            for model in models:
                model_rows = [row for row in ok_rows if row["model"] == model]
                if not model_rows:
                    continue
                mean_rate = statistics.fmean(float(row["npc_collision_rate"]) for row in model_rows)
                bags_with_collision = sum(float(row["had_npc_collision"]) > 0 for row in model_rows)
                mean_min_dist = statistics.fmean(
                    float(row["npc_min_distance_m"]) for row in model_rows
                )
                print(
                    f"  {model}: mean_collision_rate={mean_rate:.1%}, "
                    f"bags_with_collision={bags_with_collision}/{len(model_rows)}, "
                    f"mean_min_distance={mean_min_dist:.2f} m"
                )
        failed = sum(1 for row in npc_rows if row.get("npc_collision_status") != "ok")
        if failed:
            print(f"[warn] NPC collision analysis failed for {failed} trace(s)")

    if comfort_enabled:
        print(
            f"\n[info] Computing comfort metrics "
            f"(harsh_decel<{comfort_config.harsh_decel_threshold_mps2} m/s² "
            f"for>={comfort_config.harsh_decel_min_duration_sec}s)"
        )
        comfort_rows = run_comfort_analysis(output_dir, models, comfort_config)
        write_csv(out_dir / "per_model_per_bag_comfort.csv", comfort_rows)
        comfort_summary = aggregate_comfort_summary(comfort_rows)
        for row in per_model_rows:
            model_summary = comfort_summary.get(str(row["model"]))
            if model_summary:
                row.update(model_summary)
        ok_rows = [row for row in comfort_rows if row.get("comfort_status") == "ok"]
        if ok_rows:
            print("\nComfort summary (per model):")
            for model in models:
                model_rows = [row for row in ok_rows if row["model"] == model]
                if not model_rows:
                    continue
                mean_jerk = statistics.fmean(
                    float(row["rms_longitudinal_jerk_mps3"]) for row in model_rows
                )
                mean_harsh = statistics.fmean(float(row["harsh_decel_count"]) for row in model_rows)
                mean_decel = statistics.fmean(float(row["max_decel_mps2"]) for row in model_rows)
                print(
                    f"  {model}: mean_rms_long_jerk={mean_jerk:.2f} m/s³, "
                    f"mean_harsh_decel_count={mean_harsh:.1f}, "
                    f"mean_max_decel={mean_decel:.2f} m/s²"
                )
        failed = sum(1 for row in comfort_rows if row.get("comfort_status") != "ok")
        if failed:
            print(f"[warn] Comfort analysis failed for {failed} trace(s) (see comfort_status column)")

    write_csv(out_dir / "per_model_run_summary.csv", per_model_rows)

    print("\nPer-model run summary:")
    for row in per_model_rows:
        print(
            f"  {row['model']}: success={row['success_rate']:.0%}, "
            f"stuck={row['stuck_rate']:.0%}, timeout={row['timeout_rate']:.0%}, "
            f"mean_duration={row['mean_duration_sec']:.1f}s"
        )

    if len(models) == 2:
        print_pair_summary(pair_rows)

    boundary_types = (
        [name.strip() for name in args.boundary_types.split(",") if name.strip()]
        if args.boundary_types
        else None
    )
    lanelet_config = load_lanelet_analysis_config(
        args.config,
        enabled=args.lanelet_boundary_check,
        map_path=args.map_path,
        vehicle_model=args.vehicle_model,
        vehicle_info_yaml=args.vehicle_info_yaml,
        boundary_types=boundary_types,
        lanelet_sample_dt=args.lanelet_sample_dt,
        footprint_extra_margin=args.footprint_extra_margin,
        boundary_search_radius_m=args.boundary_search_radius_m,
        lanelet_map_file=args.lanelet_map_file,
    )
    if lanelet_config.enabled:
        print(
            f"\n[info] Running lanelet boundary analysis on map {lanelet_config.map_path} "
            f"(types={','.join(lanelet_config.boundary_types)}, "
            f"sample_dt={lanelet_config.lanelet_sample_dt}s)"
        )
        try:
            lanelet_rows = run_lanelet_boundary_analysis(output_dir, models, lanelet_config)
        except Exception as exc:
            print(f"[error] Lanelet boundary analysis failed: {exc}")
            sys.exit(1)
        write_csv(out_dir / "per_model_per_bag_lanelet_boundary.csv", lanelet_rows)
        if lanelet_rows:
            print("\nLanelet boundary summary (mean across bags per model):")
            for model in models:
                model_rows = [row for row in lanelet_rows if row["model"] == model]
                if not model_rows:
                    continue
                out_ratio = statistics.fmean(float(row["out_of_lane_ratio"]) for row in model_rows)
                border_ratio = statistics.fmean(
                    float(row["boundary_crossing_ratio"]) for row in model_rows
                )
                bags_with_border = sum(float(row["boundary_crossing_frames"]) > 0 for row in model_rows)
                print(
                    f"  {model}: out_of_lane_ratio={out_ratio:.1%}, "
                    f"boundary_crossing_ratio={border_ratio:.1%}, "
                    f"bags_with_boundary_crossing={bags_with_border}/{len(model_rows)}"
                )

    print(f"\n[info] Wrote CSVs to {out_dir}")
    print("  per_model_run_summary.csv  — success/stuck/timeout rates")
    print("  per_model_per_bag.csv      — path length, speed, plan stats per bag")
    print("  pairwise_per_bag.csv       — model A vs B separation metrics per bag")
    if goal_stop_enabled:
        print("  per_model_per_bag_goal_stop.csv — stop pose vs route goal per bag")
    if npc_collision_enabled:
        print("  per_model_per_bag_npc_collision.csv — ego vs NPC overlap / near-miss per bag")
    if comfort_enabled:
        print("  per_model_per_bag_comfort.csv — jerk / harsh deceleration per bag")
    if lanelet_config.enabled:
        print("  per_model_per_bag_lanelet_boundary.csv — out-of-lane / road-border crossing per bag")
    print("\nKey pairwise metrics:")
    print("  mean_separation_m — avg distance between ego paths (time-aligned)")
    print("  max_separation_m  — worst-case separation along the run")
    print("  fde_m             — final displacement (end-point distance)")
    print("  hausdorff_m       — overall path shape difference")
    if goal_stop_enabled:
        print("\nGoal stop metrics (per_model_per_bag_goal_stop.csv):")
        print("  goal_stop_lateral_m       — left(+) / right(-) offset from goal at stop")
        print("  goal_stop_longitudinal_m  — ahead(+) / behind(-) of goal at stop")
        print("  goal_stop_position_m      — Euclidean distance to goal at stop")
        print("  goal_stop_heading_deg     — yaw error vs goal at stop")
        print("\nPairwise goal stop shift (pairwise_per_bag.csv):")
        print("  stop_shift_lateral_m      — lateral stop difference (model B − A)")
        print("  stop_shift_longitudinal_m — longitudinal stop difference (model B − A)")
        print("  stop_shift_position_m     — Euclidean distance between stop poses")
    if npc_collision_enabled:
        print("\nNPC collision metrics (per_model_per_bag_npc_collision.csv):")
        print("  npc_collision_rate   — fraction of checked samples with ego/NPC box overlap")
        print("  npc_near_miss_rate   — close approach without overlap (< threshold)")
        print("  npc_min_distance_m   — closest ego-to-NPC distance along the run")
        print("  had_npc_collision    — 1 if any overlap detected in the bag")
    if comfort_enabled:
        print("\nComfort metrics (per_model_per_bag_comfort.csv):")
        print("  rms_longitudinal_jerk_mps3 — RMS of longitudinal jerk in ego frame")
        print("  rms_lateral_jerk_mps3      — RMS of lateral jerk in ego frame")
        print("  harsh_decel_count          — decel events below threshold long enough")
        print("  harsh_decel_ratio          — fraction of run time in harsh decel")
        print("  max_decel_mps2             — peak braking magnitude")
        print("\nPairwise comfort diff (pairwise_per_bag.csv):")
        print("  rms_longitudinal_jerk_diff_mps3 — model B − A")
        print("  harsh_decel_count_diff          — model B − A")


if __name__ == "__main__":
    main()
