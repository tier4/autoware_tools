"""Phase 2 — offline metrics from an output bag."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.bag_reader import BagSeries, EgoSample, downsample_ego, load_bag_series
from dp_multi_eval.geometry import (
    box_local,
    load_vehicle_footprint,
    polygon_distance,
    transform_local_polygon,
)
from dp_multi_eval.topics import TopicSet, load_topics

# Match diffusion_planner_batch_eval npc_collision_check.DEFAULT_NPC_LABELS
NPC_COLLISION_LABELS = frozenset(
    {"CAR", "TRUCK", "BUS", "TRAILER", "MOTORCYCLE", "BICYCLE"}
)


@dataclass
class Thresholds:
    stuck_speed_mps: float = 0.05
    stuck_duration_sec: float = 45.0
    goal_stop_speed_mps: float = 0.2
    goal_tolerance_m: float = 2.0
    oob_margin_m: float = 0.0
    collision_distance_m: float = 0.1
    sample_dt: float = 0.2


def load_thresholds(path: Path | None) -> Thresholds:
    if path is None or not path.expanduser().is_file():
        return Thresholds()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in Thresholds.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: float(v) for k, v in raw.items() if k in known}
    return Thresholds(**kwargs)


def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def goal_frame_errors(
    ego_x: float, ego_y: float, ego_yaw: float, goal_x: float, goal_y: float, goal_yaw: float
) -> tuple[float, float, float, float]:
    """Return lateral [m], longitudinal [m], position [m], heading [deg] in goal frame."""
    dx, dy = ego_x - goal_x, ego_y - goal_y
    pos = math.hypot(dx, dy)
    cos_g, sin_g = math.cos(goal_yaw), math.sin(goal_yaw)
    longitudinal = dx * cos_g + dy * sin_g
    lateral = -dx * sin_g + dy * cos_g
    heading = math.degrees(normalize_angle(ego_yaw - goal_yaw))
    return lateral, longitudinal, pos, heading


def find_stuck_events(
    samples: list[EgoSample],
    goal_x: float,
    goal_y: float,
    thr: Thresholds,
) -> list[dict[str, Any]]:
    """Contiguous near-zero speed away from goal lasting >= stuck_duration_sec."""
    events: list[dict[str, Any]] = []
    if not samples:
        return events

    i = 0
    n = len(samples)
    while i < n:
        if samples[i].speed_mps > thr.stuck_speed_mps:
            i += 1
            continue
        j = i
        while j < n and samples[j].speed_mps <= thr.stuck_speed_mps:
            j += 1
        dur = samples[j - 1].stamp_sec - samples[i].stamp_sec
        mid = samples[(i + j - 1) // 2]
        dist_goal = math.hypot(mid.x - goal_x, mid.y - goal_y)
        near_goal = dist_goal <= thr.goal_tolerance_m
        if dur >= thr.stuck_duration_sec and not near_goal:
            events.append(
                {
                    "start_sec": samples[i].stamp_sec,
                    "end_sec": samples[j - 1].stamp_sec,
                    "duration_sec": dur,
                    "distance_to_goal_m": dist_goal,
                }
            )
        i = j
    return events


def metric_stuck(
    series: BagSeries,
    goal_x: float,
    goal_y: float,
    thr: Thresholds,
) -> dict[str, Any]:
    """Flag near-zero speed away from goal lasting > stuck_duration_sec."""
    samples = series.ego
    if not samples:
        if not series.velocity_mps:
            return {"flagged": False, "duration_sec": 0.0, "events": [], "status": "no_ego_data"}

    events = find_stuck_events(samples, goal_x, goal_y, thr)
    total = sum(e["duration_sec"] for e in events)
    return {
        "flagged": bool(events),
        "duration_sec": total,
        "events": events,
        "status": "ok",
    }


def metric_goal_stop(
    series: BagSeries,
    goal_x: float,
    goal_y: float,
    goal_yaw: float,
    thr: Thresholds,
    *,
    stuck: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Average stop precision over low-speed samples within goal tolerance.

    Scenes flagged as stuck (away from goal) are excluded from goal precision.
    """
    empty = {
        "flagged": True,
        "position_error_m": float("nan"),
        "heading_error_deg": float("nan"),
        "lateral_m": float("nan"),
        "longitudinal_m": float("nan"),
        "stop_time_sec": float("nan"),
        "stop_speed_mps": float("nan"),
        "sample_count": 0,
        "duration_at_goal_sec": 0.0,
        "status": "no_ego_data",
    }
    if not series.ego:
        return empty

    if stuck and stuck.get("flagged"):
        return {
            **empty,
            "flagged": False,
            "status": "excluded_stuck",
        }

    at_goal: list[EgoSample] = []
    for ego in series.ego:
        if ego.speed_mps > thr.goal_stop_speed_mps:
            continue
        _lat, _lon, pos, _heading = goal_frame_errors(
            ego.x, ego.y, ego.yaw_rad, goal_x, goal_y, goal_yaw
        )
        if pos <= thr.goal_tolerance_m:
            at_goal.append(ego)

    if not at_goal:
        return {
            **empty,
            "status": "goal_not_reached",
        }

    laterals: list[float] = []
    longitudinals: list[float] = []
    positions: list[float] = []
    headings: list[float] = []
    speeds: list[float] = []
    for ego in at_goal:
        lat, lon, pos, heading = goal_frame_errors(
            ego.x, ego.y, ego.yaw_rad, goal_x, goal_y, goal_yaw
        )
        laterals.append(lat)
        longitudinals.append(lon)
        positions.append(pos)
        headings.append(heading)
        speeds.append(ego.speed_mps)

    n = len(at_goal)
    avg_pos = sum(positions) / n
    avg_lat = sum(laterals) / n
    avg_lon = sum(longitudinals) / n
    avg_heading = sum(headings) / n
    avg_speed = sum(speeds) / n
    duration_at_goal = at_goal[-1].stamp_sec - at_goal[0].stamp_sec

    return {
        "flagged": avg_pos > thr.goal_tolerance_m,
        "position_error_m": avg_pos,
        "heading_error_deg": avg_heading,
        "lateral_m": avg_lat,
        "longitudinal_m": avg_lon,
        "max_position_error_m": max(positions),
        "max_heading_error_deg": max(abs(h) for h in headings),
        "stop_time_sec": at_goal[-1].stamp_sec,
        "stop_speed_mps": avg_speed,
        "sample_count": n,
        "duration_at_goal_sec": duration_at_goal,
        "status": "ok",
    }


def metric_collision(
    series: BagSeries,
    thr: Thresholds,
    footprint_local: list[tuple[float, float]],
) -> dict[str, Any]:
    ego_ds = downsample_ego(series.ego, thr.sample_dt)
    if not ego_ds:
        return {"flagged": False, "events": [], "min_distance_m": float("nan"), "status": "no_ego"}

    # Index objects by rounded time for fast lookup
    objs = series.objects
    oi = 0
    events: list[dict[str, Any]] = []
    min_dist = float("inf")
    first_t = None

    for ego in ego_ds:
        # advance object cursor to nearby stamps
        while oi < len(objs) and objs[oi].stamp_sec < ego.stamp_sec - thr.sample_dt:
            oi += 1
        j = oi
        ego_poly = transform_local_polygon(footprint_local, ego.x, ego.y, ego.yaw_rad)
        while j < len(objs) and objs[j].stamp_sec <= ego.stamp_sec + thr.sample_dt:
            obj = objs[j]
            j += 1
            if obj.label not in NPC_COLLISION_LABELS:
                continue
            if obj.length_m <= 0.0 or obj.width_m <= 0.0:
                continue
            npc_poly = transform_local_polygon(
                box_local(obj.length_m, obj.width_m), obj.x, obj.y, obj.yaw_rad
            )
            dist = polygon_distance(ego_poly, npc_poly)
            min_dist = min(min_dist, dist)
            if dist <= thr.collision_distance_m:
                if first_t is None:
                    first_t = ego.stamp_sec
                events.append(
                    {
                        "time_sec": ego.stamp_sec,
                        "object_id": obj.object_id,
                        "label": obj.label,
                        "distance_m": dist,
                    }
                )
                break  # one collision event per ego sample
    if min_dist == float("inf"):
        min_dist = float("nan")
    # Deduplicate consecutive same object
    return {
        "flagged": bool(events),
        "events": events[:50],
        "event_count": len(events),
        "min_distance_m": min_dist,
        "first_time_sec": first_t,
        "status": "ok",
    }


def metric_out_of_boundary(
    series: BagSeries,
    map_path: Path | None,
    thr: Thresholds,
    footprint_local: list[tuple[float, float]],
) -> dict[str, Any]:
    """Lanelet footprint vs road_border; soft-fails if map libs missing."""
    ego_ds = downsample_ego(series.ego, thr.sample_dt)
    if not ego_ds:
        return {"flagged": False, "min_distance_m": float("nan"), "status": "no_ego"}
    if map_path is None:
        return {"flagged": False, "min_distance_m": float("nan"), "status": "map_not_provided"}

    try:
        import lanelet2.geometry
        from autoware_lanelet2_extension_python.projection import MGRSProjector
        from lanelet2.core import BasicPoint2d, BoundingBox2d
        from lanelet2.io import Origin, load
    except ImportError as exc:
        return {
            "flagged": False,
            "min_distance_m": float("nan"),
            "status": f"lanelet2_unavailable: {exc}",
        }

    map_path = map_path.expanduser().resolve()
    osm = map_path if map_path.suffix == ".osm" else map_path / "lanelet2_map.osm"
    if not osm.is_file():
        return {"flagged": False, "min_distance_m": float("nan"), "status": f"map_missing: {osm}"}

    try:
        projector = MGRSProjector(Origin(0.0, 0.0))
        lanelet_map = load(str(osm), projector)
    except Exception as exc:  # noqa: BLE001
        return {"flagged": False, "min_distance_m": float("nan"), "status": f"map_load_failed: {exc}"}

    min_signed = float("inf")
    worst_t = None
    violation_t = None
    try:
        for ego in ego_ds:
            poly = transform_local_polygon(footprint_local, ego.x, ego.y, ego.yaw_rad)
            # Use map point-in-lane approx: distance of corners to nearest lanelet
            for px, py in poly:
                pt = BasicPoint2d(px, py)
                # search radius
                box = BoundingBox2d(
                    BasicPoint2d(px - 30.0, py - 30.0), BasicPoint2d(px + 30.0, py + 30.0)
                )
                nearby = lanelet_map.laneletLayer.search(box)
                if not nearby:
                    # outside searchable map → treat as OOB
                    d = -1.0
                else:
                    d = min(lanelet2.geometry.distance(ll, pt) for ll in nearby)
                    # distance==0 means on/inside lanelet area-ish; use signed by inside check
                    inside = any(lanelet2.geometry.inside(ll, pt) for ll in nearby)
                    if not inside:
                        d = -abs(d) if d > 0 else d
                if d < min_signed:
                    min_signed = d
                    worst_t = ego.stamp_sec
                if d < -thr.oob_margin_m and violation_t is None:
                    violation_t = ego.stamp_sec
    except Exception as exc:  # noqa: BLE001
        return {
            "flagged": False,
            "min_distance_m": float("nan"),
            "status": f"oob_compute_failed: {exc}",
        }

    if min_signed == float("inf"):
        min_signed = float("nan")
    flagged = min_signed < -thr.oob_margin_m if not math.isnan(min_signed) else False
    return {
        "flagged": flagged,
        "min_distance_m": min_signed,
        "worst_time_sec": worst_t,
        "first_violation_sec": violation_t,
        "status": "ok",
    }


def compute_metrics(
    bag_path: Path,
    *,
    goal_x: float,
    goal_y: float,
    goal_yaw: float = 0.0,
    map_path: Path | None = None,
    thresholds: Thresholds | None = None,
    topics: TopicSet | None = None,
    vehicle_info_yaml: Path | None = None,
    output_json: Path | None = None,
    series: BagSeries | None = None,
) -> dict[str, Any]:
    thr = thresholds or Thresholds()
    topics = topics or TopicSet()
    if series is None:
        series = load_bag_series(
            bag_path,
            ego_topic=topics.ego_pose,
            objects_topic=topics.predicted_objects,
            velocity_topic=topics.vehicle_status_velocity,
            trajectory_topic=None,
            object_sample_dt=thr.sample_dt,
            skip_zero_size_objects=True,
        )
    footprint = load_vehicle_footprint(vehicle_info_yaml, margin=0.0)

    stuck = metric_stuck(series, goal_x, goal_y, thr)
    goal = metric_goal_stop(series, goal_x, goal_y, goal_yaw, thr, stuck=stuck)
    collision = metric_collision(series, thr, footprint)
    oob = metric_out_of_boundary(series, map_path, thr, footprint)

    overall_fail = bool(stuck.get("flagged") or collision.get("flagged"))
    if goal.get("status") == "ok" and goal.get("flagged"):
        overall_fail = True
    if oob.get("status") == "ok" and oob.get("flagged"):
        overall_fail = True

    result = {
        "pass": not overall_fail,
        "stuck_rate": stuck,
        "out_of_boundary": oob,
        "collision_rate": collision,
        "goal_stop_precision": goal,
        "meta": {
            "bag_path": str(bag_path),
            "ego_samples": len(series.ego),
            "object_samples": len(series.objects),
            "duration_sec": series.duration_sec,
            "goal": {"x": goal_x, "y": goal_y, "yaw": goal_yaw},
        },
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 2: compute metrics from output.bag")
    p.add_argument("--bag_path", type=Path, required=True)
    p.add_argument("--map_path", type=Path, default=None)
    p.add_argument("--goal_pose", nargs=3, type=float, metavar=("X", "Y", "YAW"), required=True)
    p.add_argument("--output", type=Path, default=None, help="metrics.json path")
    p.add_argument("--thresholds", type=Path, default=None)
    p.add_argument("--topics-yaml", type=Path, default=None)
    p.add_argument("--vehicle_info_yaml", type=Path, default=None)
    args = p.parse_args(argv)

    out = args.output
    if out is None:
        out = args.bag_path.expanduser().resolve()
        if out.is_file():
            out = out.parent
        out = out.parent / "metrics.json" if out.name == "output" else out / "metrics.json"

    result = compute_metrics(
        args.bag_path,
        goal_x=args.goal_pose[0],
        goal_y=args.goal_pose[1],
        goal_yaw=args.goal_pose[2],
        map_path=args.map_path,
        thresholds=load_thresholds(args.thresholds),
        topics=load_topics(args.topics_yaml),
        vehicle_info_yaml=args.vehicle_info_yaml,
        output_json=out,
    )
    print(f"[done] pass={result['pass']} wrote {out}")
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
