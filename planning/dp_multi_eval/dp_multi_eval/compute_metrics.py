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
    goal_lateral_tolerance_m: float = 2.0
    goal_longitudinal_tolerance_m: float = 2.0
    # Clearance to road_border/curbstone: OOB if footprint crosses or dist < margin.
    oob_margin_m: float = 0.0
    oob_search_radius_m: float = 80.0
    collision_distance_m: float = 0.1
    sample_dt: float = 0.2
    # Lanelet2 LineString ``type`` attributes treated as uncrossable borders.
    oob_boundary_types: tuple[str, ...] = ("road_border", "curbstone")


METRIC_DESCRIPTIONS = {
    "stuck_rate": (
        "Fraction of scenario time the ego is nearly stopped (speed ≤ stuck_speed_mps) "
        "away from the goal for contiguous stretches ≥ stuck_duration_sec. "
        "rate = total_stuck_duration / bag_duration."
    ),
    "collision_rate": (
        "Fraction of downsampled ego frames where the ego footprint is within "
        "collision_distance_m of a vehicle-class NPC (CAR/TRUCK/BUS/TRAILER/MOTORCYCLE/BICYCLE). "
        "rate = colliding_frames / ego_frames. Pedestrians and UNKNOWN are excluded."
    ),
    "out_of_boundary": (
        "Fraction of downsampled ego frames where the ego footprint crosses an uncrossable "
        "Lanelet2 LineString (type in oob_boundary_types, default road_border + curbstone), "
        "or comes within oob_margin_m of one. "
        "rate = oob_frames / ego_frames. Requires a loaded lanelet map."
    ),
    "goal_stop_precision": (
        "Average stop pose error in the goal frame over low-speed samples "
        "(speed ≤ goal_stop_speed_mps) within goal_tolerance_m of the goal. "
        "Reports position, lateral, longitudinal [m] and heading [deg]. "
        "Fails when |lateral|, |longitudinal|, or position exceeds the configured tolerances. "
        "Scenes flagged as stuck are excluded (status=excluded_stuck)."
    ),
}


def load_thresholds(path: Path | None) -> Thresholds:
    if path is None or not path.expanduser().is_file():
        return Thresholds()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in Thresholds.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            continue
        if key == "oob_boundary_types":
            if isinstance(value, (list, tuple)):
                kwargs[key] = tuple(str(v) for v in value)
            else:
                kwargs[key] = (str(value),)
        else:
            kwargs[key] = float(value)
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
    """Stuck time rate: near-zero speed away from goal lasting > stuck_duration_sec."""
    samples = series.ego
    duration = series.duration_sec
    if not samples:
        if not series.velocity_mps:
            return {
                "flagged": False,
                "rate": 0.0,
                "duration_sec": 0.0,
                "bag_duration_sec": 0.0,
                "events": [],
                "status": "no_ego_data",
                "description": METRIC_DESCRIPTIONS["stuck_rate"],
            }

    events = find_stuck_events(samples, goal_x, goal_y, thr)
    total = sum(e["duration_sec"] for e in events)
    rate = (total / duration) if duration > 1e-9 else 0.0
    return {
        "flagged": bool(events),
        "rate": rate,
        "duration_sec": total,
        "bag_duration_sec": duration,
        "events": events,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["stuck_rate"],
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
    """Average stop precision (position / lateral / longitudinal / heading) near goal.

    Scenes flagged as stuck (away from goal) are excluded from goal precision.
    """
    empty = {
        "flagged": True,
        "position_error_m": float("nan"),
        "lateral_m": float("nan"),
        "longitudinal_m": float("nan"),
        "abs_lateral_m": float("nan"),
        "abs_longitudinal_m": float("nan"),
        "heading_error_deg": float("nan"),
        "stop_time_sec": float("nan"),
        "stop_speed_mps": float("nan"),
        "sample_count": 0,
        "duration_at_goal_sec": 0.0,
        "status": "no_ego_data",
        "description": METRIC_DESCRIPTIONS["goal_stop_precision"],
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
    avg_abs_lat = sum(abs(v) for v in laterals) / n
    avg_abs_lon = sum(abs(v) for v in longitudinals) / n
    avg_heading = sum(headings) / n
    avg_speed = sum(speeds) / n
    duration_at_goal = at_goal[-1].stamp_sec - at_goal[0].stamp_sec

    flagged = (
        avg_pos > thr.goal_tolerance_m
        or avg_abs_lat > thr.goal_lateral_tolerance_m
        or avg_abs_lon > thr.goal_longitudinal_tolerance_m
    )

    return {
        "flagged": flagged,
        "position_error_m": avg_pos,
        "lateral_m": avg_lat,
        "longitudinal_m": avg_lon,
        "abs_lateral_m": avg_abs_lat,
        "abs_longitudinal_m": avg_abs_lon,
        "heading_error_deg": avg_heading,
        "max_position_error_m": max(positions),
        "max_abs_lateral_m": max(abs(v) for v in laterals),
        "max_abs_longitudinal_m": max(abs(v) for v in longitudinals),
        "max_heading_error_deg": max(abs(h) for h in headings),
        "stop_time_sec": at_goal[-1].stamp_sec,
        "stop_speed_mps": avg_speed,
        "sample_count": n,
        "duration_at_goal_sec": duration_at_goal,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["goal_stop_precision"],
    }


def metric_collision(
    series: BagSeries,
    thr: Thresholds,
    footprint_local: list[tuple[float, float]],
) -> dict[str, Any]:
    ego_ds = downsample_ego(series.ego, thr.sample_dt)
    if not ego_ds:
        return {
            "flagged": False,
            "rate": 0.0,
            "events": [],
            "event_count": 0,
            "frame_count": 0,
            "min_distance_m": float("nan"),
            "status": "no_ego",
            "description": METRIC_DESCRIPTIONS["collision_rate"],
        }

    # Index objects by rounded time for fast lookup
    objs = series.objects
    oi = 0
    events: list[dict[str, Any]] = []
    min_dist = float("inf")
    first_t = None
    colliding_frames = 0

    for ego in ego_ds:
        # advance object cursor to nearby stamps
        while oi < len(objs) and objs[oi].stamp_sec < ego.stamp_sec - thr.sample_dt:
            oi += 1
        j = oi
        ego_poly = transform_local_polygon(footprint_local, ego.x, ego.y, ego.yaw_rad)
        frame_hit = False
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
                if not frame_hit:
                    colliding_frames += 1
                    frame_hit = True
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
    n_frames = len(ego_ds)
    rate = colliding_frames / n_frames if n_frames else 0.0
    return {
        "flagged": colliding_frames > 0,
        "rate": rate,
        "events": events[:50],
        "event_count": colliding_frames,
        "frame_count": n_frames,
        "min_distance_m": min_dist,
        "first_time_sec": first_t,
        "status": "ok",
        "description": METRIC_DESCRIPTIONS["collision_rate"],
    }


def _linestring_type(linestring: Any) -> str:
    attrs = getattr(linestring, "attributes", None)
    if attrs is None or "type" not in attrs:
        return ""
    return str(attrs["type"])


def _orientation(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> bool:
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
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


def _point_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _footprint_boundary_hit(
    footprint: list[tuple[float, float]],
    segments: list[tuple[float, float, float, float, str]],
    *,
    ego_x: float,
    ego_y: float,
    search_radius_m: float,
    margin_m: float,
) -> tuple[bool, float, str | None]:
    """Return (hit, min_distance_m, boundary_type)."""
    min_dist = float("inf")
    hit_type: str | None = None
    crossed = False
    for x1, y1, x2, y2, btype in segments:
        if (
            math.hypot(x1 - ego_x, y1 - ego_y) > search_radius_m
            and math.hypot(x2 - ego_x, y2 - ego_y) > search_radius_m
        ):
            continue
        for i in range(len(footprint)):
            p1 = footprint[i]
            p2 = footprint[(i + 1) % len(footprint)]
            if _segments_intersect(p1, p2, (x1, y1), (x2, y2)):
                crossed = True
                hit_type = btype
                min_dist = 0.0
        for px, py in footprint:
            d = _point_segment_distance(px, py, x1, y1, x2, y2)
            if d < min_dist:
                min_dist = d
                if not crossed:
                    hit_type = btype
    if min_dist == float("inf"):
        return False, float("nan"), None
    hit = crossed or (margin_m > 0.0 and min_dist < margin_m)
    return hit, min_dist, hit_type


def _extract_boundary_segments(
    lanelet_map: Any, boundary_types: set[str]
) -> list[tuple[float, float, float, float, str]]:
    segments: list[tuple[float, float, float, float, str]] = []
    for linestring in lanelet_map.lineStringLayer:
        btype = _linestring_type(linestring)
        if btype not in boundary_types:
            continue
        points = [(float(p.x), float(p.y)) for p in linestring]
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            segments.append((x1, y1, x2, y2, btype))
    return segments


def metric_out_of_boundary(
    series: BagSeries,
    map_path: Path | None,
    thr: Thresholds,
    footprint_local: list[tuple[float, float]],
) -> dict[str, Any]:
    """Ego footprint vs road_border/curbstone LineStrings (batch-eval style)."""
    desc = METRIC_DESCRIPTIONS["out_of_boundary"]
    boundary_types = tuple(thr.oob_boundary_types) or ("road_border", "curbstone")
    ego_ds = downsample_ego(series.ego, thr.sample_dt)
    empty = {
        "flagged": False,
        "rate": 0.0,
        "min_distance_m": float("nan"),
        "frame_count": len(ego_ds),
        "oob_frame_count": 0,
        "boundary_types": list(boundary_types),
        "boundary_segment_count": 0,
        "description": desc,
    }
    if not ego_ds:
        return {**empty, "frame_count": 0, "status": "no_ego"}
    if map_path is None:
        return {**empty, "status": "map_not_provided"}

    try:
        from autoware_lanelet2_extension_python.projection import MGRSProjector
        from lanelet2.io import Origin, load
    except ImportError as exc:
        return {**empty, "status": f"lanelet2_unavailable: {exc}"}

    map_path = map_path.expanduser().resolve()
    osm = map_path if map_path.suffix == ".osm" else map_path / "lanelet2_map.osm"
    if not osm.is_file():
        return {**empty, "status": f"map_missing: {osm}"}

    try:
        projector = MGRSProjector(Origin(0.0, 0.0))
        # Prefer map_projector_info when present (same folder as osm / map dir).
        map_dir = map_path if map_path.is_dir() else map_path.parent
        projector_info = map_dir / "map_projector_info.yaml"
        if projector_info.is_file():
            info = yaml.safe_load(projector_info.read_text(encoding="utf-8")) or {}
            ptype = str(info.get("projector_type", "MGRS"))
            if ptype in ("TransverseMercator", "LocalCartesianUTM"):
                from autoware_lanelet2_extension_python._autoware_lanelet2_extension_python_boost_python_projection import (  # noqa: E501
                    TransverseMercatorProjector,
                )

                origin = info["map_origin"]
                projector = TransverseMercatorProjector(
                    Origin(float(origin["latitude"]), float(origin["longitude"]))
                )
        lanelet_map = load(str(osm), projector)
    except Exception as exc:  # noqa: BLE001
        return {**empty, "status": f"map_load_failed: {exc}"}

    try:
        segments = _extract_boundary_segments(lanelet_map, set(boundary_types))
    except Exception as exc:  # noqa: BLE001
        return {**empty, "status": f"boundary_extract_failed: {exc}"}

    if not segments:
        return {
            **empty,
            "status": "no_boundary_linestrings",
            "boundary_segment_count": 0,
        }

    min_dist_overall = float("inf")
    worst_t = None
    violation_t = None
    hit_type_first: str | None = None
    oob_frames = 0
    try:
        for ego in ego_ds:
            poly = transform_local_polygon(footprint_local, ego.x, ego.y, ego.yaw_rad)
            hit, dist, btype = _footprint_boundary_hit(
                poly,
                segments,
                ego_x=ego.x,
                ego_y=ego.y,
                search_radius_m=thr.oob_search_radius_m,
                margin_m=thr.oob_margin_m,
            )
            if isinstance(dist, float) and not math.isnan(dist) and dist < min_dist_overall:
                min_dist_overall = dist
                worst_t = ego.stamp_sec
            if hit:
                oob_frames += 1
                if violation_t is None:
                    violation_t = ego.stamp_sec
                    hit_type_first = btype
    except Exception as exc:  # noqa: BLE001
        return {
            **empty,
            "boundary_segment_count": len(segments),
            "status": f"oob_compute_failed: {exc}",
        }

    if min_dist_overall == float("inf"):
        min_dist_overall = float("nan")
    n_frames = len(ego_ds)
    rate = oob_frames / n_frames if n_frames else 0.0
    return {
        "flagged": oob_frames > 0,
        "rate": rate,
        "min_distance_m": min_dist_overall,
        "worst_time_sec": worst_t,
        "first_violation_sec": violation_t,
        "first_boundary_type": hit_type_first,
        "frame_count": n_frames,
        "oob_frame_count": oob_frames,
        "boundary_types": list(boundary_types),
        "boundary_segment_count": len(segments),
        "status": "ok",
        "description": desc,
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
        "metric_descriptions": dict(METRIC_DESCRIPTIONS),
        "meta": {
            "bag_path": str(bag_path),
            "ego_samples": len(series.ego),
            "object_samples": len(series.objects),
            "duration_sec": series.duration_sec,
            "goal": {"x": goal_x, "y": goal_y, "yaw": goal_yaw},
            "thresholds": asdict(thr),
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
