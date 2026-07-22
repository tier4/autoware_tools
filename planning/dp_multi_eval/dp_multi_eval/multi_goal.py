"""Extract multi-leg route goals from a source rosbag."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dp_multi_eval.bag_reader import EgoSample, load_bag_series, yaw_from_quat


GOAL_TOPICS = (
    "/planning/mission_planning/goal",
    "/planning/mission_planning/route_selector/main/goal",
    "/api/routing/set_route_points",  # not a topic; kept for docs only
)


@dataclass(frozen=True)
class RouteGoal:
    """One route destination in map frame."""

    x: float
    y: float
    yaw: float
    stamp_sec: float = 0.0
    source: str = "stop_segment"  # goal_topic | stop_segment | bag_end


def _xy_dist(a: RouteGoal | EgoSample, b: RouteGoal | EgoSample) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _goals_from_stop_segments(
    ego: list[EgoSample],
    *,
    speed_mps: float,
    min_sec: float,
    min_spacing_m: float,
) -> list[RouteGoal]:
    """Detect dwells (near-zero speed) and treat end-of-dwell poses as goals."""
    if not ego:
        return []

    goals: list[RouteGoal] = []
    dwell_start: int | None = None

    def _close_dwell(end_idx: int) -> None:
        nonlocal dwell_start
        if dwell_start is None:
            return
        t0 = ego[dwell_start].stamp_sec
        t1 = ego[end_idx].stamp_sec
        if (t1 - t0) < min_sec:
            dwell_start = None
            return
        # Use mid-dwell pose (more stable than last jittery sample)
        mid = dwell_start + (end_idx - dwell_start) // 2
        sample = ego[mid]
        cand = RouteGoal(
            x=sample.x,
            y=sample.y,
            yaw=sample.yaw_rad,
            stamp_sec=sample.stamp_sec,
            source="stop_segment",
        )
        if goals and _xy_dist(goals[-1], cand) < min_spacing_m:
            dwell_start = None
            return
        # Skip start-of-bag dwell (spawn)
        if not goals and mid < max(3, len(ego) // 50):
            dwell_start = None
            return
        goals.append(cand)
        dwell_start = None

    for i, sample in enumerate(ego):
        if sample.speed_mps <= speed_mps:
            if dwell_start is None:
                dwell_start = i
        else:
            if dwell_start is not None:
                _close_dwell(i - 1)
    if dwell_start is not None:
        _close_dwell(len(ego) - 1)

    return goals


def _goals_from_goal_topic(bag_path: Path) -> list[RouteGoal]:
    """Best-effort: unique PoseStamped goals published during the bag."""
    try:
        from dp_multi_eval.bag_reader import iter_topic_messages
    except Exception:  # noqa: BLE001
        return []

    topics = [
        "/planning/mission_planning/goal",
        "/planning/mission_planning/route_selector/main/goal",
    ]
    found: list[RouteGoal] = []
    try:
        for _topic, stamp_sec, msg in iter_topic_messages(bag_path, topics):
            pose = getattr(msg, "pose", None)
            if pose is None:
                continue
            pos = pose.position
            yaw = yaw_from_quat(pose.orientation)
            cand = RouteGoal(
                x=float(pos.x),
                y=float(pos.y),
                yaw=float(yaw),
                stamp_sec=float(stamp_sec),
                source="goal_topic",
            )
            if found and _xy_dist(found[-1], cand) < 2.0:
                found[-1] = cand  # keep latest near-duplicate
            else:
                found.append(cand)
    except Exception:  # noqa: BLE001
        return []
    return found


def resolve_stop_points_csv(map_path: Path, filename: str = "stop_points.csv") -> Path:
    """Resolve stop_points.csv under map_path (or an absolute file path)."""
    path = Path(filename).expanduser()
    if path.is_file():
        return path.resolve()
    return (Path(map_path).expanduser().resolve() / filename).resolve()


def load_stop_point_names(csv_path: Path) -> list[str]:
    """Return stop names in CSV row order."""
    return [s.name for s in _load_map_stop_points(csv_path)]


def _load_map_stop_points(csv_path: Path) -> list[Any]:
    """Load MapStopPoint rows; prefer batch_eval helper, else local CSV parse."""
    csv_path = csv_path.expanduser().resolve()
    try:
        from stop_points import load_stop_points  # type: ignore

        return list(load_stop_points(csv_path))
    except Exception:  # noqa: BLE001
        pass

    import csv

    if not csv_path.is_file():
        raise FileNotFoundError(f"Stop points file not found: {csv_path}")
    stops: list[SimpleStop] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            stops.append(
                SimpleStop(
                    name=name,
                    x=float(row["x"]),
                    y=float(row["y"]),
                    yaw=float(row["yaw"]),
                )
            )
    if not stops:
        raise ValueError(f"No stop points loaded from {csv_path}")
    return stops


@dataclass(frozen=True)
class SimpleStop:
    name: str
    x: float
    y: float
    yaw: float


def goals_from_stop_order(
    ordered_names: list[str],
    *,
    map_path: Path,
    stop_points_csv: str = "stop_points.csv",
) -> list[RouteGoal]:
    """Build route goals from an ordered list of stop names in stop_points.csv."""
    if not ordered_names:
        raise ValueError("multi_goal_stop_order is empty")
    csv_path = resolve_stop_points_csv(map_path, stop_points_csv)
    stops = _load_map_stop_points(csv_path)
    by_name = {s.name: s for s in stops}
    missing = [n for n in ordered_names if n not in by_name]
    if missing:
        raise ValueError(
            f"Unknown stop name(s) in multi_goal_stop_order: {missing}. "
            f"Available: {sorted(by_name)}"
        )
    goals: list[RouteGoal] = []
    for name in ordered_names:
        stop = by_name[name]
        goals.append(
            RouteGoal(
                x=float(stop.x),
                y=float(stop.y),
                yaw=float(stop.yaw),
                stamp_sec=0.0,
                source=f"stop_points:{name}",
            )
        )
    return goals


def get_bag_start_xy(bag_path: Path) -> tuple[float, float] | None:
    """Return the first ego (x, y) without scanning the whole bag."""
    try:
        from dp_multi_eval.bag_reader import open_bag_reader
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from rosbag2_py import StorageFilter
    except Exception:  # noqa: BLE001
        return None

    bag_path = bag_path.expanduser().resolve()
    try:
        reader = open_bag_reader(bag_path)
        type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
        topic = "/localization/kinematic_state"
        if topic not in type_map:
            return None
        reader.set_filter(StorageFilter(topics=[topic]))
        if not reader.has_next():
            return None
        _topic, data, _ = reader.read_next()
        msg = deserialize_message(data, get_message(type_map[topic]))
        pose = msg.pose.pose
        return float(pose.position.x), float(pose.position.y)
    except Exception:  # noqa: BLE001
        return None


def align_goals_to_bag_ego(
    goals: list[RouteGoal],
    start_xy: tuple[float, float] | None,
    *,
    near_stop_m: float = 15.0,
) -> list[RouteGoal]:
    """Pick the ordered stop list starting near bag ego, then follow list order.

    1. Find the stop closest to the bag start pose.
    2. If ego is already at that stop (within ``near_stop_m``), start from the
       *next* stop (avoid start≈goal route failures).
    3. Otherwise start from that nearest stop (drive there first).
    4. Keep the remainder of the user-selected order (including a later return
       to the same stop name).
    """
    if not goals or start_xy is None:
        return goals

    sx, sy = start_xy
    nearest_idx = min(
        range(len(goals)),
        key=lambda i: math.hypot(goals[i].x - sx, goals[i].y - sy),
    )
    nearest_dist = math.hypot(goals[nearest_idx].x - sx, goals[nearest_idx].y - sy)
    if nearest_dist < near_stop_m:
        start_idx = nearest_idx + 1
        reason = (
            f"near {goals[nearest_idx].source} ({nearest_dist:.1f}m) → next stop"
        )
    else:
        start_idx = nearest_idx
        reason = (
            f"closest {goals[nearest_idx].source} ({nearest_dist:.1f}m) → drive there"
        )

    if start_idx >= len(goals):
        print(
            f"[warn] Bag ego ({sx:.1f}, {sy:.1f}) is at/after the last stop "
            f"({goals[-1].source}); no remaining route legs"
        )
        return []

    kept = goals[start_idx:]
    print(
        f"[info] Aligned stop_points to bag ego ({sx:.1f}, {sy:.1f}): {reason}; "
        f"legs {start_idx + 1}→{len(goals)} of {len(goals)} "
        f"(first={kept[0].source}, last={kept[-1].source})"
    )
    return kept


def extract_route_goals(
    bag_path: Path,
    *,
    source: str = "auto",
    stop_speed_mps: float = 0.2,
    stop_min_sec: float = 8.0,
    min_spacing_m: float = 15.0,
    map_path: Path | None = None,
    stop_points_csv: str = "stop_points.csv",
    stop_order: list[str] | None = None,
) -> list[RouteGoal]:
    """Return ordered goals for multi-leg replay.

    ``source``:
      - ``goal_topic`` — only published goal PoseStampeds
      - ``stop_segments`` — ego dwell detection
      - ``stop_points`` — ordered names from ``stop_points.csv`` (web / config)
      - ``auto`` — goal topics if ≥1, else stop segments; always ensure bag-end goal
    """
    source = (source or "auto").strip().lower()

    if source == "stop_points":
        if map_path is None:
            raise ValueError("map_path is required when multi_goal_source=stop_points")
        goals = goals_from_stop_order(
            list(stop_order or []),
            map_path=map_path,
            stop_points_csv=stop_points_csv,
        )
        return align_goals_to_bag_ego(
            goals,
            get_bag_start_xy(bag_path),
            near_stop_m=max(5.0, float(min_spacing_m)),
        )

    bag_path = bag_path.expanduser().resolve()
    series = load_bag_series(bag_path)
    ego = series.ego
    if not ego:
        return []

    bag_end = RouteGoal(
        x=ego[-1].x,
        y=ego[-1].y,
        yaw=ego[-1].yaw_rad,
        stamp_sec=ego[-1].stamp_sec,
        source="bag_end",
    )

    goals: list[RouteGoal] = []
    if source in ("auto", "goal_topic"):
        goals = _goals_from_goal_topic(bag_path)
    if source == "stop_segments" or (source == "auto" and not goals):
        goals = _goals_from_stop_segments(
            ego,
            speed_mps=stop_speed_mps,
            min_sec=stop_min_sec,
            min_spacing_m=min_spacing_m,
        )

    # Always end at bag-end pose if last goal is far / missing
    if not goals:
        goals = [bag_end]
    elif _xy_dist(goals[-1], bag_end) >= min_spacing_m:
        goals.append(bag_end)
    else:
        # Replace last with bag_end (same place, more accurate stamp)
        goals[-1] = bag_end

    return goals


def goals_to_jsonable(goals: list[RouteGoal]) -> list[dict[str, Any]]:
    return [
        {
            "x": g.x,
            "y": g.y,
            "yaw": g.yaw,
            "stamp_sec": g.stamp_sec,
            "source": g.source,
        }
        for g in goals
    ]
