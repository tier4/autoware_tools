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


def extract_route_goals(
    bag_path: Path,
    *,
    source: str = "auto",
    stop_speed_mps: float = 0.2,
    stop_min_sec: float = 8.0,
    min_spacing_m: float = 15.0,
) -> list[RouteGoal]:
    """Return ordered goals for multi-leg replay.

    ``source``:
      - ``goal_topic`` — only published goal PoseStampeds
      - ``stop_segments`` — ego dwell detection
      - ``auto`` — goal topics if ≥1, else stop segments; always ensure bag-end goal
    """
    bag_path = bag_path.expanduser().resolve()
    source = (source or "auto").strip().lower()

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
