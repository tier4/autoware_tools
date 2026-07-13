"""Default topic names and YAML override loading."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PLANNING_FACTOR_TOPICS = [
    "/planning/planning_factors/modifier_obstacle_stop",
    "/planning/planning_factors/diffusion_planner",
    "/planning/planning_factors/stop_point_fixer",
]


@dataclass
class TopicSet:
    ego_pose: str = "/localization/kinematic_state"
    trajectory: str = "/planning/trajectory"
    predicted_objects: str = "/perception/object_recognition/tracking/objects"
    tf: str = "/tf"
    tf_static: str = "/tf_static"
    vehicle_status_velocity: str = "/vehicle/status/velocity_status"
    vehicle_status_steering: str = "/vehicle/status/steering_status"
    control_command: str = "/control/command/control_cmd"
    acceleration: str = "/localization/acceleration"
    route_state: str = "/api/routing/state"
    turn_indicators_status: str = "/vehicle/status/turn_indicators_status"
    turn_indicators_cmd: str = "/planning/turn_indicators_cmd"
    planning_factors: list[str] = field(
        default_factory=lambda: list(DEFAULT_PLANNING_FACTOR_TOPICS)
    )

    def record_list(self) -> list[str]:
        """Topics written into output.bag (Phase 1)."""
        topics = [
            self.ego_pose,
            self.trajectory,
            self.predicted_objects,
            self.tf,
            self.tf_static,
            self.vehicle_status_velocity,
            self.vehicle_status_steering,
            self.control_command,
            self.acceleration,
            self.turn_indicators_status,
            self.turn_indicators_cmd,
            *self.planning_factors,
        ]
        # Preserve order, drop duplicates
        seen: set[str] = set()
        out: list[str] = []
        for t in topics:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out


def load_topics(path: Path | None) -> TopicSet:
    if path is None:
        return TopicSet()
    path = path.expanduser().resolve()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(TopicSet)}
    kwargs: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in known:
            continue
        if k == "planning_factors":
            if isinstance(v, list):
                kwargs[k] = [str(x) for x in v]
            elif v:
                kwargs[k] = [str(v)]
        else:
            kwargs[k] = str(v)
    return TopicSet(**kwargs)
