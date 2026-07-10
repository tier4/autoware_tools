"""Default topic names and YAML override loading."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


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

    def record_list(self) -> list[str]:
        """Topics written into output.bag (Phase 1)."""
        return [
            self.ego_pose,
            self.trajectory,
            self.predicted_objects,
            self.tf,
            self.tf_static,
            self.vehicle_status_velocity,
            self.vehicle_status_steering,
            self.control_command,
            self.acceleration,
        ]


def load_topics(path: Path | None) -> TopicSet:
    if path is None:
        return TopicSet()
    path = path.expanduser().resolve()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(TopicSet)}
    kwargs = {k: str(v) for k, v in raw.items() if k in known}
    return TopicSet(**kwargs)
