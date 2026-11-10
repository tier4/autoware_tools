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
    # The single merged stream the planner reads as its turn-indicator input
    # in the replay (recorded signal when the recording has one, psim's echo
    # otherwise); see ament_overlay.set_dp_turn_indicator_input. Recorded so
    # the model's actual input is in the bag, not inferred from two writers.
    dp_turn_indicators_input: str = "/dp_multi_eval/turn_indicators_status"
    # Control-chain diagnostics. control_command alone is the vehicle_cmd_gate's
    # *output*: when a run records a driving trajectory next to a constant-decel
    # command, these three say which layer disagreed — the follower's raw command,
    # the mode the gate was in, and the mode the system reported. Without them a
    # braking gate is indistinguishable from a braking controller, and that exact
    # ambiguity cost a debugging session. All are low-rate.
    control_command_raw: str = "/control/trajectory_follower/control_cmd"
    operation_mode_state: str = "/api/operation_mode/state"
    control_mode_report: str = "/vehicle/status/control_mode"
    # Objects dominate the recording: the reproducer republishes the whole scene at
    # the source bag's odom rate (~48 Hz measured), and that one topic came to 88% of
    # a 699 MB output bag. Nothing consumes it at that rate — metrics sample objects
    # at thresholds.sample_dt (5 Hz) and the preview at video_sample_dt. Record a
    # throttled copy instead; 0 disables throttling and records the raw topic.
    objects_record_hz: float = 10.0
    objects_throttled_topic: str = "/dp_multi_eval/tracking/objects_throttled"

    def objects_record_name(self) -> str:
        """Topic the recorder writes objects under."""
        if float(self.objects_record_hz) > 0.0:
            return self.objects_throttled_topic
        return self.predicted_objects

    def objects_read_candidates(self) -> list[str]:
        """Names objects may appear under, newest scheme first.

        Bags recorded before throttling carry the raw topic, so both are tried and
        an older result stays readable.
        """
        out = [self.objects_throttled_topic, self.predicted_objects]
        return [t for i, t in enumerate(out) if t and t not in out[:i]]
    planning_factors: list[str] = field(
        default_factory=lambda: list(DEFAULT_PLANNING_FACTOR_TOPICS)
    )
    # The trajectory_modifier's input and output. `trajectory` alone is the end of
    # the chain, and it cannot say who put a stop there: a run failed as "stuck"
    # while the modifier was replacing the model's 80-point plans with 3-point
    # stop stubs, and the recording had no way to show it. With the input
    # recorded, the modifier's decision can be recomputed offline exactly — both
    # of its trigger rules are pure functions of this topic plus odometry — so
    # "the model stopped" and "the model was stopped" separate by measurement,
    # not by inference. ~5 Hz each; no throttling needed.
    # Names wired in autoware_launch/launch/e2e_components/diffusion_planner.launch.xml
    # (the modifier launch file's defaults are overridden there). A third stage, the
    # trajectory_optimizer, sits between the modifier and /planning/trajectory, so
    # recording both sides of the modifier is the only way to isolate its effect.
    # The replayed light state. Three bags stalled at the same signalized left
    # turn and nothing in the recording could say which phase the replay was
    # showing at the time — "waited at red" and "latched through green" are
    # different verdicts on the same stationary ego. ~7 Hz, tiny.
    traffic_signals: str = "/perception/traffic_light_recognition/traffic_signals"
    modifier_input_trajectories: str = (
        "/planning/generator/diffusion_planner/candidate_trajectories"
    )
    modifier_output_trajectories: str = (
        "/planning/generator/diffusion_planner/modified_candidate_trajectories"
    )

    def record_list(self) -> list[str]:
        """Topics written into output.bag (Phase 1)."""
        topics = [
            self.ego_pose,
            self.trajectory,
            self.objects_record_name(),
            # What the planner actually consumes (predicted paths included):
            # the only way to see whether an input, not the model, caused a stop.
            "/perception/object_recognition/objects",
            self.tf,
            self.tf_static,
            self.vehicle_status_velocity,
            self.vehicle_status_steering,
            self.control_command,
            self.acceleration,
            self.turn_indicators_status,
            self.turn_indicators_cmd,
            self.dp_turn_indicators_input,
            self.control_command_raw,
            self.operation_mode_state,
            self.control_mode_report,
            self.traffic_signals,
            self.modifier_input_trajectories,
            self.modifier_output_trajectories,
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
