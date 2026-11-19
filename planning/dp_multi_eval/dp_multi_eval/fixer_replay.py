# Copyright 2026 TIER IV, Inc.
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

"""Recompute StopPointFixer's decisions from a recorded bag.

The fixer's two rules are pure functions of the trajectory it receives and the
ego's twist, so with the modifier's input topic in the recording its behaviour
can be reproduced offline exactly — including the counterfactual the stack
cannot answer live: which forced stops each rule is individually responsible
for. The stack itself stays untouched; this is measurement, not modification.

The port mirrors ``stop_point_fixer.cpp`` line for line:

  - ego moving (3-D speed > velocity_threshold) → never modify
  - is_long_stop_trajectory: walking the points in order, reaching
    time_from_start > min_stop_duration before any point exceeds
    velocity_threshold means a stop; a fast point first means a plan; an
    exhausted list means a stop
  - is_stop_point_close_to_ego: ego to the *last* point < min_distance_threshold
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FixerParams:
    """Defaults from autoware_launch's neural_net_planner trajectory_modifier.param.yaml."""

    velocity_threshold: float = 0.25
    min_distance_threshold: float = 1.0
    min_stop_duration: float = 1.0
    force_stop_long_stopped_trajectories: bool = True
    force_stop_close_stopped_trajectories: bool = True


@dataclass
class TrajPoint:
    x: float
    y: float
    v: float
    t: float  # time_from_start, seconds


def is_long_stop_trajectory(points: list[TrajPoint], p: FixerParams) -> bool:
    if not points or not p.force_stop_long_stopped_trajectories:
        return False
    for pt in points:
        if pt.t > p.min_stop_duration:
            return True
        if pt.v > p.velocity_threshold:
            return False
    return True


def is_stop_point_close_to_ego(
    points: list[TrajPoint], ego_x: float, ego_y: float, p: FixerParams
) -> bool:
    if not points or not p.force_stop_close_stopped_trajectories:
        return False
    last = points[-1]
    return math.hypot(last.x - ego_x, last.y - ego_y) < p.min_distance_threshold


def fix_reason(
    points: list[TrajPoint],
    ego_speed: float,
    ego_x: float,
    ego_y: float,
    p: FixerParams,
) -> str | None:
    """Why the fixer would replace this trajectory, or None if it would not.

    Rules are evaluated in the C++ order (close first) so the named reason is
    the one the plugin itself acted on.
    """
    if not points:
        return None
    if ego_speed > p.velocity_threshold:
        return None
    if is_stop_point_close_to_ego(points, ego_x, ego_y, p):
        return "close_to_ego"
    if is_long_stop_trajectory(points, p):
        return "long_stop"
    return None


def replay_fixer(
    bag_path: Path,
    input_topic: str,
    ego_topic: str,
    params: FixerParams | None = None,
) -> dict[str, Any]:
    """Walk one recorded run and re-decide every fixer call.

    Returns per-rule counts and the two counterfactuals: how many replacements
    each rule is solely responsible for — the number that says what flipping one
    flag would have changed, measured instead of guessed.
    """
    p = params or FixerParams()
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import StorageFilter

    from dp_multi_eval.bag_reader import _type_map, open_bag_reader

    reader = open_bag_reader(bag_path)
    types = _type_map(reader)
    if input_topic not in types:
        return {"status": "unavailable", "detail": f"{input_topic} was not recorded"}
    reader.set_filter(StorageFilter(topics=[input_topic, ego_topic]))

    odom_t: list[float] = []
    odom_v: list[tuple[float, float, float]] = []  # (speed3d, x, y)
    inputs: list[tuple[float, list[list[TrajPoint]]]] = []
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        t = stamp_ns * 1e-9
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == ego_topic:
            tw = msg.twist.twist.linear
            pos = msg.pose.pose.position
            odom_t.append(t)
            odom_v.append((math.sqrt(tw.x**2 + tw.y**2 + tw.z**2), pos.x, pos.y))
        else:
            cands = []
            for cand in msg.candidate_trajectories:
                cands.append(
                    [
                        TrajPoint(
                            x=pt.pose.position.x,
                            y=pt.pose.position.y,
                            v=pt.longitudinal_velocity_mps,
                            t=pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9,
                        )
                        for pt in pt_list
                    ]
                    if (pt_list := list(cand.points))
                    else []
                )
            inputs.append((t, cands))

    if not inputs:
        return {"status": "unavailable", "detail": "no candidate trajectories in the bag"}
    if not odom_t:
        return {"status": "unavailable", "detail": f"{ego_topic} not in the bag"}

    counts = {"messages": len(inputs), "fixed": 0, "close_to_ego": 0, "long_stop": 0}
    only_long = 0
    only_close = 0
    for t, cands in inputs:
        i = min(bisect.bisect_left(odom_t, t), len(odom_t) - 1)
        speed, ex, ey = odom_v[i]
        reasons = {fix_reason(c, speed, ex, ey, p) for c in cands if c}
        reasons.discard(None)
        if not reasons:
            continue
        counts["fixed"] += 1
        for r in reasons:
            counts[r] += 1
        # A replacement survives one flag being off only if the *other* rule
        # also fires; solely-responsible counts are what a flag flip changes.
        no_long = FixerParams(**{**p.__dict__, "force_stop_long_stopped_trajectories": False})
        no_close = FixerParams(**{**p.__dict__, "force_stop_close_stopped_trajectories": False})
        if not any(fix_reason(c, speed, ex, ey, no_long) for c in cands if c):
            only_long += 1
        if not any(fix_reason(c, speed, ex, ey, no_close) for c in cands if c):
            only_close += 1

    counts["fixed_ratio"] = round(counts["fixed"] / counts["messages"], 4)
    counts["solely_long_stop"] = only_long
    counts["solely_close_to_ego"] = only_close
    counts["status"] = "ok"
    return counts
