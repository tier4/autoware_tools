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

"""Localization, routing and engage — everything that talks to Autoware's API.

Thin wrappers over ``dp_multi_eval.routing.RoutingClient`` (one node per job) that
keep the message contract the drive loop reads: ``effective_goal_xy=x,y,yaw;`` for
the pose the planner accepted and ``effective_goal=NAME; skipped_goals=A|B;`` when
a refused stop was passed as a waypoint. The stall reposition still delegates to
diffusion_planner_batch_eval's route_setup; it is off by default.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Any

from dp_multi_eval.job_config import JobConfig


def _load_route_setup_module(env: dict[str, str]):
    prefix = subprocess.check_output(
        ["ros2", "pkg", "prefix", "diffusion_planner_batch_eval"],
        env=env,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    script = Path(prefix) / "lib" / "diffusion_planner_batch_eval" / "route_setup.py"
    if not script.is_file():
        raise FileNotFoundError(f"route_setup.py not found: {script}")
    spec = importlib.util.spec_from_file_location("dp_batch_route_setup", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import route_setup from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stops_and_order(cfg: JobConfig) -> tuple[list[Any], list[str]]:
    from dp_multi_eval.multi_goal import resolve_stop_points_csv
    from dp_multi_eval.routing import load_map_stops

    stops: list[Any] = []
    try:
        stops = load_map_stops(resolve_stop_points_csv(cfg.map_path, cfg.stop_points_csv))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] stop points unavailable ({exc}); routing without map stops")
    return stops, list(cfg.multi_goal_stop_order or [])


def setup_route(
    cfg: JobConfig,
    env: dict[str, str],
    *,
    goal_xy_yaw: tuple[float, float, float] | None = None,
    allow_goal_substitution: bool = True,
) -> tuple[bool, str]:
    """Spawn the ego at the recording's start and set the first goal.

    ``goal_xy_yaw`` None routes to the recording's last pose. A start within
    SPAWN_SNAP_M of a map stop spawns on the stop: the bag ego is often a metre
    off-lane and the planner then finds no lanelet under it.
    """
    if cfg.skip_route_setup:
        return True, "route_setup_skipped"
    from dp_multi_eval.routing import SPAWN_SNAP_M, bag_start_pose, get_client, nearest_stop

    try:
        bag = cfg.bag_path.expanduser().resolve()  # type: ignore[union-attr]
        start = bag_start_pose(bag)
        if start is None:
            return False, "route_setup_failed: no ego pose in the bag"
        stops, order = _stops_and_order(cfg)
        spawn, spawn_note = start, "spawn=bag_start"
        stop, d = nearest_stop(stops, start[0], start[1])
        if stop is not None and d <= SPAWN_SNAP_M:
            spawn, spawn_note = stop.pose, f"spawn={stop.name}({d:.1f}m)"
        goal = goal_xy_yaw
        if goal is None:
            from dp_multi_eval.multi_goal import bag_end_pose

            end = bag_end_pose(bag)
            if end is None:
                return False, "route_setup_failed: no ego pose in the bag"
            goal = end[:3]

        client = get_client(env)
        ok, msg = client.initialize_ego(spawn, timeout=float(cfg.route_setup_timeout_sec))
        print(f"[route] {spawn_note}: {msg}")
        if not ok:
            return False, f"route_setup_failed: {msg}"
        res = client.route(goal, stops, order, allow_walk=allow_goal_substitution, map_path=cfg.map_path)
        if not res.accepted:
            return False, f"route_setup_failed: {res.message}"
        message = f"{spawn_note}; {res.describe()}"
        print(f"Route setup succeeded: {message}")
        return True, message
    except Exception as exc:  # noqa: BLE001
        return False, f"route_setup_exception: {exc}"


def advance_route(
    cfg: JobConfig,
    env: dict[str, str],
    goal_xy_yaw: tuple[float, float, float],
    *,
    allow_goal_substitution: bool = True,
) -> tuple[bool, str]:
    """Clear the ARRIVED route and set the next leg's goal from where the ego stands."""
    print(f"[info] Advancing route to next goal ({goal_xy_yaw[0]:.1f}, {goal_xy_yaw[1]:.1f})")
    from dp_multi_eval.routing import get_client

    try:
        stops, order = _stops_and_order(cfg)
        res = get_client(env).route(
            goal_xy_yaw, stops, order, allow_walk=allow_goal_substitution, map_path=cfg.map_path
        )
        if not res.accepted:
            print(f"[error] advance failed: {res.message}")
            return False, f"advance_set_failed: {res.message}"
        message = f"{res.describe()}; advance_ok"
        print(f"[info] {message}")
        return True, message
    except Exception as exc:  # noqa: BLE001
        return False, f"advance_route_exception: {exc}"


def reposition_stuck_ego(
    cfg: JobConfig,
    env: dict[str, str],
    distance_m: float | None = None,
) -> tuple[bool, str]:
    """Shift planning-simulator ego forward while preserving the active route.

    ``distance_m`` 0.0 is an in-place recovery: no position is fabricated, the
    ego is re-initialized where it stands and handed its speed nudge. That is
    the right escape from a StopPointFixer hold, which motion releases and a
    teleport re-enters.
    """
    if distance_m is None:
        distance_m = float(cfg.stuck_reposition_forward_m)
        if distance_m <= 0.0:
            return False, "stuck_reposition_disabled"

    what = f"move ego {distance_m:.1f}m forward" if distance_m else "nudge ego in place"
    print(f"[warn] Attempting stuck recovery: {what}")
    old_env = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        module = _load_route_setup_module(env)
        if not hasattr(module, "shift_ego_longitudinal"):
            return False, "stuck_reposition_unsupported (rebuild diffusion_planner_batch_eval)"
        ok, message = module.shift_ego_longitudinal(
            distance_m,
            nudge_speed_mps=float(getattr(cfg, "reposition_nudge_mps", 0.0)),
            # Repositioning re-initializes localization, exactly as route setup does,
            # so it needs the same budget. The old 45 s cap was tight enough to fail
            # under two concurrent workers — "localization_initialize_timeout
            # (dist=5.00m)" with the ego never reaching the shifted pose.
            timeout_sec=float(cfg.route_setup_timeout_sec),
        )
        if ok:
            print(f"[info] {message}")
        else:
            print(f"[error] {message}")
        return bool(ok), str(message)
    except Exception as exc:  # noqa: BLE001
        return False, f"stuck_reposition_exception: {exc}"
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def engage_autonomous(env: dict[str, str], timeout_sec: float = 60.0) -> tuple[bool, str]:
    """Enable Autoware control and switch to AUTONOMOUS; confirms the mode is held."""
    from dp_multi_eval.routing import get_client

    try:
        ok, message = get_client(env).engage(float(timeout_sec))
    except Exception as exc:  # noqa: BLE001
        return False, f"engage_exception: {exc}"
    print(f"[info] engage: {message}" if ok else f"[error] engage: {message}")
    return bool(ok), str(message)


def close_routing(env: dict[str, str]) -> None:
    """Drop the job's routing node (called from the job's teardown)."""
    from dp_multi_eval.routing import close_routing as _close

    try:
        _close(env)
    except Exception:  # noqa: BLE001
        pass
