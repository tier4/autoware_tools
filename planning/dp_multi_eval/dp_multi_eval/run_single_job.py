"""Phase 1 — single evaluation job (psim + reproducer + bag record).

Importable as ``from dp_multi_eval.run_single_job import run_single_job``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.ament_overlay import env_with_overlay, make_model_overlay
from dp_multi_eval.job_status import utc_now_iso, write_job_status
from dp_multi_eval.process_utils import close_log_handle, popen, popen_log_file, stop_process_group
from dp_multi_eval.scenario_sim import (
    build_record_command_sim_time,
    build_scenario_runner_command,
    scenario_test_runner_available,
)
from dp_multi_eval.topics import TopicSet, load_topics


def update_job_progress(
    output_dir: Path,
    *,
    phase: str,
    start_time: str,
    domain_id: int,
    bag_path: str | None = None,
    scenario_path: str | None = None,
    model_config: str | None = None,
) -> None:
    status_path = output_dir / "job_status.json"
    phase_start_time = utc_now_iso()
    if status_path.is_file():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if existing.get("phase") == phase and existing.get("phase_start_time"):
                phase_start_time = str(existing["phase_start_time"])
        except (json.JSONDecodeError, OSError):
            pass
    write_job_status(
        output_dir,
        {
            "status": "running",
            "phase": phase,
            "start_time": start_time,
            "phase_start_time": phase_start_time,
            "end_time": None,
            "error": None,
            "domain_id": domain_id,
            "bag_path": bag_path,
            "scenario_path": scenario_path,
            "model_config": model_config,
        },
    )


@dataclass
class JobConfig:
    model_config: Path
    output_dir: Path
    domain_id: int
    bag_path: Path | None = None
    scenario_path: Path | None = None
    mode: str = "reproducer"  # reproducer | scenario_simulator
    map_path: Path = Path("/opt/autoware/maps")
    vehicle_model: str = "lv828l"
    sensor_model: str = "aip_x2_gen2"
    vehicle_id: str | None = None
    topics: TopicSet = field(default_factory=TopicSet)
    end_condition: str = "route_arrived"  # route_arrived | bag_duration
    bag_duration_margin_sec: float = 30.0
    route_timeout_sec: float = 900.0
    psim_startup_sec: float = 120.0
    route_setup_timeout_sec: float = 90.0
    post_arrival_sec: float = 3.0
    auto_engage_timeout_sec: float = 60.0
    perception_warmup_sec: float = 5.0
    perception_ready_timeout_sec: float = 45.0
    perception_ready_min_objects: int = 1
    perception_ready_stable_sec: float = 2.0
    # Fail the job instead of engaging Auto with empty objects (common when the
    # full input bag is still loading into perception_reproducer).
    require_perception_ready: bool = True
    # Extra wall-time budget for large bags: timeout =
    # max(perception_ready_timeout_sec, bag_load_base_sec + bag_duration * bag_load_duration_factor).
    bag_load_base_sec: float = 120.0
    bag_load_duration_factor: float = 1.0
    bag_load_timeout_cap_sec: float = 3600.0
    # perception_reproducer -r: 0 = always nearest bag pose (needed when ego
    # diverges from the recording; 1.5 leaves objects empty off-path).
    reproducer_search_radius_m: float = 0.0
    reproducer_cool_down_sec: float = 80.0
    # Multi-goal bag replay (bus stop → arrive → next goal). Off = single start→end route.
    multi_goal: bool = False
    multi_goal_source: str = "auto"  # auto | goal_topic | stop_segments | stop_points
    multi_goal_stop_speed_mps: float = 0.2
    multi_goal_stop_min_sec: float = 8.0
    multi_goal_min_spacing_m: float = 15.0
    multi_goal_stop_order: list[str] | None = None
    stop_points_csv: str = "stop_points.csv"
    # When stuck (near-zero speed), publish left/right blinker to DP input to try
    # triggering a new trajectory (experimental recovery for lane-change stalls).
    stuck_blinker_nudge: bool = True
    stuck_blinker_speed_mps: float = 0.15
    stuck_blinker_trigger_sec: float = 12.0
    stuck_blinker_hold_sec: float = 3.0
    stuck_blinker_cooldown_sec: float = 25.0
    # Abort a leg if ego stays near-stopped this long when forward-shift is disabled.
    stuck_abort_sec: float = 90.0
    # While stopped (red light / lead vehicle), shift ego forward and continue.
    # trigger_sec: how long to wait stopped before each shift. forward_m: 0 disables.
    stuck_reposition_trigger_sec: float = 30.0
    stuck_reposition_forward_m: float = 5.0
    # 0 = unlimited (bounded by the current leg wait timeout).
    stuck_reposition_max_count: int = 0
    # Mission planner can remain SET when the bus stops just short of ARRIVED.
    multi_goal_arrival_tolerance_m: float = 5.0
    # Cap each multi-goal ARRIVED wait (full bag_duration is only the overall budget).
    multi_goal_leg_timeout_sec: float = 300.0
    # Low-CPU web live view: sample ego pose → live_drive.json for dashboard.
    live_web_monitor: bool = True
    live_web_sample_sec: float = 2.0
    live_drive_root: Path | None = None  # usually results_root (next to dashboard.html)
    # Scenario Simulator v2
    architecture_type: str = "awf/universe/20250130"
    scenario_timeout_sec: float = 300.0
    scenario_record_warmup_sec: float = 15.0
    dry_run: bool = False
    skip_route_setup: bool = False
    rviz: bool = False
    planning_setting: str = "diffusion_planner"


@dataclass
class JobResult:
    status: str  # done | failed
    start_time: str
    end_time: str
    error: str | None = None
    output_bag: str | None = None
    duration_sec: float = 0.0


def get_bag_duration_sec(bag_path: Path) -> float | None:
    """Best-effort duration from metadata.yaml or ros2 bag info."""
    bag_path = bag_path.expanduser().resolve()
    meta_candidates = []
    if bag_path.is_dir():
        meta_candidates.append(bag_path / "metadata.yaml")
    else:
        meta_candidates.append(bag_path.parent / "metadata.yaml")
        meta_candidates.append(bag_path.with_suffix("") / "metadata.yaml")

    for meta in meta_candidates:
        if not meta.is_file():
            continue
        try:
            raw = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            info = raw.get("rosbag2_bagfile_information") or raw
            duration = info.get("duration")
            if isinstance(duration, dict) and "nanoseconds" in duration:
                return float(duration["nanoseconds"]) * 1e-9
            # Some metadata store starting/ending time
            start = info.get("starting_time", {}).get("nanoseconds_since_epoch")
            # Fallback: sum files duration if present
            files = info.get("files") or info.get("relative_file_paths")
            if files and isinstance(files, list) and isinstance(files[0], dict):
                ns = files[0].get("duration", {}).get("nanoseconds")
                if ns is not None:
                    return float(ns) * 1e-9
            if start is not None:
                # cannot compute without end; try ros2
                break
        except (OSError, yaml.YAMLError, TypeError, ValueError, KeyError):
            continue

    if shutil.which("ros2") is None:
        return None
    try:
        out = subprocess.check_output(
            ["ros2", "bag", "info", str(bag_path)],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30.0,
        )
        for line in out.splitlines():
            if "Duration:" in line:
                # e.g. Duration: 123.456s
                token = line.split("Duration:", 1)[1].strip().rstrip("s")
                return float(token.split()[0])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None
    return None


def _ros_call_env(env: dict[str, str]) -> dict[str, str]:
    return {**env, "ROS2_DISABLE_DAEMON": "1"}


def _list_ros_nodes(env: dict[str, str], timeout_sec: float = 8.0) -> set[str]:
    out = subprocess.check_output(
        ["ros2", "node", "list"],
        env=_ros_call_env(env),
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=timeout_sec,
    )
    return set(line.strip() for line in out.splitlines() if line.strip())


def planning_stack_nodes_ready(nodes: set[str]) -> bool:
    """Match component_state_monitor planning/map/api modules being up."""
    has_planning = any(
        "mission_planner" in node or "route_selector" in node for node in nodes
    )
    has_map = any("/map/" in node or node.endswith("/map") for node in nodes)
    has_api = any("adapi" in node for node in nodes)
    return has_planning and has_map and has_api


def _list_ros_services(env: dict[str, str], timeout_sec: float = 8.0) -> set[str]:
    out = subprocess.check_output(
        ["ros2", "service", "list"],
        env=_ros_call_env(env),
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=timeout_sec,
    )
    return set(line.strip() for line in out.splitlines() if line.strip())


def wait_for_services(
    env: dict[str, str],
    timeout_sec: float,
    psim: subprocess.Popen[Any] | None = None,
) -> tuple[bool, str]:
    """Wait until psim routing/localization services are up.

    Node-list checks are best-effort: polluted DDS / slow discovery often makes
    ``ros2 node list`` hang even while services are already usable.
    """
    adapi_required = {"/api/routing/clear_route", "/api/routing/set_route_points"}
    mission_required = {
        "/planning/mission_planning/route_selector/main/clear_route",
        "/planning/mission_planning/route_selector/main/set_waypoint_route",
    }
    localize_required = {"/localization/initialize"}
    deadline = time.time() + timeout_sec
    last_log = 0.0
    services_ok_streak = 0
    while time.time() < deadline:
        if psim is not None and psim.poll() is not None:
            return False, f"planning simulator exited early (code={psim.returncode})"

        services: set[str] | None = None
        nodes: set[str] | None = None
        try:
            services = _list_ros_services(env, timeout_sec=8.0)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            services = None
        try:
            nodes = _list_ros_nodes(env, timeout_sec=5.0)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            nodes = None

        routing_ready = False
        localize_ready = False
        stack_ready = False
        if services is not None:
            routing_ready = adapi_required.issubset(services) or mission_required.issubset(
                services
            )
            localize_ready = localize_required.issubset(services)
        if nodes is not None:
            stack_ready = planning_stack_nodes_ready(nodes)

        services_ok = localize_ready and routing_ready
        if services_ok:
            services_ok_streak += 1
        else:
            services_ok_streak = 0

        # Prefer full stack, but accept stable service readiness when node list
        # is unavailable (common after leftover domain pollution).
        if services_ok and (stack_ready or services_ok_streak >= 2):
            time.sleep(3.0)
            detail = "planning_stack_ready" if stack_ready else "services_ready"
            return True, detail

        now = time.time()
        if now - last_log >= 15.0:
            elapsed = now - (deadline - timeout_sec)
            if nodes is None and services is None:
                detail = " nodes(unavailable) services(unavailable)"
            elif nodes is None:
                detail = (
                    f" nodes(unavailable)"
                    f" services(localize={localize_ready}, routing={routing_ready},"
                    f" streak={services_ok_streak})"
                )
            else:
                detail = (
                    f" nodes(planning={any('mission_planner' in n or 'route_selector' in n for n in nodes)},"
                    f" map={any('/map/' in n for n in nodes)},"
                    f" api={any('adapi' in n for n in nodes)})"
                    f" services(localize={localize_ready}, routing={routing_ready})"
                )
            print(
                f"[info] Waiting for planning simulator stack "
                f"({elapsed:.0f}/{timeout_sec:.0f}s){detail}…"
            )
            last_log = now
        time.sleep(2.0)
    return False, "planning_stack_not_ready"


def tail_text_file(path: Path, max_chars: int = 6000) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _routing_services_in_graph(env: dict[str, str]) -> bool:
    """True when ros2 service list shows a usable routing backend."""
    adapi_required = {"/api/routing/clear_route", "/api/routing/set_route_points"}
    mission_required = {
        "/planning/mission_planning/route_selector/main/clear_route",
        "/planning/mission_planning/route_selector/main/set_waypoint_route",
    }
    try:
        out = subprocess.check_output(
            ["ros2", "service", "list"],
            env=_ros_call_env(env),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
        services = set(out.splitlines())
        return adapi_required.issubset(services) or mission_required.issubset(services)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def wait_for_routing_callable(env: dict[str, str], timeout_sec: float = 45.0) -> bool:
    """Routing is ready when services appear in the ROS graph (rclpy may still lag)."""
    del timeout_sec  # kept for API compatibility; graph check is immediate after psim wait
    if _routing_services_in_graph(env):
        return True
    return False


def _route_state_label(state: int | None) -> str:
    try:
        from autoware_adapi_v1_msgs.msg import RouteState

        names = {
            RouteState.UNSET: "UNSET",
            RouteState.SET: "SET",
            RouteState.ARRIVED: "ARRIVED",
        }
        if state is None:
            return "unknown"
        return names.get(state, str(state))
    except ImportError:
        return "unknown" if state is None else str(state)


def wait_for_route_arrived(
    env: dict[str, str],
    timeout_sec: float,
    *,
    goal_xy_yaw: tuple[float, float, float] | None = None,
    near_goal_arrival_m: float = 5.0,
    stuck_blinker_nudge: bool = False,
    stuck_blinker_speed_mps: float = 0.15,
    stuck_blinker_trigger_sec: float = 12.0,
    stuck_blinker_hold_sec: float = 3.0,
    stuck_blinker_cooldown_sec: float = 25.0,
    stuck_abort_sec: float = 90.0,
) -> tuple[str, str]:
    """Poll /api/routing/state. Optionally nudge DP with blinkers when stuck.

    Returns status ``done`` | ``timeout`` | ``stuck``.
    """
    try:
        import rclpy
        from autoware_adapi_v1_msgs.msg import RouteState
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError:
        return "timeout", "rclpy_or_adapi_msgs_unavailable"

    try:
        from autoware_vehicle_msgs.msg import TurnIndicatorsReport
    except ImportError:
        TurnIndicatorsReport = None  # type: ignore[misc, assignment]

    if not rclpy.ok():
        rclpy.init()

    # Keep env for this process (ROS_DOMAIN_ID already set by caller via os.environ
    # when used from setup_route; here we rely on caller's env being current).
    old_env = os.environ.copy()
    os.environ.update(env)

    class Waiter(Node):
        def __init__(self) -> None:
            super().__init__("dp_multi_eval_route_waiter")
            self.state: int | None = None
            self.speed_mps: float = float("nan")
            self.x: float = float("nan")
            self.y: float = float("nan")
            qos_route = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(RouteState, "/api/routing/state", self._cb_route, qos_route)
            self.create_subscription(
                Odometry,
                "/localization/kinematic_state",
                self._cb_odom,
                10,
            )
            self._blinker_pub = None
            if stuck_blinker_nudge and TurnIndicatorsReport is not None:
                self._blinker_pub = self.create_publisher(
                    TurnIndicatorsReport,
                    "/vehicle/status/turn_indicators_status",
                    10,
                )

        def _cb_route(self, msg: RouteState) -> None:
            self.state = msg.state

        def _cb_odom(self, msg: Odometry) -> None:
            vx = float(msg.twist.twist.linear.x)
            vy = float(msg.twist.twist.linear.y)
            self.speed_mps = (vx * vx + vy * vy) ** 0.5
            self.x = float(msg.pose.pose.position.x)
            self.y = float(msg.pose.pose.position.y)

        def publish_blinker(self, report: int) -> None:
            if self._blinker_pub is None or TurnIndicatorsReport is None:
                return
            msg = TurnIndicatorsReport()
            msg.stamp = self.get_clock().now().to_msg()
            msg.report = report
            self._blinker_pub.publish(msg)

    node = Waiter()
    deadline = time.time() + timeout_sec
    last_log = 0.0
    stuck_since: float | None = None
    last_nudge_end = 0.0
    nudge_count = 0
    abort_after = max(0.0, float(stuck_abort_sec or 0.0))
    # Cycle: LEFT → DISABLE → RIGHT → DISABLE
    nudge_phases: list[tuple[str, int]] = []
    if TurnIndicatorsReport is not None:
        nudge_phases = [
            ("LEFT", int(TurnIndicatorsReport.ENABLE_LEFT)),
            ("DISABLE", int(TurnIndicatorsReport.DISABLE)),
            ("RIGHT", int(TurnIndicatorsReport.ENABLE_RIGHT)),
            ("DISABLE", int(TurnIndicatorsReport.DISABLE)),
        ]

    detail_state: int | None = None
    try:
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.25)
            now = time.time()
            if node.state == RouteState.ARRIVED:
                detail = "route_arrived"
                if nudge_count:
                    detail += f" (blinker_nudges={nudge_count})"
                return "done", detail

            speed = node.speed_mps
            if (
                goal_xy_yaw is not None
                and speed == speed
                and node.x == node.x
                and speed <= stuck_blinker_speed_mps
            ):
                goal_dist = math.hypot(
                    node.x - float(goal_xy_yaw[0]),
                    node.y - float(goal_xy_yaw[1]),
                )
                if goal_dist <= max(0.0, float(near_goal_arrival_m)):
                    return (
                        "done",
                        f"near_goal_arrival dist={goal_dist:.2f}m "
                        f"state={_route_state_label(node.state)}",
                    )

            if speed == speed:  # not NaN
                if speed <= stuck_blinker_speed_mps:
                    if stuck_since is None:
                        stuck_since = now
                else:
                    stuck_since = None

            if (
                abort_after > 0.0
                and stuck_since is not None
                and (now - stuck_since) >= abort_after
            ):
                detail = (
                    f"stuck_abort v={speed:.2f} m/s for {now - stuck_since:.0f}s "
                    f"state={_route_state_label(node.state)}"
                )
                if nudge_count:
                    detail += f" blinker_nudges={nudge_count}"
                print(f"[warn] {detail} — ending leg early")
                return "stuck", detail

            if stuck_blinker_nudge and nudge_phases and speed == speed:
                ready_to_nudge = (
                    stuck_since is not None
                    and (now - stuck_since) >= stuck_blinker_trigger_sec
                    and (now - last_nudge_end) >= stuck_blinker_cooldown_sec
                )
                if ready_to_nudge:
                    nudge_count += 1
                    print(
                        f"[info] Stuck (v={speed:.2f} m/s for "
                        f"{now - stuck_since:.0f}s) — blinker nudge #{nudge_count} "
                        "(LEFT then RIGHT) to try DP trajectory generation"
                    )
                    for label, report in nudge_phases:
                        phase_deadline = time.time() + stuck_blinker_hold_sec
                        print(f"[info]   blinker → {label} ({stuck_blinker_hold_sec:.1f}s)")
                        while time.time() < phase_deadline and rclpy.ok():
                            node.publish_blinker(report)
                            rclpy.spin_once(node, timeout_sec=0.1)
                            if node.state == RouteState.ARRIVED:
                                return "done", f"route_arrived (during blinker={label})"
                        node.publish_blinker(report)
                    last_nudge_end = time.time()
                    # Keep stuck_since so abort_after still accumulates across nudges.

            if now - last_log >= 15.0:
                speed_s = f"{speed:.2f}" if speed == speed else "?"
                stuck_s = (
                    f" stuck={now - stuck_since:.0f}s"
                    if stuck_since is not None
                    else ""
                )
                print(
                    f"[info] waiting for route ARRIVED "
                    f"({timeout_sec - (deadline - now):.0f}/{timeout_sec:.0f}s) "
                    f"state={_route_state_label(node.state)} v={speed_s} m/s"
                    + (f" nudges={nudge_count}" if nudge_count else "")
                    + stuck_s
                )
                last_log = now
            detail_state = node.state
    finally:
        node.destroy_node()
        os.environ.clear()
        os.environ.update(old_env)

    detail = f"route_not_arrived state={_route_state_label(detail_state)}"
    if nudge_count:
        detail += f" blinker_nudges={nudge_count}"
    return "timeout", detail


def wait_for_perception_ready(
    cfg: JobConfig,
    *,
    timeout_sec: float | None = None,
) -> tuple[bool, str]:
    """Wait until reproducer publishes a stable set of tracked objects."""
    try:
        import rclpy
        from autoware_perception_msgs.msg import TrackedObjects
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
    except ImportError:
        return False, "perception_msgs_unavailable"

    if not rclpy.ok():
        rclpy.init()

    wait_budget = float(
        timeout_sec if timeout_sec is not None else cfg.perception_ready_timeout_sec
    )

    class Monitor(Node):
        def __init__(self) -> None:
            super().__init__("dp_multi_eval_perception_ready")
            # Match perception_reproducer's RELIABLE publisher (BEST_EFFORT can miss).
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            self.object_count = 0
            self.message_count = 0
            self.create_subscription(
                TrackedObjects,
                "/perception/object_recognition/tracking/objects",
                self._on_objects,
                qos,
            )

        def _on_objects(self, msg: TrackedObjects) -> None:
            self.message_count += 1
            self.object_count = len(msg.objects)

    monitor = Monitor()
    t0 = time.time()
    deadline = t0 + wait_budget
    warmup_until = t0 + cfg.perception_warmup_sec
    stable_since: float | None = None
    last_count = -1
    last_log = 0.0
    try:
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(monitor, timeout_sec=0.25)
            now = time.time()
            if now < warmup_until:
                continue

            count = monitor.object_count
            if count < cfg.perception_ready_min_objects:
                stable_since = None
                last_count = count
            elif count == last_count:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= cfg.perception_ready_stable_sec:
                    return (
                        True,
                        f"perception_ready objects={count} after_{now - t0:.0f}s",
                    )
            else:
                stable_since = None
                last_count = count

            if now - last_log >= 15.0:
                elapsed = now - t0
                print(
                    f"[info] Waiting for perception_reproducer… "
                    f"{elapsed:.0f}/{wait_budget:.0f}s "
                    f"objects={count} msgs={monitor.message_count} "
                    f"(full bags can take many minutes to load)"
                )
                last_log = now
                # Keep clearing empty publishers from psim.
                kill_conflicting_perception_nodes()
    finally:
        monitor.destroy_node()

    return (
        False,
        f"perception_ready_timeout after_{wait_budget:.0f}s "
        f"objects={monitor.object_count} msgs={monitor.message_count}",
    )


def perception_load_timeout_sec(cfg: JobConfig, bag_duration: float | None) -> float:
    """Wall-time budget for full-bag perception_reproducer load + object publish."""
    base = max(float(cfg.perception_ready_timeout_sec), float(cfg.bag_load_base_sec))
    if bag_duration is None:
        return min(float(cfg.bag_load_timeout_cap_sec), base)
    scaled = float(cfg.bag_load_base_sec) + float(bag_duration) * float(
        cfg.bag_load_duration_factor
    )
    return min(float(cfg.bag_load_timeout_cap_sec), max(base, scaled))


def build_psim_command(cfg: JobConfig) -> list[str]:
    cmd = [
        "ros2",
        "launch",
        "autoware_launch",
        "planning_simulator.launch.xml",
        f"map_path:={cfg.map_path}",
        f"vehicle_model:={cfg.vehicle_model}",
        f"sensor_model:={cfg.sensor_model}",
        f"planning_setting:={cfg.planning_setting}",
        f"rviz:={'true' if cfg.rviz else 'false'}",
    ]
    vehicle_id = cfg.vehicle_id or os.environ.get("VEHICLE_ID")
    if vehicle_id:
        cmd.append(f"vehicle_id:={vehicle_id}")
    return cmd


def build_reproducer_command(
    bag_path: Path,
    search_radius: float = 0.0,
    reproduce_cool_down_sec: float = 80.0,
) -> list[str]:
    """Build perception_reproducer command.

    ``search_radius`` 0 = always nearest bag pose (needed when ego path diverges
    from the recording; otherwise objects go empty outside the radius).
    """
    cmd = [
        "ros2",
        "run",
        "planning_debug_tools",
        "perception_reproducer.py",
        "-b",
        str(bag_path),
        "-t",
        "-r",
        str(float(search_radius)),
    ]
    if float(search_radius) > 0.0:
        cmd.extend(["-c", str(float(reproduce_cool_down_sec))])
    return cmd


def kill_conflicting_perception_nodes() -> None:
    """Stop psim perception publishers that fight with perception_reproducer."""
    patterns = (
        "multi_object_tracker",
        "map_based_prediction",
        "dummy_perception_publisher",
        "autoware_multi_object_tracker",
    )
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-f", pattern],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def build_record_command(output_bag: Path, topics: list[str]) -> list[str]:
    return [
        "ros2",
        "bag",
        "record",
        "-o",
        str(output_bag),
        "--storage",
        "sqlite3",
        *topics,
    ]


def _load_route_setup_module(env: dict[str, str]):
    import importlib.util

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


def setup_route(
    cfg: JobConfig,
    env: dict[str, str],
    *,
    goal_xy_yaw: tuple[float, float, float] | None = None,
) -> tuple[bool, str]:
    """Localization + route in-process (same Python interpreter, shared DDS env)."""
    if cfg.skip_route_setup:
        return True, "route_setup_skipped"

    print(f"[info] Route setup in-process ROS_DOMAIN_ID={env.get('ROS_DOMAIN_ID', '?')}")
    old_env = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        module = _load_route_setup_module(env)
        goal_pose = None
        if goal_xy_yaw is not None and hasattr(module, "pose_from_xy_yaw"):
            goal_pose = module.pose_from_xy_yaw(*goal_xy_yaw)
        route_order = list(cfg.multi_goal_stop_order or []) or None
        ok, message = module.setup_route_for_bag(
            bag_path=cfg.bag_path.expanduser().resolve(),  # type: ignore[union-attr]
            timeout_sec=cfg.route_setup_timeout_sec,
            service_wait_sec=max(cfg.route_setup_timeout_sec, 60.0),
            backend_preference="adapi",
            auto_engage=False,
            map_path=cfg.map_path,
            stop_point_goal_fallback=True,
            start_pose_stop_fallback=True,
            stop_points_csv=cfg.stop_points_csv,
            route_stop_order=route_order,
            goal_pose=goal_pose,
        )
        if ok:
            print(f"Route setup succeeded: {message}")
            return True, message
        print(f"Route setup failed:\n{message}")
        return False, f"route_setup_failed: {message}"
    except Exception as exc:  # noqa: BLE001
        return False, f"route_setup_exception: {exc}"
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def advance_route(
    cfg: JobConfig,
    env: dict[str, str],
    goal_xy_yaw: tuple[float, float, float],
) -> tuple[bool, str]:
    """Clear ARRIVED route and set the next multi-goal destination."""
    print(
        f"[info] Advancing route to next goal "
        f"({goal_xy_yaw[0]:.1f}, {goal_xy_yaw[1]:.1f})"
    )
    old_env = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        module = _load_route_setup_module(env)
        if not hasattr(module, "advance_route_to_goal") or not hasattr(module, "pose_from_xy_yaw"):
            return False, "advance_route_unsupported (rebuild diffusion_planner_batch_eval)"
        goal_pose = module.pose_from_xy_yaw(*goal_xy_yaw)
        route_order = list(cfg.multi_goal_stop_order or []) or None
        ok, message = module.advance_route_to_goal(
            goal_pose,
            timeout_sec=cfg.route_setup_timeout_sec,
            service_wait_sec=max(cfg.route_setup_timeout_sec, 60.0),
            backend_preference="adapi",
            map_path=cfg.map_path,
            stop_point_goal_fallback=True,
            route_stop_order=route_order,
        )
        if ok:
            print(f"[info] {message}")
            return True, message
        print(f"[error] advance failed: {message}")
        return False, message
    except Exception as exc:  # noqa: BLE001
        return False, f"advance_route_exception: {exc}"
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def reposition_stuck_ego(
    cfg: JobConfig,
    env: dict[str, str],
) -> tuple[bool, str]:
    """Shift planning-simulator ego forward while preserving the active route."""
    distance_m = float(cfg.stuck_reposition_forward_m)
    if distance_m <= 0.0:
        return False, "stuck_reposition_disabled"

    print(f"[warn] Attempting stuck recovery: move ego {distance_m:.1f}m forward")
    old_env = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        module = _load_route_setup_module(env)
        if not hasattr(module, "shift_ego_longitudinal"):
            return False, "stuck_reposition_unsupported (rebuild diffusion_planner_batch_eval)"
        ok, message = module.shift_ego_longitudinal(
            distance_m,
            timeout_sec=min(float(cfg.route_setup_timeout_sec), 45.0),
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
    """Enable Autoware control + switch to Autonomous (after reproducer is up)."""
    services = (
        "/api/operation_mode/enable_autoware_control",
        "/api/operation_mode/change_to_autonomous",
    )
    for service in services:
        cmd = [
            "ros2",
            "service",
            "call",
            service,
            "autoware_adapi_v1_msgs/srv/ChangeOperationMode",
            "{}",
        ]
        print(f"[info] Engage: {service}")
        try:
            completed = subprocess.run(
                cmd,
                env=env,
                text=True,
                timeout=timeout_sec,
                check=False,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            return False, f"engage_timeout:{service}"
        if completed.returncode != 0:
            detail = (completed.stdout or "") + (completed.stderr or "")
            return False, f"engage_failed:{service}:{detail[-200:]}"
    return True, "autonomous_engaged"


def run_single_job(cfg: JobConfig) -> JobResult:
    mode = (cfg.mode or "reproducer").strip().lower()
    if mode == "scenario_simulator":
        return run_scenario_simulator_job(cfg)
    return run_reproducer_job(cfg)


def run_scenario_simulator_job(cfg: JobConfig) -> JobResult:
    """Run one Scenario Simulator v2 case with diffusion planner + bag record."""
    start_iso = utc_now_iso()
    t0 = time.time()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_key = hashlib.sha1(str(cfg.output_dir.resolve()).encode()).hexdigest()[:16]
    overlay_root = Path("/tmp/dp_multi_eval_overlays") / overlay_key
    output_bag = cfg.output_dir / "output"
    ss2_out = cfg.output_dir / "ss2_out"
    if output_bag.exists():
        shutil.rmtree(output_bag)
    ss2_out.mkdir(parents=True, exist_ok=True)

    status = "failed"
    error: str | None = None
    procs: list[subprocess.Popen[Any]] = []
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(cfg.domain_id)
    env.setdefault("ROS2_DISABLE_DAEMON", "1")
    scenario_path = (cfg.scenario_path or Path("")).expanduser().resolve()

    try:
        if not cfg.model_config.expanduser().resolve().is_file():
            raise FileNotFoundError(f"model_config not found: {cfg.model_config}")
        if not scenario_path.is_file():
            raise FileNotFoundError(f"scenario_path not found: {scenario_path}")
        if not scenario_test_runner_available(env):
            raise RuntimeError(
                "scenario_test_runner package not found. "
                "Install Scenario Simulator v2 (simulator.repos) and source the workspace."
            )

        make_model_overlay(cfg.model_config, overlay_root)
        env = env_with_overlay(env, overlay_root, cfg.domain_id)
        vehicle_id = cfg.vehicle_id or env.get("VEHICLE_ID")
        if vehicle_id:
            env["VEHICLE_ID"] = vehicle_id

        run_timeout = float(cfg.scenario_timeout_sec or cfg.route_timeout_sec)
        ss2_cmd = build_scenario_runner_command(
            scenario_path=scenario_path,
            output_directory=ss2_out,
            vehicle_model=cfg.vehicle_model,
            sensor_model=cfg.sensor_model,
            architecture_type=cfg.architecture_type,
            planning_setting=cfg.planning_setting,
            initialize_duration_sec=cfg.psim_startup_sec,
            global_timeout_sec=run_timeout,
            rviz=cfg.rviz,
            record=False,
            vehicle_id=vehicle_id,
        )
        record_cmd = build_record_command_sim_time(output_bag, cfg.topics.record_list())

        print(f"[info] mode=scenario_simulator ROS_DOMAIN_ID={cfg.domain_id}")
        print(f"[info] scenario={scenario_path}")
        print(f"[info] model_config={cfg.model_config}")
        print(f"[info] timeout={run_timeout:.0f}s")

        if cfg.dry_run:
            print("[dry-run] Would launch:")
            print(f"  ss2:      {' '.join(ss2_cmd)}")
            print(f"  record:   {' '.join(record_cmd)}")
            write_job_status(
                cfg.output_dir,
                {
                    "status": "dry_run",
                    "mode": "scenario_simulator",
                    "start_time": start_iso,
                    "end_time": utc_now_iso(),
                    "error": None,
                    "dry_run": True,
                    "domain_id": cfg.domain_id,
                    "commands": {"scenario_test_runner": ss2_cmd, "record": record_cmd},
                },
            )
            return JobResult("dry_run", start_iso, utc_now_iso(), None, None, time.time() - t0)

        update_job_progress(
            cfg.output_dir,
            phase="scenario_startup",
            start_time=start_iso,
            domain_id=cfg.domain_id,
            scenario_path=str(scenario_path),
            model_config=str(cfg.model_config),
        )
        ss2_log = cfg.output_dir / "scenario_runner.log"
        print(f"[info] Launching scenario_test_runner...")
        print(f"[info] log: {ss2_log}")
        ss2, _ = popen_log_file(ss2_cmd, ss2_log, env=env)
        procs.append(ss2)

        # Wait briefly for /clock then start recording with use_sim_time
        warmup = max(0.0, float(cfg.scenario_record_warmup_sec))
        print(f"[info] Waiting {warmup:.0f}s before bag record (sim time)...")
        time.sleep(warmup)
        if ss2.poll() is not None:
            raise RuntimeError(
                f"scenario_test_runner exited early (code={ss2.returncode}); "
                f"see {ss2_log}"
            )

        update_job_progress(
            cfg.output_dir,
            phase="recording",
            start_time=start_iso,
            domain_id=cfg.domain_id,
            scenario_path=str(scenario_path),
            model_config=str(cfg.model_config),
        )
        print(f"[info] Starting ros2 bag record (use_sim_time) → {output_bag}")
        recorder = popen(record_cmd, env=env)
        procs.append(recorder)
        time.sleep(1.0)
        if recorder.poll() is not None:
            raise RuntimeError("ros2 bag record exited immediately (check use_sim_time / clock)")

        update_job_progress(
            cfg.output_dir,
            phase="driving",
            start_time=start_iso,
            domain_id=cfg.domain_id,
            scenario_path=str(scenario_path),
            model_config=str(cfg.model_config),
        )
        print(f"[info] Waiting for scenario_test_runner (timeout {run_timeout:.0f}s)...")
        try:
            code = ss2.wait(timeout=run_timeout + float(cfg.psim_startup_sec))
        except subprocess.TimeoutExpired:
            error = f"scenario_timeout after {run_timeout:.0f}s"
            status = "failed"
            code = None
        else:
            if code == 0:
                status = "done"
                error = None
                time.sleep(cfg.post_arrival_sec)
            else:
                status = "failed"
                error = f"scenario_test_runner exit={code}"
                # Soft: keep bag for offline metrics if anything was recorded
                if output_bag.exists():
                    print(f"[warn] {error} — keeping recorded bag for metrics")

    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        print(f"[error] {error}")

    finally:
        for proc in reversed(procs):
            stop_process_group(proc, grace_sec=10.0)
            close_log_handle(proc)

    end_iso = utc_now_iso()
    bag_path_out = str(output_bag) if output_bag.exists() else None
    # If runner succeeded but bag missing, still fail
    if status == "done" and not bag_path_out:
        status = "failed"
        error = (error or "") + "; output_bag_missing"

    write_job_status(
        cfg.output_dir,
        {
            "status": status,
            "mode": "scenario_simulator",
            "start_time": start_iso,
            "end_time": end_iso,
            "error": error,
            "domain_id": cfg.domain_id,
            "scenario_path": str(scenario_path),
            "output_bag": bag_path_out,
            "duration_sec": time.time() - t0,
        },
    )
    return JobResult(status, start_iso, end_iso, error, bag_path_out, time.time() - t0)


def run_reproducer_job(cfg: JobConfig) -> JobResult:
    start_iso = utc_now_iso()
    t0 = time.time()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep overlay on local disk — cloning autoware_launch to /media is very slow.
    overlay_key = hashlib.sha1(str(cfg.output_dir.resolve()).encode()).hexdigest()[:16]
    overlay_root = Path("/tmp/dp_multi_eval_overlays") / overlay_key
    output_bag = cfg.output_dir / "output"
    # ros2 bag record creates a directory named output/ (or fails if exists)
    if output_bag.exists():
        shutil.rmtree(output_bag)

    status = "failed"
    error: str | None = None
    procs: list[subprocess.Popen[Any]] = []
    live_monitor = None
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(cfg.domain_id)
    env.setdefault("ROS2_DISABLE_DAEMON", "1")
    bag_path = (cfg.bag_path or Path("")).expanduser().resolve()

    try:
        if not cfg.model_config.expanduser().resolve().is_file():
            raise FileNotFoundError(f"model_config not found: {cfg.model_config}")
        if not bag_path.exists():
            raise FileNotFoundError(f"bag_path not found: {bag_path}")

        make_model_overlay(cfg.model_config, overlay_root)
        env = env_with_overlay(env, overlay_root, cfg.domain_id)
        vehicle_id = cfg.vehicle_id or env.get("VEHICLE_ID")
        if vehicle_id:
            env["VEHICLE_ID"] = vehicle_id

        bag_duration = get_bag_duration_sec(bag_path)
        # route_timeout_sec is the ARRIVED / stuck budget for route_arrived mode.
        # For bag_duration, drive for the full input bag (plus margin) — do NOT
        # cap with route_timeout_sec (that was cutting 20‑min bags down to ~2 min).
        if cfg.end_condition == "bag_duration" and bag_duration is not None:
            run_timeout = float(bag_duration) + float(cfg.bag_duration_margin_sec)
        else:
            run_timeout = float(cfg.route_timeout_sec)

        # Ensure setup_route sees bag_path
        cfg.bag_path = bag_path
        psim_cmd = build_psim_command(cfg)
        repro_cmd = build_reproducer_command(
            bag_path,
            search_radius=float(cfg.reproducer_search_radius_m),
            reproduce_cool_down_sec=float(cfg.reproducer_cool_down_sec),
        )
        record_cmd = build_record_command(output_bag, cfg.topics.record_list())

        print(f"[info] mode=reproducer ROS_DOMAIN_ID={cfg.domain_id}")
        if vehicle_id:
            print(f"[info] vehicle_id={vehicle_id}")
        print(f"[info] model_config={cfg.model_config}")
        print(f"[info] bag_path={bag_path} duration={bag_duration}")
        print(f"[info] output_dir={cfg.output_dir}")
        print(f"[info] run_timeout={run_timeout:.0f}s end_condition={cfg.end_condition}")

        if cfg.dry_run:
            print("[dry-run] Would launch:")
            print(f"  psim:     {' '.join(psim_cmd)}")
            print(f"  repro:    {' '.join(repro_cmd)}")
            print(f"  record:   {' '.join(record_cmd)}")
            print(f"  topics:   {cfg.topics.record_list()}")
            write_job_status(
                cfg.output_dir,
                {
                    "status": "dry_run",
                    "mode": "reproducer",
                    "start_time": start_iso,
                    "end_time": utc_now_iso(),
                    "error": None,
                    "dry_run": True,
                    "domain_id": cfg.domain_id,
                    "commands": {
                        "psim": psim_cmd,
                        "reproducer": repro_cmd,
                        "record": record_cmd,
                    },
                },
            )
            return JobResult("dry_run", start_iso, utc_now_iso(), None, None, time.time() - t0)

        update_job_progress(
            cfg.output_dir,
            phase="psim_startup",
            start_time=start_iso,
            domain_id=cfg.domain_id,
            bag_path=str(bag_path),
            model_config=str(cfg.model_config),
        )
        # Leftover psim/rosbridge from a previous failed job can make
        # ``ros2 node list`` hang and block readiness forever.
        from dp_multi_eval.process_utils import cleanup_evaluation_processes

        print("[info] Cleaning leftover evaluation processes before psim launch...")
        cleanup_evaluation_processes()
        time.sleep(2.0)

        psim_log = cfg.output_dir / "psim_launch.log"
        print(f"[info] Launching planning simulator...")
        print(f"[info] psim log: {psim_log}")
        print(f"[info] psim cmd: {' '.join(psim_cmd)}")
        psim, _psim_log_handle = popen_log_file(psim_cmd, psim_log, env=env)
        procs.append(psim)

        psim_ready, psim_detail = wait_for_services(env, cfg.psim_startup_sec, psim=psim)
        if not psim_ready:
            psim_tail = tail_text_file(psim_log)
            if psim_tail.strip():
                print(f"[error] psim_launch.log (tail):\n{psim_tail}")
            raise RuntimeError(
                f"planning simulator not ready within {cfg.psim_startup_sec:.0f}s: {psim_detail}"
            )
        print(f"[info] Planning simulator stack ready ({psim_detail})")

        update_job_progress(
            cfg.output_dir,
            phase="route_setup",
            start_time=start_iso,
            domain_id=cfg.domain_id,
            bag_path=str(bag_path),
            model_config=str(cfg.model_config),
        )
        if not wait_for_routing_callable(env, timeout_sec=45.0):
            raise RuntimeError(
                "routing services not visible in ROS graph after startup waits"
            )

        # Resolve route legs (single-goal by default)
        route_goals: list[tuple[float, float, float]] = []
        if cfg.multi_goal:
            from dp_multi_eval.multi_goal import extract_route_goals, goals_to_jsonable

            extracted = extract_route_goals(
                bag_path,
                source=cfg.multi_goal_source,
                stop_speed_mps=cfg.multi_goal_stop_speed_mps,
                stop_min_sec=cfg.multi_goal_stop_min_sec,
                min_spacing_m=cfg.multi_goal_min_spacing_m,
                map_path=cfg.map_path,
                stop_points_csv=cfg.stop_points_csv,
                stop_order=list(cfg.multi_goal_stop_order or []),
            )
            if not extracted:
                raise RuntimeError(
                    "multi_goal produced no remaining stops for this bag start pose"
                )
            route_goals = [(g.x, g.y, g.yaw) for g in extracted]
            (cfg.output_dir / "route_goals.json").write_text(
                json.dumps(
                    {
                        "goals": goals_to_jsonable(extracted),
                        "source": cfg.multi_goal_source,
                        "stop_order": list(cfg.multi_goal_stop_order or []),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[info] multi_goal=on source={cfg.multi_goal_source} "
                f"legs={len(route_goals)}"
            )
            for i, g in enumerate(extracted):
                print(
                    f"[info]   leg {i + 1}/{len(extracted)}: "
                    f"({g.x:.1f}, {g.y:.1f}) source={g.source}"
                )

        first_goal = route_goals[0] if route_goals else None
        ok, route_msg = setup_route(cfg, env, goal_xy_yaw=first_goal)
        if not ok:
            raise RuntimeError(f"route setup failed: {route_msg}")
        print(f"[info] {route_msg}")

        # If route setup had to skip unroutable stop goals (empty lanelet path),
        # drop those legs so ARRIVED waits match the effective goal.
        if cfg.multi_goal and route_goals and "effective_goal=" in route_msg:
            effective = route_msg.split("effective_goal=", 1)[1].split(";", 1)[0].strip()
            if effective:
                matched_idx = None
                for i, g in enumerate(extracted):
                    if g.source.endswith(effective) or f"stop_points:{effective}" == g.source:
                        matched_idx = i
                        break
                if matched_idx is not None and matched_idx > 0:
                    print(
                        f"[info] Trimming {matched_idx} unroutable stop leg(s); "
                        f"effective first goal={effective}"
                    )
                    extracted = extracted[matched_idx:]
                    route_goals = [(g.x, g.y, g.yaw) for g in extracted]
                    (cfg.output_dir / "route_goals.json").write_text(
                        json.dumps(
                            {
                                "goals": goals_to_jsonable(extracted),
                                "source": cfg.multi_goal_source,
                                "stop_order": list(cfg.multi_goal_stop_order or []),
                                "effective_first_goal": effective,
                                "route_setup": route_msg,
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )

        update_job_progress(
            cfg.output_dir,
            phase="reproducer",
            start_time=start_iso,
            domain_id=cfg.domain_id,
            bag_path=str(bag_path),
            model_config=str(cfg.model_config),
        )

        # Start live map early (ego pose during bag load) so Streamlit / dashboard
        # can show something before Auto engage.
        if cfg.live_web_monitor:
            from dp_multi_eval.live_monitor import LiveDriveMonitor

            goal_markers = [
                {
                    "x": float(g[0]),
                    "y": float(g[1]),
                    "yaw": float(g[2]),
                    "label": f"leg{i + 1}",
                }
                for i, g in enumerate(route_goals)
                if g is not None and len(g) >= 3
            ]
            root_path = (
                Path(cfg.live_drive_root) / "live_drive.json"
                if cfg.live_drive_root
                else cfg.output_dir / "live_drive.json"
            )
            extra = []
            if cfg.live_drive_root:
                extra.append(cfg.output_dir / "live_drive.json")
            live_monitor = LiveDriveMonitor(
                output_path=root_path,
                extra_paths=extra,
                sample_sec=float(cfg.live_web_sample_sec),
                job_id=cfg.output_dir.name,
                bag_key=bag_path.name,
                goals=goal_markers,
            )
            live_monitor.set_phase(
                "loading_perception",
                note="waiting for full bag load",
            )
            live_monitor.start()
            print(
                f"[info] Live web monitor → {root_path} "
                f"(sample every {cfg.live_web_sample_sec:.1f}s; "
                "Live Progress / dashboard.html)"
            )

        print(
            f"[info] Starting perception_reproducer "
            f"(search_radius={cfg.reproducer_search_radius_m:.1f}m)..."
        )
        repro = popen(repro_cmd, env=env)
        procs.append(repro)
        # Psim's tracker publishes empty TrackedObjects; kill it so the
        # reproducer owns the topic. Full bags load slowly — poll until objects
        # appear instead of a short fixed sleep.
        time.sleep(2.0)
        kill_conflicting_perception_nodes()
        load_timeout = perception_load_timeout_sec(cfg, bag_duration)
        print(
            f"[info] Waiting up to {load_timeout:.0f}s for perception_reproducer "
            f"to finish loading the full bag and publish objects "
            f"(input duration={bag_duration})..."
        )
        ready_ok, ready_msg = wait_for_perception_ready(cfg, timeout_sec=load_timeout)
        kill_conflicting_perception_nodes()
        if ready_ok:
            print(f"[info] {ready_msg}")
        elif cfg.require_perception_ready:
            raise RuntimeError(
                f"{ready_msg} — not engaging Auto with empty perception. "
                "Full bags can take ~bag_duration wall time to load; "
                "increase bag_load_timeout_cap_sec / bag_load_duration_factor if needed."
            )
        else:
            print(f"[warn] {ready_msg} — engaging Auto anyway (require_perception_ready=false)")

        update_job_progress(
            cfg.output_dir,
            phase="recording",
            start_time=start_iso,
            domain_id=cfg.domain_id,
            bag_path=str(bag_path),
            model_config=str(cfg.model_config),
        )
        print(f"[info] Starting ros2 bag record → {output_bag}")
        recorder = popen(record_cmd, env=env)
        procs.append(recorder)
        time.sleep(1.0)
        if recorder.poll() is not None:
            raise RuntimeError("ros2 bag record exited immediately")

        ok, engage_msg = engage_autonomous(env, timeout_sec=cfg.auto_engage_timeout_sec)
        if not ok:
            print(f"[warn] {engage_msg} — continuing; run may end as stuck/timeout")
        else:
            print(f"[info] {engage_msg}")

        update_job_progress(
            cfg.output_dir,
            phase="driving",
            start_time=start_iso,
            domain_id=cfg.domain_id,
            bag_path=str(bag_path),
            model_config=str(cfg.model_config),
        )
        if live_monitor is not None:
            live_monitor.set_phase("driving", note="auto engaged")

        legs = route_goals if (cfg.multi_goal and len(route_goals) > 1) else [None]
        drive_deadline = time.time() + run_timeout
        status = "done"
        error = None
        leg_cap = max(
            5.0,
            float(cfg.multi_goal_leg_timeout_sec or cfg.route_timeout_sec or 300.0),
        )
        stuck_kw = dict(
            stuck_blinker_nudge=bool(cfg.stuck_blinker_nudge),
            stuck_blinker_speed_mps=float(cfg.stuck_blinker_speed_mps),
            stuck_blinker_trigger_sec=float(cfg.stuck_blinker_trigger_sec),
            stuck_blinker_hold_sec=float(cfg.stuck_blinker_hold_sec),
            stuck_blinker_cooldown_sec=float(cfg.stuck_blinker_cooldown_sec),
        )
        reposition_enabled = float(cfg.stuck_reposition_forward_m) > 0.0
        reposition_trigger = max(
            5.0,
            float(cfg.stuck_reposition_trigger_sec or 30.0),
        )
        reposition_max = max(0, int(cfg.stuck_reposition_max_count or 0))

        def wait_with_stuck_recovery(
            wait_sec: float,
            current_goal: tuple[float, float, float] | None,
        ) -> tuple[str, str]:
            """Wait for ARRIVED; if stopped too long, shift ego forward and retry.

            Used for red-light / lead-vehicle stalls: every
            ``stuck_reposition_trigger_sec`` of near-zero speed, move ego
            ``stuck_reposition_forward_m`` and continue the same leg.
            """
            deadline = time.time() + wait_sec
            shifts = 0
            last_detail = "wait_not_started"

            while True:
                remaining = deadline - time.time()
                if remaining < 5.0:
                    return "timeout", (
                        f"{last_detail}; leg_wait_exhausted "
                        f"forward_shifts={shifts}"
                    )

                abort_after = (
                    reposition_trigger
                    if reposition_enabled
                    else float(cfg.stuck_abort_sec)
                )
                wait_status, wait_detail = wait_for_route_arrived(
                    env,
                    remaining,
                    goal_xy_yaw=current_goal,
                    near_goal_arrival_m=float(cfg.multi_goal_arrival_tolerance_m),
                    stuck_abort_sec=abort_after,
                    **stuck_kw,
                )
                last_detail = wait_detail
                if wait_status == "done":
                    if shifts:
                        wait_detail = (
                            f"{wait_detail}; forward_shifts={shifts}"
                        )
                    return wait_status, wait_detail
                if wait_status != "stuck" or not reposition_enabled:
                    return wait_status, wait_detail
                if reposition_max and shifts >= reposition_max:
                    return (
                        "stuck",
                        f"{wait_detail}; reposition_max={reposition_max}",
                    )

                shifts += 1
                print(
                    f"[warn] Stopped ≥{reposition_trigger:.0f}s "
                    f"(red light / lead vehicle?) — forward shift "
                    f"#{shifts} by {cfg.stuck_reposition_forward_m:.1f}m"
                )
                recovered, recovery_detail = reposition_stuck_ego(cfg, env)
                if not recovered:
                    return "stuck", f"{wait_detail}; {recovery_detail}"

                if current_goal is not None:
                    route_ok, route_detail = advance_route(cfg, env, current_goal)
                    if not route_ok:
                        return "stuck", (
                            f"{wait_detail}; {recovery_detail}; "
                            f"route_reset_failed: {route_detail}"
                        )

                engaged, engage_detail = engage_autonomous(
                    env, timeout_sec=cfg.auto_engage_timeout_sec
                )
                if not engaged:
                    return "stuck", (
                        f"{wait_detail}; {recovery_detail}; "
                        f"reengage_failed: {engage_detail}"
                    )
                print(f"[info] {engage_detail} after forward shift #{shifts}")
                last_detail = (
                    f"{wait_detail}; {recovery_detail}; "
                    f"forward_shifts={shifts}"
                )

        for leg_idx, _leg in enumerate(legs):
            if live_monitor is not None:
                live_monitor.set_leg(leg_idx + 1, len(legs))
                live_monitor.set_phase("driving", note=f"leg {leg_idx + 1}/{len(legs)}")
            remaining = max(5.0, drive_deadline - time.time())
            is_intermediate = (
                cfg.multi_goal and len(legs) > 1 and leg_idx < len(legs) - 1
            )
            # Intermediate multi-goal legs must not burn the full bag_duration budget.
            wait_timeout = min(remaining, leg_cap) if is_intermediate else remaining
            if cfg.multi_goal and len(legs) > 1:
                print(
                    f"[info] Driving leg {leg_idx + 1}/{len(legs)} "
                    f"(leg_wait={wait_timeout:.0f}s, overall remaining={remaining:.0f}s)..."
                )
            if cfg.end_condition == "route_arrived" or is_intermediate:
                end_status, detail = wait_with_stuck_recovery(wait_timeout, _leg)
                if end_status != "done":
                    if (
                        cfg.end_condition == "bag_duration"
                        and time.time() >= drive_deadline
                    ):
                        print(f"[info] bag_duration complete ({detail})")
                        status = "done"
                        error = None
                        break
                    error = detail if len(legs) == 1 else f"leg_{leg_idx + 1}:{detail}"
                    status = "failed"
                    break
                print(f"[info] Leg {leg_idx + 1} ARRIVED ({detail})")
                time.sleep(cfg.post_arrival_sec)
            else:
                # Final leg with bag_duration: wait remaining time; ARRIVED early-exits.
                # stuck_abort_sec still ends early if ego never moves.
                print(
                    f"[info] Waiting bag_duration timeout {wait_timeout:.0f}s "
                    f"(blinker nudge enabled={cfg.stuck_blinker_nudge}, "
                    f"forward_shift={cfg.stuck_reposition_forward_m:.1f}m "
                    f"after {cfg.stuck_reposition_trigger_sec:.0f}s stop)..."
                )
                end_status, detail = wait_with_stuck_recovery(wait_timeout, _leg)
                if end_status == "done":
                    print(f"[info] ARRIVED before bag_duration ({detail})")
                    time.sleep(cfg.post_arrival_sec)
                    status = "done"
                    error = None
                elif end_status == "stuck":
                    print(f"[warn] Ending early — {detail}")
                    status = "failed"
                    error = detail
                else:
                    print(f"[info] bag_duration complete ({detail})")
                    status = "done"
                    error = None
                break

            # More legs? Clear + set next goal, then re-engage.
            if leg_idx + 1 < len(legs) and legs[leg_idx + 1] is not None:
                next_goal = route_goals[leg_idx + 1]
                adv_ok, adv_msg = advance_route(cfg, env, next_goal)
                if not adv_ok:
                    error = f"leg_{leg_idx + 2}_advance:{adv_msg}"
                    status = "failed"
                    break
                ok, engage_msg = engage_autonomous(
                    env, timeout_sec=cfg.auto_engage_timeout_sec
                )
                if not ok:
                    print(f"[warn] {engage_msg} after advance — continuing")
                else:
                    print(f"[info] {engage_msg} (leg {leg_idx + 2})")

        if status == "done":
            error = None

    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        print(f"[error] {error}")

    finally:
        if live_monitor is not None:
            live_monitor.stop(final_note=status if status else "stopped")
        # Shutdown order: record → reproducer → psim
        for proc in reversed(procs):
            stop_process_group(proc, grace_sec=10.0)
            close_log_handle(proc)

    end_iso = utc_now_iso()
    bag_path_out = None
    if output_bag.exists():
        bag_path_out = str(output_bag)

    write_job_status(
        cfg.output_dir,
        {
            "status": status,
            "mode": "reproducer",
            "start_time": start_iso,
            "end_time": end_iso,
            "error": error,
            "domain_id": cfg.domain_id,
            "bag_path": str(bag_path),
            "output_bag": bag_path_out,
            "duration_sec": time.time() - t0,
        },
    )
    return JobResult(status, start_iso, end_iso, error, bag_path_out, time.time() - t0)



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1: run one Diffusion Planner evaluation job "
        "(psim + perception_reproducer + ros2 bag record)."
    )
    p.add_argument("--model_config", type=Path, required=True, help="diffusion_planner.param.yaml")
    p.add_argument("--bag_path", type=Path, default=None, help="Input rosbag (reproducer mode)")
    p.add_argument("--scenario_path", type=Path, default=None, help="Scenario file (scenario_simulator mode)")
    p.add_argument(
        "--mode",
        choices=("reproducer", "scenario_simulator"),
        default="reproducer",
    )
    p.add_argument("--output_dir", type=Path, required=True, help="Per-job output directory")
    p.add_argument("--domain_id", type=int, required=True, help="ROS_DOMAIN_ID for this job")
    p.add_argument("--map_path", type=Path, default=Path("/opt/autoware/maps"))
    p.add_argument("--vehicle_model", default="lv828l")
    p.add_argument("--sensor_model", default="aip_x2_gen2")
    p.add_argument(
        "--vehicle_id",
        default=None,
        help="Vehicle-specific ID for individual_params (e.g. 6_lv828l). "
        "Also sets VEHICLE_ID in the job environment.",
    )
    p.add_argument("--topics-yaml", type=Path, help="Override default topic names")
    p.add_argument(
        "--end_condition",
        choices=("route_arrived", "bag_duration"),
        default="route_arrived",
    )
    p.add_argument("--bag_duration_margin_sec", type=float, default=30.0)
    p.add_argument("--route_timeout_sec", type=float, default=900.0)
    p.add_argument("--psim_startup_sec", type=float, default=120.0)
    p.add_argument("--route_setup_timeout_sec", type=float, default=90.0)
    p.add_argument("--auto_engage_timeout_sec", type=float, default=60.0)
    p.add_argument("--perception_warmup_sec", type=float, default=5.0)
    p.add_argument("--perception_ready_timeout_sec", type=float, default=45.0)
    p.add_argument("--perception_ready_min_objects", type=int, default=1)
    p.add_argument("--perception_ready_stable_sec", type=float, default=2.0)
    p.add_argument(
        "--require_perception_ready",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail the job if tracked objects never appear (default: true).",
    )
    p.add_argument(
        "--bag_load_base_sec",
        type=float,
        default=120.0,
        help="Base wait budget for full-bag perception_reproducer load.",
    )
    p.add_argument(
        "--bag_load_duration_factor",
        type=float,
        default=1.0,
        help="Extra wait = bag_duration * this factor (added to bag_load_base_sec).",
    )
    p.add_argument(
        "--bag_load_timeout_cap_sec",
        type=float,
        default=3600.0,
        help="Hard cap on perception load wait (seconds).",
    )
    p.add_argument(
        "--reproducer_search_radius_m",
        type=float,
        default=0.0,
        help=(
            "perception_reproducer -r (meters). 0 = always nearest bag pose "
            "(recommended when ego path may diverge; avoids empty objects)."
        ),
    )
    p.add_argument(
        "--reproducer_cool_down_sec",
        type=float,
        default=80.0,
        help="perception_reproducer -c (only used when search_radius > 0).",
    )
    p.add_argument(
        "--multi_goal",
        action="store_true",
        help="Replay multiple goals from one bag (bus-stop legs).",
    )
    p.add_argument(
        "--multi_goal_source",
        choices=("auto", "goal_topic", "stop_segments", "stop_points"),
        default="auto",
    )
    p.add_argument("--multi_goal_stop_speed_mps", type=float, default=0.2)
    p.add_argument("--multi_goal_stop_min_sec", type=float, default=8.0)
    p.add_argument("--multi_goal_min_spacing_m", type=float, default=15.0)
    p.add_argument(
        "--multi_goal_stop_order",
        default="",
        help="Comma-separated stop names from stop_points.csv (order = route legs).",
    )
    p.add_argument(
        "--stop_points_csv",
        default="stop_points.csv",
        help="Stop points file under map_path (or absolute path).",
    )
    p.add_argument("--stuck_abort_sec", type=float, default=90.0)
    p.add_argument("--stuck_reposition_trigger_sec", type=float, default=30.0)
    p.add_argument("--stuck_reposition_forward_m", type=float, default=5.0)
    p.add_argument("--stuck_reposition_max_count", type=int, default=0)
    p.add_argument("--multi_goal_arrival_tolerance_m", type=float, default=5.0)
    p.add_argument("--multi_goal_leg_timeout_sec", type=float, default=300.0)
    p.add_argument("--architecture_type", default="awf/universe/20250130")
    p.add_argument("--scenario_timeout_sec", type=float, default=300.0)
    p.add_argument("--skip_route_setup", action="store_true")
    p.add_argument("--rviz", action="store_true", help="Launch RViz (off by default for headless)")
    p.add_argument("--dry_run", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> JobConfig:
    mode = args.mode
    if mode == "reproducer" and not args.bag_path:
        raise SystemExit("--bag_path is required for mode=reproducer")
    if mode == "scenario_simulator" and not args.scenario_path:
        raise SystemExit("--scenario_path is required for mode=scenario_simulator")
    return JobConfig(
        model_config=args.model_config.expanduser(),
        bag_path=args.bag_path.expanduser() if args.bag_path else None,
        scenario_path=args.scenario_path.expanduser() if args.scenario_path else None,
        mode=mode,
        output_dir=args.output_dir.expanduser(),
        domain_id=args.domain_id,
        map_path=args.map_path.expanduser(),
        vehicle_model=args.vehicle_model,
        sensor_model=args.sensor_model,
        vehicle_id=args.vehicle_id,
        topics=load_topics(args.topics_yaml),
        end_condition=args.end_condition,
        bag_duration_margin_sec=args.bag_duration_margin_sec,
        route_timeout_sec=args.route_timeout_sec,
        psim_startup_sec=args.psim_startup_sec,
        route_setup_timeout_sec=args.route_setup_timeout_sec,
        auto_engage_timeout_sec=args.auto_engage_timeout_sec,
        perception_warmup_sec=args.perception_warmup_sec,
        perception_ready_timeout_sec=args.perception_ready_timeout_sec,
        perception_ready_min_objects=args.perception_ready_min_objects,
        perception_ready_stable_sec=args.perception_ready_stable_sec,
        require_perception_ready=bool(args.require_perception_ready),
        bag_load_base_sec=float(args.bag_load_base_sec),
        bag_load_duration_factor=float(args.bag_load_duration_factor),
        bag_load_timeout_cap_sec=float(args.bag_load_timeout_cap_sec),
        reproducer_search_radius_m=args.reproducer_search_radius_m,
        reproducer_cool_down_sec=float(args.reproducer_cool_down_sec),
        multi_goal=bool(args.multi_goal),
        multi_goal_source=str(args.multi_goal_source),
        multi_goal_stop_speed_mps=float(args.multi_goal_stop_speed_mps),
        multi_goal_stop_min_sec=float(args.multi_goal_stop_min_sec),
        multi_goal_min_spacing_m=float(args.multi_goal_min_spacing_m),
        multi_goal_stop_order=[
            s.strip() for s in str(args.multi_goal_stop_order or "").split(",") if s.strip()
        ]
        or None,
        stop_points_csv=str(args.stop_points_csv),
        stuck_abort_sec=float(args.stuck_abort_sec),
        stuck_reposition_trigger_sec=float(args.stuck_reposition_trigger_sec),
        stuck_reposition_forward_m=float(args.stuck_reposition_forward_m),
        stuck_reposition_max_count=int(args.stuck_reposition_max_count),
        multi_goal_arrival_tolerance_m=float(args.multi_goal_arrival_tolerance_m),
        multi_goal_leg_timeout_sec=float(args.multi_goal_leg_timeout_sec),
        architecture_type=args.architecture_type,
        scenario_timeout_sec=args.scenario_timeout_sec,
        dry_run=args.dry_run,
        skip_route_setup=args.skip_route_setup,
        rviz=args.rviz,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_single_job(config_from_args(args))
    print(
        f"[done] status={result.status} duration={result.duration_sec:.1f}s "
        f"bag={result.output_bag} error={result.error}"
    )
    return 0 if result.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
