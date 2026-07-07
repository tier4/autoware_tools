#!/usr/bin/env python3

# Copyright 2025 TIER IV, Inc.
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

"""Batch-evaluate diffusion planner models with planning simulator and perception reproducer."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import rclpy
import yaml
from autoware_adapi_v1_msgs.msg import OperationModeState
from autoware_adapi_v1_msgs.msg import RouteState as AdapiRouteState
from autoware_perception_msgs.msg import TrackedObjects
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rosbag_utils import bag_label
from rosbag_utils import bag_trace_dir
from rosbag_utils import bag_video_path
from rosbag_utils import discover_rosbags
from rosbag_utils import get_bag_duration_sec
from rosbag_utils import trace_logs_complete
from route_setup import disengage_for_next_bag
from route_setup import engage_autonomous_mode
from route_setup import setup_route_for_bag
from route_setup import verify_route_is_set
from route_setup import wait_for_autonomous_mode
from tier4_planning_msgs.msg import RouteState as MissionRouteState


@dataclass
class EvalConfig:
    rosbag_dir: Path
    output_dir: Path
    map_path: str
    vehicle_model: str
    sensor_model: str
    manage_psim: bool
    psim_startup_sec: float
    route_timeout_sec: float
    route_setup_timeout_sec: float
    route_service_wait_sec: float
    route_backend: str
    goal_min_move_m: float
    start_pose_offset_m: float
    stop_point_goal_fallback: bool
    stop_points_csv: str
    stop_point_goal_min_dist_m: float
    stop_point_snap_goal_m: float
    stop_point_waypoint_fallback: bool
    route_stop_order: list[str] | None
    start_pose_stop_fallback: bool
    start_pose_stop_min_dist_m: float
    post_arrival_sec: float
    bag_duration_margin_sec: float
    skip_existing: bool
    record_video: bool
    display: str
    video_size: str
    video_capture: str
    video_window_name: str
    video_capture_offset_x: int
    video_capture_offset_y: int
    record_trajectories: bool
    auto_engage: bool
    auto_engage_timeout_sec: float
    perception_ready_before_engage: bool
    perception_warmup_sec: float
    perception_ready_timeout_sec: float
    perception_ready_min_objects: int
    perception_ready_stable_sec: float
    reproducer_search_radius: float
    reproducer_cool_down: float
    stuck_timeout_sec: float
    stuck_speed_threshold: float
    run_grace_sec: float
    video_start_after_engage: bool
    models: list[dict[str, Any]]
    param_template_path: Path | None
    diffusion_planner_param_deploy_path: Path | None


@dataclass
class BagRunResult:
    model_name: str
    bag_path: Path
    status: str
    duration_sec: float
    video_path: str
    trace_path: str
    message: str


class RouteWaiter(Node):
    def __init__(self) -> None:
        super().__init__("diffusion_planner_batch_eval_route_waiter")
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.adapi_route_state: int | None = None
        self.mission_route_state: int | None = None
        self.create_subscription(
            AdapiRouteState, "/api/routing/state", self._on_adapi_route_state, qos
        )
        self.create_subscription(
            MissionRouteState,
            "/planning/mission_planning/route_selector/main/state",
            self._on_mission_route_state,
            qos,
        )

    def _on_adapi_route_state(self, msg: AdapiRouteState) -> None:
        self.adapi_route_state = msg.state

    def _on_mission_route_state(self, msg: MissionRouteState) -> None:
        self.mission_route_state = msg.state

    def has_arrived(self) -> bool:
        return (
            self.adapi_route_state == AdapiRouteState.ARRIVED
            or self.mission_route_state == MissionRouteState.ARRIVED
        )


class EgoMonitor(Node):
    def __init__(self) -> None:
        super().__init__("diffusion_planner_batch_eval_ego_monitor")
        durable_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.speed_mps = 0.0
        self.operation_mode: int | None = None
        self.autoware_control_enabled = False
        self.create_subscription(
            Odometry, "/localization/kinematic_state", self._on_odom, 10
        )
        self.create_subscription(
            OperationModeState,
            "/api/operation_mode/state",
            self._on_operation_mode,
            durable_qos,
        )

    def _on_odom(self, msg: Odometry) -> None:
        twist = msg.twist.twist.linear
        self.speed_mps = math.hypot(twist.x, twist.y)

    def _on_operation_mode(self, msg: OperationModeState) -> None:
        self.operation_mode = msg.mode
        self.autoware_control_enabled = msg.is_autoware_control_enabled

    def mode_label(self) -> str:
        labels = {
            OperationModeState.UNKNOWN: "UNKNOWN",
            OperationModeState.STOP: "STOP",
            OperationModeState.AUTONOMOUS: "AUTO",
            OperationModeState.LOCAL: "LOCAL",
            OperationModeState.REMOTE: "REMOTE",
        }
        if self.operation_mode is None:
            return "?"
        return labels.get(self.operation_mode, str(self.operation_mode))


class PerceptionReadyMonitor(Node):
    def __init__(self) -> None:
        super().__init__("diffusion_planner_batch_eval_perception_ready")
        sensor_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.object_count = 0
        self.message_count = 0
        self.create_subscription(
            TrackedObjects,
            "/perception/object_recognition/tracking/objects",
            self._on_objects,
            sensor_qos,
        )

    def _on_objects(self, msg: TrackedObjects) -> None:
        self.message_count += 1
        self.object_count = len(msg.objects)


def wait_for_perception_ready(config: EvalConfig) -> tuple[bool, str]:
    """Wait until reproducer has published a stable set of tracked objects."""
    if not rclpy.ok():
        rclpy.init()

    monitor = PerceptionReadyMonitor()
    deadline = time.time() + config.perception_ready_timeout_sec
    warmup_until = time.time() + config.perception_warmup_sec
    stable_since: float | None = None
    last_count = -1
    last_log = 0.0

    try:
        while time.time() < deadline:
            rclpy.spin_once(monitor, timeout_sec=0.25)
            now = time.time()
            if now < warmup_until:
                continue

            count = monitor.object_count
            if count < config.perception_ready_min_objects:
                stable_since = None
                last_count = count
            elif count == last_count:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= config.perception_ready_stable_sec:
                    return (
                        True,
                        f"perception_ready objects={count} "
                        f"after_{now - (deadline - config.perception_ready_timeout_sec):.0f}s",
                    )
            else:
                stable_since = None
                last_count = count

            if now - last_log >= 10.0:
                elapsed = now - (deadline - config.perception_ready_timeout_sec)
                print(
                    f"[info] Waiting for perception... {elapsed:.0f}s "
                    f"objects={count} msgs={monitor.message_count}"
                )
                last_log = now
    finally:
        monitor.destroy_node()

    return (
        False,
        f"perception_ready_timeout objects={monitor.object_count} "
        f"msgs={monitor.message_count}",
    )


def wait_for_bag_run_end(config: EvalConfig, timeout_sec: float) -> tuple[str, str]:
    """Wait until route arrives, ego is stuck, or timeout. Returns (status, detail)."""
    if not rclpy.ok():
        rclpy.init()

    route_waiter = RouteWaiter()
    ego_monitor = EgoMonitor()
    deadline = time.time() + timeout_sec
    grace_until = time.time() + config.run_grace_sec
    stuck_since: float | None = None
    not_auto_since: float | None = None
    last_log = 0.0

    # Let DDS deliver operation_mode before the first status line.
    spin_until = time.time() + 3.0
    while time.time() < spin_until and rclpy.ok():
        rclpy.spin_once(route_waiter, timeout_sec=0.25)
        rclpy.spin_once(ego_monitor, timeout_sec=0.25)

    try:
        while time.time() < deadline:
            rclpy.spin_once(route_waiter, timeout_sec=0.25)
            rclpy.spin_once(ego_monitor, timeout_sec=0.25)

            if route_waiter.has_arrived():
                return "success", "route_arrived"

            now = time.time()
            if now > grace_until:
                if ego_monitor.operation_mode != OperationModeState.AUTONOMOUS:
                    if not_auto_since is None:
                        not_auto_since = now
                    elif now - not_auto_since >= 10.0:
                        return (
                            "stuck",
                            f"operation_mode_{ego_monitor.mode_label()}_not_autonomous_for_10s",
                        )
                else:
                    not_auto_since = None

                if ego_monitor.speed_mps < config.stuck_speed_threshold:
                    if stuck_since is None:
                        stuck_since = now
                    elif now - stuck_since >= config.stuck_timeout_sec:
                        return (
                            "stuck",
                            f"ego_speed_below_{config.stuck_speed_threshold}mps_for_"
                            f"{config.stuck_timeout_sec:.0f}s "
                            f"(mode={ego_monitor.mode_label()})",
                        )
                else:
                    stuck_since = None
            else:
                stuck_since = None
                not_auto_since = None

            if now - last_log >= 15.0:
                elapsed = now - (deadline - timeout_sec)
                print(
                    f"[info] Running bag... {elapsed:.0f}s / {timeout_sec:.0f}s "
                    f"speed={ego_monitor.speed_mps:.2f}m/s "
                    f"mode={ego_monitor.mode_label()} "
                    f"(adapi={route_waiter.adapi_route_state}, "
                    f"mission={route_waiter.mission_route_state})"
                )
                last_log = now
    finally:
        route_waiter.destroy_node()
        ego_monitor.destroy_node()

    return "timeout", f"no_arrival_after_{timeout_sec:.0f}s"


def load_config(path: Path) -> EvalConfig:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return EvalConfig(
        rosbag_dir=Path(raw["rosbag_dir"]).expanduser(),
        output_dir=Path(raw["output_dir"]).expanduser(),
        map_path=str(raw.get("map_path", "/opt/autoware/maps")),
        vehicle_model=str(raw.get("vehicle_model", "lv828l")),
        sensor_model=str(raw.get("sensor_model", "aip_x2_gen2")),
        manage_psim=bool(raw.get("manage_psim", False)),
        psim_startup_sec=float(raw.get("psim_startup_sec", 90.0)),
        route_timeout_sec=float(raw.get("route_timeout_sec", 900.0)),
        route_setup_timeout_sec=float(raw.get("route_setup_timeout_sec", 60.0)),
        route_service_wait_sec=float(raw.get("route_service_wait_sec", 120.0)),
        route_backend=str(raw.get("route_backend", "auto")),
        goal_min_move_m=float(raw.get("goal_min_move_m", 0.1)),
        start_pose_offset_m=float(raw.get("start_pose_offset_m", 0.0)),
        stop_point_goal_fallback=bool(raw.get("stop_point_goal_fallback", True)),
        stop_points_csv=str(raw.get("stop_points_csv", "stop_points.csv")),
        stop_point_goal_min_dist_m=float(raw.get("stop_point_goal_min_dist_m", 0.5)),
        stop_point_snap_goal_m=float(raw.get("stop_point_snap_goal_m", 2.0)),
        stop_point_waypoint_fallback=bool(raw.get("stop_point_waypoint_fallback", True)),
        route_stop_order=list(raw["route_stop_order"]) if raw.get("route_stop_order") else None,
        start_pose_stop_fallback=bool(raw.get("start_pose_stop_fallback", True)),
        start_pose_stop_min_dist_m=float(raw.get("start_pose_stop_min_dist_m", 1.0)),
        post_arrival_sec=float(raw.get("post_arrival_sec", 5.0)),
        bag_duration_margin_sec=float(raw.get("bag_duration_margin_sec", 30.0)),
        skip_existing=bool(raw.get("skip_existing", True)),
        record_video=bool(raw.get("record_video", True)),
        display=str(raw.get("display", "auto")),
        video_size=str(raw.get("video_size", "auto")),
        video_capture=str(raw.get("video_capture", "rviz")),
        video_window_name=str(raw.get("video_window_name", "rviz")),
        video_capture_offset_x=int(raw.get("video_capture_offset_x", 0)),
        video_capture_offset_y=int(raw.get("video_capture_offset_y", 0)),
        record_trajectories=bool(raw.get("record_trajectories", True)),
        auto_engage=bool(raw.get("auto_engage", True)),
        auto_engage_timeout_sec=float(raw.get("auto_engage_timeout_sec", 60.0)),
        perception_ready_before_engage=bool(raw.get("perception_ready_before_engage", False)),
        perception_warmup_sec=float(raw.get("perception_warmup_sec", 5.0)),
        perception_ready_timeout_sec=float(raw.get("perception_ready_timeout_sec", 45.0)),
        perception_ready_min_objects=int(raw.get("perception_ready_min_objects", 1)),
        perception_ready_stable_sec=float(raw.get("perception_ready_stable_sec", 2.0)),
        reproducer_search_radius=float(raw.get("reproducer_search_radius", 0.0)),
        reproducer_cool_down=float(raw.get("reproducer_cool_down", 80.0)),
        stuck_timeout_sec=float(raw.get("stuck_timeout_sec", 45.0)),
        stuck_speed_threshold=float(raw.get("stuck_speed_threshold", 0.2)),
        run_grace_sec=float(raw.get("run_grace_sec", 20.0)),
        video_start_after_engage=bool(raw.get("video_start_after_engage", False)),
        models=list(raw.get("models", [])),
        param_template_path=(
            Path(raw["param_template_path"]).expanduser() if raw.get("param_template_path") else None
        ),
        diffusion_planner_param_deploy_path=(
            Path(raw["diffusion_planner_param_deploy_path"]).expanduser()
            if raw.get("diffusion_planner_param_deploy_path")
            else None
        ),
    )


def default_param_template_path() -> Path:
    prefix = subprocess.check_output(
        ["ros2", "pkg", "prefix", "autoware_launch"], text=True, stderr=subprocess.DEVNULL
    ).strip()
    return (
        Path(prefix)
        / "share/autoware_launch/config/planning/neural_net_planner/diffusion_planner.param.yaml"
    )


def default_param_deploy_path() -> Path:
    return default_param_template_path()


def patch_param_yaml(
    template_path: Path,
    output_path: Path,
    onnx_path: str,
    args_path: str,
    model: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for line in template_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("onnx_model_path:"):
            lines.append(f"    onnx_model_path: {onnx_path}")
        elif stripped.startswith("args_path:"):
            lines.append(f"    args_path: {args_path}")
        elif stripped.startswith("ignore_neighbors:") and "ignore_neighbors" in model:
            lines.append(f"    ignore_neighbors: {str(model['ignore_neighbors']).lower()}")
        else:
            lines.append(line)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_model_param(
    model: dict[str, Any],
    cache_dir: Path,
    config: EvalConfig,
) -> Path:
    model_name = str(model["name"])
    if "param_yaml" in model:
        source = Path(model["param_yaml"]).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"param_yaml not found for model {model_name}: {source}")
        target = cache_dir / f"{model_name}.param.yaml"
        shutil.copy2(source, target)
        return target

    onnx_path = str(model["onnx_model_path"])
    args_path = str(model["args_path"])
    template = config.param_template_path or default_param_template_path()
    if not template.is_file():
        raise FileNotFoundError(f"Diffusion planner param template not found: {template}")

    target = cache_dir / f"{model_name}.param.yaml"
    patch_param_yaml(template, target, onnx_path, args_path, model)
    return target


def deploy_param_file(generated_param: Path, deploy_path: Path) -> Path | None:
    backup_path = deploy_path.with_suffix(deploy_path.suffix + ".batch_eval_backup")
    if not backup_path.exists():
        shutil.copy2(deploy_path, backup_path)
    shutil.copy2(generated_param, deploy_path)
    return backup_path


def restore_param_file(deploy_path: Path, backup_path: Path | None) -> None:
    if backup_path is None or not backup_path.exists():
        return
    shutil.copy2(backup_path, deploy_path)
    backup_path.unlink()


def wait_for_psim_services(timeout_sec: float) -> bool:
    required_services = {"/localization/initialize", "/api/routing/set_route_points"}
    required_topics = {"/localization/kinematic_state", "/tf"}
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            service_output = subprocess.check_output(
                ["ros2", "service", "list"], text=True, stderr=subprocess.DEVNULL
            )
            services = set(service_output.splitlines())
            if not required_services.issubset(services):
                time.sleep(2.0)
                continue

            topic_output = subprocess.check_output(
                ["ros2", "topic", "list"], text=True, stderr=subprocess.DEVNULL
            )
            topics = set(topic_output.splitlines())
            if required_topics.issubset(topics):
                time.sleep(5.0)
                return True
        except subprocess.CalledProcessError:
            pass
        time.sleep(2.0)
    return False


def psim_services_available() -> bool:
    return wait_for_psim_services(3.0)


def launch_psim(config: EvalConfig) -> subprocess.Popen[Any]:
    command = [
        "ros2",
        "launch",
        "autoware_launch",
        "planning_simulator.launch.xml",
        f"map_path:={config.map_path}",
        f"vehicle_model:={config.vehicle_model}",
        f"sensor_model:={config.sensor_model}",
        "planning_setting:=diffusion_planner",
    ]
    print(f"[info] Launching planning simulator: {' '.join(command)}")
    return subprocess.Popen(command, preexec_fn=os.setsid)


def ensure_psim_for_batch(config: EvalConfig, dry_run: bool) -> tuple[subprocess.Popen[Any] | None, bool]:
    """Return (psim_process, should_manage_psim). Never launch a second psim if one is running."""
    already_running = psim_services_available()

    if config.manage_psim:
        if already_running:
            print(
                "[warn] Planning simulator is already running — reusing it.\n"
                "       Do NOT launch psim manually when manage_psim: true.\n"
                "       Set manage_psim: false in your config if you prefer to start psim yourself."
            )
            return None, False
        if dry_run:
            return None, True
        return launch_psim(config), True

    if not already_running:
        print(
            "[error] manage_psim is false but planning simulator is not running.\n"
            "        Terminal 1: ros2 launch autoware_launch planning_simulator.launch.xml "
            "planning_setting:=diffusion_planner\n"
            "        Then rerun batch_diffusion_eval.py"
        )
        return None, False

    print("[info] Using planning simulator already running in another terminal.")
    return None, False


def stop_process_group(process: subprocess.Popen[Any] | None, grace_sec: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        # SIGTERM avoids rclpy double-shutdown tracebacks from SIGINT in child nodes.
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=grace_sec)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _display_variants(display: str) -> list[str]:
    variants = [display]
    if display.endswith(".0") and len(display) > 2:
        variants.append(display[:-2])
    elif display.startswith(":") and display[1:].isdigit():
        variants.append(f"{display}.0")
    return variants


def find_working_display(display: str) -> tuple[str | None, list[str]]:
    """Return the first X11 display that responds to xdpyinfo."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    if display in ("", "auto"):
        env_display = os.environ.get("DISPLAY", "")
        for variant in _display_variants(env_display) if env_display else []:
            add(variant)
    else:
        for variant in _display_variants(display):
            add(variant)

    for fallback in (":1", ":1.0", ":0", ":0.0"):
        add(fallback)

    tried: list[str] = []
    for candidate in candidates:
        tried.append(candidate)
        if display_is_available(candidate):
            return candidate, tried
    return None, tried


def find_display_for_video_capture(config: EvalConfig) -> tuple[str | None, list[str], str]:
    """Pick the X display to capture. With dual monitors, prefer where RViz actually is."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    if config.display not in ("", "auto"):
        for variant in _display_variants(config.display):
            add(variant)
    env_display = os.environ.get("DISPLAY", "")
    if env_display:
        for variant in _display_variants(env_display):
            add(variant)
    for fallback in (":0", ":0.0", ":1", ":1.0"):
        add(fallback)

    tried: list[str] = []
    if config.video_capture.lower() == "rviz":
        for candidate in candidates:
            tried.append(candidate)
            if not display_is_available(candidate):
                continue
            if find_window_geometry(candidate, config.video_window_name) is not None:
                note = f"rviz window found on {candidate}"
                if env_display and candidate not in _display_variants(env_display):
                    note += f" (Terminal $DISPLAY={env_display})"
                return candidate, tried, note

    working, tried_working = find_working_display(config.display)
    tried.extend(item for item in tried_working if item not in tried)
    if working is not None:
        return working, tried, f"using display {working}"
    return None, tried, "no display available"


@dataclass
class VideoCaptureSpec:
    display_input: str
    video_size: str
    grab_x: int
    grab_y: int
    description: str
    window_id: str | None = None


@dataclass
class WindowMatch:
    window_id: str
    x_pos: int
    y_pos: int
    width: int
    height: int
    title: str


_WINDOW_GEOM_RE = re.compile(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)")
_WINDOW_ID_RE = re.compile(r'^\s*(0x[0-9a-fA-F]+)\s+"([^"]+)"')


def _query_window_absolute_geometry(
    display: str, window_id: str
) -> tuple[int, int, int, int] | None:
    try:
        result = subprocess.run(
            ["xwininfo", "-id", window_id, "-display", display],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None

    abs_x: int | None = None
    abs_y: int | None = None
    width: int | None = None
    height: int | None = None
    for line in result.stdout.splitlines():
        if "Absolute upper-left X:" in line:
            abs_x = int(line.split(":", 1)[1].strip())
        elif "Absolute upper-left Y:" in line:
            abs_y = int(line.split(":", 1)[1].strip())
        elif line.strip().startswith("Width:"):
            width = int(line.split(":", 1)[1].strip())
        elif line.strip().startswith("Height:"):
            height = int(line.split(":", 1)[1].strip())

    if None in (abs_x, abs_y, width, height):
        return None
    return abs_x, abs_y, width, height


def find_window_geometry(display: str, name_hint: str) -> WindowMatch | None:
    """Return absolute geometry and X11 window id for the largest matching window."""
    if not shutil.which("xwininfo"):
        return None
    try:
        result = subprocess.run(
            ["xwininfo", "-root", "-tree", "-display", display],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None

    hint = name_hint.lower()
    matches: list[WindowMatch] = []
    for line in result.stdout.splitlines():
        id_match = _WINDOW_ID_RE.match(line)
        if id_match is None or hint not in line.lower():
            continue
        window_id, title = id_match.groups()
        geometry = _query_window_absolute_geometry(display, window_id)
        if geometry is None:
            # Tree-line geometry is relative to the parent — do not use for x11grab.
            continue
        x_pos, y_pos, width, height = geometry
        if width < 200 or height < 200:
            continue
        matches.append(
            WindowMatch(
                window_id=window_id,
                x_pos=x_pos,
                y_pos=y_pos,
                width=width,
                height=height,
                title=title,
            )
        )

    if not matches:
        return None
    best = max(matches, key=lambda item: item.width * item.height)
    return best


def normalize_x11_display(display: str) -> str:
    if display.startswith(":") and "." not in display:
        return f"{display}.0"
    return display


def _apply_capture_offsets(
    x_pos: int, y_pos: int, width: int, height: int, config: EvalConfig
) -> tuple[int, int, int, int]:
    x_pos += config.video_capture_offset_x
    y_pos += config.video_capture_offset_y
    width = _even_dimension(width)
    height = _even_dimension(height)
    return x_pos, y_pos, width, height


def _even_dimension(value: int) -> int:
    return max(2, value - (value % 2))


def get_display_dimensions(display: str) -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            ["xdpyinfo", "-display", display],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if "dimensions:" not in line:
            continue
        token = line.split()[1]
        width_str, height_str = token.split("x", 1)
        return int(width_str), int(height_str)
    return None


def resolve_video_capture(config: EvalConfig, display: str) -> VideoCaptureSpec:
    capture_mode = config.video_capture.lower()
    if capture_mode == "rviz":
        window = find_window_geometry(display, config.video_window_name)
        if window is not None:
            x_pos, y_pos, width, height = _apply_capture_offsets(
                window.x_pos, window.y_pos, window.width, window.height, config
            )
            return VideoCaptureSpec(
                display_input=normalize_x11_display(display),
                video_size=f"{width}x{height}",
                grab_x=x_pos,
                grab_y=y_pos,
                window_id=window.window_id,
                description=(
                    f"rviz window '{window.title}' id={window.window_id} "
                    f"(window_id capture, fallback region +{x_pos},{y_pos} {width}x{height})"
                ),
            )
        print(
            f"[warn] RViz window matching {config.video_window_name!r} not found — "
            "recording full display instead. Launch RViz before batch run."
        )
        capture_mode = "display"

    if config.video_size not in ("", "auto"):
        width_str, height_str = config.video_size.lower().split("x", 1)
        width = _even_dimension(int(width_str))
        height = _even_dimension(int(height_str))
        return VideoCaptureSpec(
            display_input=normalize_x11_display(display),
            video_size=f"{width}x{height}",
            grab_x=config.video_capture_offset_x,
            grab_y=config.video_capture_offset_y,
            description=f"display {display} at configured {width}x{height}",
        )

    dims = get_display_dimensions(display)
    if dims is None:
        return VideoCaptureSpec(
            display_input=normalize_x11_display(display),
            video_size="1920x1080",
            grab_x=config.video_capture_offset_x,
            grab_y=config.video_capture_offset_y,
            description=f"display {display} (fallback 1920x1080)",
        )
    width = _even_dimension(dims[0])
    height = _even_dimension(dims[1])
    return VideoCaptureSpec(
        display_input=normalize_x11_display(display),
        video_size=f"{width}x{height}",
        grab_x=config.video_capture_offset_x,
        grab_y=config.video_capture_offset_y,
        description=f"full display {display} at native {width}x{height}",
    )


def build_ffmpeg_record_command(
    output_path: Path,
    capture: VideoCaptureSpec,
    *,
    duration_sec: float | None = None,
    preset: str = "fast",
) -> list[str]:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "x11grab",
        "-framerate",
        "30",
        "-draw_mouse",
        "0",
    ]
    if capture.window_id:
        command.extend(["-window_id", capture.window_id])
    else:
        command.extend(
            [
                "-grab_x",
                str(capture.grab_x),
                "-grab_y",
                str(capture.grab_y),
                "-video_size",
                capture.video_size,
            ]
        )
    command.extend(["-i", capture.display_input])
    if duration_sec is not None:
        command.extend(["-t", str(duration_sec)])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    return command


def resolve_display(display: str) -> str:
    working, _ = find_working_display(display)
    if working is not None:
        return working
    if display in ("", "auto"):
        return os.environ.get("DISPLAY", ":0")
    return display


def display_is_available(display: str) -> bool:
    try:
        result = subprocess.run(
            ["xdpyinfo", "-display", display],
            capture_output=True,
            timeout=3.0,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return bool(os.environ.get("DISPLAY"))


def mean_video_luma(path: Path, sample_time: float = 0.5) -> float | None:
    """Return mean grayscale [0-255] of one frame; low values suggest a black capture."""
    if not shutil.which("ffmpeg"):
        return None
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-ss",
                str(sample_time),
                "-i",
                str(path),
                "-vframes",
                "1",
                "-vf",
                "scale=32:32,format=gray",
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
            timeout=20.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    data = result.stdout
    return sum(data) / len(data)


def validate_video_file(path: Path, min_bytes: int = 4096) -> tuple[bool, str]:
    if not path.exists():
        return False, "file_missing"
    size = path.stat().st_size
    if size < min_bytes:
        return False, f"too_small ({size} bytes)"

    duration_sec: float | None = None
    if shutil.which("ffprobe"):
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                duration_sec = float(result.stdout.strip())
        except (ValueError, subprocess.TimeoutExpired, OSError):
            duration_sec = None

    luma = mean_video_luma(path, sample_time=0.5)
    if luma is not None and luma < 8.0:
        return False, f"mostly_black (mean_luma={luma:.1f}/255)"

    if duration_sec is not None:
        return True, f"{size / (1024 * 1024):.2f} MB, {duration_sec:.1f}s"

    return True, f"{size / (1024 * 1024):.2f} MB"


def run_ffmpeg_preflight(config: EvalConfig) -> bool:
    if not shutil.which("ffmpeg"):
        print("[error] ffmpeg not found in PATH. Install ffmpeg or set record_video: false.")
        return False

    resolved_display, tried_displays, display_note = find_display_for_video_capture(config)
    if resolved_display is None:
        env_display = os.environ.get("DISPLAY", "<unset>")
        tried = ", ".join(tried_displays) if tried_displays else "(none)"
        print(
            f"[error] No working X11 display found — cannot record video.\n"
            f"       config display={config.display!r}, $DISPLAY={env_display}\n"
            f"       Tried: {tried}\n"
            "       Dual monitor: launch RViz on the external screen, then in Terminal 2 run\n"
            "       `echo $DISPLAY` from the terminal where RViz works and `export DISPLAY=<it>`.\n"
            "       Or set display: :1 (or :0) explicitly in diffusion_batch_config.yaml."
        )
        return False

    if config.video_capture.lower() == "rviz" and "rviz window found" in display_note:
        print(f"[info] {display_note}")
    elif config.display in ("", "auto"):
        env_display = os.environ.get("DISPLAY", "<unset>")
        if env_display != resolved_display:
            print(
                f"[info] Using display {resolved_display} "
                f"(auto-detected; $DISPLAY={env_display})"
            )

    test_path = config.output_dir / ".ffmpeg_preflight.mp4"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    capture = resolve_video_capture(config, resolved_display)
    command = build_ffmpeg_record_command(
        test_path, capture, duration_sec=2.0, preset="ultrafast"
    )
    print(
        f"[info] ffmpeg preflight: recording 2s test clip to {test_path}\n"
        f"       Capture: {capture.description}"
    )
    try:
        subprocess.run(command, check=True, timeout=15.0)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"[error] ffmpeg preflight failed: {error}")
        return False

    ok, detail = validate_video_file(test_path, min_bytes=5000)
    if not ok:
        print(f"[error] ffmpeg preflight produced invalid video ({detail})")
        if "mostly_black" in detail:
            print(
                "[hint] Black capture usually means:\n"
                "       - RViz is on a different monitor ($DISPLAY mismatch) → set display: :0 or :1\n"
                "       - RViz is minimized / behind other windows → keep it visible\n"
                "       - Wayland session (x11grab unreliable) → use X11 session or video_capture: display\n"
                "       - Batch terminal has no access to the GUI display → run batch on the same desktop"
            )
        return False

    print(f"[info] ffmpeg preflight OK ({detail})")
    print(
        "[info] Per-bag videos will be saved as:\n"
        f"       {config.output_dir}/<model_name>/<rosbag_relative_path>.mp4\n"
        "       Example: .../model_baseline/ID1/94eca28e-..._p0900_27.mp4"
    )
    return True


def start_video_recording(config: EvalConfig, output_path: Path) -> subprocess.Popen[Any] | None:
    resolved_display, tried_displays, display_note = find_display_for_video_capture(config)
    if resolved_display is None:
        env_display = os.environ.get("DISPLAY", "<unset>")
        tried = ", ".join(tried_displays) if tried_displays else "(none)"
        print(
            f"[warn] No working X11 display — skipping video recording.\n"
            f"       config display={config.display!r}, $DISPLAY={env_display}, tried: {tried}"
        )
        return None

    if "rviz window found" in display_note:
        print(f"[info] {display_note}")

    capture = resolve_video_capture(config, resolved_display)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_record_command(output_path, capture)
    print(f"[info] Recording video: {output_path}\n       Capture: {capture.description}")
    process = subprocess.Popen(command, preexec_fn=os.setsid, stderr=subprocess.PIPE)
    time.sleep(0.5)
    if process.poll() is not None:
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        detail = f": {stderr}" if stderr else ""
        print(f"[warn] ffmpeg exited immediately — skipping video for {output_path.name}{detail}")
        return None
    return process


def stop_video_recording(process: subprocess.Popen[Any] | None) -> None:
    stop_process_group(process, grace_sec=5.0)


def start_trajectory_logger(
    trace_dir: Path, model_name: str, bag_path: Path
) -> subprocess.Popen[Any]:
    command = [
        "ros2",
        "run",
        "diffusion_planner_batch_eval",
        "trajectory_logger.py",
        "-o",
        str(trace_dir),
        "--model-name",
        model_name,
        "--bag-path",
        str(bag_path),
    ]
    print(f"[info] Starting trajectory logger: {trace_dir}")
    return subprocess.Popen(command, preexec_fn=os.setsid)


def start_perception_reproducer(bag_path: Path, config: EvalConfig) -> subprocess.Popen[Any]:
    command = [
        "ros2",
        "run",
        "planning_debug_tools",
        "perception_reproducer.py",
        "-b",
        str(bag_path),
        "-t",
        "-r",
        str(config.reproducer_search_radius),
        "-c",
        str(config.reproducer_cool_down),
    ]
    print(f"[info] Starting perception reproducer: {' '.join(command)}")
    return subprocess.Popen(command, preexec_fn=os.setsid)


def run_single_bag(
    config: EvalConfig,
    model_name: str,
    bag_path: Path,
    video_path: Path,
    trace_dir: Path,
) -> BagRunResult:
    start = time.time()
    reproducer: subprocess.Popen[Any] | None = None
    recorder: subprocess.Popen[Any] | None = None
    trace_logger: subprocess.Popen[Any] | None = None
    message = ""
    status = "error"

    try:
        engage_in_route_setup = False  # engage after reproducer; early engage is dropped by reproducer startup
        route_ok, route_message = setup_route_for_bag(
            bag_path,
            timeout_sec=config.route_setup_timeout_sec,
            min_move_m=config.goal_min_move_m,
            service_wait_sec=config.route_service_wait_sec,
            backend_preference=config.route_backend,
            auto_engage=engage_in_route_setup,
            start_pose_offset_m=config.start_pose_offset_m,
            map_path=Path(config.map_path),
            stop_point_goal_fallback=config.stop_point_goal_fallback,
            stop_points_csv=config.stop_points_csv,
            stop_point_goal_min_dist_m=config.stop_point_goal_min_dist_m,
            stop_point_snap_goal_m=config.stop_point_snap_goal_m,
            stop_point_waypoint_fallback=config.stop_point_waypoint_fallback,
            route_stop_order=config.route_stop_order,
            start_pose_stop_fallback=config.start_pose_stop_fallback,
            start_pose_stop_min_dist_m=config.start_pose_stop_min_dist_m,
        )
        if not route_ok:
            return BagRunResult(
                model_name,
                bag_path,
                "route_setup_failed",
                time.time() - start,
                "",
                "",
                route_message,
            )

        if config.record_trajectories:
            trace_logger = start_trajectory_logger(trace_dir, model_name, bag_path)

        reproducer = start_perception_reproducer(bag_path, config)

        if config.record_video and not config.video_start_after_engage:
            recorder = start_video_recording(config, video_path)

        autonomous_ready = False
        if config.auto_engage:
            print(
                f"[info] Waiting for perception (warmup {config.perception_warmup_sec:.0f}s, "
                f"stable {config.perception_ready_stable_sec:.0f}s) before Auto..."
            )
            ready_ok, ready_message = wait_for_perception_ready(config)
            if ready_ok:
                print(f"[info] {ready_message}")
            else:
                print(
                    f"[warn] {ready_message} — engaging Auto anyway; "
                    "increase perception_ready_timeout_sec if objects are still missing."
                )

            print(
                f"[info] Engaging Auto (up to {config.auto_engage_timeout_sec:.0f}s)..."
            )
            route_ok, route_verify_message = verify_route_is_set(
                min(config.route_setup_timeout_sec, 15.0)
            )
            if not route_ok:
                print(
                    f"[warn] Skipping Auto engage — route is not SET ({route_verify_message}). "
                    "Check RViz goal marker and route_setup logs."
                )
                route_message = f"{route_message}; engage_skipped: {route_verify_message}"
            else:
                engage_ok, engage_message = engage_autonomous_mode(config.auto_engage_timeout_sec)
                if engage_ok:
                    autonomous_ready = True
                    route_message = f"{route_message}; {route_verify_message}; {engage_message}"
                    print(f"[info] Auto mode engaged: {engage_message}")
                else:
                    print(
                        f"[warn] Auto engage failed ({engage_message}). "
                        "Click Auto in RViz if the vehicle does not move."
                    )
                    route_message = f"{route_message}; engage_warn: {engage_message}"
        elif not config.auto_engage:
            print(
                f"[info] Waiting for Auto mode (up to {config.auto_engage_timeout_sec:.0f}s)..."
            )
            wait_ok, wait_message = wait_for_autonomous_mode(config.auto_engage_timeout_sec)
            if wait_ok:
                autonomous_ready = True
                route_message = f"{route_message}; {wait_message}"
                print(f"[info] Auto mode detected: {wait_message}")
            else:
                print(
                    f"[warn] Auto mode not detected ({wait_message}). "
                    "Engage Auto in RViz to start driving."
                )
                route_message = f"{route_message}; engage_warn: {wait_message}"

        if config.record_video and config.video_start_after_engage:
            if autonomous_ready:
                recorder = start_video_recording(config, video_path)
            else:
                print("[warn] Skipping video — Auto was not engaged.")

        if config.auto_engage and not autonomous_ready:
            print(
                "[warn] Run continues without Auto — ego will likely stay at speed=0. "
                "Check engage_warn in setup message above."
            )

        bag_duration = get_bag_duration_sec(bag_path)
        timeout = config.route_timeout_sec
        if bag_duration is not None:
            timeout = min(timeout, bag_duration + config.bag_duration_margin_sec)
            print(f"[info] Run timeout: {timeout:.0f}s (bag duration {bag_duration:.0f}s)")

        status, run_detail = wait_for_bag_run_end(config, timeout)
        message = f"{run_detail} (setup: {route_message})"
        if status == "success":
            time.sleep(config.post_arrival_sec)
        elif status == "stuck":
            print(
                "[warn] Ego stuck — check run logs for mode=STOP (engage failed) vs mode=AUTO "
                "(perception/planner blockage). Try reproducer_search_radius: 0."
            )

    except Exception as error:  # noqa: BLE001
        status = "error"
        message = str(error)
    finally:
        stop_process_group(reproducer)
        stop_video_recording(recorder)
        stop_process_group(trace_logger)
        if recorder is not None:
            time.sleep(0.5)
        disengage_ok, disengage_message = disengage_for_next_bag(
            min(config.route_setup_timeout_sec, 25.0)
        )
        if disengage_ok:
            print(f"[info] Disengaged for next bag: {disengage_message}")
        else:
            print(
                f"[warn] Could not fully disengage before next bag ({disengage_message}). "
                "Next route setup may fail — stop the vehicle manually if needed."
            )

    duration = time.time() - start
    video_result = ""
    trace_result = ""
    if config.record_video:
        if video_path.exists():
            ok, detail = validate_video_file(video_path)
            if ok:
                video_result = str(video_path)
                print(f"[info] Video saved: {video_path} ({detail})")
            else:
                print(f"[warn] Video invalid after recording: {video_path} ({detail})")
        elif recorder is not None:
            print(f"[warn] No video file created: {video_path}")
    if config.record_trajectories and trace_logs_complete(trace_dir):
        trace_result = str(trace_dir)
        print(f"[info] Trajectory logs saved: {trace_dir}")
    elif config.record_trajectories:
        print(f"[warn] Trajectory logs incomplete: {trace_dir}")
    return BagRunResult(model_name, bag_path, status, duration, video_result, trace_result, message)


def append_log_row(log_path: Path, result: BagRunResult) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(
                [
                    "model_name",
                    "bag_path",
                    "status",
                    "duration_sec",
                    "video_path",
                    "trace_path",
                    "message",
                ]
            )
        writer.writerow(
            [
                result.model_name,
                str(result.bag_path),
                result.status,
                f"{result.duration_sec:.1f}",
                result.video_path,
                result.trace_path,
                result.message,
            ]
        )


def run_batch(config: EvalConfig, dry_run: bool = False) -> int:
    if not config.models:
        print("[error] No models configured.")
        return 1

    bags = discover_rosbags(config.rosbag_dir)
    print(f"[info] Found {len(bags)} rosbag(s) in {config.rosbag_dir}")
    print(f"[info] Evaluating {len(config.models)} model(s)")

    if config.record_video and not dry_run:
        if not run_ffmpeg_preflight(config):
            print("[warn] Continuing without video recording.")
            config = replace(config, record_video=False)

    if not config.manage_psim and len(config.models) > 1:
        print(
            "[warn] manage_psim is false with multiple models: launch psim and swap models "
            "manually between model blocks, or enable manage_psim."
        )

    cache_dir = config.output_dir / ".generated_params"
    log_path = config.output_dir / "batch_eval_log.csv"
    deploy_path = config.diffusion_planner_param_deploy_path or default_param_deploy_path()
    param_backup: Path | None = None
    psim_process: subprocess.Popen[Any] | None = None
    psim_launched_by_tool = False
    failures = 0

    try:
        for model in config.models:
            model_name = str(model["name"])
            model_output_dir = config.output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            generated_param = prepare_model_param(model, cache_dir, config)
            print(f"[info] Model '{model_name}' param: {generated_param}")

            if config.manage_psim:
                if dry_run:
                    print("[dry-run] Would deploy param and start psim if not already running")
                else:
                    param_backup = deploy_param_file(generated_param, deploy_path)
                    psim_process, psim_launched_by_tool = ensure_psim_for_batch(config, dry_run=False)
                    if psim_process is None and not psim_services_available():
                        print(f"[error] planning simulator not available for model {model_name}")
                        failures += 1
                        continue
                    if psim_launched_by_tool and not wait_for_psim_services(config.psim_startup_sec):
                        print(f"[error] planning simulator not ready for model {model_name}")
                        failures += 1
                        stop_process_group(psim_process)
                        psim_process = None
                        psim_launched_by_tool = False
                        continue
            elif not dry_run and not psim_services_available():
                print("[error] planning simulator is not running. Start it first or set manage_psim: true.")
                return 1

            for bag_path in bags:
                label = bag_label(bag_path)
                video_path = bag_video_path(bag_path, config.rosbag_dir, model_output_dir)
                trace_dir = bag_trace_dir(bag_path, config.rosbag_dir, model_output_dir)

                video_ok = (
                    not config.record_video
                    or (video_path.exists() and validate_video_file(video_path)[0])
                )
                trace_ok = not config.record_trajectories or trace_logs_complete(trace_dir)
                if config.skip_existing and video_ok and trace_ok:
                    print(f"[skip] {model_name}/{label} (outputs already exist)")
                    continue

                print(f"[info] Running {model_name} / {label}")
                if config.record_video:
                    print(f"[info] Video output: {video_path}")
                if config.record_trajectories:
                    print(f"[info] Trajectory log output: {trace_dir}")
                if dry_run:
                    print(f"[dry-run] Would run reproducer on {bag_path}")
                    continue

                result = run_single_bag(config, model_name, bag_path, video_path, trace_dir)
                append_log_row(log_path, result)
                print(
                    f"[done] {model_name}/{label}: status={result.status}, "
                    f"duration={result.duration_sec:.1f}s, "
                    f"video={'yes' if result.video_path else 'no'}, "
                    f"trace={'yes' if result.trace_path else 'no'}, "
                    f"message={result.message}"
                )
                if result.status != "success":
                    failures += 1

            if psim_launched_by_tool and psim_process is not None:
                print(f"[info] Stopping planning simulator launched by batch tool after model {model_name}")
                stop_process_group(psim_process)
                psim_process = None
                psim_launched_by_tool = False
                time.sleep(5.0)

    finally:
        if psim_launched_by_tool:
            stop_process_group(psim_process)
        restore_param_file(deploy_path, param_backup)

    if dry_run:
        print("[dry-run] Completed without executing bags.")
        return 0

    print(f"[info] Log written to {log_path}")
    if failures:
        print(f"[warn] Completed with {failures} failure(s).")
        return 1
    print("[info] All runs completed successfully.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate diffusion planner models using planning simulator, "
            "perception reproducer (-p -t), and optional RViz screen recording."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="YAML config file (see share/diffusion_planner_batch_eval/config/example_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without launching psim, reproducer, or ffmpeg",
    )
    parser.add_argument(
        "--test-ffmpeg",
        action="store_true",
        help="Record a 2s test clip and exit (verify DISPLAY + ffmpeg before batch run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.test_ffmpeg:
        sys.exit(0 if run_ffmpeg_preflight(config) else 1)
    sys.exit(run_batch(config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
