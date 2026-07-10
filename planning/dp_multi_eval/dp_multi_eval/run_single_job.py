"""Phase 1 — single evaluation job (psim + reproducer + bag record).

Importable as ``from dp_multi_eval.run_single_job import run_single_job``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from dp_multi_eval.topics import TopicSet, load_topics


def update_job_progress(
    output_dir: Path,
    *,
    phase: str,
    start_time: str,
    domain_id: int,
    bag_path: str | None = None,
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
            "model_config": model_config,
        },
    )


@dataclass
class JobConfig:
    model_config: Path
    bag_path: Path
    output_dir: Path
    domain_id: int
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


def _list_ros_nodes(env: dict[str, str]) -> set[str]:
    out = subprocess.check_output(
        ["ros2", "node", "list"],
        env=_ros_call_env(env),
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=15.0,
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


def wait_for_services(
    env: dict[str, str],
    timeout_sec: float,
    psim: subprocess.Popen[Any] | None = None,
) -> tuple[bool, str]:
    """Wait until psim nodes + routing/localization services are actually up."""
    adapi_required = {"/api/routing/clear_route", "/api/routing/set_route_points"}
    mission_required = {
        "/planning/mission_planning/route_selector/main/clear_route",
        "/planning/mission_planning/route_selector/main/set_waypoint_route",
    }
    localize_required = {"/localization/initialize"}
    deadline = time.time() + timeout_sec
    last_log = 0.0
    while time.time() < deadline:
        if psim is not None and psim.poll() is not None:
            return False, f"planning simulator exited early (code={psim.returncode})"

        try:
            out = subprocess.check_output(
                ["ros2", "service", "list"],
                env=_ros_call_env(env),
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10.0,
            )
            services = set(out.splitlines())
            routing_ready = adapi_required.issubset(services) or mission_required.issubset(
                services
            )
            nodes = _list_ros_nodes(env)
            stack_ready = planning_stack_nodes_ready(nodes)
            if (
                localize_required.issubset(services)
                and routing_ready
                and stack_ready
            ):
                time.sleep(10.0)
                return True, "planning_stack_ready"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            pass
        now = time.time()
        if now - last_log >= 15.0:
            elapsed = now - (deadline - timeout_sec)
            detail = ""
            try:
                nodes = _list_ros_nodes(env)
                detail = (
                    f" nodes(planning={any('mission_planner' in n or 'route_selector' in n for n in nodes)},"
                    f" map={any('/map/' in n for n in nodes)},"
                    f" api={any('adapi' in n for n in nodes)})"
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                detail = " nodes(unavailable)"
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


def wait_for_route_arrived(env: dict[str, str], timeout_sec: float) -> tuple[str, str]:
    """Poll /api/routing/state via ros2 topic echo. Returns (status, detail)."""
    # Prefer python rclpy when available; fall back to timeout by bag duration caller.
    try:
        import rclpy
        from autoware_adapi_v1_msgs.msg import RouteState
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError:
        return "timeout", "rclpy_or_adapi_msgs_unavailable"

    if not rclpy.ok():
        rclpy.init()

    class Waiter(Node):
        def __init__(self) -> None:
            super().__init__("dp_multi_eval_route_waiter")
            self.state: int | None = None
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(RouteState, "/api/routing/state", self._cb, qos)

        def _cb(self, msg: RouteState) -> None:
            self.state = msg.state

    node = Waiter()
    deadline = time.time() + timeout_sec
    last_log = 0.0
    try:
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            if node.state == RouteState.ARRIVED:
                return "done", "route_arrived"
            now = time.time()
            if now - last_log >= 15.0:
                print(
                    f"[info] waiting for route ARRIVED "
                    f"({timeout_sec - (deadline - now):.0f}/{timeout_sec:.0f}s) "
                    f"state={_route_state_label(node.state)}"
                )
                last_log = now
    finally:
        node.destroy_node()
    return "timeout", f"route_not_arrived state={_route_state_label(node.state)}"


def wait_for_perception_ready(cfg: JobConfig) -> tuple[bool, str]:
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

    class Monitor(Node):
        def __init__(self) -> None:
            super().__init__("dp_multi_eval_perception_ready")
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
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
    deadline = time.time() + cfg.perception_ready_timeout_sec
    warmup_until = time.time() + cfg.perception_warmup_sec
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
                        f"perception_ready objects={count} "
                        f"after_{now - (deadline - cfg.perception_ready_timeout_sec):.0f}s",
                    )
            else:
                stable_since = None
                last_count = count

            if now - last_log >= 10.0:
                elapsed = now - (deadline - cfg.perception_ready_timeout_sec)
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


def build_reproducer_command(bag_path: Path, search_radius: float = 0.0) -> list[str]:
    return [
        "ros2",
        "run",
        "planning_debug_tools",
        "perception_reproducer.py",
        "-b",
        str(bag_path),
        "-t",
        "-r",
        str(search_radius),
    ]


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


def setup_route(cfg: JobConfig, env: dict[str, str]) -> tuple[bool, str]:
    """Localization + route in-process (same Python interpreter, shared DDS env)."""
    if cfg.skip_route_setup:
        return True, "route_setup_skipped"

    print(f"[info] Route setup in-process ROS_DOMAIN_ID={env.get('ROS_DOMAIN_ID', '?')}")
    old_env = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        module = _load_route_setup_module(env)
        ok, message = module.setup_route_for_bag(
            bag_path=cfg.bag_path.expanduser().resolve(),
            timeout_sec=cfg.route_setup_timeout_sec,
            service_wait_sec=max(cfg.route_setup_timeout_sec, 60.0),
            backend_preference="adapi",
            auto_engage=False,
            map_path=cfg.map_path,
            stop_point_goal_fallback=True,
            start_pose_stop_fallback=True,
        )
        if ok:
            print(f"Route setup succeeded: {message}")
            return True, "route_setup_ok"
        print(f"Route setup failed:\n{message}")
        return False, f"route_setup_failed: {message}"
    except Exception as exc:  # noqa: BLE001
        return False, f"route_setup_exception: {exc}"
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
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(cfg.domain_id)
    env.setdefault("ROS2_DISABLE_DAEMON", "1")

    try:
        if not cfg.model_config.expanduser().resolve().is_file():
            raise FileNotFoundError(f"model_config not found: {cfg.model_config}")
        if not cfg.bag_path.expanduser().resolve().exists():
            raise FileNotFoundError(f"bag_path not found: {cfg.bag_path}")

        make_model_overlay(cfg.model_config, overlay_root)
        env = env_with_overlay(env, overlay_root, cfg.domain_id)
        vehicle_id = cfg.vehicle_id or env.get("VEHICLE_ID")
        if vehicle_id:
            env["VEHICLE_ID"] = vehicle_id

        bag_duration = get_bag_duration_sec(cfg.bag_path)
        run_timeout = cfg.route_timeout_sec
        if cfg.end_condition == "bag_duration" and bag_duration is not None:
            run_timeout = min(run_timeout, bag_duration + cfg.bag_duration_margin_sec)

        psim_cmd = build_psim_command(cfg)
        repro_cmd = build_reproducer_command(cfg.bag_path)
        record_cmd = build_record_command(output_bag, cfg.topics.record_list())

        print(f"[info] ROS_DOMAIN_ID={cfg.domain_id}")
        if vehicle_id:
            print(f"[info] vehicle_id={vehicle_id}")
        print(f"[info] model_config={cfg.model_config}")
        print(f"[info] bag_path={cfg.bag_path} duration={bag_duration}")
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
            bag_path=str(cfg.bag_path),
            model_config=str(cfg.model_config),
        )
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
            bag_path=str(cfg.bag_path),
            model_config=str(cfg.model_config),
        )
        if not wait_for_routing_callable(env, timeout_sec=45.0):
            raise RuntimeError(
                "routing services not visible in ROS graph after startup waits"
            )

        ok, route_msg = setup_route(cfg, env)
        if not ok:
            raise RuntimeError(f"route setup failed: {route_msg}")
        print(f"[info] {route_msg}")

        update_job_progress(
            cfg.output_dir,
            phase="reproducer",
            start_time=start_iso,
            domain_id=cfg.domain_id,
            bag_path=str(cfg.bag_path),
            model_config=str(cfg.model_config),
        )
        print(f"[info] Starting perception_reproducer...")
        repro = popen(repro_cmd, env=env)
        procs.append(repro)

        print(
            f"[info] Waiting for perception (warmup {cfg.perception_warmup_sec:.0f}s, "
            f"stable {cfg.perception_ready_stable_sec:.0f}s) before Auto..."
        )
        ready_ok, ready_msg = wait_for_perception_ready(cfg)
        if ready_ok:
            print(f"[info] {ready_msg}")
        else:
            print(f"[warn] {ready_msg} — engaging Auto anyway")

        update_job_progress(
            cfg.output_dir,
            phase="recording",
            start_time=start_iso,
            domain_id=cfg.domain_id,
            bag_path=str(cfg.bag_path),
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
            bag_path=str(cfg.bag_path),
            model_config=str(cfg.model_config),
        )
        if cfg.end_condition == "route_arrived":
            end_status, detail = wait_for_route_arrived(env, run_timeout)
            if end_status != "done":
                # Soft failure: still keep the bag for offline metrics
                error = detail
                status = "failed"
            else:
                status = "done"
                time.sleep(cfg.post_arrival_sec)
        else:
            print(f"[info] Waiting bag_duration timeout {run_timeout:.0f}s...")
            time.sleep(run_timeout)
            status = "done"

        if status == "done":
            error = None

    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        print(f"[error] {error}")

    finally:
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
            "start_time": start_iso,
            "end_time": end_iso,
            "error": error,
            "domain_id": cfg.domain_id,
            "output_bag": bag_path_out,
            "model_config": str(cfg.model_config),
            "bag_path": str(cfg.bag_path),
            "duration_sec": round(time.time() - t0, 1),
        },
    )
    return JobResult(status, start_iso, end_iso, error, bag_path_out, time.time() - t0)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1: run one Diffusion Planner evaluation job "
        "(psim + perception_reproducer + ros2 bag record)."
    )
    p.add_argument("--model_config", type=Path, required=True, help="diffusion_planner.param.yaml")
    p.add_argument("--bag_path", type=Path, required=True, help="Input rosbag path or directory")
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
    p.add_argument("--skip_route_setup", action="store_true")
    p.add_argument("--rviz", action="store_true", help="Launch RViz (off by default for headless)")
    p.add_argument("--dry_run", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> JobConfig:
    return JobConfig(
        model_config=args.model_config.expanduser(),
        bag_path=args.bag_path.expanduser(),
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
