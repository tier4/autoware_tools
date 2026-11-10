"""Phase 1 — single evaluation job (psim + reproducer + bag record).

Importable as ``from dp_multi_eval.run_single_job import run_single_job``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.ament_overlay import (
    add_vehicle_dynamics_overlay,
    env_with_overlay,
    make_model_overlay,
)
from dp_multi_eval.build_provenance import build_provenance
from dp_multi_eval.job_status import utc_now_iso, write_job_status
from dp_multi_eval.process_utils import (
    cleanup_evaluation_processes,
    report_gpu_zombies,
    wait_for_gpu_headroom,
    close_log_handle,
    popen,
    popen_log_file,
    stop_process_group,
)
from dp_multi_eval.scenario_sim import (
    build_record_command_sim_time,
    build_scenario_runner_command,
    scenario_test_runner_available,
)
from dp_multi_eval.topics import TopicSet, load_topics
from dp_multi_eval.job_config import JobConfig, ros_call_env
from dp_multi_eval.job_waiters import (
    kill_conflicting_perception_nodes,
    planning_stack_nodes_ready,
    wait_for_perception_ready,
    wait_for_first_trajectory,
    wait_for_route_arrived,
    wait_for_services,
)
from dp_multi_eval.route_control import (
    advance_route,
    engage_autonomous,
    reposition_stuck_ego,
    setup_route,
)
from dp_multi_eval.route_control import close_routing
from dp_multi_eval.resync import resync_ego
from dp_multi_eval.ros_env import launch_failure_reason, workspace_env
from dp_multi_eval.window_eval import find_cuts, plan_legs, run_legs

# Re-syncs a job may make before it is abandoned: a model that diverges every
# few seconds is not being measured any more, only chased.
RESYNC_MAX = 30


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
            # Which planner build and which checkpoint produced this. The
            # Autoware tree is not under version control here, so a local edit to
            # WEIGHT_MAJOR_VERSION leaves no trace anywhere else, and a result
            # can quietly depend on a patched build.
            "build": build_provenance(model_config),
        },
    )




@dataclass
class JobResult:
    status: str  # done | failed
    start_time: str
    end_time: str
    error: str | None = None
    output_bag: str | None = None
    duration_sec: float = 0.0
    # Harness interventions during the drive (blinker nudges, forward shifts, …).
    # Empty dict = clean run. See ``clean_run``.
    interventions: dict[str, float] = field(default_factory=dict)

    @property
    def clean_run(self) -> bool:
        """True when the ego drove without any harness assistance."""
        # stall_x / stall_y are the last stall position, not a count.
        return not any(
            float(v) > 0
            for k, v in (self.interventions or {}).items()
            if not str(k).startswith("stall_")
        )


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
            env=ros_call_env(env),
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


def build_objects_throttle_command(topics: TopicSet) -> list[str] | None:
    """Republish the scene at a rate something actually consumes, or None.

    The reproducer publishes the whole object set at the source bag's odom rate,
    which is roughly 48 Hz and made that one topic 88% of a 699 MB output bag.
    Metrics read objects at 5 Hz and the preview slower still, so the extra
    messages cost disk and post-processing time and buy nothing.
    """
    hz = float(topics.objects_record_hz)
    if hz <= 0.0:
        return None
    return [
        "ros2", "run", "topic_tools", "throttle", "messages",
        topics.predicted_objects, str(hz), topics.objects_throttled_topic,
    ]


def _parse_ego_from_detail(detail: str) -> tuple[float, float] | None:
    m = re.search(r"ego=\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)", detail or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def _parse_ahead_blocked(detail: str) -> float | None:
    """Metres to the object ahead, when the waiter reported one."""
    m = re.search(r"ahead_blocked=(\d+(?:\.\d+)?)m", detail or "")
    return float(m.group(1)) if m else None


def build_reproducer_command(
    bag_path: Path,
    npc_reaction: bool = False,
    npc_reaction_decel: float = 3.0,
    indicator_merge_topic: str | None = None,
    window_clock: bool = False,
    objects: bool = True,
    map_path: Path | None = None,
    shield_log: Path | None = None,
    departure_hold: bool = True,
) -> list[str]:
    """Build the scene player command: planning_debug_tools' original
    perception_reproducer, driven by dp_multi_eval's scene clock (scene_driver.py).

    ``indicator_merge_topic`` makes the replay the single writer of the
    planner's turn-indicator input (the launch overlay points the planner at
    that topic); None keeps the replay on the vehicle topic beside psim.
    """
    cmd = [
        "ros2",
        "run",
        "dp_multi_eval",
        "scene_driver.py",
        "-b",
        str(bag_path),
        "-t",
    ]
    if npc_reaction:
        cmd.extend(["--npc-reaction", "--npc-reaction-decel", str(float(npc_reaction_decel))])
    if indicator_merge_topic:
        cmd.extend(["--indicator-merge-topic", str(indicator_merge_topic)])
    if window_clock:
        cmd.append("--window-clock")
    if not objects:
        cmd.append("--no-objects")
    if map_path is not None:
        cmd.extend(["--map-path", str(map_path)])
    if shield_log is not None:
        cmd.extend(["--shield-log", str(shield_log)])
    if window_clock and not departure_hold:
        cmd.append("--no-departure-hold")
    return cmd




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




# ros2 launch runs its own shutdown ladder on the nodes it spawned: SIGINT, then
# SIGTERM after sigterm_timeout (5 s), then SIGKILL after sigkill_timeout (5 s more).
# Killing the group at 10 s cut it off at exactly that boundary, so a node still
# unwinding — a planner releasing its CUDA context, for one — was SIGKILLed mid-exit
# instead of finishing. Leave margin for the full sequence.
PSIM_STOP_GRACE_SEC = 20.0

PLANNER_DISABLED_MARKERS = (
    "Model not loaded. Inference is disabled",
    "Failed to setup TRT engine",
)


def planner_inference_disabled(psim_log: Path) -> str | None:
    """The reason the planner will publish nothing, if it already said so.

    A planner whose engine failed to build stays up and silent: no trajectory is
    ever published, the ego never moves, and the run is recorded as "stuck" after
    burning the full stall budget. The cause is in the log within seconds of
    launch — reading it turns a five-minute wrong answer into an immediate right
    one. This is not a pre-flight check; nothing is predicted, the planner has
    already reported the failure.
    """
    try:
        text = psim_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # Report the planner's own reason, not the symptom. "Model not loaded" is what it
    # prints once it has given up; the ERROR immediately before says why — a version
    # mismatch, a missing file, a TensorRT failure. Quoting the symptom sent one
    # investigation to the GPU while the card sat almost empty.
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not any(marker in line for marker in PLANNER_DISABLED_MARKERS):
            continue
        for prev in reversed(lines[max(0, idx - 40) : idx]):
            if "[ERROR" in prev and "diffusion_planner" in prev:
                cause = prev.split("]:", 1)[-1].strip()
                return cause[:300]
        return line.strip()[-300:]
    return None


def runs_single_goal(n_legs: int, want_multi: bool | None) -> bool:
    """Whether a run with legs extracted takes the single-goal path.

    Fewer than two legs is a single-goal scenario under "on" as under "auto":
    that path routes a cut recording to the next stop and has the goal-backoff
    fallback, and "on" used to skip both for one leg and fail on none — the GUI
    writes "on" for its checkbox, so its runs differed from the CLI's "auto".
    """
    return want_multi is not False and n_legs < 2


def leg_cuts_on(cfg) -> bool:
    """Planned cuts (window_eval.find_cuts) for this job: leg windows and
    ``leg_cuts`` not off. YAML reads a bare ``off`` as False, so false / no / 0
    are off too."""
    value = str(getattr(cfg, "leg_cuts", "auto")).strip().lower()
    return bool(getattr(cfg, "window_eval", False)) and value not in ("off", "false", "no", "0", "none")


def drop_road_waits(goals: list, map_path) -> list:
    """Leg goals without the recording's waits on the road. A stand the
    recording made away from any bus stop (traffic, a queue, a parked car it
    could not pass) became a leg goal and the route ended there: H81's second
    leg ended on a crosswalk where the recorded bus stood 55 s behind a parked
    car, and the model was made to stop in the middle of the crosswalk
    (2026-10-29). Only stands at a map stop (within BUS_STOP_NEAR_M) and the
    recording's end stay; the scene clock catches up past a dropped wait at
    most 1.5x real time. Without a stop list nothing is dropped."""
    from dp_multi_eval.compute_metrics import BUS_STOP_NEAR_M, map_bus_stops

    stops = map_bus_stops(map_path)
    if stops is None or len(goals) < 2:
        return goals
    def at_stop(g) -> bool:
        return any(math.hypot(float(s["x"]) - g.x, float(s["y"]) - g.y) <= BUS_STOP_NEAR_M for s in stops)
    kept = [g for g in goals[:-1] if g.source != "stop_segment" or at_stop(g)] + [goals[-1]]
    for g in goals[:-1]:
        if g not in kept:
            print(f"[info] leg goal dropped: a wait on the road at ({g.x:.1f}, {g.y:.1f}), no bus stop within {BUS_STOP_NEAR_M:.0f} m")
    return kept


def _multi_goal_requested(value: bool | str | None) -> bool | None:
    """True = on, False = off, None = "auto". On and auto extract the same legs
    and fall back to single-goal the same way (runs_single_goal)."""
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("auto", ""):
            return None
        if token in ("true", "yes", "on", "1"):
            return True
        if token in ("false", "no", "off", "0"):
            return False
        return None
    return bool(value) if value is not None else None


def _parse_effective_goal(route_msg: str) -> tuple[float, float, float] | None:
    """Coordinates of the goal route_setup actually got accepted, if it reported them."""
    if "effective_goal_xy=" not in route_msg:
        return None
    raw = route_msg.split("effective_goal_xy=", 1)[1].split(";", 1)[0].strip()
    try:
        x, y, yaw = (float(v) for v in raw.split(","))
    except ValueError:
        return None
    return x, y, yaw


def leg_is_bus_stop(
    leg: tuple[float, float, float] | None,
    bus_stops_xy: list[tuple[float, float]],
    radius_m: float = 10.0,
) -> bool:
    """A route leg that sits on a map bus stop must be stopped at, not driven through.

    Intermediate legs are otherwise waypoints (a recorded-route goal at a
    junction is not a place to halt). On the vehicle a bus stop IS the
    operator's goal: arrive, doors, next goal. 2026-08-24: legs 教会前 and
    松風公園入口 were passed at 3.6 and 5.3 m/s where the recording stood 11 s.
    """
    if leg is None:
        return False
    return any(math.hypot(leg[0] - x, leg[1] - y) <= radius_m for x, y in bus_stops_xy)


def extract_bag_goal_xy(bag_path: Path) -> tuple[float, float] | None:
    """Last ego position in the source bag — what the goal would be without fallbacks."""
    try:
        from dp_multi_eval.bag_reader import load_reference_ego

        ego = load_reference_ego(bag_path)
        if ego:
            return ego[-1].x, ego[-1].y
    except Exception:  # noqa: BLE001
        pass
    return None














def _reference_track(bag_path: Path):
    """(t, x, y, yaw, dwells) of the recording's ego (lag.reference_track)."""
    from dp_multi_eval.bag_reader import load_reference_ego
    from dp_multi_eval.lag import reference_track

    return reference_track(load_reference_ego(bag_path))


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
    env = workspace_env(dict(os.environ))   # packages built since the GUI started count
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

        make_model_overlay(
            cfg.model_config, overlay_root,
            disable_long_stop=bool(cfg.disable_long_stop_fixer),
            planning_setting=str(cfg.planning_setting),
        )
        if bool(cfg.match_vehicle_dynamics):
            add_vehicle_dynamics_overlay(overlay_root, cfg.vehicle_model)
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
        recorder, _rec_log_handle = popen_log_file(
            record_cmd, cfg.output_dir / "record.log", env=env
        )
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
        stragglers = []
        for proc in reversed(procs):
            if not stop_process_group(proc, grace_sec=PSIM_STOP_GRACE_SEC):
                stragglers.append(proc.pid)
            close_log_handle(proc)
        if stragglers:
            print(f"[warn] process group(s) {stragglers} outlived their stop request")
        # Same reason as the reproducer path: a surviving planner/container process
        # keeps its CUDA context, and the next job cannot allocate its engine.
        if not cleanup_evaluation_processes(domain_id=cfg.domain_id):
            print("[warn] leftover evaluation processes remain after this job")

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
    # Every harness intervention lands here. Reported with the result so an
    # assisted arrival is never mistaken for an unassisted one. Declared before
    # the try so the final status write can read it even on an early failure.
    interventions: dict[str, float] = {
        "blinker_nudges": 0,
        "forward_shifts": 0,
        "reposition_total_m": 0.0,
        "route_resets": 0,
        "resyncs": 0,
    }
    resync_events: list[dict[str, Any]] = []
    # How far the accepted goal ended up from the bag's own goal. Recorded rather
    # than counted as an intervention: it changes what the run measures, and the
    # metrics follow it, so it needs to be visible without failing the run.
    goal_substituted_m: float = 0.0
    # The recording was cut while moving and the mission goal was extended to
    # the bus's next map stop; arrival is measured drive-through at the cut
    # pose. See multi_goal.next_stop_ahead.
    cut_end_drive_through = False
    # Multi-goal: the last extracted leg is the cut pose of a recording that
    # ended moving (source bag_end:cut).
    cut_final_leg = False
    # Intermediate legs dropped because their goal could not be routed to. Recorded
    # so a run that quietly lost half its stops is not read as a clean traversal.
    legs_skipped: int = 0
    env = workspace_env(dict(os.environ))   # packages built since the GUI started count
    env["ROS_DOMAIN_ID"] = str(cfg.domain_id)
    env.setdefault("ROS2_DISABLE_DAEMON", "1")
    bag_path = (cfg.bag_path or Path("")).expanduser().resolve()

    try:
        if not cfg.model_config.expanduser().resolve().is_file():
            raise FileNotFoundError(f"model_config not found: {cfg.model_config}")
        if not bag_path.exists():
            raise FileNotFoundError(f"bag_path not found: {bag_path}")

        indicator_merge_topic = (
            cfg.topics.dp_turn_indicators_input
            if bool(getattr(cfg, "indicator_merge", True))
            else None
        )
        make_model_overlay(
            cfg.model_config, overlay_root,
            disable_long_stop=bool(cfg.disable_long_stop_fixer),
            planning_setting=str(cfg.planning_setting),
            dp_turn_indicator_topic=indicator_merge_topic,
        )
        if bool(cfg.match_vehicle_dynamics):
            add_vehicle_dynamics_overlay(overlay_root, cfg.vehicle_model)
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
            npc_reaction=bool(cfg.npc_reaction),
            npc_reaction_decel=float(cfg.npc_reaction_decel),
            indicator_merge_topic=indicator_merge_topic,
            window_clock=bool(cfg.window_eval),
            objects=bool(getattr(cfg, "scene_objects", True)),
            map_path=cfg.map_path,
            shield_log=cfg.output_dir / "shield_log.jsonl",
            # With planned cuts every recorded stand closes a window at its
            # departure: the 6 s freeze there only froze the scene (walkers
            # standing mid-crosswalk at 0 m/s, H78 2026-11-10).
            departure_hold=not leg_cuts_on(cfg),
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
        # ``ros2 node list`` hang and block readiness forever, and a leftover
        # planner node still holds its CUDA context.
        report_gpu_zombies()
        gpu_ok, gpu_detail = wait_for_gpu_headroom(
            float(cfg.gpu_min_free_mib), float(cfg.gpu_wait_timeout_sec)
        )
        if not gpu_ok:
            raise RuntimeError(f"planner would not fit on the GPU: {gpu_detail}")
        print(f"[info] GPU check: {gpu_detail}")

        print("[info] Cleaning leftover evaluation processes before psim launch...")
        if not cleanup_evaluation_processes(domain_id=cfg.domain_id):
            print("[warn] Some evaluation processes survived cleanup — this job may "
                  "fail to allocate GPU memory")

        # The scene driver replays the recorded turn indicators, which the
        # reproducer's own load does not read. A first read is one more pass
        # over the bag: done here, its cost does not eat into the reproducer's
        # load budget, and every later job on this bag reads the cache.
        try:
            from dp_multi_eval.scene_driver import recorded_indicators

            print(f"[info] Recorded turn indicators: {len(recorded_indicators(bag_path))} reports")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Recorded turn indicators not cached ({exc}); the scene driver reads them itself")

        psim_log = cfg.output_dir / "psim_launch.log"
        print(f"[info] Launching planning simulator...")
        print(f"[info] psim log: {psim_log}")
        print(f"[info] psim cmd: {' '.join(psim_cmd)}")
        psim, _psim_log_handle = popen_log_file(psim_cmd, psim_log, env=env)
        procs.append(psim)
        # The reproducer's bag load (~20 s on a 79 GB bag) needs nothing from
        # psim, so it runs beside psim's own startup instead of after the route
        # was set; objects are only checked for once the ego is localized.
        print("[info] Starting the scene driver (original perception_reproducer)...")
        repro, _repro_log_handle = popen_log_file(
            repro_cmd, cfg.output_dir / "reproducer.log", env=env
        )
        procs.append(repro)

        psim_ready, psim_detail = wait_for_services(env, cfg.psim_startup_sec, psim=psim)
        if not psim_ready:
            reason = launch_failure_reason(psim_log) if psim.poll() is not None else None
            if reason:
                psim_detail = f"{psim_detail}: {reason}"
            psim_tail = tail_text_file(psim_log)
            if psim_tail.strip():
                print(f"[error] psim_launch.log (tail):\n{psim_tail}")
            raise RuntimeError(
                f"planning simulator not ready within {cfg.psim_startup_sec:.0f}s: {psim_detail}"
            )
        print(f"[info] Planning simulator stack ready ({psim_detail})")

        # The planner comes up even when its engine failed to build, and then never
        # publishes a trajectory. Without this the job looks stuck and the real cause
        # sits unread in the log.
        disabled = planner_inference_disabled(psim_log)
        if disabled:
            raise RuntimeError(f"planner inference disabled: {disabled}")

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

        # Resolve route legs. multi_goal may be "auto", in which case the bag decides.
        route_goals: list[tuple[float, float, float]] = []
        want_multi = _multi_goal_requested(cfg.multi_goal)
        multi_goal_active = False
        if want_multi is not False:
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
            if cfg.window_eval:
                extracted = drop_road_waits(extracted, cfg.map_path)
            if runs_single_goal(len(extracted), want_multi):
                print(
                    f"[info] multi_goal={cfg.multi_goal}: {len(extracted)} leg(s) found in the bag "
                    "— running single-goal"
                )
                # Single-goal, but not goal_from_bag: the extraction has already
                # merged the settled bag end with the recorded mission goals, and
                # discarding it here handed the route the raw cut-off pose again.
                # Observed as "keeping the mission goal" in one line and
                # goal=(50295.74, 9849.29) — the intersection the merge had just
                # rejected — in the next, with every set_route "empty".
                if extracted:
                    single = extracted[0]
                    route_goals = [(single.x, single.y, single.yaw)]
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
                        f"[info] single goal from the merged extraction: "
                        f"({single.x:.1f}, {single.y:.1f}) source={single.source}"
                    )
                extracted = []
        if want_multi is not False and extracted:
            multi_goal_active = True
            print(f"[info] multi_goal active: {len(extracted)} leg(s)")
            route_goals = [(g.x, g.y, g.yaw) for g in extracted]
            cut_final_leg = bool(extracted) and extracted[-1].source == "bag_end:cut"
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

        # Intermediate legs are waypoints, not requirements. stop_segments cannot tell
        # a bus stop from a wait at a traffic light, and an incidental dwell makes a
        # poor goal: the vehicle stopped straddling lanelets, so the mission planner
        # rejects it with "Goal's footprint exceeds lane!" — and no amount of pulling
        # the goal back can fix that. Skipping such a leg costs nothing: the ego still
        # drives through that point, it just does not stop and register there. Failing
        # the whole bag over it costs everything.
        first_goal = route_goals[0] if route_goals else None
        # A recording cut while the bus was still moving ends at an arbitrary
        # pose — on 2026-04-28 the stop line of a green-lit junction. The bus's
        # real destination was its next map stop, past the junction. Routing
        # to that stop gives the model the road context the vehicle had (the
        # route continues, so it sets up for the turn instead of parking on
        # the line); the run still ends at the cut pose, drive-through.
        cut_end_goal = None
        cut_end_pose = None
        if not multi_goal_active or cut_final_leg:
            try:
                from dp_multi_eval.multi_goal import (
                    bag_end_pose, bag_path_xy, next_stop_ahead,
                )

                end_pose = bag_end_pose(bag_path)
                if end_pose is not None and end_pose[3] > 1.0:
                    cand = next_stop_ahead(
                        end_pose[:3], map_path=cfg.map_path,
                        stop_points_csv=cfg.stop_points_csv,
                        stop_order=list(cfg.multi_goal_stop_order or []),
                        recorded_path_xy=bag_path_xy(bag_path),
                    )
                    if cand is not None:
                        cut_end_goal = (cand.x, cand.y, cand.yaw)
                        cut_end_pose = end_pose[:3]
                        print(
                            f"[info] recording was cut while moving "
                            f"({end_pose[3]:.1f} m/s); mission goal extended to "
                            f"{cand.source.split(':', 1)[-1]} "
                            f"{math.hypot(cand.x - end_pose[0], cand.y - end_pose[1]):.0f} m "
                            "ahead — arrival measured drive-through at the cut pose"
                        )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] cut-end goal extension skipped: {exc}")
        if cut_end_goal is not None and not multi_goal_active:
            first_goal = cut_end_goal
        # An intermediate leg goal may NOT be substituted with an earlier map
        # stop: the walk-back invented a full stop at a bay the recording
        # drove past at 3 m/s, and v4 does not restart from an artificial
        # at-goal state it never sees on the vehicle. Unroutable intermediates
        # merge forward instead; only the final goal may be relocated.
        ok, route_msg = setup_route(
            cfg, env, goal_xy_yaw=first_goal,
            allow_goal_substitution=(not multi_goal_active or len(route_goals) <= 1),
        )
        if ok and cut_end_goal is not None and not multi_goal_active:
            cut_end_drive_through = True
        if not ok and first_goal is not None and not multi_goal_active:
            # The merged goal is the better destination when the planner takes
            # it — but at a stop bay it can fail the footprint check outright
            # ("Goal's footprint exceeds lane!"), while the raw bag-end path
            # with its own backoff machinery was being accepted here for four
            # passes straight. Acceptance belongs to the live planner, not to
            # any approximation of it: try the good goal first, fall back to
            # the proven one.
            print(
                f"[warn] merged goal was refused ({route_msg[:120]}) — "
                "retrying with the raw bag-end goal"
            )
            ok, route_msg = setup_route(cfg, env, goal_xy_yaw=None)
            if ok:
                route_goals.clear()
        while not ok and multi_goal_active and len(route_goals) > 1:
            skipped = route_goals.pop(0)
            extracted = extracted[1:]
            print(
                f"[warn] Leg goal ({skipped[0]:.1f}, {skipped[1]:.1f}) is not routable "
                f"— skipping it and routing to the next of {len(route_goals)} leg(s)"
            )
            legs_skipped += 1
            first_goal = route_goals[0]
            ok, route_msg = setup_route(
                cfg, env, goal_xy_yaw=first_goal,
                allow_goal_substitution=(len(route_goals) <= 1),
            )
        if not ok:
            raise RuntimeError(f"route setup failed: {route_msg}")
        print(f"[info] {route_msg}")

        # Score the run against the goal the planner accepted, not the one taken from
        # the bag. Snapping, the stop fallback and the goal backoff all move it, and
        # until now only multi-goal runs recorded the result — so a single-goal run
        # whose goal was relocated was measured against a pose the ego was never
        # routed to, and could not pass however well it drove.
        effective_goal = _parse_effective_goal(route_msg)
        # Used only for the near-goal arrival check below, never for re-routing.
        arrival_goal = effective_goal
        if cut_end_drive_through and cut_end_pose is not None:
            # The mission goal is the next stop; the run ends at the cut pose.
            arrival_goal = cut_end_pose
            (cfg.output_dir / "route_goals.json").write_text(
                json.dumps(
                    {
                        "goals": [
                            {
                                "x": cut_end_pose[0],
                                "y": cut_end_pose[1],
                                "yaw": cut_end_pose[2],
                                "source": "bag_end:cut_while_moving",
                            }
                        ],
                        "mission_goal": (
                            {
                                "x": effective_goal[0],
                                "y": effective_goal[1],
                                "yaw": effective_goal[2],
                                "source": "route_setup:effective (next map stop)",
                            }
                            if effective_goal is not None
                            else None
                        ),
                        "arrival": "drive_through_at_cut_pose",
                        "route_setup": route_msg,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        elif effective_goal is not None and not multi_goal_active:
            bag_goal = extract_bag_goal_xy(bag_path)
            moved = (
                math.hypot(effective_goal[0] - bag_goal[0], effective_goal[1] - bag_goal[1])
                if bag_goal is not None
                else float("nan")
            )
            goal_substituted_m = moved
            (cfg.output_dir / "route_goals.json").write_text(
                json.dumps(
                    {
                        "goals": [
                            {
                                "x": effective_goal[0],
                                "y": effective_goal[1],
                                "yaw": effective_goal[2],
                                "source": "route_setup:effective",
                            }
                        ],
                        "bag_goal": (
                            {"x": bag_goal[0], "y": bag_goal[1]} if bag_goal else None
                        ),
                        "goal_moved_m": None if moved != moved else round(moved, 3),
                        "route_setup": route_msg,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            if moved == moved and moved > 1.0:
                print(
                    f"[warn] routing goal differs from the bag goal by {moved:.1f} m — "
                    f"metrics will score against the routed goal"
                )

        # Multi-goal: the first leg is driven to whatever pose route setup got
        # accepted, which is not always the one it was handed — a rejected goal is
        # backed off along its lane or replaced by a nearby map stop. Autoware then
        # drives to the accepted pose while the harness watches the recorded one, so
        # the ego sails past its real goal (measured: 12 m away at closest approach,
        # tolerance 5 m), never registers ARRIVED, and the leg fails as
        # "route_not_arrived" after driving perfectly. Adopt the accepted pose.
        if multi_goal_active and effective_goal is not None and route_goals:
            moved_leg = math.hypot(
                effective_goal[0] - route_goals[0][0],
                effective_goal[1] - route_goals[0][1],
            )
            if moved_leg > 0.5:
                print(
                    f"[info] Leg 1 goal was relocated {moved_leg:.1f} m by route setup "
                    "— tracking arrival against the accepted pose"
                )
                goal_substituted_m = max(goal_substituted_m, moved_leg)
            route_goals[0] = effective_goal
            if extracted:
                extracted[0] = replace(
                    extracted[0],
                    x=effective_goal[0],
                    y=effective_goal[1],
                    yaw=effective_goal[2],
                )

        # If route setup had to skip unroutable stop goals (empty lanelet path),
        # drop those legs so ARRIVED waits match the effective goal.
        if multi_goal_active and route_goals and "effective_goal=" in route_msg:
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
                env=env,
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

        # Psim's tracker publishes empty TrackedObjects; kill it so the
        # reproducer owns the topic. Full bags load slowly — poll until objects
        # appear instead of a short fixed sleep.
        kill_conflicting_perception_nodes()
        load_timeout = perception_load_timeout_sec(cfg, bag_duration)
        print(
            f"[info] Waiting up to {load_timeout:.0f}s for perception_reproducer "
            f"to finish loading the full bag and publish objects "
            f"(input duration={bag_duration})..."
        )
        ready_ok, ready_msg = wait_for_perception_ready(
            cfg, timeout_sec=load_timeout, reproducer=repro
        )
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
        throttle_cmd = build_objects_throttle_command(cfg.topics)
        if throttle_cmd is not None:
            print(
                f"[info] Throttling objects to {cfg.topics.objects_record_hz:.0f} Hz "
                f"→ {cfg.topics.objects_throttled_topic}"
            )
            throttle_proc, _thr_log = popen_log_file(
                throttle_cmd, cfg.output_dir / "objects_throttle.log", env=env
            )
            procs.append(throttle_proc)
            # The recorder must find the topic advertised before it subscribes.
            time.sleep(2.0)

        print(f"[info] Starting ros2 bag record → {output_bag}")
        recorder, _rec_log_handle = popen_log_file(
            record_cmd, cfg.output_dir / "record.log", env=env
        )
        procs.append(recorder)
        time.sleep(1.0)
        if recorder.poll() is not None:
            raise RuntimeError("ros2 bag record exited immediately")

        if float(cfg.planner_warmup_timeout_sec) > 0.0:
            warm_ok, warm_msg = wait_for_first_trajectory(
                cfg, float(cfg.planner_warmup_timeout_sec)
            )
            print(f"[info] {warm_msg}")
            if not warm_ok:
                raise RuntimeError(warm_msg)
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

        # What the recording did at each place, so a stall can be judged against the
        # wait that actually belongs there rather than one number for the whole run.
        from dp_multi_eval.dwell_profile import load_dwell_profile, stall_budget_sec

        dwell_profile = load_dwell_profile(bag_path, stop_speed_mps=cfg.stall_speed_mps)
        if dwell_profile is None:
            print("[warn] no dwell profile from the source bag; stalls fall back to "
                  "stuck_abort_sec everywhere")
        reference_track = None
        if float(cfg.resync_lag_sec) > 0.0:
            reference_track = _reference_track(bag_path)
            if reference_track is None:
                print("[warn] no reference ego track from the source bag; re-sync is off")
            else:
                print(
                    f"[info] re-sync on: the ego is set back onto the recording once it is "
                    f"{cfg.resync_lag_sec:.0f}s behind it ({len(reference_track[0])} track samples)"
                )

        legs = route_goals if (multi_goal_active and len(route_goals) > 1) else [None]
        bus_stops_xy: list[tuple[float, float]] = []
        try:
            from dp_multi_eval.multi_goal import _load_map_stop_points, resolve_stop_points_csv

            bus_stops_xy = [
                (float(st.x), float(st.y))
                for st in _load_map_stop_points(
                    resolve_stop_points_csv(cfg.map_path, cfg.stop_points_csv)
                )
            ]
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] stop points unavailable ({exc}); intermediate legs stay drive-through")
        leg_cap = max(
            5.0,
            float(cfg.multi_goal_leg_timeout_sec or cfg.route_timeout_sec or 300.0),
        )
        # The whole drive shared one budget while every leg was promised leg_cap, so a
        # multi-leg run could not honour its own configuration: with two legs and both
        # values at 300 s, leg 1 consumed the budget and leg 2 started with 5 s left and
        # timed out on arrival. Give the drive at least what the legs are promised.
        if len(legs) > 1:
            needed = leg_cap * len(legs)
            if needed > run_timeout:
                print(
                    f"[info] multi_goal: raising the drive budget {run_timeout:.0f}s → "
                    f"{needed:.0f}s ({len(legs)} legs × {leg_cap:.0f}s per leg)"
                )
                run_timeout = needed
        drive_deadline = time.time() + run_timeout
        status = "done"
        error = None
        stuck_kw = dict(
            stuck_blinker_nudge=bool(cfg.stuck_blinker_nudge),
            stuck_blinker_speed_mps=float(cfg.stuck_blinker_speed_mps),
            stall_speed_mps=float(cfg.stall_speed_mps),
            stuck_blinker_trigger_sec=float(cfg.stuck_blinker_trigger_sec),
            stuck_blinker_hold_sec=float(cfg.stuck_blinker_hold_sec),
            stuck_blinker_cooldown_sec=float(cfg.stuck_blinker_cooldown_sec),
            stack_hold_detect_sec=float(cfg.stack_hold_detect_sec),
            resync_lag_sec=float(cfg.resync_lag_sec) if reference_track is not None else 0.0,
            reference_track=reference_track,
            counters=interventions,
        )
        reposition_trigger = max(
            5.0,
            float(cfg.stuck_reposition_trigger_sec or 30.0),
        )
        # stuck_reposition_max_count: 0 = never reposition, N>0 = at most N shifts,
        # negative = unlimited. This used to read 0 as "unlimited", the opposite of
        # what the name says, so a hopeless scenario teleported until the route
        # timeout expired instead of being abandoned.
        reposition_max = int(cfg.stuck_reposition_max_count)
        reposition_unlimited = reposition_max < 0
        reposition_enabled = float(cfg.stuck_reposition_forward_m) > 0.0 and (
            reposition_unlimited or reposition_max > 0
        )

        def wait_with_stuck_recovery(
            wait_sec: float,
            current_goal: tuple[float, float, float] | None,
            require_stop_at_goal: bool = True,
            arrival_tolerance_m: float | None = None,
        ) -> tuple[str, str]:
            """Wait for ARRIVED; if stopped too long, shift ego forward and retry.

            Used for red-light / lead-vehicle stalls: every
            ``stuck_reposition_trigger_sec`` of near-zero speed, move ego
            ``stuck_reposition_forward_m`` and continue the same leg.

            ``stuck_abort_sec`` is enforced here, as a budget accumulated across
            repositions. Each call to ``wait_for_route_arrived`` builds a fresh
            waiter whose stall timer starts at zero, so a per-call abort can never
            be reached once repositioning restarts the wait — the job would run to
            the full route timeout instead of giving up.
            """
            deadline = time.time() + wait_sec
            # Time provably lost to the departure hold is credited back to the
            # leg, up to a cap. The hold is measured and attributed on its own
            # (stack_intervention, green_latched_sec); charging it against the
            # route clock as well ended a leg as "leg_wait_exhausted" with 81%
            # of its trajectories stubbed — the timeout was measuring the latch,
            # not the driving. The cap keeps a run that is nothing but holds
            # from living forever.
            hold_credit_left = 240.0
            creep_marks: list[tuple[float, float, float]] = []
            creep_extension_left = max(60.0, 0.5 * float(wait_sec))
            shifts = 0
            stalled_sec = 0.0
            # Red-phase backoff for the in-place nudge. A nudge whose speed is
            # gone again within seconds was braked away on purpose — the model
            # wants to stay stopped, likeliest at a red light — and repeating it
            # every cycle burned the whole nudge budget inside one red phase.
            # Each ineffective nudge doubles the pause before the next; the
            # first effective one clears it.
            nudge_backoff_sec = 0.0
            last_nudge_t = 0.0
            # Where the previous hold happened. Ego movement between holds means
            # the escapes are working — the run reached 8.33 m/s between two of
            # them and was still charged 6 s per hold until stuck_abort_sec
            # filled at nudges=5. Progress resets the stall ledger; only holds
            # at the same spot accumulate toward giving up.
            hold_x = float("nan")
            hold_y = float("nan")
            holds_here = 0
            yield_frozen_sec = 0.0
            last_blocked_m: float | None = None
            global_abort = max(0.0, float(cfg.stuck_abort_sec or 0.0))
            last_detail = "wait_not_started"

            while True:
                remaining = deadline - time.time()
                if remaining < 5.0:
                    # A creeping final approach is still arriving. The waiter's
                    # own extension never sees it — a stack_hold returns every
                    # ~6 s, so the wait re-enters here where the clock actually
                    # runs out. Same rule, same bound: within 60 m of the goal,
                    # >2 m of motion across recent hold events earns 30 s more,
                    # to at most +50% of the leg budget in total.
                    goal_for_creep = current_goal if current_goal is not None else arrival_goal
                    hx0 = float(interventions.get("stall_x", float("nan")))
                    hy0 = float(interventions.get("stall_y", float("nan")))
                    old_mark = next(
                        (m for m in creep_marks if time.time() - m[0] >= 55.0),
                        creep_marks[0] if creep_marks else None,
                    )
                    if (
                        creep_extension_left > 0.0
                        and goal_for_creep is not None
                        and hx0 == hx0
                        and old_mark is not None
                        and math.hypot(hx0 - goal_for_creep[0], hy0 - goal_for_creep[1]) < 60.0
                        and math.hypot(hx0 - old_mark[1], hy0 - old_mark[2]) > 2.0
                    ):
                        deadline += 30.0
                        creep_extension_left -= 30.0
                        print(
                            f"[info] creeping toward the goal — extending leg "
                            f"wait by 30s ({creep_extension_left:.0f}s left)"
                        )
                        continue
                    lx = float(interventions.get("stall_x", float("nan")))
                    ly = float(interventions.get("stall_y", float("nan")))
                    return "timeout", (
                        f"{last_detail}; leg_wait_exhausted "
                        f"forward_shifts={shifts}"
                        + (f" ego=({lx:.1f},{ly:.1f})" if lx == lx else "")
                    )

                abort_after = (
                    reposition_trigger
                    if reposition_enabled
                    else float(cfg.stuck_abort_sec)
                )
                # current_goal is None on single-goal runs, which used to disable the
                # near-goal arrival check entirely: the run then depended solely on
                # /api/routing/state reaching ARRIVED. A planner that stops a few
                # metres short — likeliest when the goal sits mid-road rather than at
                # a stop line — was recorded as a timeout despite having arrived.
                # arrival_goal is read-only here; re-routing still keys off current_goal.
                wait_status, wait_detail = wait_for_route_arrived(
                    env,
                    remaining,
                    goal_xy_yaw=current_goal if current_goal is not None else arrival_goal,
                    near_goal_arrival_m=float(
                        arrival_tolerance_m
                        if arrival_tolerance_m is not None
                        else cfg.multi_goal_arrival_tolerance_m
                    ),
                    require_stop_at_goal=require_stop_at_goal,
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
                # "stale" means the simulator stopped publishing: repositioning a
                # vehicle nothing is simulating cannot help, so end the leg here.
                # Handled regardless of the forward-shift setting: this branch
                # is the patient one (dwell budgets, yield freeze, in-place
                # nudge) and must not vanish with the teleport.
                if wait_status == "resync":
                    # The ego fell behind the recording. Set it back onto the
                    # recorded track (resync.py) and keep the leg; the event is
                    # the measurement. Harness time spent here is credited back
                    # to the leg.
                    done = int(interventions.get("resyncs", 0))
                    if done >= RESYNC_MAX:
                        return "stuck", f"{wait_detail}; resync_budget={RESYNC_MAX}"
                    print(f"[warn] {wait_detail} — re-sync #{done + 1}")
                    t_sync = time.time()
                    ok_sync, sync_detail, event = resync_ego(env)
                    deadline += time.time() - t_sync
                    if not ok_sync:
                        return "stuck", f"{wait_detail}; {sync_detail}"
                    interventions["resyncs"] = done + 1
                    lag_m = re.search(r"ego ([\d.]+)s behind", wait_detail)
                    event.update(
                        {
                            "n": done + 1,
                            "t_wall": round(t_sync, 3),
                            "lag_sec": float(lag_m.group(1)) if lag_m else None,
                            "leg": current_goal,
                        }
                    )
                    resync_events.append(event)
                    (cfg.output_dir / "resyncs.json").write_text(
                        json.dumps(resync_events, indent=1) + "\n", encoding="utf-8"
                    )
                    print(f"[info] {sync_detail}")
                    last_detail = f"{wait_detail}; {sync_detail}"
                    continue
                if wait_status == "stack_hold":
                    # The fixer's hold, caught by its live signature instead of
                    # ridden out by the stall timer. Escape in place: motion
                    # releases the hold, a 5 m teleport lands at rest and re-arms
                    # it — measured as four consecutive ~40 s bites through one
                    # segment. The ego has not moved, so the route it is on is
                    # still the route: no re-route, just motion and re-engage.
                    hx = float(interventions.get("stall_x", float("nan")))
                    hy = float(interventions.get("stall_y", float("nan")))
                    if hx == hx:
                        now_m = time.time()
                        creep_marks.append((now_m, hx, hy))
                        creep_marks[:] = [m for m in creep_marks if now_m - m[0] <= 65.0]
                    moved = (
                        math.hypot(hx - hold_x, hy - hold_y)
                        if hold_x == hold_x and hx == hx
                        else float("inf")
                    )
                    blocked_m = _parse_ahead_blocked(wait_detail)
                    # Following a queue is progress, not a stall. The ego died
                    # 30 s into a queue whose lead car had covered 49 m in the
                    # previous minute: each creep was 1-2 m, under the 5 m reset,
                    # and the dwell radius missed the recording's wait 20 m up
                    # the same queue. A blocked hold resets on queue-sized creep,
                    # or whenever the car ahead has visibly pulled away.
                    # 2 m unblocked: v4 departs bus stops by creeping (measured
                    # 9 m over 174 s before pulling away); a 5 m bar let the
                    # ledger fill mid-creep and killed departures in progress.
                    # A dead stop still moves 0 m and still dies on budget.
                    reset_at = 1.5 if blocked_m is not None else 2.0
                    pulled_away = (
                        blocked_m is not None
                        and last_blocked_m is not None
                        and blocked_m > last_blocked_m + 1.0
                    )
                    if moved > reset_at or pulled_away:
                        stalled_sec = 0.0
                        holds_here = 0
                        nudge_backoff_sec = 0.0
                        yield_frozen_sec = 0.0
                        # The anchor moves only on reset: measuring hold-to-hold
                        # displacement made a 0.05 m/s creep invisible (0.4 m
                        # between events, 3+ m over the ledger's life) and a
                        # departure in progress was still budget-killed 30 m
                        # from its goal. Cumulative-from-anchor sees it.
                        hold_x, hold_y = hx, hy
                    if hold_x != hold_x:
                        hold_x, hold_y = hx, hy
                    last_blocked_m = blocked_m
                    if (
                        "front_traffic" in wait_detail
                        and blocked_m is None
                        and yield_frozen_sec < 90.0
                    ):
                        # Moving traffic crosses the ego's front: the hold is
                        # yielding, and yielding spends no hold budget. The
                        # 2026-04-28 junction died 30 s into a wait for cross
                        # traffic the recording never faced (it hit the gap).
                        # The freeze is bounded: governed NPCs run their paths
                        # once (monotonic shield clocks), and the leg deadline
                        # still rules over everything.
                        credit = min(hold_credit_left, float(cfg.stack_hold_detect_sec))
                        if credit > 0.0:
                            deadline += credit
                            hold_credit_left -= credit
                        yield_frozen_sec += float(cfg.stack_hold_detect_sec)
                        print(
                            f"[info] yielding to moving traffic in view — "
                            f"hold budget frozen ({yield_frozen_sec:.0f}s/90s)"
                        )
                        last_nudge_t = time.time()
                        continue
                    holds_here += 1
                    stalled_sec += float(cfg.stack_hold_detect_sec)
                    credit = min(hold_credit_left, float(cfg.stack_hold_detect_sec) + 10.0)
                    if credit > 0.0:
                        deadline += credit
                        hold_credit_left -= credit
                    # A place the recording waited earns the wait, in this branch
                    # too. The regular stall path already budgets from the dwell
                    # profile; this one compared against the raw 30 s abort and
                    # gave up 33 s into a queue the recording itself sat in for
                    # 56 s — with the car ahead about to move.
                    hold_budget = global_abort
                    if hx == hx:
                        # A queue is one wait spread over its own length: the
                        # recording's 56 s dwell sat 19 m up the road from where
                        # this ego joined the same queue, and the default 6 m
                        # radius missed it — the ledger filled while the replayed
                        # queue phase, which lasts as long as the recorded one,
                        # was still standing. Search at queue scale.
                        here_budget, recorded_dwell = stall_budget_sec(
                            dwell_profile, hx, hy, floor_sec=global_abort,
                            radius_m=25.0,
                        )
                        if recorded_dwell == recorded_dwell and here_budget > hold_budget:
                            if holds_here == 1:
                                print(
                                    f"[info] the recording waited "
                                    f"{recorded_dwell:.0f}s here; allowing "
                                    f"{here_budget:.0f}s of hold before giving up"
                                )
                            hold_budget = here_budget
                    max_holds = max(8, int(hold_budget / max(1.0, float(cfg.stack_hold_detect_sec))) + 2)
                    if (hold_budget > 0.0 and stalled_sec >= hold_budget) or holds_here > max_holds:
                        return "stuck", (
                            f"{wait_detail}; stack_hold_budget "
                            f"held_here={stalled_sec:.0f}s/{hold_budget:.0f}s "
                            f"nudges_here={holds_here} "
                            f"total_nudges={int(interventions.get('stack_holds', 0))}"
                            + (f" ego=({hx:.1f},{hy:.1f})" if hx == hx else "")
                        )
                    if "ahead_blocked" in wait_detail:
                        # Something stands on the ego's nose. Speed cannot escape
                        # this hold — it can only manufacture a contact, and did:
                        # a nudged ego kissed the bumper of a car stopped 2 m
                        # ahead, scored ego_caused. Wait like a red light instead.
                        nudge_backoff_sec = min(24.0, max(6.0, nudge_backoff_sec * 2.0))
                        print(
                            f"[info] hold with an object ahead — no nudge; "
                            f"waiting (next check >={nudge_backoff_sec:.0f}s)"
                        )
                        last_nudge_t = time.time()
                        continue
                    goal_for_nudge = (
                        current_goal if current_goal is not None else arrival_goal
                    )
                    if (
                        goal_for_nudge is not None
                        and hx == hx
                        and math.hypot(hx - goal_for_nudge[0], hy - goal_for_nudge[1]) < 40.0
                    ):
                        # An ego slowing inside the arrival zone is arriving,
                        # not stuck. Nudging it produced the mid-road lurch
                        # visible on 2026-04-28's video: the model plans its
                        # goal stop, the harness injects speed, the model
                        # brakes it away — repeat. Wait; the arrival check and
                        # the near-goal completion own this zone.
                        nudge_backoff_sec = min(24.0, max(6.0, nudge_backoff_sec * 2.0))
                        print(
                            f"[info] hold inside the arrival zone — no nudge; "
                            f"waiting (next check >={nudge_backoff_sec:.0f}s)"
                        )
                        last_nudge_t = time.time()
                        continue
                    since_last = time.time() - last_nudge_t
                    if since_last < nudge_backoff_sec:
                        # Still inside the pause an ineffective nudge earned.
                        # The wait loop keeps watching; a red light needs time,
                        # not another shove.
                        continue
                    print(f"[warn] {wait_detail} — in-place nudge")
                    last_nudge_t = time.time()
                    recovered, recovery_detail = reposition_stuck_ego(
                        cfg, env, distance_m=0.0
                    )
                    if not recovered:
                        return "stuck", f"{wait_detail}; {recovery_detail}"
                    reached = re.search(r"ego now (\d+(?:\.\d+)?)", recovery_detail)
                    if reached is not None and float(reached.group(1)) < 0.3:
                        nudge_backoff_sec = min(
                            24.0, max(6.0, nudge_backoff_sec * 2.0)
                        )
                        print(
                            f"[info] nudge was braked away immediately — the "
                            f"model is holding this stop; next nudge in "
                            f">={nudge_backoff_sec:.0f}s"
                        )
                    else:
                        nudge_backoff_sec = 0.0
                    engaged, engage_detail = engage_autonomous(
                        env, timeout_sec=cfg.auto_engage_timeout_sec
                    )
                    if not engaged:
                        return "stuck", (
                            f"{wait_detail}; {recovery_detail}; "
                            f"reengage_failed: {engage_detail}"
                        )
                    continue
                if wait_status != "stuck":
                    return wait_status, wait_detail
                # Patient budgeting below runs whether or not the forward
                # shift is enabled: disabling the 5 m teleport used to also
                # disable the dwell-aware budget, and a 40 s model hesitation
                # was then killed by the raw 30 s abort ("stuck_abort v=0.00
                # for 30s"). Only the shift ACTION is gated on forward_m.

                # A "stuck" return means the ego held near-zero speed for at least
                # abort_after seconds.
                stall_x = float(interventions.get("stall_x", float("nan")))
                stall_y = float(interventions.get("stall_y", float("nan")))
                # The ledger is per PLACE, exactly like the stack_hold branch:
                # without the reset, 30 s at a 129 s-budget junction plus 30 s
                # somewhere new read as cumulative=60 against the new place's
                # 30 s floor, and a run that had driven on was killed for a
                # wait it had already survived.
                moved_stall = (
                    math.hypot(stall_x - hold_x, stall_y - hold_y)
                    if hold_x == hold_x and stall_x == stall_x
                    else float("inf")
                )
                if moved_stall > 2.0:
                    stalled_sec = 0.0
                    hold_x, hold_y = stall_x, stall_y
                stalled_sec += abort_after
                # Judge the stall where it happened. A place the recording waited at
                # earns that wait back, with margin; a place it rolled through earns
                # only the floor. One number for the whole route is wrong at both ends.
                # The budget bounds time spent stalled; repositioning is the recovery
                # we are allowed to attempt inside it. A budget shorter than one
                # reposition cycle abandons the leg before the recovery can run at
                # all — measured as "cumulative=30s>=30s forward_shifts=0", giving up
                # without once trying the thing that exists to break a stall. So the
                # floor never drops below a cycle plus the trigger that starts it.
                # Cover every reposition the configuration permits, not just one.
                # The floor was twice the trigger, which buys a single attempt, while
                # stuck_reposition_max_count allowed three — so a leg was abandoned
                # with shifts still on the table ("cumulative=60s>=60s
                # forward_shifts=1"). Promising N recoveries and budgeting for one is
                # the same mistake in a smaller size.
                allowed_shifts = (
                    reposition_max if not reposition_unlimited else 3
                )
                recovery_floor = (
                    reposition_trigger * (max(1, allowed_shifts) + 1)
                    if reposition_enabled
                    else global_abort
                )
                floor_here = max(global_abort, recovery_floor)
                here_budget, recorded_dwell = (
                    stall_budget_sec(
                        dwell_profile, stall_x, stall_y, floor_sec=floor_here,
                        # Queue scale, same as the stack_hold branch: the
                        # default 6 m missed a 48 s recorded dwell sitting a
                        # dozen metres up the same approach and applied the
                        # bare 30 s floor to a wait the recording itself made.
                        radius_m=25.0,
                    )
                    if stall_x == stall_x
                    else (floor_here, float("nan"))
                )
                if recorded_dwell == recorded_dwell and here_budget > global_abort:
                    print(
                        f"[info] the recording waited {recorded_dwell:.0f}s here; "
                        f"allowing {here_budget:.0f}s before giving up"
                    )
                if here_budget > 0.0 and stalled_sec >= here_budget:
                    global_abort = here_budget
                    return "stuck", (
                        f"{wait_detail}; stuck_abort "
                        f"cumulative={stalled_sec:.0f}s>={global_abort:.0f}s "
                        f"forward_shifts={shifts}"
                    )

                if not reposition_enabled:
                    # No teleports: keep waiting inside the budget. The model's
                    # hesitation is the measurement; the budget above is the
                    # only honest way out of it.
                    continue

                if not reposition_unlimited and shifts >= reposition_max:
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
                interventions["forward_shifts"] += 1
                interventions["reposition_total_m"] += float(
                    cfg.stuck_reposition_forward_m
                )

                if current_goal is not None:
                    route_ok, route_detail = advance_route(
                        cfg, env, current_goal,
                        allow_goal_substitution=(
                            bool(route_goals) and current_goal == route_goals[-1]
                        ),
                    )
                    interventions["route_resets"] += 1
                    # The re-route is a fresh set_route and can be relocated just like
                    # the original — and this is the path a stuck leg actually takes.
                    # Keep watching the pose Autoware was given, or the ego drives to
                    # the accepted goal while arrival is measured somewhere else.
                    reroute_goal = _parse_effective_goal(route_detail)
                    if route_ok and reroute_goal is not None:
                        moved_reroute = math.hypot(
                            reroute_goal[0] - current_goal[0],
                            reroute_goal[1] - current_goal[1],
                        )
                        if moved_reroute > 0.5:
                            print(
                                f"[info] Re-route relocated the goal {moved_reroute:.1f} m "
                                "— tracking arrival against the accepted pose"
                            )
                        # legs *is* route_goals for multi-goal runs, so updating the
                        # slot keeps the outer leg loop in step with the re-route.
                        for _i, _g in enumerate(route_goals):
                            if _g == current_goal:
                                route_goals[_i] = reroute_goal
                                break
                        current_goal = reroute_goal
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

        if cfg.window_eval:
            # Leg windows (window_eval.py): the recording is the timeline, one
            # leg at a time; the stall / nudge / re-sync loop below does not run.
            from dp_multi_eval.bag_reader import load_reference_ego

            reference_ego = load_reference_ego(bag_path)
            leg_goals = (
                list(route_goals)
                if (multi_goal_active and len(route_goals) > 1)
                else ([arrival_goal] if arrival_goal is not None else [])
            )
            cuts = find_cuts(reference_ego, leg_goals) if leg_cuts_on(cfg) else None
            windows = plan_legs(
                reference_ego, leg_goals,
                cut_final=bool(cut_final_leg or cut_end_drive_through),
                cut_end_goal=cut_end_goal,
                cuts=cuts,
            )
            n_parents = len({w.parent for w in windows})
            print(
                f"[info] leg-window evaluation: {len(windows)} window(s) over {n_parents} leg(s)"
                + (f", {len(windows) - n_parents} planned cut(s) of {len(cuts)} recorded stand(s)" if cuts is not None else ", planned cuts off")
            )

            def _on_leg(idx: int, total: int) -> None:
                if live_monitor is not None:
                    live_monitor.set_leg(idx, total)
                    live_monitor.set_phase("driving", note="final" if idx == total else f"leg {idx}/{total}")

            status, error = run_legs(
                cfg, env, legs=windows, reference=reference_ego,
                track=reference_track if reference_track is not None else _reference_track(bag_path),
                interventions=interventions, output_dir=cfg.output_dir,
                on_leg=_on_leg,
            )
            legs = []
        for leg_idx, _leg in enumerate(legs):
            if live_monitor is not None:
                live_monitor.set_leg(leg_idx + 1, len(legs))
                live_monitor.set_phase("driving", note="final" if leg_idx + 1 == len(legs) else f"leg {leg_idx + 1}/{len(legs)}")
            remaining = max(5.0, drive_deadline - time.time())
            is_intermediate = (
                multi_goal_active and len(legs) > 1 and leg_idx < len(legs) - 1
            )
            # Intermediate multi-goal legs must not burn the full bag_duration budget.
            wait_timeout = min(remaining, leg_cap) if is_intermediate else remaining
            if multi_goal_active and len(legs) > 1:
                print(
                    f"[info] Driving leg {leg_idx + 1}/{len(legs)} "
                    f"(leg_wait={wait_timeout:.0f}s, overall remaining={remaining:.0f}s)..."
                )
            if cfg.end_condition == "route_arrived" or is_intermediate:
                # A SUBSTITUTED final goal is a place the recorded bus drove
                # past, not a stop it made: demanding a full stop there hands
                # the model a route-end context the vehicle never saw, and v4
                # answers it with a two-minute oscillating crawl (verified
                # with byte-identical lights/objects — the crawl is the
                # model's route-end behaviour). Arrival at such a goal is
                # drive-through: reaching it counts, stopping is not required.
                # The FINAL goal of a bag replay marks where the recording
                # ends; the terminal parking micro-approach after that point
                # measures nothing (verified on 2026-04-28: the model pulses
                # its own diff_plan stop wall on its nose for ~70 s over the
                # last 15-40 m, then parks 1 m from the recorded spot). The
                # run ends when the ego reaches the goal area; the pulsing
                # stays visible in stuck_rate/reference_speed as a report.
                # Drive-through only for a goal that was actually SUBSTITUTED
                # (relocated >10 m to a routable spot the recording drove
                # past). The 70 s pulse-crawl that motivated the cutoff was the
                # reproducer streaming cross-traffic in front of the mistimed
                # ego; the progress clock removed it, so the real goal is now a
                # clean approach and the ego must drive all the way to it. With
                # the cutoff always on, the run ended 12-15 m short of the real
                # bag goal while the ego was still moving at 2.4 m/s — it looked
                # like it stopped near the goal, but the run had just cut off.
                substituted_final = (
                    not is_intermediate
                    and bool(getattr(cfg, "final_drive_through_arrival", True))
                    and float(goal_substituted_m or 0.0) > 10.0
                )
                if substituted_final:
                    print(
                        "[info] substituted final goal — drive-through arrival "
                        "(reaching the goal area ends the run; no stop required)"
                    )
                # A cut recording: the route continues to the next stop, the
                # ego passes the cut pose at road speed, and the run ends
                # there. A tight radius is enough — it is not slowing down.
                cut_final = not is_intermediate and cut_end_drive_through
                if cut_final:
                    print(
                        "[info] cut-while-moving recording — the run ends when the "
                        "ego passes the cut pose (6 m, no stop required)"
                    )
                bus_stop_leg = is_intermediate and leg_is_bus_stop(_leg, bus_stops_xy)
                if bus_stop_leg:
                    print(
                        f"[info] leg {leg_idx + 1}/{len(legs)} is a map bus stop — "
                        "the ego must stop there (no drive-through)"
                    )
                end_status, detail = wait_with_stuck_recovery(
                    wait_timeout, _leg,
                    require_stop_at_goal=(
                        (not is_intermediate and not substituted_final and not cut_final)
                        or bus_stop_leg
                    ),
                    # Wide enough to register while the ego still moves at its
                    # route-end deceleration, before the pulse-glide begins.
                    arrival_tolerance_m=(
                        20.0 if substituted_final else (6.0 if cut_final else None)
                    ),
                )
                if end_status != "done":
                    if (
                        cfg.end_condition == "bag_duration"
                        and time.time() >= drive_deadline
                    ):
                        print(f"[info] bag_duration complete ({detail})")
                        status = "done"
                        error = None
                        break
                    # A run that expires on the LAST leg within 60 m of its
                    # goal has driven the route; the last metres were a crawl
                    # (measured: 0.06 m/s for six minutes toward a
                    # substituted mid-road goal). That is model behaviour to
                    # measure, not a tool DNF: complete the run and let
                    # goal_stop_precision and reference_speed judge it.
                    is_last = leg_idx == len(legs) - 1
                    gp = _parse_ego_from_detail(detail)
                    near_goal = (
                        is_last
                        and gp is not None
                        and arrival_goal is not None
                        and math.hypot(gp[0] - arrival_goal[0], gp[1] - arrival_goal[1]) < 60.0
                    )
                    if near_goal:
                        print(
                            f"[warn] time expired {math.hypot(gp[0] - arrival_goal[0], gp[1] - arrival_goal[1]):.0f} m "
                            f"from the final goal — completing the run; the goal "
                            f"metrics will judge the shortfall ({detail})"
                        )
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
                # Unroutable intermediates merge forward here too; legs and
                # route_goals are the same list, so popping keeps them aligned
                # and enumerate resumes on the merged-forward goal.
                adv_ok, adv_msg = False, "no_legs_left"
                while leg_idx + 1 < len(route_goals):
                    next_goal = route_goals[leg_idx + 1]
                    final_leg = leg_idx + 1 == len(route_goals) - 1
                    routing_goal = next_goal
                    cut_leg_now = False
                    if final_leg and cut_final_leg and cut_end_goal is not None:
                        # The recording was cut while moving: route on to the
                        # bus's next stop so the model keeps the road context,
                        # and end the run when the ego passes the cut pose.
                        routing_goal = cut_end_goal
                        cut_leg_now = True
                        print(
                            f"[info] final leg: recording cut while moving — routing to "
                            f"({routing_goal[0]:.1f}, {routing_goal[1]:.1f}), arrival "
                            "drive-through at the cut pose"
                        )
                    adv_ok, adv_msg = advance_route(
                        cfg, env, routing_goal,
                        allow_goal_substitution=final_leg,
                    )
                    if adv_ok or final_leg:
                        break
                    print(
                        f"[warn] Leg goal ({next_goal[0]:.1f}, {next_goal[1]:.1f}) "
                        f"is not routable from here — merging it into the next leg"
                    )
                    route_goals.pop(leg_idx + 1)
                # Same substitution applies here, and the same mismatch follows if
                # the harness keeps watching the pose it asked for.
                adv_effective = _parse_effective_goal(adv_msg)
                if adv_ok and cut_leg_now:
                    cut_end_drive_through = True
                    cut_end_pose = next_goal
                    adv_effective = None  # arrival stays at the cut pose
                if adv_ok and adv_effective is not None:
                    adv_moved = math.hypot(
                        adv_effective[0] - next_goal[0], adv_effective[1] - next_goal[1]
                    )
                    if adv_moved > 0.5:
                        print(
                            f"[info] Leg {leg_idx + 2} goal was relocated "
                            f"{adv_moved:.1f} m by route setup — tracking the accepted pose"
                        )
                    route_goals[leg_idx + 1] = adv_effective
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
        close_routing(env)
        # Shutdown order: record → reproducer → psim
        stragglers = []
        for proc in reversed(procs):
            if not stop_process_group(proc, grace_sec=PSIM_STOP_GRACE_SEC):
                stragglers.append(proc.pid)
            close_log_handle(proc)
        if stragglers:
            print(f"[warn] process group(s) {stragglers} outlived their stop request")
        # Sweep anything the launch spawned outside those groups. Until every
        # planner/container process is gone the driver keeps their GPU memory, and
        # the next job's TensorRT engine fails with "Cuda Runtime (out of memory)".
        if not cleanup_evaluation_processes(domain_id=cfg.domain_id):
            print("[warn] leftover evaluation processes remain after this job")

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
            "interventions": interventions,
            # stall_x / stall_y are a position, not a count (see run_orchestrator).
            "clean_run": not any(
                float(v) > 0 for k, v in interventions.items() if not str(k).startswith("stall_")
            ),
            "legs_skipped": legs_skipped,
            "goal_substituted_m": (
                None if goal_substituted_m != goal_substituted_m else round(goal_substituted_m, 3)
            ),
        },
    )
    return JobResult(
        status,
        start_iso,
        end_iso,
        error,
        bag_path_out,
        time.time() - t0,
        interventions=interventions,
    )



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
    p.add_argument("--resync_lag_sec", type=float, default=0.0)
    p.add_argument("--window_eval", action="store_true", help="Leg-window evaluation (window_eval.py)")
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
        resync_lag_sec=float(args.resync_lag_sec),
        window_eval=bool(args.window_eval),
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
