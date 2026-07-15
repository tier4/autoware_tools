"""Scenario Simulator v2 helpers for dp_multi_eval (OpenSCENARIO / T4 YAML)."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

SCENARIO_EXTENSIONS = (".yaml", ".yml", ".xosc")


def discover_scenarios(
    scenario_dir: Path,
    *,
    extensions: tuple[str, ...] = SCENARIO_EXTENSIONS,
) -> list[Path]:
    """Find scenario files under scenario_dir (one level of nesting allowed)."""
    scenario_dir = scenario_dir.expanduser().resolve()
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"scenario_dir missing: {scenario_dir}")

    found: list[Path] = []
    ext_set = {e.lower() for e in extensions}

    def _maybe_add(path: Path) -> None:
        if path.is_file() and path.suffix.lower() in ext_set:
            # Skip pipeline / config YAMLs that are not scenarios
            name = path.name.lower()
            if name in ("metadata.yaml", "pipeline_config.yaml", "args.json"):
                return
            found.append(path.resolve())

    for child in sorted(scenario_dir.iterdir()):
        if child.is_file():
            _maybe_add(child)
        elif child.is_dir():
            for sub in sorted(child.iterdir()):
                if sub.is_file():
                    _maybe_add(sub)
                elif sub.is_dir():
                    for nested in sorted(sub.iterdir()):
                        if nested.is_file():
                            _maybe_add(nested)
    # Deduplicate
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def scenario_key(scenario: Path, scenario_dir: Path) -> str:
    try:
        rel = scenario.resolve().relative_to(scenario_dir.resolve())
        return str(rel.with_suffix(""))
    except ValueError:
        return scenario.stem


def scenario_test_runner_available(env: dict[str, str] | None = None) -> bool:
    if shutil.which("ros2") is None:
        return False
    try:
        subprocess.check_output(
            ["ros2", "pkg", "prefix", "scenario_test_runner"],
            env=env,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15.0,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def build_scenario_runner_command(
    *,
    scenario_path: Path,
    output_directory: Path,
    vehicle_model: str,
    sensor_model: str,
    architecture_type: str = "awf/universe/20250130",
    planning_setting: str = "diffusion_planner",
    initialize_duration_sec: float = 120.0,
    global_timeout_sec: float = 300.0,
    rviz: bool = False,
    record: bool = False,
    vehicle_id: str | None = None,
) -> list[str]:
    """Launch Scenario Simulator v2 test runner with Autoware + diffusion planner."""
    cmd = [
        "ros2",
        "launch",
        "scenario_test_runner",
        "scenario_test_runner.launch.py",
        f"scenario:={scenario_path.expanduser().resolve()}",
        f"output_directory:={output_directory.expanduser().resolve()}",
        f"architecture_type:={architecture_type}",
        f"sensor_model:={sensor_model}",
        f"vehicle_model:={vehicle_model}",
        f"initialize_duration:={int(initialize_duration_sec)}",
        f"global_timeout:={int(global_timeout_sec)}",
        f"launch_rviz:={'true' if rviz else 'false'}",
        f"record:={'true' if record else 'false'}",
        # Autoware launch selection (CI / SS2 conventions)
        "autoware_launch_package:=autoware_launch",
        "autoware_launch_file:=planning_simulator.launch.xml",
        f"planning_setting:={planning_setting}",
        # Dotted overrides used by SS2 / Web.Auto (match manual launches).
        # planning_simulator.launch.xml defaults rviz:=true — must override explicitly
        # or Autoware still opens RViz even when launch_rviz:=false.
        f"autoware.planning_setting:={planning_setting}",
        f"autoware.rviz:={'true' if rviz else 'false'}",
    ]
    if vehicle_id:
        cmd.append(f"vehicle_id:={vehicle_id}")
        cmd.append(f"autoware.vehicle_id:={vehicle_id}")
    return cmd


def build_record_command_sim_time(output_bag: Path, topics: list[str]) -> list[str]:
    """ros2 bag record with sim time (required when scenario_simulation:=true)."""
    return [
        "ros2",
        "bag",
        "record",
        "-o",
        str(output_bag),
        "--storage",
        "sqlite3",
        "--use-sim-time",
        *topics,
    ]


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pose_from_mapping(node: dict[str, Any]) -> tuple[float, float, float] | None:
    """Extract (x, y, yaw) from common pose dict shapes (incl. WorldPosition)."""
    if not isinstance(node, dict):
        return None
    # Nested Position / WorldPosition / Orientation
    pos = node.get("Position") or node.get("position") or node
    if isinstance(pos, dict):
        wp = pos.get("WorldPosition") or pos.get("worldPosition")
        if isinstance(wp, dict):
            pos = wp
    if not isinstance(pos, dict):
        return None
    x = _as_float(pos.get("x"))
    y = _as_float(pos.get("y"))
    if x is None or y is None:
        return None
    yaw = None
    for key in ("yaw", "heading", "h", "theta"):
        yaw = _as_float(pos.get(key))
        if yaw is not None:
            break
        yaw = _as_float(node.get(key))
        if yaw is not None:
            break
    orient = node.get("Orientation") or node.get("orientation")
    if yaw is None and isinstance(orient, dict):
        yaw = _as_float(orient.get("yaw") or orient.get("h"))
    return x, y, float(yaw or 0.0)


def _walk_goal_candidates(node: Any, found: list[tuple[float, float, float]], depth: int = 0) -> None:
    if depth > 12 or node is None:
        return
    if isinstance(node, dict):
        keys_lower = {str(k).lower() for k in node}
        goalish = keys_lower & {
            "destination",
            "goal",
            "goalpose",
            "goal_pose",
            "routegoal",
            "route_goal",
            "endpose",
            "end_pose",
            "acquirepositionaction",
        }
        if goalish or ("x" in node and "y" in node and ("yaw" in node or "h" in node)):
            for key, value in node.items():
                kl = str(key).lower()
                if kl in {
                    "destination",
                    "goal",
                    "goalpose",
                    "goal_pose",
                    "routegoal",
                    "route_goal",
                    "endpose",
                    "end_pose",
                    "acquirepositionaction",
                }:
                    pose = _pose_from_mapping(value if isinstance(value, dict) else node)
                    if pose is not None:
                        found.append(pose)
            if "x" in node and "y" in node and goalish:
                pose = _pose_from_mapping(node)
                if pose is not None:
                    found.append(pose)
        for value in node.values():
            _walk_goal_candidates(value, found, depth + 1)
    elif isinstance(node, list):
        for item in node:
            _walk_goal_candidates(item, found, depth + 1)


def _ego_acquire_position_goal(raw: Any) -> tuple[float, float, float] | None:
    """T4 / OpenSCENARIO YAML: ego Init → RoutingAction → AcquirePositionAction."""
    root = raw.get("OpenSCENARIO", raw) if isinstance(raw, dict) else None
    if not isinstance(root, dict):
        return None
    storyboard = root.get("Storyboard") or root.get("storyboard")
    if not isinstance(storyboard, dict):
        return None
    init = storyboard.get("Init") or storyboard.get("init")
    if not isinstance(init, dict):
        return None
    actions = init.get("Actions") or init.get("actions") or {}
    privates = []
    if isinstance(actions, dict):
        privates = actions.get("Private") or actions.get("private") or []
    if not isinstance(privates, list):
        return None

    for private in privates:
        if not isinstance(private, dict):
            continue
        entity = str(private.get("entityRef") or private.get("entityref") or "").lower()
        if entity not in ("ego", "ego_vehicle", "hero"):
            continue
        for pact in private.get("PrivateAction") or private.get("privateAction") or []:
            if not isinstance(pact, dict):
                continue
            routing = pact.get("RoutingAction") or pact.get("routingAction") or pact
            if not isinstance(routing, dict):
                continue
            acquire = (
                routing.get("AcquirePositionAction")
                or routing.get("acquirePositionAction")
                or pact.get("AcquirePositionAction")
            )
            if isinstance(acquire, dict):
                pose = _pose_from_mapping(acquire)
                if pose is not None:
                    return pose
    return None


def _goal_from_xosc(text: str) -> tuple[float, float, float] | None:
    # Prefer AcquirePositionAction / Destination WorldPosition
    acquire = re.findall(
        r"(?is)AcquirePositionAction.*?WorldPosition[^>]*x\s*=\s*\"([^\"]+)\"[^>]*y\s*=\s*\"([^\"]+)\"(?:[^>]*h\s*=\s*\"([^\"]+)\")?",
        text,
    )
    if acquire:
        x, y, h = acquire[0]
        return float(x), float(y), float(h or 0.0)
    dest_blocks = re.findall(
        r"(?is)Destination.*?WorldPosition[^>]*x\s*=\s*\"([^\"]+)\"[^>]*y\s*=\s*\"([^\"]+)\"(?:[^>]*h\s*=\s*\"([^\"]+)\")?",
        text,
    )
    if dest_blocks:
        x, y, h = dest_blocks[-1]
        return float(x), float(y), float(h or 0.0)
    positions = re.findall(
        r"(?is)WorldPosition[^>]*x\s*=\s*\"([^\"]+)\"[^>]*y\s*=\s*\"([^\"]+)\"(?:[^>]*h\s*=\s*\"([^\"]+)\")?",
        text,
    )
    if positions:
        x, y, h = positions[-1]
        return float(x), float(y), float(h or 0.0)
    return None


def extract_goal_from_scenario(scenario_path: Path) -> tuple[float, float, float] | None:
    """Best-effort goal pose from T4 YAML or OpenSCENARIO XML."""
    scenario_path = scenario_path.expanduser().resolve()
    if not scenario_path.is_file():
        return None
    suffix = scenario_path.suffix.lower()
    try:
        text = scenario_path.read_text(encoding="utf-8")
    except OSError:
        return None

    if suffix == ".xosc":
        return _goal_from_xosc(text)

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        return _goal_from_xosc(text)

    # Preferred: ego AcquirePositionAction (Scenario Editor / SS2 export)
    ego_goal = _ego_acquire_position_goal(raw)
    if ego_goal is not None:
        return ego_goal

    found: list[tuple[float, float, float]] = []
    _walk_goal_candidates(raw, found)
    if found:
        return found[-1]
    return _goal_from_xosc(text)


def extract_goal_from_output_bag(bag_path: Path) -> tuple[float, float, float]:
    """Fallback: last ego pose in the recorded output bag."""
    try:
        from dp_multi_eval.bag_reader import load_bag_series

        series = load_bag_series(bag_path)
        if series.ego:
            e = series.ego[-1]
            return e.x, e.y, e.yaw_rad
    except Exception:  # noqa: BLE001
        pass
    return 0.0, 0.0, 0.0


def normalize_yaw(yaw: float) -> float:
    while yaw > math.pi:
        yaw -= 2.0 * math.pi
    while yaw < -math.pi:
        yaw += 2.0 * math.pi
    return yaw
