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

"""Per-job AMENT overlay so each parallel worker can use its own model param.

Autoware's diffusion planner launch resolves:
  $(find-pkg-share autoware_launch)/config/planning/neural_net_planner/diffusion_planner.param.yaml

The overlay clones ``share/autoware_launch`` from the real install (hardlinks when
possible) and replaces only the job's planner param (``diffusion_planner.param.yaml``,
or ``ml_planner.param.yaml`` for ``planning_setting=ml_planner``). A minimal overlay that
only contained the param file would shadow the whole package and break
``ros2 launch autoware_launch planning_simulator.launch.xml``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

PARAM_REL = Path("config/planning/neural_net_planner/diffusion_planner.param.yaml")
ML_PARAM_REL = Path("config/planning/neural_net_planner/ml_planner.param.yaml")
TRAJ_MOD_REL = Path("config/planning/neural_net_planner/trajectory_modifier.param.yaml")


def resolve_autoware_launch_share() -> Path:
    try:
        prefix = subprocess.check_output(
            ["ros2", "pkg", "prefix", "autoware_launch"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        share = Path(prefix) / "share" / "autoware_launch"
        if share.is_dir():
            return share
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    fallback = Path("/opt/autoware/share/autoware_launch")
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(
        "autoware_launch share directory not found. Source the Autoware workspace first."
    )


def _copy_file_hardlink_or_copy(src: str, dst: str) -> None:
    """Prefer hardlinks on the same filesystem; fall back to a regular copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_overlay_file(src: Path, dst: Path) -> None:
    """Write ``src`` to ``dst`` as a fresh, private file.

    The install tree ships these params as symlinks back into ``src/autoware``,
    and ``copytree(symlinks=True)`` faithfully reproduces them, so writing to the
    overlay path followed the link and rewrote the Autoware source file. Two
    workers evaluating different models then wrote the same file and both loaded
    whichever landed last, which would silently invalidate any M x N comparison.
    Unlinking first also breaks the hardlinks ``_copy_file_hardlink_or_copy``
    leaves behind, so the overlay never shares an inode with the install tree.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    shutil.copyfile(src, dst)


def _clone_share_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    # symlinks=True keeps install-tree symlinks as symlinks. Without it, copytree
    # dereferences symlinks and fails on dangling ones (e.g. trajectory_modifier).
    # copy_function must not be os.link alone: results on another drive (e.g. /media)
    # raise EINVAL and the old fallback then hit the broken symlink.
    shutil.copytree(
        src,
        dst,
        copy_function=_copy_file_hardlink_or_copy,
        symlinks=True,
        dirs_exist_ok=True,
    )


def _disable_long_stop_in_overlay(overlay_share: Path) -> None:
    """Turn off ONLY the long_stop rule in the overlay's trajectory_modifier.

    Measured on 2026-08-25: during 81 s and 48 s forced-stop windows the raw
    model plans reached 2.5-6.5 m/s with a first second of 0.02-0.08 m/s —
    the gate looks only at the first 1.0 s, replaced every slow-starting
    departure with a stop stub, and the ego, never moving, kept producing
    from-rest plans: a self-locking loop broken only when the first second
    happened to clear 0.25 m/s. The vehicle's own config is untouched (the
    overlay lives in /tmp per job), close_to_ego and obstacle_stop stay on,
    and fixer_replay still reports offline what the rule would have done.
    """
    import yaml

    path = overlay_share / TRAJ_MOD_REL
    data = yaml.safe_load(path.read_text())
    sp = data["/**"]["ros__parameters"]["stop_point_fixer"]
    sp["force_stop_long_stopped_trajectories"] = False
    # close_to_ego turned out to be the same latch wearing another hat: with
    # long_stop off it kept stubbing the model's own legitimate stop plans
    # (goal approach, hesitation), the stub detector read that as a stack
    # hold and nudged the ego 7 times mid-road — the "nonsensical lurch"
    # visible on 2026-04-28's video. Only the fixer's stop-stub rules go;
    # the obstacle_stop plugin (real collision guard) stays armed.
    sp["force_stop_close_stopped_trajectories"] = False
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    path.unlink()
    tmp.rename(path)


DP_LAUNCH_REL = Path("launch/e2e_components/diffusion_planner.launch.xml")
VEHICLE_TURN_INDICATOR_TOPIC = "/vehicle/status/turn_indicators_status"


def set_dp_turn_indicator_input(overlay_share: Path, topic: str) -> None:
    """Point the diffusion planner's turn-indicator INPUT at ``topic`` (sim-only).

    On the vehicle the status topic has one writer, the vehicle interface. In
    the replay it had two: psim's report (an echo of the model's own command)
    and the reproducer's recorded signal, interleaved on the model's input.
    Measured at 2026-04-28's bus stop: LEFT/DISABLE alternating 50/40 per
    second while the model was deciding whether to pull out. The reproducer
    merges the two into ``topic`` (recorded signal when the recording has one,
    psim's echo otherwise) and the model reads that single stream. The vehicle
    launch file is untouched; only the overlay copy is rewritten.
    """
    path = overlay_share / DP_LAUNCH_REL
    text = path.read_text(encoding="utf-8")
    needle = f'<arg name="input_turn_indicators" value="{VEHICLE_TURN_INDICATOR_TOPIC}"/>'
    if needle not in text:
        raise RuntimeError(f"input_turn_indicators arg not found in {path}")
    text = text.replace(needle, f'<arg name="input_turn_indicators" value="{topic}"/>')
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    path.unlink()
    tmp.rename(path)


def add_vehicle_dynamics_overlay(overlay_root: Path, vehicle_model: str) -> None:
    """Make the vehicle description's own delays effective in the simulator.

    The Erga description ships vehicle_model_type IDEAL_STEER_ACC_GEARED next
    to acc_time_delay 0.6 / acc_time_constant 0.1 and steer_time_delay 0.24 /
    steer_time_constant 0.27 — and the IDEAL model ignores all four. The
    overlay switches the type to DELAY_STEER_ACC_GEARED and changes nothing
    else, so the description's numbers, not invented ones, shape the ego.
    Measured on 2026-04-28 with the ideal model: the optimizer's 0.5 m/s
    engage floor turned every wait into 1.0-1.6 m/s creep bursts (84 s of
    stop-go on a bus-stop approach); with delayed dynamics the bursts stay
    at ~0.6 m/s.
    """
    import subprocess
    import yaml

    pkg = f"{vehicle_model}_description"
    try:
        prefix = subprocess.check_output(
            ["ros2", "pkg", "prefix", pkg], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"[warn] {pkg} not found — vehicle dynamics overlay skipped")
        return
    real_share = Path(prefix) / "share" / pkg
    param_rel = Path("config/simulator_model.param.yaml")
    if not (real_share / param_rel).is_file():
        print(f"[warn] {pkg} has no simulator_model.param.yaml — overlay skipped")
        return
    overlay_share = overlay_root / "share" / pkg
    _clone_share_tree(real_share, overlay_share)
    path = overlay_share / param_rel
    data = yaml.safe_load(path.read_text())
    rp = data["/**"]["ros__parameters"]
    rp["vehicle_model_type"] = "DELAY_STEER_ACC_GEARED"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    path.unlink()
    tmp.rename(path)
    marker_dir = overlay_root / "share" / "ament_index" / "resource_index" / "packages"
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / pkg).write_text("", encoding="utf-8")


def make_model_overlay(
    model_config: Path, overlay_root: Path, *, disable_long_stop: bool = False,
    dp_turn_indicator_topic: str | None = None,
    planning_setting: str = "diffusion_planner",
) -> Path:
    """
    Create overlay_root with a full autoware_launch share tree and the job param.
    Returns overlay_root (prepend this to AMENT_PREFIX_PATH).
    """
    model_config = model_config.expanduser().resolve()
    if not model_config.is_file():
        raise FileNotFoundError(f"model_config not found: {model_config}")

    real_share = resolve_autoware_launch_share()
    overlay_share = overlay_root / "share" / "autoware_launch"
    _clone_share_tree(real_share, overlay_share)
    param_rel = ML_PARAM_REL if planning_setting == "ml_planner" else PARAM_REL
    _write_overlay_file(model_config, overlay_share / param_rel)
    if disable_long_stop:
        _disable_long_stop_in_overlay(overlay_share)
    if dp_turn_indicator_topic:
        set_dp_turn_indicator_input(overlay_share, dp_turn_indicator_topic)

    (overlay_root / "share" / "ament_index" / "resource_index" / "packages").mkdir(
        parents=True, exist_ok=True
    )
    marker = (
        overlay_root
        / "share"
        / "ament_index"
        / "resource_index"
        / "packages"
        / "autoware_launch"
    )
    marker.write_text("", encoding="utf-8")

    return overlay_root


def env_with_overlay(
    base_env: dict[str, str] | None,
    overlay_root: Path,
    domain_id: int,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env["ROS_DOMAIN_ID"] = str(domain_id)
    prefix = str(overlay_root.resolve())
    existing = env.get("AMENT_PREFIX_PATH", "")
    env["AMENT_PREFIX_PATH"] = f"{prefix}:{existing}" if existing else prefix
    return env
