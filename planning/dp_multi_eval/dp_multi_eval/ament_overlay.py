"""Per-job AMENT overlay so each parallel worker can use its own model param.

Autoware's diffusion planner launch resolves:
  $(find-pkg-share autoware_launch)/config/planning/neural_net_planner/diffusion_planner.param.yaml

The overlay clones ``share/autoware_launch`` from the real install (hardlinks when
possible) and replaces only ``diffusion_planner.param.yaml``. A minimal overlay that
only contained the param file would shadow the whole package and break
``ros2 launch autoware_launch planning_simulator.launch.xml``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

PARAM_REL = Path("config/planning/neural_net_planner/diffusion_planner.param.yaml")


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


def make_model_overlay(model_config: Path, overlay_root: Path) -> Path:
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
    shutil.copy2(model_config, overlay_share / PARAM_REL)

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
