"""The workspace environment a job's processes start in, and why a launch died."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_LAUNCH_ERROR = re.compile(r"\[ERROR\] \[launch\]: (.+)")
_PKG_MISSING = re.compile(r"package '([\w-]+)' not found")


def workspace_env(base: dict[str, str]) -> dict[str, str]:
    """``base`` with every workspace on COLCON_PREFIX_PATH sourced again.

    A job inherits the environment of whoever started it, frozen at that
    moment. A package built after the GUI started is not on its
    AMENT_PREFIX_PATH, and psim died with "package ... not found"
    (2026-11-05: autoware_trajectory_selector, built 31 min after the GUI
    started). Sourcing the install's setup again adds what is new and keeps
    the rest (colcon prepends unique values).
    """
    prefixes = [p for p in base.get("COLCON_PREFIX_PATH", "").split(os.pathsep) if p]
    # COLCON_PREFIX_PATH lists the overlay first: source the underlay first.
    scripts = [Path(p) / "setup.bash" for p in reversed(prefixes)]
    scripts = [s for s in scripts if s.is_file()]
    if not scripts:
        return dict(base)
    cmd = "".join(f'source "{s}" >/dev/null 2>&1; ' for s in scripts) + "env -0"
    try:
        out = subprocess.run(
            ["bash", "-c", cmd], env=base, capture_output=True, timeout=120, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[warn] could not source the workspace again ({exc}); using the inherited environment")
        return dict(base)
    env: dict[str, str] = {}
    for item in out.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep and key:
            env[key.decode()] = value.decode(errors="replace")
    return env or dict(base)


def launch_failure_reason(log_path: Path) -> str | None:
    """The launch's own error from its log, readable; None when there is none."""
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _LAUNCH_ERROR.search(text)
    if not match:
        return None
    line = match.group(1)
    pkg = _PKG_MISSING.search(line)
    if pkg:
        name = pkg.group(1)
        return (
            f"package '{name}' not found — not built in this workspace, or the source "
            f"checkout lacks it (colcon build --symlink-install --packages-select {name})"
        )
    return line[:300]
