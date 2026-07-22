"""Helpers for process groups and timeouts."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def popen(cmd: list[str], *, env: dict[str, str] | None = None, cwd: str | None = None) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def popen_log_file(
    cmd: list[str],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> tuple[subprocess.Popen[Any], Any]:
    """Launch a process with stdout/stderr appended to log_path (avoids PIPE deadlock)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8", buffering=1)  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        preexec_fn=os.setsid,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._dp_multi_eval_log_handle = log_handle  # type: ignore[attr-defined]
    return proc, log_handle


def close_log_handle(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    handle = getattr(proc, "_dp_multi_eval_log_handle", None)
    if handle is not None:
        try:
            handle.close()
        except OSError:
            pass
        proc._dp_multi_eval_log_handle = None  # type: ignore[attr-defined]


def stop_process_group(proc: subprocess.Popen[Any] | None, grace_sec: float = 8.0) -> None:
    if proc is None:
        return
    # The ros2/launch parent may exit before its child nodes. Always target its
    # process group, even when Popen.poll() says the parent is already gone.
    stop_pid_group(proc.pid, grace_sec=grace_sec)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def stop_pid_group(pid: int, grace_sec: float = 8.0) -> bool:
    """Send SIGTERM then SIGKILL to the process group containing pid."""
    if pid <= 0:
        return False
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        pgid = pid
    sent = False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
            sent = True
        except ProcessLookupError:
            try:
                os.kill(pid, sig)
                sent = True
            except ProcessLookupError:
                return sent
        except PermissionError:
            return sent
        if sig == signal.SIGTERM:
            deadline = time.time() + grace_sec
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return sent
                time.sleep(0.2)
    return sent


def cleanup_evaluation_processes() -> None:
    """Best-effort cleanup of sim/reproducer/bag processes left after a stop."""
    patterns = (
        "planning_simulator.launch",
        "perception_reproducer.py",
        "scenario_test_runner",
        "openscenario_interpreter",
        "simple_planning_simulator",
        "rosbridge_websocket",
        "ros2 bag record",
        "dp_multi_eval.run_single_job",
    )
    groups: set[int] = set()
    own_group = os.getpgrp()
    for pattern in patterns:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            check=False,
            capture_output=True,
            text=True,
        )
        for text_pid in result.stdout.split():
            try:
                pid = int(text_pid)
                pgid = os.getpgid(pid)
            except (ValueError, ProcessLookupError):
                continue
            if pgid != own_group:
                groups.add(pgid)

    for sig, delay in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 0.0)):
        for pgid in groups:
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        if delay:
            time.sleep(delay)


def wait_with_timeout(proc: subprocess.Popen[Any], timeout_sec: float) -> int | None:
    """Return exit code, or None on timeout (process still running)."""
    try:
        return proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return None
