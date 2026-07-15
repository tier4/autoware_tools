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
    if proc.poll() is not None:
        return
    stop_pid_group(proc.pid, grace_sec=grace_sec)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def stop_pid_group(pid: int, grace_sec: float = 8.0) -> bool:
    """Send SIGTERM then SIGKILL to the process group led by pid."""
    if pid <= 0:
        return False
    sent = False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
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
        "ros2 bag record",
        "dp_multi_eval.run_single_job",
    )
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-f", pattern],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def wait_with_timeout(proc: subprocess.Popen[Any], timeout_sec: float) -> int | None:
    """Return exit code, or None on timeout (process still running)."""
    try:
        return proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return None
