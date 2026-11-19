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

"""Helpers for process groups and timeouts."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def popen(cmd: list[str], *, env: dict[str, str] | None = None, cwd: str | None = None) -> subprocess.Popen[Any]:
    """Launch a detached process, discarding its output.

    Deliberately NOT a pipe. A pipe nobody reads is a time bomb: the child blocks
    forever on write() once it has filled the ~64 KB kernel buffer, and a blocked
    ros2 bag record neither records nor shuts down cleanly, so the recording ends
    up without its metadata.yaml and unreadable. Nothing here ever drained these
    pipes. Use popen_log_file when the output is worth keeping — which, for
    anything long-lived, it is.
    """
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
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


def _is_zombie(pid: int) -> bool:
    """A process that has exited and holds nothing — leader dead, no threads left.

    A zombie *leader* is not the same as a finished process. The planner exits its
    main thread while worker threads stay behind, sleeping, and those threads keep
    the CUDA context open: caught on this machine as three "dead" processes holding
    816 MiB each, invisible to a cleanup that took the leader's Z state as proof the
    process was gone. Sending SIGKILL to the thread group reaped all three and
    returned the memory, so the threads were killable the whole time — nothing was
    stuck in the driver, we simply stopped asking.

    So a leader in Z counts as a zombie only when no live thread remains.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            leader_z = fh.read().rpartition(")")[2].split()[0] == "Z"
    except (OSError, IndexError):
        return False
    if not leader_z:
        return False
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            if int(tid) == pid:
                continue
            try:
                with open(f"/proc/{pid}/task/{tid}/stat", encoding="utf-8") as fh:
                    if fh.read().rpartition(")")[2].split()[0] != "Z":
                        return False  # a live thread: the process still holds things
            except (OSError, IndexError):
                continue
    except OSError:
        pass
    return True


def group_alive(pgid: int) -> bool:
    """True while any process in the group is still running.

    Probing the parent pid instead would report the group dead as soon as ros2
    launch exits, while its node processes — including the one holding the CUDA
    context — are still running.

    Zombies do not count. killpg(pgid, 0) succeeds against them because they still
    occupy a pid, so a stack that shut down perfectly looks alive until whoever
    inherited the children reaps them, and the caller reports a straggler that has
    in fact already released everything it held.
    """
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    members = _group_members(pgid)
    if not members:
        return False
    return any(not _is_zombie(pid) for pid in members)


def stop_process_group(proc: subprocess.Popen[Any] | None, grace_sec: float = 8.0) -> bool:
    """Stop proc's whole process group. True when the group is confirmed gone."""
    if proc is None:
        return True
    # The ros2/launch parent may exit before its child nodes. Always target its
    # process group, even when Popen.poll() says the parent is already gone.
    gone = stop_pid_group(proc.pid, grace_sec=grace_sec)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass
    return gone


def stop_pid_group(
    pid: int,
    grace_sec: float = 8.0,
    kill_wait_sec: float = 5.0,
    stall_grace_sec: float = 6.0,
) -> bool:
    """SIGTERM then SIGKILL the process group containing pid.

    Returns True once no process remains in the group.

    Waiting on the group rather than on ``pid`` matters: ros2 launch exits before
    the nodes it spawned, so a parent-only wait returned success while the node
    processes lived on. They kept their CUDA contexts, and the next job's
    TensorRT engine then failed to allocate with "Cuda Runtime (out of memory)"
    on a GPU that looked busy with nothing.

    ``grace_sec`` is the outer bound; within it the wait follows progress, and
    ``stall_grace_sec`` is how long the group may go without a single process
    exiting before it counts as stuck. Stopping the stack by hand works because a
    person waits for it to finish — a fixed timer does not.
    """
    if pid <= 0:
        return True
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        pgid = pid

    if not group_alive(pgid):
        return True

    # SIGINT first, which is what Ctrl+C delivers and what ros2 launch expects to
    # start its own ladder with. rclcpp installs handlers for both, so the signal
    # itself is not the difference — but matching the interactive path costs nothing.
    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return True
    except PermissionError:
        return not group_alive(pgid)

    # Wait on *progress*, not on a fixed budget. A stack tearing down in order keeps
    # shrinking; killing it on a timer cuts whichever node happens to be unwinding
    # when the timer expires. That is how a planner gets SIGKILLed mid-exit while
    # still holding its CUDA context — the GPU memory stays out until it is reaped.
    # Only a group that has stopped shrinking is actually stuck.
    deadline = time.time() + grace_sec
    remaining = len(_group_members(pgid))
    last_change = time.time()
    while time.time() < deadline:
        if not group_alive(pgid):
            return True
        now_remaining = len(_group_members(pgid))
        if now_remaining < remaining:
            remaining = now_remaining
            last_change = time.time()          # still making progress, keep waiting
        elif time.time() - last_change >= stall_grace_sec:
            break                              # no exits for a while: it is stuck
        time.sleep(0.2)

    # Anything still up ignored SIGTERM (or is mid-shutdown). Escalate to the
    # whole group — never just the parent, which is usually already gone.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return not group_alive(pgid)

    deadline = time.time() + kill_wait_sec
    while time.time() < deadline:
        if not group_alive(pgid):
            return True
        time.sleep(0.1)
    return not group_alive(pgid)


# Processes a job leaves behind. The GPU-holding ones matter most: the planner
# node and the component containers own CUDA contexts, and until they exit the
# driver does not reclaim their memory. They used to be absent from this list, so
# cleanup swept the simulator and the reproducer while the process actually
# holding ~7 GB survived into the next job.
LEFTOVER_PROCESS_PATTERNS = (
    "autoware_diffusion_planner_node",
    "component_container_mt",
    "component_container",
    "planning_simulator.launch",
    "scene_driver.py",           # the original perception_reproducer runs inside it
    "perception_reproducer.py",
    "scenario_test_runner",
    "openscenario_interpreter",
    "simple_planning_simulator",
    "autoware_multi_object_tracker",
    "multi_object_tracker_node",
    "rosbridge_websocket",
    "ros2 bag record",
    # The objects throttle this harness starts. Matched by its output topic, not by
    # "topic_tools": Autoware runs relay nodes of its own that must survive a sweep.
    "objects_throttled",
    "dp_multi_eval.run_single_job",
)


def _own_process_tree() -> set[int]:
    """This process and its ancestors, which must never be killed as leftovers."""
    pids: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
                # ppid is field 4, after the comm field which may contain spaces
                pid = int(fh.read().rpartition(")")[2].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def describe_processes(pids: set[int]) -> list[str]:
    """`pid exe (args…)` for each pid still alive, for naming what refused to exit."""
    out: list[str] = []
    for pid in sorted(pids):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().decode("utf-8", "replace").replace("\0", " ").strip()
        except OSError:
            continue
        try:
            exe = os.path.basename(os.readlink(f"/proc/{pid}/exe"))
        except OSError:
            exe = "?"
        out.append(f"{pid} {exe} ({cmd[:90]})")
    return out


def _pgrep(args: list[str]) -> str:
    """pgrep with a hard bound.

    pgrep -f reads every process's /proc cmdline, and reading it from a process
    stuck in uninterruptible IO blocks on its mmap lock — observed as a 30 s+
    hang of the whole leftover scan while an unrelated indexer thrashed the
    machine. A scan that cannot finish must report nothing rather than hang the
    job that asked for it.
    """
    try:
        return subprocess.run(
            ["pgrep", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout
    except subprocess.TimeoutExpired:
        return ""


def leftover_process_pids(domain_id: int | None = None) -> set[int]:
    """Pids in leftover groups — the processes themselves, not just their groups."""
    pids: set[int] = set()
    for pgid in find_leftover_groups(domain_id):
        pids |= _group_members(pgid)
    return pids


def find_leftover_groups(domain_id: int | None = None) -> set[int]:
    """Process groups matching a leftover pattern, excluding our own.

    ``pgrep -f`` matches the whole command line, so it also hits processes that
    merely *mention* a pattern in their arguments — a shell running the batch, a
    log tail. Excluding our own process tree keeps a stray match from turning
    cleanup into self-termination.

    ``domain_id`` restricts the sweep to process groups running on that
    ``ROS_DOMAIN_ID``. Patterns match by node name, which is identical for every
    worker, so an unscoped sweep from one worker's teardown kills the simulator
    another worker is still driving — the ego pose simply stops updating
    mid-route. Workers must always pass their own domain; only a whole-run stop
    should sweep everything.
    """
    groups: set[int] = set()
    own_group = os.getpgrp()
    protected = _own_process_tree()
    for pattern in LEFTOVER_PROCESS_PATTERNS:
        for text_pid in _pgrep(["-f", pattern]).split():
            try:
                pid = int(text_pid)
                pgid = os.getpgid(pid)
            except (ValueError, ProcessLookupError):
                continue
            if pid in protected or pgid == own_group:
                continue
            if protected & _group_members(pgid):
                # Killing this group would take us with it.
                continue
            if domain_id is not None and domain_id not in _group_domain_ids(pgid):
                if not _group_is_orphaned(pgid):
                    # Another worker's live process group. Leaving a stray behind
                    # is recoverable; killing a live simulator is not.
                    continue
            groups.add(pgid)

    groups |= _orphaned_groups_by_name(protected, own_group)
    return groups


def _orphaned_groups_by_name(protected: set[int], own_group: int) -> set[int]:
    """Orphaned groups that command-line matching cannot see.

    The kernel empties /proc/<pid>/cmdline when a process dies, so ``pgrep -f``
    is blind to exactly the leaders we most need to reclaim: a node whose leader
    went zombie while its threads kept its CUDA context alive. Only ``comm``
    survives, truncated to 15 characters. Matching on it is broader than a
    command line, so a hit only counts when the group holds nothing live.
    """
    groups: set[int] = set()
    for pattern in LEFTOVER_PROCESS_PATTERNS:
        name = pattern[:15]
        if " " in name:
            continue  # comm never contains spaces
        for text_pid in _pgrep([name]).split():
            try:
                pid = int(text_pid)
                pgid = os.getpgid(pid)
            except (ValueError, ProcessLookupError):
                continue
            if pid in protected or pgid == own_group or pgid in groups:
                continue
            if protected & _group_members(pgid):
                continue
            if _group_is_orphaned(pgid):
                groups.add(pgid)
    return groups


def _process_domain_id(pid: int) -> int | None:
    """The ROS_DOMAIN_ID a process was launched with, or None if unreadable."""
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    for entry in raw.split(b"\0"):
        if entry.startswith(b"ROS_DOMAIN_ID="):
            try:
                return int(entry.split(b"=", 1)[1])
            except ValueError:
                return None
    return None


def _group_domain_ids(pgid: int) -> set[int]:
    """Every ROS_DOMAIN_ID seen among a group's members."""
    domains: set[int] = set()
    for pid in _group_members(pgid):
        domain = _process_domain_id(pid)
        if domain is not None:
            domains.add(domain)
    return domains


def _environ_readable(pid: int) -> bool:
    """Whether /proc/<pid>/environ can still be read.

    The kernel tears down a zombie's address space, so its environ turns into
    EACCES while every live process of ours stays readable. That difference is
    the only way to tell "another worker's simulator" from "a leader that already
    died", and the two must not share a fate.
    """
    try:
        with open(f"/proc/{pid}/environ", "rb") as handle:
            handle.read(1)
    except OSError:
        return False
    return True


def _group_is_orphaned(pgid: int) -> bool:
    """True when a group holds no live process of ours.

    A multi-threaded node whose leader died leaves the leader in Z while its
    threads keep running — and keep holding their CUDA context. Such a group can
    never be attributed to a domain (the environ is gone with the address space),
    so a domain-scoped sweep skipped it and its GPU memory outlived the run,
    starving later jobs at the "planner would not fit on the GPU" gate. Nothing
    live can be hiding here: a running simulator always has a readable environ.
    """
    members = _group_members(pgid)
    if not members:
        return False
    return not any(_environ_readable(pid) for pid in members)


def _group_members(pgid: int) -> set[int]:
    out: set[int] = set()
    for token in _pgrep(["-g", str(pgid)]).split():
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def gpu_free_mib() -> float | None:
    """Free GPU memory in MiB, or None when it cannot be read."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=8.0,
        )
    except Exception:  # noqa: BLE001
        return None
    values = [float(line.strip()) for line in out.splitlines() if line.strip()]
    return min(values) if values else None


def gpu_zombie_holders() -> list[tuple[int, float]]:
    """Dead processes that still hold GPU memory: (pid, MiB).

    A planner that exits without unwinding its CUDA context leaves the memory
    allocated to a process that no longer exists. The kernel shows it as a zombie —
    threads still parked in the driver, never fully torn down — and nvidia-smi keeps
    billing it. Measured on this machine: two of them holding 816 MiB each, a fifth
    of the card, while cleanup reported success because a zombie is not "alive".

    Nothing can kill what is already dead, so this exists to be reported rather than
    resolved: a job that cannot fit should say the card is held by corpses instead of
    waiting for room that will never appear.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True,
            timeout=8.0,
        )
    except Exception:  # noqa: BLE001
        return []
    held: list[tuple[int, float]] = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, mib = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if not Path(f"/proc/{pid}").exists() or _is_zombie(pid):
            held.append((pid, mib))
    return held


def report_gpu_zombies() -> float:
    """Warn about GPU memory held by dead processes; returns the MiB held."""
    zombies = gpu_zombie_holders()
    if not zombies:
        return 0.0
    held = sum(m for _, m in zombies)
    names = ", ".join(f"{pid} ({m:.0f}MiB)" for pid, m in zombies)
    print(
        f"[warn] {held:.0f}MiB of GPU memory is held by {len(zombies)} dead "
        f"process(es): {names}. Nothing frees it short of a driver reset; expect "
        "less headroom for this run."
    )
    return held


def wait_for_gpu_headroom(
    need_mib: float,
    timeout_sec: float,
    *,
    poll_sec: float = 5.0,
) -> tuple[bool, str]:
    """Hold until the GPU has room for another planner, or give up saying so.

    Every worker builds its own TensorRT engine. Start two at the same instant on
    one card and the second gets "Cuda Runtime (out of memory)" — the planner then
    stays up and silent, publishes no trajectory, and the run reads as a stall that
    has nothing to do with the planner. Queueing on memory is what keeps parallel
    workers honest: a job waits for room rather than failing for the lack of it.
    """
    if need_mib <= 0:
        return True, "gpu_gate_disabled"
    free = gpu_free_mib()
    if free is None:
        return True, "gpu_unreadable"
    deadline = time.time() + max(0.0, timeout_sec)
    last_log = 0.0
    while True:
        free = gpu_free_mib()
        if free is None:
            return True, "gpu_unreadable"
        if free >= need_mib:
            return True, f"gpu_free={free:.0f}MiB"
        now = time.time()
        if now >= deadline:
            zombies = gpu_zombie_holders()
            if zombies:
                held = sum(m for _, m in zombies)
                names = ", ".join(str(pid) for pid, _ in zombies)
                return False, (
                    f"gpu_busy free={free:.0f}MiB need={need_mib:.0f}MiB — "
                    f"{held:.0f}MiB is held by {len(zombies)} dead process(es) "
                    f"({names}) whose CUDA context never unwound; that memory does "
                    "not come back on its own"
                )
            return False, f"gpu_busy free={free:.0f}MiB need={need_mib:.0f}MiB"
        if now - last_log >= 20.0:
            print(
                f"[info] waiting for GPU room: {free:.0f} MiB free, "
                f"need {need_mib:.0f} MiB"
            )
            last_log = now
        time.sleep(poll_sec)


def cleanup_evaluation_processes(
    grace_sec: float = 5.0,
    kill_wait_sec: float = 10.0,
    verbose: bool = True,
    domain_id: int | None = None,
) -> bool:
    """Terminate every leftover evaluation process group and confirm they are gone.

    Returns True when nothing matching is left running. The confirmation is the
    point: the previous version sent SIGTERM, slept 2 s, sent SIGKILL and returned
    without ever checking, so a node that took longer than that to die was still
    holding its CUDA context when the next job started.
    """
    groups = find_leftover_groups(domain_id)
    if not groups:
        return True
    if verbose:
        scope = "" if domain_id is None else f" on domain {domain_id}"
        print(
            f"[info] Cleaning {len(groups)} leftover process group(s){scope}: "
            f"{sorted(groups)}"
        )

    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    deadline = time.time() + grace_sec
    while time.time() < deadline and any(group_alive(g) for g in groups):
        time.sleep(0.2)

    stubborn = [g for g in groups if group_alive(g)]
    for pgid in stubborn:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    deadline = time.time() + kill_wait_sec
    while time.time() < deadline and any(group_alive(g) for g in groups):
        time.sleep(0.2)

    # pgrep again rather than trusting the group list: a leftover may have been
    # reparented into a group we never saw.
    remaining = find_leftover_groups(domain_id)
    if remaining and verbose:
        # Name them. "process group 12345 survived" is not actionable; the package
        # that ignores shutdown is, and it is the only way to tell a tool bug from a
        # node that genuinely refuses to exit.
        print(
            f"[warn] {len(remaining)} evaluation process group(s) survived cleanup "
            f"— GPU memory may still be held:"
        )
        for line in describe_processes(leftover_process_pids(domain_id)):
            print(f"[warn]   {line}")
    return not remaining


def wait_with_timeout(proc: subprocess.Popen[Any], timeout_sec: float) -> int | None:
    """Return exit code, or None on timeout (process still running)."""
    try:
        return proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return None
