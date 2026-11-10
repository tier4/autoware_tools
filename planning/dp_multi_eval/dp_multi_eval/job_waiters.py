"""Waiting for the stack to become ready, and for a drive to end.

Each of these blocks until a condition holds or a budget runs out, and each returns
why it stopped rather than raising — the caller records that reason, so the wording
is part of the result, not a debug aid.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node

from dp_multi_eval.job_config import JobConfig, ros_call_env

# The model "means to move" when a candidate plan reaches GO_MPS within
# GO_HORIZON_S. A stopped ego fed nothing but stop stubs while such a plan sits
# under them is the StopPointFixer's veto (its gate judges only the first
# second: 0.05/0.08/0.11/1.71 m/s at 0/1/2/4 s was replaced by stubs for 30 s on
# 2026-04-28 at y=9904, in a static scene that could never change the answer).
# A hold whose plan stays flat is the model's own stop and is not nudged.
GO_HORIZON_S = 4.0
GO_MPS = 1.0


def plan_says_go(candidates, horizon_s: float = GO_HORIZON_S, go_mps: float = GO_MPS) -> bool:
    """``candidates``: per candidate a list of (time_from_start_s, velocity_mps)."""
    return any(v >= go_mps for cand in candidates for t, v in cand if t <= horizon_s)


# The ego's lag behind the recording lives in lag.py (no ROS there); re-exported
# for the callers that always found it here.
from dp_multi_eval.lag import (  # noqa: E402,F401
    RESYNC_DWELL_M,
    RESYNC_FUTURE_S,
    RESYNC_REC_MOVING_MPS,
    RESYNC_TOL_M,
    recorded_lag_sec,
)


def open_job_context(env: dict[str, str] | None = None, *, domain_id: int | None = None) -> Any:
    """A ROS context on this job's domain.

    The domain is read from the environment when a context is initialized, so it has
    to be in place first. These waiters used to call rclpy.init() and set
    ROS_DOMAIN_ID afterwards, which only worked because route setup happened to
    have initialized the default context earlier, while the job's env was applied.
    When route setup moved to its own context that accident ended: the waiters
    initialized the default context on domain 0 and heard nothing, while psim
    published on domain 10 — a job then failed as "objects=0" with a perfectly
    healthy simulator running beside it.

    Own the context, and set the domain before opening it.
    """
    old_env = os.environ.copy()
    try:
        if env:
            os.environ.update(env)
        if domain_id is not None:
            os.environ["ROS_DOMAIN_ID"] = str(int(domain_id))
        context = rclpy.Context()
        rclpy.init(context=context)
        return context
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def job_executor(context: Any, node: Any) -> Any:
    """An executor on this job's context, holding one node.

    rclpy.spin_once(node) reaches for the *global* executor, which is built on the
    default context — with an owned context that raises before a single message is
    spun. The executor has to be created on the same context as the node.
    """
    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    return executor


def close_job_context(context: Any) -> None:
    try:
        if context is not None and context.ok():
            rclpy.shutdown(context=context)
    except Exception:  # noqa: BLE001
        pass


def _list_ros_nodes(env: dict[str, str], timeout_sec: float = 8.0) -> set[str]:
    out = subprocess.check_output(
        ["ros2", "node", "list"],
        env=ros_call_env(env),
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=timeout_sec,
    )
    return set(line.strip() for line in out.splitlines() if line.strip())
def planning_stack_nodes_ready(nodes: set[str]) -> bool:
    """Match component_state_monitor planning/map/api modules being up.

    The simulator itself counts. Route setup's next move is to place the ego and wait
    for /localization/kinematic_state, and simple_planning_simulator is what publishes
    it — a gate that passes without it sends route setup to wait on a node that has
    not started. Under two workers that showed up as "localization_failed: no
    /localization/kinematic_state" after the full route-setup budget.
    """
    has_planning = any(
        "mission_planner" in node or "route_selector" in node for node in nodes
    )
    has_map = any("/map/" in node or node.endswith("/map") for node in nodes)
    has_api = any("adapi" in node for node in nodes)
    has_sim = any("simple_planning_simulator" in node for node in nodes)
    return has_planning and has_map and has_api and has_sim
def _list_ros_services(env: dict[str, str], timeout_sec: float = 8.0) -> set[str]:
    out = subprocess.check_output(
        ["ros2", "service", "list"],
        env=ros_call_env(env),
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=timeout_sec,
    )
    return set(line.strip() for line in out.splitlines() if line.strip())
def wait_for_services(
    env: dict[str, str],
    timeout_sec: float,
    psim: subprocess.Popen[Any] | None = None,
) -> tuple[bool, str]:
    """Wait until psim routing/localization services are up.

    Node-list checks are best-effort: polluted DDS / slow discovery often makes
    ``ros2 node list`` hang even while services are already usable.
    """
    adapi_required = {"/api/routing/clear_route", "/api/routing/set_route_points"}
    mission_required = {
        "/planning/mission_planning/route_selector/main/clear_route",
        "/planning/mission_planning/route_selector/main/set_waypoint_route",
    }
    localize_required = {"/localization/initialize"}
    deadline = time.time() + timeout_sec
    last_log = 0.0
    services_ok_streak = 0
    while time.time() < deadline:
        if psim is not None and psim.poll() is not None:
            return False, f"planning simulator exited early (code={psim.returncode})"

        services: set[str] | None = None
        nodes: set[str] | None = None
        try:
            services = _list_ros_services(env, timeout_sec=8.0)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            services = None
        try:
            nodes = _list_ros_nodes(env, timeout_sec=5.0)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            nodes = None

        routing_ready = False
        localize_ready = False
        stack_ready = False
        if services is not None:
            routing_ready = adapi_required.issubset(services) or mission_required.issubset(
                services
            )
            localize_ready = localize_required.issubset(services)
        if nodes is not None:
            stack_ready = planning_stack_nodes_ready(nodes)

        services_ok = localize_ready and routing_ready
        if services_ok:
            services_ok_streak += 1
        else:
            services_ok_streak = 0

        # Prefer full stack, but accept stable service readiness when node list
        # is unavailable (common after leftover domain pollution).
        if services_ok and (stack_ready or services_ok_streak >= 2):
            time.sleep(3.0)
            detail = "planning_stack_ready" if stack_ready else "services_ready"
            return True, detail

        now = time.time()
        if now - last_log >= 15.0:
            elapsed = now - (deadline - timeout_sec)
            if nodes is None and services is None:
                detail = " nodes(unavailable) services(unavailable)"
            elif nodes is None:
                detail = (
                    f" nodes(unavailable)"
                    f" services(localize={localize_ready}, routing={routing_ready},"
                    f" streak={services_ok_streak})"
                )
            else:
                detail = (
                    f" nodes(planning={any('mission_planner' in n or 'route_selector' in n for n in nodes)},"
                    f" map={any('/map/' in n for n in nodes)},"
                    f" api={any('adapi' in n for n in nodes)})"
                    f" services(localize={localize_ready}, routing={routing_ready})"
                )
            print(
                f"[info] Waiting for planning simulator stack "
                f"({elapsed:.0f}/{timeout_sec:.0f}s){detail}…"
            )
            last_log = now
        time.sleep(2.0)
    return False, "planning_stack_not_ready"
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
def wait_for_route_arrived(
    env: dict[str, str],
    timeout_sec: float,
    *,
    goal_xy_yaw: tuple[float, float, float] | None = None,
    near_goal_arrival_m: float = 5.0,
    # Whether reaching the goal means stopping at it. True for the last leg, which is
    # a destination; false for the ones before it, which are waypoints the route
    # passes through.
    require_stop_at_goal: bool = True,
    stuck_blinker_nudge: bool = False,
    stuck_blinker_speed_mps: float = 0.15,
    stall_speed_mps: float = 0.05,
    stuck_blinker_trigger_sec: float = 12.0,
    stuck_blinker_hold_sec: float = 3.0,
    stuck_blinker_cooldown_sec: float = 25.0,
    stuck_abort_sec: float = 90.0,
    # Seconds without an ego pose before the run is called dead rather than stopped.
    ego_stale_sec: float = 15.0,
    # Near-zero speed while /planning/trajectory carries only stop stubs is the
    # StopPointFixer hold, and it never resolves on its own; report it as its own
    # status after this long instead of letting the stall timer ride it out.
    stack_hold_detect_sec: float = 0.0,
    # Re-sync: return ``resync`` once the ego has been this many seconds behind
    # the recording (recorded_lag_sec against ``reference_track``) for a full
    # second. 0 disables.
    resync_lag_sec: float = 0.0,
    reference_track: Any = None,
    counters: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Poll /api/routing/state. Optionally nudge DP with blinkers when stuck.

    Returns status ``done`` | ``timeout`` | ``stuck`` | ``stack_hold`` | ``resync``.

    ``counters`` accumulates interventions this harness performed on the system
    under test, so a run that only arrived because it was nudged can be told apart
    from one that arrived on its own. Counts survive the return value, which only
    carries a human-readable detail string.
    """
    try:
        import rclpy
        from autoware_adapi_v1_msgs.msg import RouteState
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError:
        return "timeout", "rclpy_or_adapi_msgs_unavailable"

    try:
        from autoware_vehicle_msgs.msg import TurnIndicatorsReport
    except ImportError:
        TurnIndicatorsReport = None  # type: ignore[misc, assignment]

    try:
        from autoware_planning_msgs.msg import Trajectory
    except ImportError:
        Trajectory = None  # type: ignore[misc, assignment]

    try:
        from autoware_perception_msgs.msg import TrackedObjects
    except ImportError:
        TrackedObjects = None  # type: ignore[misc, assignment]

    try:
        from autoware_internal_planning_msgs.msg import CandidateTrajectories
    except ImportError:
        CandidateTrajectories = None  # type: ignore[misc, assignment]

    context = open_job_context(env)

    old_env = os.environ.copy()
    os.environ.update(env)

    class Waiter(Node):
        def __init__(self) -> None:
            super().__init__("dp_multi_eval_route_waiter", context=context)
            self.state: int | None = None
            self.speed_mps: float = float("nan")
            self.x: float = float("nan")
            self.y: float = float("nan")
            # When the last ego pose arrived. A pose that stops arriving is not a
            # vehicle holding its speed — it is a simulator that has stopped
            # publishing, and reading the cached value calls it motion.
            self.last_odom_sec: float | None = None
            qos_route = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(RouteState, "/api/routing/state", self._cb_route, qos_route)
            self.create_subscription(
                Odometry,
                "/localization/kinematic_state",
                self._cb_odom,
                10,
            )
            # Stub streak: since when the controller has been fed only stop
            # points while the ego is stopped. None whenever a real plan arrives
            # or the ego moves.
            self.stub_since: float | None = None
            self.yaw_rad: float = float("nan")
            # Distance to the nearest object in the cone ahead. The in-place
            # nudge injects speed blindly, and it pushed a creeping ego into the
            # bumper of a car stopped two metres in front of it — a contact the
            # harness manufactured, scored as ego_caused. A hold with something
            # parked on the nose is not escapable by speed.
            self.ahead_clear_m: float = float("inf")
            # Nearest object anywhere in the front half-disc (±100°, 20 m).
            # Junction cross-traffic sits outside the nudge cone but is exactly
            # what a yielding planner waits for; a hold with such traffic in
            # view is yielding, not stalling, and must not spend hold budget.
            self.front_traffic_m: float = float("inf")
            # The model's latest raw plan: does it mean to move (plan_says_go)?
            self.plan_go: bool = False
            self.plan_v_max: float = float("nan")
            if Trajectory is not None and stack_hold_detect_sec > 0.0:
                self.create_subscription(
                    Trajectory, "/planning/trajectory", self._cb_traj, 1
                )
                if CandidateTrajectories is not None:
                    self.create_subscription(
                        CandidateTrajectories,
                        "/planning/generator/diffusion_planner/candidate_trajectories",
                        self._cb_candidates,
                        1,
                    )
                if TrackedObjects is not None:
                    self.create_subscription(
                        TrackedObjects,
                        "/perception/object_recognition/tracking/objects",
                        self._cb_objects,
                        QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
                    )
            # The scene's own ego: the recorded vehicle at the instant the
            # reproducer is showing. Its stamp is recorded time; the ego's lag
            # behind it is measured against reference_track.
            self.rec_t: float | None = None
            self.rec_x: float = float("nan")
            self.rec_y: float = float("nan")
            self.rec_v: float = float("nan")
            self.lag_sec: float = float("nan")
            self.lag_since: float | None = None
            self._lag_checked = 0.0
            if resync_lag_sec > 0.0 and reference_track is not None:
                self.create_subscription(
                    Odometry, "/perception_reproducer/rosbag_ego_odom", self._cb_rec, 1
                )
            self._blinker_pub = None
            if stuck_blinker_nudge and TurnIndicatorsReport is not None:
                self._blinker_pub = self.create_publisher(
                    TurnIndicatorsReport,
                    "/vehicle/status/turn_indicators_status",
                    10,
                )

        def _cb_route(self, msg: RouteState) -> None:
            self.state = msg.state

        def _cb_traj(self, msg) -> None:
            stopped = self.speed_mps == self.speed_mps and self.speed_mps <= max(
                stall_speed_mps, 0.25
            )
            if len(msg.points) <= 5 and stopped:
                if self.stub_since is None:
                    self.stub_since = time.time()
            else:
                self.stub_since = None

        def _cb_candidates(self, msg) -> None:
            cands = [
                [
                    (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9,
                     float(pt.longitudinal_velocity_mps))
                    for pt in cand.points
                ]
                for cand in msg.candidate_trajectories
            ]
            self.plan_go = plan_says_go(cands)
            within = [v for cand in cands for t, v in cand if t <= GO_HORIZON_S]
            self.plan_v_max = max(within) if within else float("nan")

        def _cb_objects(self, msg) -> None:
            import math as _m

            if not (self.x == self.x and self.yaw_rad == self.yaw_rad):
                return
            best = float("inf")
            front = float("inf")
            for obj in msg.objects:
                pos = obj.kinematics.pose_with_covariance.pose.position
                dx, dy = pos.x - self.x, pos.y - self.y
                dist = _m.hypot(dx, dy)
                if dist > 20.0:
                    continue
                bearing = abs(
                    (_m.atan2(dy, dx) - self.yaw_rad + _m.pi) % (2.0 * _m.pi) - _m.pi
                )
                tw = obj.kinematics.twist_with_covariance.twist.linear
                moving = _m.hypot(tw.x, tw.y) > 0.5
                if moving and bearing < _m.pi * 100.0 / 180.0:
                    # Oncoming traffic in the adjacent lane is not something a
                    # departing ego yields to, and on a two-way road it never
                    # runs out — it froze a hold for minutes. Anti-parallel
                    # heading with a lateral offset means the opposite lane.
                    q = obj.kinematics.pose_with_covariance.pose.orientation
                    oyaw = _m.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                    )
                    hdiff = abs(
                        (oyaw - self.yaw_rad + _m.pi) % (2.0 * _m.pi) - _m.pi
                    )
                    ly = -_m.sin(self.yaw_rad) * dx + _m.cos(self.yaw_rad) * dy
                    oncoming = hdiff > _m.pi * 140.0 / 180.0 and abs(ly) > 1.5
                    if not oncoming:
                        front = min(front, dist)
                if dist <= 12.0 and bearing < _m.pi / 5.0:
                    best = min(best, dist)
            self.ahead_clear_m = best
            self.front_traffic_m = front

        def _cb_odom(self, msg: Odometry) -> None:
            import math as _m

            q = msg.pose.pose.orientation
            self.yaw_rad = _m.atan2(
                2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            vx = float(msg.twist.twist.linear.x)
            vy = float(msg.twist.twist.linear.y)
            self.speed_mps = (vx * vx + vy * vy) ** 0.5
            self.x = float(msg.pose.pose.position.x)
            self.y = float(msg.pose.pose.position.y)
            self.last_odom_sec = time.time()
            if self.rec_t is not None and self.last_odom_sec - self._lag_checked >= 0.25:
                self._lag_checked = self.last_odom_sec
                lag = recorded_lag_sec(reference_track, self.x, self.y, self.yaw_rad, self.rec_t)
                self.lag_sec = float("nan") if lag is None else lag
                # A standing recorded vehicle is not running away from the
                # ego, and a re-sync onto it would only hand back a stopped
                # ego that has to depart on its own all over again.
                if (
                    lag is not None
                    and lag >= resync_lag_sec
                    and self.rec_v >= RESYNC_REC_MOVING_MPS
                ):
                    if self.lag_since is None:
                        self.lag_since = self.last_odom_sec
                else:
                    self.lag_since = None

        def _cb_rec(self, msg: Odometry) -> None:
            self.rec_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.rec_x = float(msg.pose.pose.position.x)
            self.rec_y = float(msg.pose.pose.position.y)
            tw = msg.twist.twist.linear
            self.rec_v = math.hypot(float(tw.x), float(tw.y))

        def publish_blinker(self, report: int) -> None:
            if self._blinker_pub is None or TurnIndicatorsReport is None:
                return
            msg = TurnIndicatorsReport()
            msg.stamp = self.get_clock().now().to_msg()
            msg.report = report
            self._blinker_pub.publish(msg)

    node = Waiter()
    executor = job_executor(context, node)
    deadline = time.time() + timeout_sec
    last_log = 0.0
    stuck_since: float | None = None
    last_nudge_end = 0.0
    nudge_count = 0
    abort_after = max(0.0, float(stuck_abort_sec or 0.0))
    # Cycle: LEFT → DISABLE → RIGHT → DISABLE
    nudge_phases: list[tuple[str, int]] = []
    if TurnIndicatorsReport is not None:
        nudge_phases = [
            ("LEFT", int(TurnIndicatorsReport.ENABLE_LEFT)),
            ("DISABLE", int(TurnIndicatorsReport.DISABLE)),
            ("RIGHT", int(TurnIndicatorsReport.ENABLE_RIGHT)),
            ("DISABLE", int(TurnIndicatorsReport.DISABLE)),
        ]

    detail_state: int | None = None
    # A run still covering ground when the clock runs out is driving, not
    # failing: 2026-04-28 was killed at 4.8 m/s, 61 m from its goal, after
    # two long (legitimate) holds ate the budget. While the ego has moved
    # more than 20 m in the last 60 s, the deadline stretches — by at most
    # half the original budget in total, so a circling or crawling run
    # still ends.
    progress_extension_left = max(60.0, 0.5 * float(timeout_sec))
    progress_marks: list[tuple[float, float, float]] = []
    try:
        while time.time() < deadline and context.ok():
            executor.spin_once(timeout_sec=0.25)
            now = time.time()

            if node.x == node.x:
                if not progress_marks or now - progress_marks[-1][0] >= 5.0:
                    progress_marks.append((now, node.x, node.y))
                    progress_marks[:] = [
                        m for m in progress_marks if now - m[0] <= 65.0
                    ]
            if deadline - now < 10.0 and progress_extension_left > 0.0:
                old = next((m for m in progress_marks if now - m[0] >= 55.0),
                           progress_marks[0] if progress_marks else None)
                if old is not None and node.x == node.x:
                    moved = math.hypot(node.x - old[1], node.y - old[2])
                    # Terminal creep counts too: 25 m from its goal at
                    # 0.15 m/s the ego was still arriving, just slowly —
                    # cutting it there converts slowness into DNF while
                    # reference_speed already reports the slowness honestly.
                    near_goal_creep = (
                        goal_xy_yaw is not None
                        and math.hypot(node.x - goal_xy_yaw[0], node.y - goal_xy_yaw[1]) < 60.0
                        and moved > 2.0
                    )
                    if moved > 20.0 or near_goal_creep:
                        ext = min(30.0, progress_extension_left)
                        deadline += ext
                        progress_extension_left -= ext
                        print(
                            f"[info] still driving ({moved:.0f} m in the last "
                            f"minute) — extending arrival wait by {ext:.0f}s "
                            f"({progress_extension_left:.0f}s of extension left)"
                        )

            if node.state == RouteState.ARRIVED:
                detail = "route_arrived"
                if nudge_count:
                    detail += f" (blinker_nudges={nudge_count})"
                return "done", detail

            # Nothing arriving means nothing is driving.
            #
            # The stack stopped publishing mid-run once and this loop carried on for
            # four more minutes reporting "state=SET v=1.01 m/s" — the same figure to
            # two decimals every time, because it was the last value ever received.
            # A constant speed reads as motion, so the stall detector never fired, no
            # recovery ran, and the leg burned its whole budget before failing with a
            # reason that described none of it.
            if node.last_odom_sec is not None and ego_stale_sec > 0.0:
                silent_for = now - node.last_odom_sec
                if silent_for >= ego_stale_sec:
                    return "stale", (
                        f"ego pose stopped updating {silent_for:.0f}s ago "
                        f"(last v={node.speed_mps:.2f} m/s); the simulator is no "
                        "longer publishing"
                    )

            # The ego has fallen behind the recording: the scene the model is
            # looking at belongs to a vehicle that is no longer here. End the
            # segment; the caller sets the ego back onto the recording.
            if (
                resync_lag_sec > 0.0
                and node.lag_since is not None
                and now - node.lag_since >= 1.0
            ):
                if counters is not None and node.x == node.x:
                    counters["stall_x"] = float(node.x)
                    counters["stall_y"] = float(node.y)
                return "resync", (
                    f"resync: ego {node.lag_sec:.1f}s behind the recording at "
                    f"({node.x:.1f},{node.y:.1f}) v={node.speed_mps:.2f} m/s; recorded "
                    f"ego at ({node.rec_x:.1f},{node.rec_y:.1f}) v={node.rec_v:.2f} m/s"
                )

            # Only the fixer's veto is escaped: stubs under a plan that means to
            # move. Stubs under a flat plan are the model's stop and fall through
            # to the stall timer like any other standstill.
            if stack_hold_detect_sec > 0.0 and node.stub_since is not None and node.plan_go:
                held = now - node.stub_since
                if held >= stack_hold_detect_sec:
                    if counters is not None:
                        counters["stack_holds"] = int(counters.get("stack_holds", 0)) + 1
                        counters["stall_x"] = node.x
                        counters["stall_y"] = node.y
                    blocked = node.ahead_clear_m < 8.0
                    traffic = node.front_traffic_m < 20.0
                    return "stack_hold", (
                        f"stack_hold: only stop stubs reached the controller for "
                        f"{held:.0f}s at v={node.speed_mps:.2f} m/s while the model's "
                        f"plan reaches {node.plan_v_max:.2f} m/s within {GO_HORIZON_S:.0f}s (fixer veto)"
                        + (f"; ahead_blocked={node.ahead_clear_m:.1f}m" if blocked else "")
                        + (
                            f"; front_traffic={node.front_traffic_m:.1f}m"
                            if traffic and not blocked
                            else ""
                        )
                    )

            speed = node.speed_mps
            # An intermediate leg is a waypoint: the route runs through it and the
            # ego is meant to keep going. Requiring it to stop there meant a leg it
            # had driven straight over never registered — measured at 0.3 m from the
            # goal doing 4.4 m/s, and again at 0.5 m doing 7.7 m/s, on runs that
            # reached every goal on the route and were failed anyway. Only the last
            # leg is a destination, where stopping is what arriving means.
            if (
                goal_xy_yaw is not None
                and speed == speed
                and node.x == node.x
                and (speed <= stall_speed_mps or not require_stop_at_goal)
            ):
                goal_dist = math.hypot(
                    node.x - float(goal_xy_yaw[0]),
                    node.y - float(goal_xy_yaw[1]),
                )
                if goal_dist <= max(0.0, float(near_goal_arrival_m)):
                    how = "stopped at" if speed <= stall_speed_mps else "passed"
                    return (
                        "done",
                        f"near_goal_arrival {how} dist={goal_dist:.2f}m "
                        f"v={speed:.2f} state={_route_state_label(node.state)}",
                    )

            if speed == speed:  # not NaN
                if speed <= stall_speed_mps:
                    if stuck_since is None:
                        stuck_since = now
                else:
                    stuck_since = None

            if (
                abort_after > 0.0
                and stuck_since is not None
                and (now - stuck_since) >= abort_after
            ):
                # Where it stalled, so the caller can weigh it against what the
                # recording did at that place rather than one budget for the route.
                if counters is not None and node.x == node.x:
                    counters["stall_x"] = float(node.x)
                    counters["stall_y"] = float(node.y)
                detail = (
                    f"stuck_abort v={speed:.2f} m/s for {now - stuck_since:.0f}s "
                    f"state={_route_state_label(node.state)}"
                )
                if nudge_count:
                    detail += f" blinker_nudges={nudge_count}"
                print(f"[warn] {detail} — ending leg early")
                return "stuck", detail

            if stuck_blinker_nudge and nudge_phases and speed == speed:
                ready_to_nudge = (
                    stuck_since is not None
                    and (now - stuck_since) >= stuck_blinker_trigger_sec
                    and (now - last_nudge_end) >= stuck_blinker_cooldown_sec
                )
                if ready_to_nudge:
                    nudge_count += 1
                    if counters is not None:
                        counters["blinker_nudges"] = (
                            counters.get("blinker_nudges", 0) + 1
                        )
                    print(
                        f"[info] Stuck (v={speed:.2f} m/s for "
                        f"{now - stuck_since:.0f}s) — blinker nudge #{nudge_count} "
                        "(LEFT then RIGHT) to try DP trajectory generation"
                    )
                    for label, report in nudge_phases:
                        phase_deadline = time.time() + stuck_blinker_hold_sec
                        print(f"[info]   blinker → {label} ({stuck_blinker_hold_sec:.1f}s)")
                        while time.time() < phase_deadline and context.ok():
                            node.publish_blinker(report)
                            executor.spin_once(timeout_sec=0.1)
                            if node.state == RouteState.ARRIVED:
                                return "done", f"route_arrived (during blinker={label})"
                        node.publish_blinker(report)
                    last_nudge_end = time.time()
                    # Keep stuck_since so abort_after still accumulates across nudges.

            if now - last_log >= 15.0:
                speed_s = f"{speed:.2f}" if speed == speed else "?"
                stuck_s = (
                    f" stuck={now - stuck_since:.0f}s"
                    if stuck_since is not None
                    else ""
                )
                print(
                    f"[info] waiting for route ARRIVED "
                    f"({timeout_sec - (deadline - now):.0f}/{timeout_sec:.0f}s) "
                    f"state={_route_state_label(node.state)} v={speed_s} m/s"
                    + (f" nudges={nudge_count}" if nudge_count else "")
                    + stuck_s
                )
                last_log = now
            detail_state = node.state
    finally:
        executor.shutdown()
        node.destroy_node()
        close_job_context(context)
        os.environ.clear()
        os.environ.update(old_env)

    detail = f"route_not_arrived state={_route_state_label(detail_state)}"
    if node.x == node.x:
        detail += f" ego=({node.x:.1f},{node.y:.1f})"
    if nudge_count:
        detail += f" blinker_nudges={nudge_count}"
    return "timeout", detail
def reproducer_death_reason(proc: Any, log_path: Path) -> str | None:
    """A one-line cause when the reproducer process has already exited.

    A crashed reproducer looks exactly like a slow one from the topic's side —
    both publish nothing — so waiting on objects alone burned the whole budget
    before reporting "objects=0 msgs=0", which names the symptom and hides the
    cause. The process table settles it in one call.
    """
    if proc is None or proc.poll() is None:
        return None
    tail = ""
    try:
        lines = [ln.strip() for ln in log_path.read_text(errors="replace").splitlines() if ln.strip()]
        # The last line is usually "[ros2run]: Process exited with failure 1";
        # the exception above it is the part worth quoting.
        for line in reversed(lines[-12:]):
            if "ros2run" not in line:
                tail = line
                break
    except OSError:
        pass
    detail = f": {tail}" if tail else ""
    return f"perception_reproducer exited with code {proc.returncode}{detail}"


def wait_for_perception_ready(
    cfg: JobConfig,
    *,
    timeout_sec: float | None = None,
    reproducer: Any = None,
) -> tuple[bool, str]:
    """Wait until reproducer publishes a stable set of tracked objects."""
    try:
        import rclpy
        from autoware_perception_msgs.msg import TrackedObjects
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
    except ImportError:
        return False, "perception_msgs_unavailable"

    context = open_job_context(domain_id=cfg.domain_id)

    wait_budget = float(
        timeout_sec if timeout_sec is not None else cfg.perception_ready_timeout_sec
    )

    class Monitor(Node):
        def __init__(self) -> None:
            super().__init__("dp_multi_eval_perception_ready", context=context)
            # Match perception_reproducer's RELIABLE publisher (BEST_EFFORT can miss).
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
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
    executor = job_executor(context, monitor)
    t0 = time.time()
    deadline = t0 + wait_budget

    # Extend the wait while the reproducer is visibly still loading.
    #
    # The budget is computed from the recording's *duration*, but loading time is set
    # by its size on disk and by how many jobs are reading at once. A 79 GB bag that
    # loads in 20 s alone did not finish inside a 360 s budget with a second worker
    # reading beside it, and the job failed as "objects=0" with a healthy reproducer
    # halfway through opening its databases. A clock cannot tell those apart; the
    # log can — it grows while frames are being read and stops when they are not.
    repro_log = cfg.output_dir / "reproducer.log"
    hard_deadline = t0 + max(wait_budget, float(cfg.bag_load_timeout_cap_sec))
    last_log_size = -1
    last_growth = t0
    grace_after_growth = 60.0
    warmup_until = t0 + cfg.perception_warmup_sec
    stable_since: float | None = None
    last_count = -1
    last_log = 0.0
    try:
        while time.time() < deadline and context.ok():
            executor.spin_once(timeout_sec=0.25)
            now = time.time()

            died = reproducer_death_reason(reproducer, repro_log)
            if died is not None:
                return False, died

            # Still reading? Then the budget has not really expired.
            try:
                size = repro_log.stat().st_size if repro_log.is_file() else -1
            except OSError:
                size = -1
            if size > last_log_size:
                last_log_size = size
                last_growth = now
            if (
                now + 5.0 >= deadline
                and now - last_growth < grace_after_growth
                and deadline < hard_deadline
            ):
                deadline = min(hard_deadline, now + grace_after_growth)
                print(
                    f"[info] perception_reproducer still loading "
                    f"({now - t0:.0f}s elapsed); extending the wait"
                )
            if now < warmup_until:
                continue

            # Ready means objects keep arriving, not that their number stops changing.
            #
            # This used to wait for an identical count across perception_ready_stable_sec,
            # which a live scene rarely gives — counts ran 24, 28, 31, 33, 39, 32 as
            # vehicles came and went, and the wait ran to its full budget beside a
            # perfectly healthy reproducer. Worse, the only thing that reliably satisfied
            # it was a scene that had stopped moving, which is the pathology this harness
            # exists to catch.
            count = monitor.object_count
            last_count = count
            if count < cfg.perception_ready_min_objects:
                stable_since = None
            else:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= cfg.perception_ready_stable_sec:
                    return (
                        True,
                        f"perception_ready objects={count} after_{now - t0:.0f}s",
                    )

            if now - last_log >= 15.0:
                elapsed = now - t0
                print(
                    f"[info] Waiting for perception_reproducer… "
                    f"{elapsed:.0f}/{wait_budget:.0f}s "
                    f"objects={count} msgs={monitor.message_count} "
                    f"(full bags can take many minutes to load)"
                )
                last_log = now
                # Keep clearing empty publishers from psim.
                kill_conflicting_perception_nodes()
    finally:
        executor.shutdown()
        monitor.destroy_node()
        close_job_context(context)

    return (
        False,
        f"perception_ready_timeout after_{wait_budget:.0f}s "
        f"objects={monitor.object_count} msgs={monitor.message_count}",
    )
def wait_for_first_trajectory(
    cfg: JobConfig, timeout_sec: float
) -> tuple[bool, str]:
    """Block until the planner has produced one trajectory.

    A model's first job builds its TensorRT engine, and the build used to run
    inside the drive: the planner published nothing for minutes, every metric
    counted it as a stalled vehicle, and a 127.6 s "departure resistance" was
    diagnosed before anyone noticed it was compilation. Engage waits for proof
    of inference now, so a run's clock starts when the model can actually
    drive.
    """
    try:
        import rclpy
        from autoware_planning_msgs.msg import Trajectory
        from rclpy.node import Node
    except ImportError:
        return True, "planning_msgs_unavailable (gate skipped)"

    context = open_job_context(domain_id=cfg.domain_id)

    class Probe(Node):
        def __init__(self) -> None:
            super().__init__("dp_multi_eval_traj_probe", context=context)
            self.got = False
            self.create_subscription(
                Trajectory, "/planning/trajectory", self._cb, 1
            )

        def _cb(self, _msg) -> None:
            self.got = True

    node = Probe()
    executor = job_executor(context, node)
    deadline = time.time() + max(1.0, float(timeout_sec))
    last_note = time.time()
    try:
        while time.time() < deadline and context.ok():
            executor.spin_once(timeout_sec=0.25)
            if node.got:
                return True, "planner_warm (first trajectory received)"
            now = time.time()
            if now - last_note >= 30.0:
                last_note = now
                print(
                    f"[info] waiting for the planner's first trajectory "
                    f"({now - (deadline - timeout_sec):.0f}/{timeout_sec:.0f}s) "
                    "— a fresh model builds its TensorRT engine here"
                )
        return False, (
            f"planner_never_warmed after {timeout_sec:.0f}s — no trajectory was "
            "ever published; engine build too slow, or inference is dead "
            "(check the planner's own ERROR lines in psim_launch.log)"
        )
    finally:
        executor.remove_node(node)
        node.destroy_node()
        close_job_context(context)


def kill_conflicting_perception_nodes() -> None:
    """Stop psim perception publishers that fight with perception_reproducer.

    ``map_based_prediction`` is deliberately kept alive: the reproducer runs with
    ``-t`` and only publishes ``/perception/object_recognition/tracking/objects``,
    so map_based_prediction is the only remaining publisher of
    ``/perception/object_recognition/objects``. Killing it leaves that topic dead,
    ``topic_state_monitor`` latches an ERROR in the diagnostic graph, and ADAPI then
    rejects ``change_to_autonomous`` ("The target mode is not available") — ego never
    engages and every job looks "stuck". It does not conflict with the reproducer:
    it consumes tracking/objects and produces objects.
    """
    patterns = (
        "multi_object_tracker",
        "dummy_perception_publisher",
        "autoware_multi_object_tracker",
    )
    for pattern in patterns:
        # Bounded for the same reason as _pgrep in process_utils: pkill -f reads
        # every process's cmdline, and a process wedged in exit_mm blocks that
        # read in D state. This exact call sat for 7 minutes between "Starting
        # perception_reproducer" and the next line while two half-dead planners
        # from a killed run finished tearing down their CUDA address space.
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10.0,
            )
        except subprocess.TimeoutExpired:
            print(f"[warn] pkill -f {pattern} did not finish in 10s; moving on")
