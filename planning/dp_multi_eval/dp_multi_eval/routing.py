"""Localization, routing and engage for one evaluation job — one node, one verdict.

The job keeps a single ROS node on its own context for its whole life and asks
Autoware's AD API directly. The previous seam (diffusion_planner_batch_eval's
route_setup) opened a fresh context per call, refreshed route state through
``ros2 topic echo`` / ``ros2 service list`` subprocesses and spun 3 s before
every clear, so one planner verdict cost ~9.5 s; a refused goal then walked a
ladder of 2 backends x 5 pull-backs x a stop-order walk that re-tried the same
goal — 22 calls and 180 s (the budget cap) at 教会前 on 2026-04-28, of which
two were informative. The planner itself answers in under a second.

Routing policy, one planner call per candidate, the verdict final:
  1. the goal as given (snapped onto a map stop within SNAP_STOP_M);
  2. the goal projected onto the centreline of the stop's lanelet (the csv's
     lane_id). Every refusal measured on map v206 was the mission planner's
     "Goal's footprint exceeds lane!": the stop-point poses sit a metre off
     the lane centre and the bus footprint crosses the bay boundary. On the
     vehicle the operator's goal is modified by Autoware's goal planner; the
     e2e pipeline has none, so the harness does the equivalent here;
  3. the base pulled back PULL_BACK_M along its heading;
  4. only when the caller allows it: the next map stops in route order with
     the refused stop as a waypoint. The caller learns which stops were
     skipped and scores the run accordingly.
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ADAPI_SET_ROUTE = "/api/routing/set_route_points"
ADAPI_CLEAR_ROUTE = "/api/routing/clear_route"
ADAPI_ROUTE_STATE = "/api/routing/state"
ADAPI_OPERATION_MODE = "/api/operation_mode/state"
ADAPI_ENABLE_CONTROL = "/api/operation_mode/enable_autoware_control"
ADAPI_TO_AUTONOMOUS = "/api/operation_mode/change_to_autonomous"
ADAPI_TO_STOP = "/api/operation_mode/change_to_stop"
LOCALIZATION_INITIALIZE = "/localization/initialize"
ADAPI_LOCALIZATION_INITIALIZE = "/api/localization/initialize"
KINEMATIC_STATE = "/localization/kinematic_state"

ROUTE_UNSET, ROUTE_SET, ROUTE_ARRIVED = 1, 2, 3
MODE_STOP, MODE_AUTONOMOUS = 1, 2

SNAP_STOP_M = 2.0  # a goal this close to a map stop is that stop
SPAWN_SNAP_M = 5.0  # a bag start this close to a map stop spawns on it
PULL_BACK_M = (1.0, 3.0)
WALK_MAX = 3  # stops tried beyond a refused one
INIT_TOLERANCE_M = 2.0

XYYaw = tuple[float, float, float]


@dataclass(frozen=True)
class MapStop:
    name: str
    x: float
    y: float
    yaw: float
    lane_id: str = ""

    @property
    def pose(self) -> XYYaw:
        return (self.x, self.y, self.yaw)


@dataclass(frozen=True)
class Candidate:
    goal: XYYaw
    waypoints: tuple[XYYaw, ...]
    label: str  # exact | pull_back=1m | walk
    name: str | None  # map stop the goal is, if any
    skipped: tuple[str, ...] = ()


@dataclass
class RouteResult:
    accepted: bool
    message: str
    goal: XYYaw | None = None
    name: str | None = None
    skipped: tuple[str, ...] = ()
    waypoints: int = 0
    label: str = ""
    calls: int = 0

    def describe(self) -> str:
        """The contract the drive loop reads: the accepted pose first, then the
        stop name and skipped stops when the walk changed the goal."""
        if not self.accepted or self.goal is None:
            return self.message
        text = f"effective_goal_xy={self.goal[0]:.4f},{self.goal[1]:.4f},{self.goal[2]:.6f}; "
        if self.skipped and self.name:
            text += f"effective_goal={self.name}; skipped_goals={'|'.join(self.skipped)}; "
        return text + (
            f"route_set_via_adapi ({self.label}, waypoints={self.waypoints}, "
            f"planner_calls={self.calls})"
        )


# -- pure helpers ---------------------------------------------------------------
def load_map_stops(csv_path: Path) -> list[MapStop]:
    csv_path = Path(csv_path).expanduser()
    if not csv_path.is_file():
        return []
    out: list[MapStop] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("name") or "").strip()
            if not name:
                continue
            out.append(
                MapStop(name, float(row["x"]), float(row["y"]), float(row["yaw"]), (row.get("lane_id") or "").strip())
            )
    return out


_CENTERLINES: dict[str, dict[str, list[tuple[float, float]]]] = {}


def lane_centerlines(map_path: Path | None) -> dict[str, list[tuple[float, float]]]:
    """Lanelet id -> centreline points, parsed once per map from lanelet2_map.osm."""
    if map_path is None:
        return {}
    osm = Path(map_path).expanduser().resolve() / "lanelet2_map.osm"
    key = str(osm)
    if key in _CENTERLINES:
        return _CENTERLINES[key]
    out: dict[str, list[tuple[float, float]]] = {}
    if osm.is_file():
        import xml.etree.ElementTree as ET

        root = ET.parse(osm).getroot()
        nodes: dict[str, tuple[float, float]] = {}
        for n in root.iter("node"):
            t = {tag.get("k"): tag.get("v") for tag in n.findall("tag")}
            if "local_x" in t and "local_y" in t:
                nodes[n.get("id")] = (float(t["local_x"]), float(t["local_y"]))
        ways = {w.get("id"): [nd.get("ref") for nd in w.findall("nd")] for w in root.iter("way")}
        for rel in root.iter("relation"):
            if not any(t.get("k") == "type" and t.get("v") == "lanelet" for t in rel.findall("tag")):
                continue
            m = {x.get("role"): x.get("ref") for x in rel.findall("member") if x.get("type") == "way"}
            if "left" not in m or "right" not in m:
                continue
            left = [nodes[n] for n in ways.get(m["left"], []) if n in nodes]
            right = [nodes[n] for n in ways.get(m["right"], []) if n in nodes]
            if len(left) < 2 or len(right) < 2:
                continue
            # bounds may be stored in opposite directions; orient like lanelet2's loader
            if math.hypot(left[0][0] - right[0][0], left[0][1] - right[0][1]) > math.hypot(
                left[0][0] - right[-1][0], left[0][1] - right[-1][1]
            ):
                right = right[::-1]
            n = max(len(left), len(right), 8)
            L, R = _resample(left, n), _resample(right, n)
            out[rel.get("id")] = [((L[i][0] + R[i][0]) / 2, (L[i][1] + R[i][1]) / 2) for i in range(n)]
    _CENTERLINES[key] = out
    return out


def _resample(line: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """``n`` points evenly spaced along ``line`` by arc length."""
    seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(line, line[1:])]
    total = sum(seg)
    if total <= 0.0:
        return [line[0]] * n
    out: list[tuple[float, float]] = []
    i, acc = 0, 0.0
    for k in range(n):
        target = total * k / (n - 1)
        while i < len(seg) - 1 and acc + seg[i] < target:
            acc += seg[i]
            i += 1
        t = (target - acc) / seg[i] if seg[i] > 0 else 0.0
        a, b = line[i], line[i + 1]
        out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def project_on_centerline(goal: XYYaw, line: list[tuple[float, float]]) -> XYYaw | None:
    """The nearest centreline point to ``goal`` with the line's heading there."""
    best: tuple[float, float, float, float] | None = None
    for (ax, ay), (bx, by) in zip(line, line[1:]):
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 <= 0.0:
            continue
        t = max(0.0, min(1.0, ((goal[0] - ax) * dx + (goal[1] - ay) * dy) / seg2))
        px, py = ax + t * dx, ay + t * dy
        d = math.hypot(goal[0] - px, goal[1] - py)
        if best is None or d < best[0]:
            best = (d, px, py, math.atan2(dy, dx))
    if best is None:
        return None
    return (best[1], best[2], best[3])


def nearest_stop(stops: list[MapStop], x: float, y: float) -> tuple[MapStop | None, float]:
    best, best_d = None, float("inf")
    for s in stops:
        d = math.hypot(s.x - x, s.y - y)
        if d < best_d:
            best, best_d = s, d
    return best, best_d


def pull_back(goal: XYYaw, metres: float) -> XYYaw:
    x, y, yaw = goal
    return (x - metres * math.cos(yaw), y - metres * math.sin(yaw), yaw)


def route_candidates(
    goal: XYYaw,
    stops: list[MapStop],
    order: list[str],
    allow_walk: bool,
    centerlines: dict[str, list[tuple[float, float]]] | None = None,
) -> list[Candidate]:
    """What to ask the planner, in order. See the module docstring."""
    stop, d = nearest_stop(stops, goal[0], goal[1])
    name = None
    if stop is not None and d <= SNAP_STOP_M:
        goal, name = stop.pose, stop.name
    out = [Candidate(goal, (), "exact", name)]
    base = goal
    line = (centerlines or {}).get(stop.lane_id) if stop is not None and name is not None else None
    centred = project_on_centerline(goal, line) if line else None
    if centred is not None and math.hypot(centred[0] - goal[0], centred[1] - goal[1]) > 0.05:
        out.append(Candidate(centred, (), "lane_centre", name))
        base = centred
    for m in PULL_BACK_M:
        out.append(Candidate(pull_back(base, m), (), f"pull_back={m:g}m", name))
    if allow_walk and name is not None and name in order:
        by_name = {s.name: s for s in stops}
        idx = order.index(name)
        for j in range(idx + 1, min(idx + 1 + WALK_MAX, len(order))):
            nxt = by_name.get(order[j])
            if nxt is None:
                continue
            via = tuple(by_name[n].pose for n in order[idx:j] if n in by_name)
            out.append(Candidate(nxt.pose, via, "walk", nxt.name, tuple(order[idx:j])))
    return out


def bag_start_pose(bag_path: Path) -> XYYaw | None:
    """The recording's first ego pose, from the first kinematic_state message."""
    from rclpy.serialization import deserialize_message
    from rosbag2_py import StorageFilter
    from rosidl_runtime_py.utilities import get_message

    from dp_multi_eval.bag_reader import open_bag_reader

    reader = open_bag_reader(Path(bag_path).expanduser().resolve())
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if KINEMATIC_STATE not in type_map:
        return None
    reader.set_filter(StorageFilter(topics=[KINEMATIC_STATE]))
    if not reader.has_next():
        return None
    _topic, data, _stamp = reader.read_next()
    pose = deserialize_message(data, get_message(type_map[KINEMATIC_STATE])).pose.pose
    return (float(pose.position.x), float(pose.position.y), _yaw_of(pose.orientation))


def _yaw_of(q: Any) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# -- the client -----------------------------------------------------------------
class RoutingClient:
    """One node on the job's domain; every wait spins this node's executor."""

    def __init__(self, env: dict[str, str]):
        from autoware_adapi_v1_msgs.msg import OperationModeState, RouteState
        from autoware_adapi_v1_msgs.srv import ChangeOperationMode, ClearRoute, SetRoutePoints
        from autoware_adapi_v1_msgs.srv import InitializeLocalization as AdapiInitialize
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from tier4_localization_msgs.srv import InitializeLocalization

        from dp_multi_eval.job_waiters import job_executor, open_job_context

        self._msgs = {
            "SetRoutePoints": SetRoutePoints,
            "ClearRoute": ClearRoute,
            "ChangeOperationMode": ChangeOperationMode,
            "InitializeLocalization": InitializeLocalization,
            "AdapiInitialize": AdapiInitialize,
            "PoseWithCovarianceStamped": PoseWithCovarianceStamped,
        }
        self.domain = str(env.get("ROS_DOMAIN_ID", "0"))
        self.context = open_job_context(env)
        self.node = Node("dp_multi_eval_routing", context=self.context)
        self.executor = job_executor(self.context, self.node)
        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.route_state: int | None = None
        self.mode: int | None = None
        self.control_enabled = False
        self.auto_available = False
        self.ego: XYYaw | None = None
        self.node.create_subscription(RouteState, ADAPI_ROUTE_STATE, self._on_route, latched)
        self.node.create_subscription(
            OperationModeState, ADAPI_OPERATION_MODE, self._on_mode, latched
        )
        self.node.create_subscription(Odometry, KINEMATIC_STATE, self._on_odom, 10)
        self.set_route = self.node.create_client(SetRoutePoints, ADAPI_SET_ROUTE)
        self.clear_route = self.node.create_client(ClearRoute, ADAPI_CLEAR_ROUTE)
        self.init_loc = self.node.create_client(InitializeLocalization, LOCALIZATION_INITIALIZE)
        self.init_loc_adapi = self.node.create_client(
            AdapiInitialize, ADAPI_LOCALIZATION_INITIALIZE
        )
        self.enable_control = self.node.create_client(ChangeOperationMode, ADAPI_ENABLE_CONTROL)
        self.to_autonomous = self.node.create_client(ChangeOperationMode, ADAPI_TO_AUTONOMOUS)
        self.to_stop = self.node.create_client(ChangeOperationMode, ADAPI_TO_STOP)

    # -- callbacks
    def _on_route(self, msg: Any) -> None:
        self.route_state = int(msg.state)

    def _on_mode(self, msg: Any) -> None:
        self.mode = int(msg.mode)
        self.control_enabled = bool(msg.is_autoware_control_enabled)
        self.auto_available = bool(msg.is_autonomous_mode_available)

    def _on_odom(self, msg: Any) -> None:
        p = msg.pose.pose
        self.ego = (float(p.position.x), float(p.position.y), _yaw_of(p.orientation))

    # -- plumbing
    def spin(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline and self.context.ok():
            self.executor.spin_once(timeout_sec=0.1)

    def wait_until(self, pred: Callable[[], bool], timeout: float) -> bool:
        deadline = time.time() + timeout
        while self.context.ok():
            self.executor.spin_once(timeout_sec=0.1)
            if pred():
                return True
            if time.time() >= deadline:
                return False
        return False

    def call(self, client: Any, request: Any, timeout: float = 30.0) -> Any:
        """The response, or None when the service is absent or silent."""
        if not client.wait_for_service(timeout_sec=min(timeout, 10.0)):
            return None
        future = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline and self.context.ok():
            self.executor.spin_once(timeout_sec=0.1)
        return future.result() if future.done() else None

    @staticmethod
    def _log(text: str) -> None:
        print(f"[route] {text}", flush=True)

    def _pose(self, xyyaw: XYYaw) -> Any:
        from geometry_msgs.msg import Pose

        pose = Pose()
        pose.position.x, pose.position.y = float(xyyaw[0]), float(xyyaw[1])
        pose.orientation.z = math.sin(xyyaw[2] / 2.0)
        pose.orientation.w = math.cos(xyyaw[2] / 2.0)
        return pose

    # -- localization
    def initialize_ego(self, spawn: XYYaw, timeout: float) -> tuple[bool, str]:
        """Put the simulator's ego at ``spawn`` and wait until it reports there."""
        msg = self._msgs["PoseWithCovarianceStamped"]()
        msg.header.frame_id = "map"
        msg.pose.pose = self._pose(spawn)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.25
        cov[35] = 0.06853892326654787
        msg.pose.covariance = cov

        deadline = time.time() + timeout
        attempt = 0
        while time.time() < deadline and self.context.ok():
            attempt += 1
            msg.header.stamp = self.node.get_clock().now().to_msg()
            if self.init_loc.wait_for_service(timeout_sec=2.0):
                req = self._msgs["InitializeLocalization"].Request()
                req.pose_with_covariance = [msg]
                req.method = req.DIRECT
                res = self.call(self.init_loc, req, 20.0)
                via = LOCALIZATION_INITIALIZE
            else:
                req = self._msgs["AdapiInitialize"].Request()
                req.pose = [msg]
                res = self.call(self.init_loc_adapi, req, 20.0)
                via = ADAPI_LOCALIZATION_INITIALIZE
            if res is not None and not res.status.success:
                self._log(f"initialize rejected ({res.status.message}); retrying")
            arrived = self.wait_until(
                lambda: self.ego is not None
                and math.hypot(self.ego[0] - spawn[0], self.ego[1] - spawn[1]) <= INIT_TOLERANCE_M,
                5.0,
            )
            if arrived:
                return True, f"ego_initialized via {via} (attempt {attempt})"
        where = "no kinematic_state yet" if self.ego is None else (
            f"ego {math.hypot(self.ego[0] - spawn[0], self.ego[1] - spawn[1]):.1f} m off"
        )
        return False, f"localization_initialize_timeout ({where})"

    # -- routing
    def clear(self, timeout: float = 15.0) -> tuple[bool, str]:
        self.wait_until(lambda: self.route_state is not None, 3.0)
        if self.route_state in (None, ROUTE_UNSET):
            return True, "route_already_unset"
        for attempt in (1, 2):
            res = self.call(self.clear_route, self._msgs["ClearRoute"].Request(), 15.0)
            if res is None:
                return False, "clear_route_no_response"
            if res.status.success:
                break
            # AD API refuses to clear a route the vehicle is driving; after
            # ARRIVED Autoware has already switched to STOP, so this is the
            # rare case of a leg cut short while moving.
            if "in use" in res.status.message and attempt == 1:
                self.call(self.to_stop, self._msgs["ChangeOperationMode"].Request(), 10.0)
                self.wait_until(lambda: self.mode == MODE_STOP, 5.0)
                continue
            return False, f"clear_route_failed: {res.status.message}"
        if self.wait_until(lambda: self.route_state == ROUTE_UNSET, timeout):
            return True, "route_cleared"
        return False, f"clear_route_timeout (state={self.route_state})"

    def try_route(self, goal: XYYaw, waypoints: tuple[XYYaw, ...] = ()) -> tuple[bool, str]:
        """One planner verdict for one goal."""
        ok, msg = self.clear()
        if not ok:
            return False, msg
        req = self._msgs["SetRoutePoints"].Request()
        req.header.frame_id = "map"
        req.header.stamp = self.node.get_clock().now().to_msg()
        req.option.allow_goal_modification = True  # what the vehicle's operator sends
        req.goal = self._pose(goal)
        req.waypoints = [self._pose(w) for w in waypoints]
        res = self.call(self.set_route, req, 60.0)
        if res is None:
            return False, "set_route_no_response"
        if not res.status.success:
            return False, f"set_route_rejected: {res.status.message}"
        if not self.wait_until(lambda: self.route_state == ROUTE_SET, 20.0):
            return False, f"route_set_timeout (state={self.route_state})"
        return True, "route_set"

    def route(
        self,
        goal: XYYaw,
        stops: list[MapStop],
        order: list[str],
        *,
        allow_walk: bool,
        map_path: Path | None = None,
    ) -> RouteResult:
        last = "no_candidates"
        calls = 0
        for cand in route_candidates(goal, stops, order, allow_walk, lane_centerlines(map_path)):
            for attempt in (1, 2):
                calls += 1
                t0 = time.time()
                ok, msg = self.try_route(cand.goal, cand.waypoints)
                where = f"{cand.name or 'goal'} ({cand.goal[0]:.1f}, {cand.goal[1]:.1f})"
                self._log(
                    f"{cand.label} -> {where} waypoints={len(cand.waypoints)}: "
                    f"{'accepted' if ok else msg} [{time.time() - t0:.1f}s]"
                )
                if ok:
                    return RouteResult(
                        True, msg, cand.goal, cand.name, cand.skipped,
                        len(cand.waypoints), cand.label, calls,
                    )
                last = msg
                # A silent service is transient; a rejection is the verdict.
                if "no_response" not in msg and "timeout" not in msg:
                    break
        return RouteResult(False, last, calls=calls)

    # -- engage
    def engage(self, timeout: float) -> tuple[bool, str]:
        self.wait_until(lambda: self.route_state is not None and self.mode is not None, 5.0)
        if self.route_state != ROUTE_SET:
            return False, f"route_not_set_before_engage (state={self.route_state})"
        if self.mode == MODE_AUTONOMOUS:
            return True, "already_autonomous"
        deadline = time.time() + timeout
        last = "autonomous_mode_not_available_yet"
        last_log = 0.0
        while time.time() < deadline and self.context.ok():
            self.spin(0.5)
            if self.mode == MODE_AUTONOMOUS:
                break
            if time.time() - last_log >= 10.0:
                self._log(
                    f"waiting to engage (available={self.auto_available}, "
                    f"control={self.control_enabled}, mode={self.mode})"
                )
                last_log = time.time()
            if not self.auto_available:
                continue
            if not self.control_enabled:
                res = self.call(self.enable_control, self._msgs["ChangeOperationMode"].Request(), 10.0)
                if res is None or not res.status.success:
                    last = "enable_autoware_control_" + (
                        "no_response" if res is None else f"rejected: {res.status.message}"
                    )
                    self.spin(2.0)
                    continue
                self.spin(1.0)
            res = self.call(self.to_autonomous, self._msgs["ChangeOperationMode"].Request(), 10.0)
            if res is None or not res.status.success:
                last = "change_to_autonomous_" + (
                    "no_response" if res is None else f"rejected: {res.status.message}"
                )
                self.spin(3.0)
                continue
            self.wait_until(lambda: self.mode == MODE_AUTONOMOUS, 15.0)
        # Confirm the mode is held, not just reported once during a transition.
        if self.mode == MODE_AUTONOMOUS:
            held = self.wait_until(lambda: self.mode != MODE_AUTONOMOUS, 2.0)
            if not held:
                return True, f"autonomous_engaged (mode={self.mode}, control={self.control_enabled})"
            last = f"autonomous_engage_not_held (mode={self.mode})"
        return False, last

    def close(self) -> None:
        from dp_multi_eval.job_waiters import close_job_context

        try:
            self.executor.remove_node(self.node)
            self.node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        close_job_context(self.context)


# -- one client per job domain in this process -----------------------------------
_CLIENTS: dict[str, RoutingClient] = {}


def get_client(env: dict[str, str]) -> RoutingClient:
    key = str(env.get("ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "0")))
    client = _CLIENTS.get(key)
    if client is None or not client.context.ok():
        client = RoutingClient(env)
        _CLIENTS[key] = client
    return client


def close_routing(env: dict[str, str]) -> None:
    key = str(env.get("ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "0")))
    client = _CLIENTS.pop(key, None)
    if client is not None:
        client.close()
