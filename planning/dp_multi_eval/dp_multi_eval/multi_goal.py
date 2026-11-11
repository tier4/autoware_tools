"""Extract multi-leg route goals from a source rosbag."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dp_multi_eval.bag_reader import EgoSample, load_reference_ego, yaw_from_quat


GOAL_TOPICS = (
    "/planning/mission_planning/goal",
    "/planning/mission_planning/route_selector/main/goal",
    "/api/routing/set_route_points",  # not a topic; kept for docs only
)


@dataclass(frozen=True)
class RouteGoal:
    """One route destination in map frame."""

    x: float
    y: float
    yaw: float
    stamp_sec: float = 0.0
    source: str = "stop_segment"  # goal_topic | stop_segment | bag_end


def _xy_dist(a: RouteGoal | EgoSample, b: RouteGoal | EgoSample) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _goals_from_stop_segments(
    ego: list[EgoSample],
    *,
    speed_mps: float,
    min_sec: float,
    min_spacing_m: float,
) -> list[RouteGoal]:
    """Detect dwells (near-zero speed) and treat end-of-dwell poses as goals."""
    if not ego:
        return []

    goals: list[RouteGoal] = []
    dwell_start: int | None = None

    def _close_dwell(end_idx: int) -> None:
        nonlocal dwell_start
        if dwell_start is None:
            return
        t0 = ego[dwell_start].stamp_sec
        t1 = ego[end_idx].stamp_sec
        if (t1 - t0) < min_sec:
            dwell_start = None
            return
        # Use mid-dwell pose (more stable than last jittery sample)
        mid = dwell_start + (end_idx - dwell_start) // 2
        sample = ego[mid]
        cand = RouteGoal(
            x=sample.x,
            y=sample.y,
            yaw=sample.yaw_rad,
            stamp_sec=sample.stamp_sec,
            source="stop_segment",
        )
        if goals and _xy_dist(goals[-1], cand) < min_spacing_m:
            dwell_start = None
            return
        goals.append(cand)
        dwell_start = None

    for i, sample in enumerate(ego):
        if sample.speed_mps <= speed_mps:
            if dwell_start is None:
                dwell_start = i
        else:
            if dwell_start is not None:
                _close_dwell(i - 1)
    if dwell_start is not None:
        _close_dwell(len(ego) - 1)

    return goals


ROUTE_TOPIC = "/planning/mission_planning/route"


def _recorded_route_goals(bag_path: Path) -> list[RouteGoal]:
    """The goal of every LaneletRoute the recording carries, in publish order
    (near-duplicate republishes of one goal collapsed)."""
    try:
        import sqlite3

        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception:  # noqa: BLE001
        return []

    bag_path = Path(bag_path).expanduser().resolve()
    files = sorted(bag_path.glob("*.db3")) if bag_path.is_dir() else [bag_path]
    found: list[RouteGoal] = []
    for f in files:
        try:
            con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        except Exception:  # noqa: BLE001
            continue
        try:
            row = con.execute(
                "SELECT id,type FROM topics WHERE name=?", (ROUTE_TOPIC,)
            ).fetchone()
            if not row:
                continue
            tid, ttype = row
            cls = get_message(ttype)
            for ts, data in con.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
                (tid,),
            ):
                msg = deserialize_message(bytes(data), cls)
                pose = msg.goal_pose
                found.append(
                    RouteGoal(
                        x=float(pose.position.x),
                        y=float(pose.position.y),
                        yaw=float(yaw_from_quat(pose.orientation)),
                        stamp_sec=float(ts) / 1e9,
                        source="recorded_route",
                    )
                )
        except Exception:  # noqa: BLE001
            continue
        finally:
            con.close()

    # Slice files sort lexicographically (_10 before _2), so order by publish time,
    # which is the leg order. Collapse near-duplicate republishes of the same goal.
    found.sort(key=lambda g: g.stamp_sec)
    deduped: list[RouteGoal] = []
    for goal in found:
        if deduped and _xy_dist(deduped[-1], goal) < 2.0:
            deduped[-1] = goal
        else:
            deduped.append(goal)
    found = deduped

    return found


def _goals_from_recorded_route(
    bag_path: Path,
    ego: list[EgoSample],
    *,
    reached_radius_m: float,
) -> list[RouteGoal]:
    """Goals taken from the LaneletRoute the recorded vehicle actually drove.

    Every other source guesses. A dwell says where the vehicle stopped, which is not
    the same as where it was going: it also stops at lights and in queues, and those
    poses sit wherever the vehicle happened to halt — often straddling lanelets, where
    the mission planner rejects the goal outright. The recorded route has no such
    ambiguity. Its goal was produced and accepted by a real mission planner, carries
    the heading that planner agreed with, and is republished as each leg completes, so
    a multi-stop recording yields its legs directly.

    Goals the ego never reached are dropped: the last route of a recording is usually
    published just before the cut and aims past the end of the data, where the
    reproducer has no scene to replay.
    """
    found = _recorded_route_goals(bag_path)
    if not found or not ego:
        return found

    kept: list[RouteGoal] = []
    for goal in found:
        nearest = min(math.hypot(e.x - goal.x, e.y - goal.y) for e in ego)
        if nearest <= reached_radius_m:
            kept.append(goal)
        else:
            print(
                f"[info] Dropping recorded-route goal ({goal.x:.0f}, {goal.y:.0f}) — "
                f"ego never came closer than {nearest:.0f} m; it lies past the bag"
            )
    return kept


def _goals_from_goal_topic(bag_path: Path) -> list[RouteGoal]:
    """Best-effort: unique PoseStamped goals published during the bag."""
    try:
        from dp_multi_eval.bag_reader import iter_topic_messages
    except Exception:  # noqa: BLE001
        return []

    topics = [
        "/planning/mission_planning/goal",
        "/planning/mission_planning/route_selector/main/goal",
    ]
    found: list[RouteGoal] = []
    try:
        for _topic, stamp_sec, msg in iter_topic_messages(bag_path, topics):
            pose = getattr(msg, "pose", None)
            if pose is None:
                continue
            pos = pose.position
            yaw = yaw_from_quat(pose.orientation)
            cand = RouteGoal(
                x=float(pos.x),
                y=float(pos.y),
                yaw=float(yaw),
                stamp_sec=float(stamp_sec),
                source="goal_topic",
            )
            if found and _xy_dist(found[-1], cand) < 2.0:
                found[-1] = cand  # keep latest near-duplicate
            else:
                found.append(cand)
    except Exception:  # noqa: BLE001
        return []
    return found


def resolve_stop_points_csv(map_path: Path, filename: str = "stop_points.csv") -> Path:
    """Resolve stop_points.csv under map_path (or an absolute file path)."""
    path = Path(filename).expanduser()
    if path.is_file():
        return path.resolve()
    return (Path(map_path).expanduser().resolve() / filename).resolve()


def load_stop_point_names(csv_path: Path) -> list[str]:
    """Return stop names in CSV row order."""
    return [s.name for s in _load_map_stop_points(csv_path)]


def _load_map_stop_points(csv_path: Path) -> list[Any]:
    """Load MapStopPoint rows; prefer batch_eval helper, else local CSV parse."""
    csv_path = csv_path.expanduser().resolve()
    try:
        from stop_points import load_stop_points  # type: ignore

        return list(load_stop_points(csv_path))
    except Exception:  # noqa: BLE001
        pass

    import csv

    if not csv_path.is_file():
        raise FileNotFoundError(f"Stop points file not found: {csv_path}")
    stops: list[SimpleStop] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            stops.append(
                SimpleStop(
                    name=name,
                    x=float(row["x"]),
                    y=float(row["y"]),
                    yaw=float(row["yaw"]),
                )
            )
    if not stops:
        raise ValueError(f"No stop points loaded from {csv_path}")
    return stops


@dataclass(frozen=True)
class SimpleStop:
    name: str
    x: float
    y: float
    yaw: float


def goals_from_stop_order(
    ordered_names: list[str],
    *,
    map_path: Path,
    stop_points_csv: str = "stop_points.csv",
) -> list[RouteGoal]:
    """Build route goals from an ordered list of stop names in stop_points.csv."""
    if not ordered_names:
        raise ValueError("multi_goal_stop_order is empty")
    csv_path = resolve_stop_points_csv(map_path, stop_points_csv)
    stops = _load_map_stop_points(csv_path)
    by_name = {s.name: s for s in stops}
    missing = [n for n in ordered_names if n not in by_name]
    if missing:
        raise ValueError(
            f"Unknown stop name(s) in multi_goal_stop_order: {missing}. "
            f"Available: {sorted(by_name)}"
        )
    goals: list[RouteGoal] = []
    for name in ordered_names:
        stop = by_name[name]
        goals.append(
            RouteGoal(
                x=float(stop.x),
                y=float(stop.y),
                yaw=float(stop.yaw),
                stamp_sec=0.0,
                source=f"stop_points:{name}",
            )
        )
    return goals


def get_bag_start_xy(bag_path: Path) -> tuple[float, float] | None:
    """Return the first ego (x, y) without scanning the whole bag."""
    try:
        from dp_multi_eval.bag_reader import open_bag_reader
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from rosbag2_py import StorageFilter
    except Exception:  # noqa: BLE001
        return None

    bag_path = bag_path.expanduser().resolve()
    try:
        reader = open_bag_reader(bag_path)
        type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
        topic = "/localization/kinematic_state"
        if topic not in type_map:
            return None
        reader.set_filter(StorageFilter(topics=[topic]))
        if not reader.has_next():
            return None
        _topic, data, _ = reader.read_next()
        msg = deserialize_message(data, get_message(type_map[topic]))
        pose = msg.pose.pose
        return float(pose.position.x), float(pose.position.y)
    except Exception:  # noqa: BLE001
        return None


def start_alignment_radius(min_spacing_m: float) -> float:
    """Radius within which a goal counts as "the ego is already there".

    Shared by both alignment paths so they cannot drift apart: a goal this close to
    the spawn pose is not a destination, and routing to it returns an empty route.
    """
    return max(5.0, float(min_spacing_m))


def drop_goals_at_bag_start(
    goals: list[RouteGoal],
    start_xy: tuple[float, float] | None,
    *,
    near_start_m: float,
) -> list[RouteGoal]:
    """Drop leading goals the ego spawns on top of.

    For bag-derived goals (dwells, published goal topics) the list is already in
    traversal order, so only *leading* entries can be ones we are standing at —
    unlike an externally ordered stop list, where the ego may join partway through
    and align_goals_to_bag_ego has to search. Searching here would be wrong: a route
    that passes near its own start later would have its opening legs discarded.
    """
    if not goals or start_xy is None:
        return goals
    sx, sy = start_xy
    kept = list(goals)
    while len(kept) > 1 and math.hypot(kept[0].x - sx, kept[0].y - sy) < near_start_m:
        dropped = kept.pop(0)
        print(
            f"[info] Dropping first leg ({dropped.source}) — ego spawns there "
            f"({math.hypot(dropped.x - sx, dropped.y - sy):.1f}m < {near_start_m:.1f}m)"
        )
    return kept


def align_goals_to_bag_ego(
    goals: list[RouteGoal],
    start_xy: tuple[float, float] | None,
    *,
    near_stop_m: float = 15.0,
) -> list[RouteGoal]:
    """Pick the ordered stop list starting near bag ego, then follow list order.

    1. Find the stop closest to the bag start pose.
    2. If ego is already at that stop (within ``near_stop_m``), start from the
       *next* stop (avoid start≈goal route failures).
    3. Otherwise start from that nearest stop (drive there first).
    4. Keep the remainder of the user-selected order (including a later return
       to the same stop name).
    """
    if not goals or start_xy is None:
        return goals

    sx, sy = start_xy
    nearest_idx = min(
        range(len(goals)),
        key=lambda i: math.hypot(goals[i].x - sx, goals[i].y - sy),
    )
    nearest_dist = math.hypot(goals[nearest_idx].x - sx, goals[nearest_idx].y - sy)
    if nearest_dist < near_stop_m:
        start_idx = nearest_idx + 1
        reason = (
            f"near {goals[nearest_idx].source} ({nearest_dist:.1f}m) → next stop"
        )
    else:
        start_idx = nearest_idx
        reason = (
            f"closest {goals[nearest_idx].source} ({nearest_dist:.1f}m) → drive there"
        )

    if start_idx >= len(goals):
        print(
            f"[warn] Bag ego ({sx:.1f}, {sy:.1f}) is at/after the last stop "
            f"({goals[-1].source}); no remaining route legs"
        )
        return []

    kept = goals[start_idx:]
    print(
        f"[info] Aligned stop_points to bag ego ({sx:.1f}, {sy:.1f}): {reason}; "
        f"legs {start_idx + 1}→{len(goals)} of {len(goals)} "
        f"(first={kept[0].source}, last={kept[-1].source})"
    )
    return kept


def _settled_bag_end(ego: list, speed_mps: float = 0.3, min_sec: float = 2.0,
                     max_walkback_m: float = 200.0) -> RouteGoal:
    """The recording's last real stop, not the frame the recording was cut at.

    ego[-1] is wherever someone pressed stop, and two of the Hiratsuka bags were
    cut moving through a signalized intersection. Their bag_end goals landed 4 m
    and 14 m from a traffic light's stop line, so a replay that stopped at the
    light — correct driving — sat metres outside the 5 m arrival tolerance and
    failed as stuck at its own destination.

    When the recording ends at rest, that resting pose is the goal, unchanged.
    When it ends moving, walk back to the last spot the recorded vehicle
    demonstrably stopped (>= min_sec below speed_mps): a place it stopped is a
    place a bus can stop. The walk is capped — a recording whose only stop was
    its departure point keeps its raw end rather than losing the whole route,
    and that run's arrival is at least explainable.
    """
    end = ego[-1]
    raw = RouteGoal(x=end.x, y=end.y, yaw=end.yaw_rad,
                    stamp_sec=end.stamp_sec, source="bag_end")
    if end.speed_mps < speed_mps:
        return raw
    i = len(ego) - 1
    while i >= 0:
        if ego[i].speed_mps < speed_mps:
            j = i
            while j > 0 and ego[j - 1].speed_mps < speed_mps:
                j -= 1
            if ego[i].stamp_sec - ego[j].stamp_sec >= min_sec:
                mid = ego[(j + i) // 2]
                walked = math.hypot(mid.x - end.x, mid.y - end.y)
                if walked > max_walkback_m:
                    break
                print(
                    f"[info] recording was cut while moving "
                    f"({end.speed_mps:.1f} m/s); final goal pulled back "
                    f"{walked:.0f}m to its last settled stop"
                )
                return RouteGoal(x=mid.x, y=mid.y, yaw=mid.yaw_rad,
                                 stamp_sec=mid.stamp_sec, source="bag_end_settled")
            i = j - 1
        else:
            i -= 1
    return raw


def bag_end_pose(bag_path: Path) -> tuple[float, float, float, float] | None:
    """The recording's last ego pose and speed: (x, y, yaw, speed_mps)."""
    ego = load_reference_ego(Path(bag_path).expanduser().resolve())
    if not ego:
        return None
    end = ego[-1]
    return float(end.x), float(end.y), float(end.yaw_rad), float(end.speed_mps)


def _map_lanelet_ids(map_path: Path) -> set[str] | None:
    """Ids of all lanelets in the map, or None when the map file is absent.

    Whether the mission planner can route to a stop is the planner's call,
    not this tool's: a predecessor-based guess rejected 松風公園入口 (lanelet
    87169, no predecessor) although the vehicle's own recorded routes end
    exactly there. Only stops whose lanelet does not exist at all are skipped.
    """
    osm = Path(map_path).expanduser().resolve() / "lanelet2_map.osm"
    if not osm.is_file():
        return None
    import xml.etree.ElementTree as ET

    root = ET.parse(osm).getroot()
    return {
        r.get("id")
        for r in root.iter("relation")
        if any(t.get("k") == "type" and t.get("v") == "lanelet" for t in r.findall("tag"))
    }


def bag_path_xy(bag_path: Path, step_m: float = 5.0) -> list[tuple[float, float]]:
    """The recorded ego path, thinned to about one point per ``step_m``."""
    out: list[tuple[float, float]] = []
    for e in load_reference_ego(Path(bag_path).expanduser().resolve()):
        if not out or math.hypot(e.x - out[-1][0], e.y - out[-1][1]) >= step_m:
            out.append((float(e.x), float(e.y)))
    return out


def recorded_destination(
    bag_path: Path, end_xy: tuple[float, float], *, min_m: float = 10.0, max_m: float = 1500.0,
) -> RouteGoal | None:
    """Where the recorded vehicle was going when the recording was cut: the
    goal of the last route it published, when that lies beyond the cut. The
    vehicle's own answer — no stop list, no guess; None when the recording
    carries no route (one set before it began) or its last goal is the end."""
    goals = _recorded_route_goals(bag_path)
    if not goals:
        return None
    last = goals[-1]
    dist = math.hypot(last.x - float(end_xy[0]), last.y - float(end_xy[1]))
    if dist < min_m or dist > max_m:
        return None
    return RouteGoal(x=last.x, y=last.y, yaw=last.yaw, stamp_sec=last.stamp_sec,
                     source="cut_end_recorded_route")


def recorded_stop_successors(
    bag_dir: Path, map_path: Path, stop_points_csv: str = "stop_points.csv",
) -> dict[str, str]:
    """The line's stop sequence as the vehicle itself drove it: for each map
    stop, the stop its next route went to — read from the routes recorded in
    the recordings of ``bag_dir`` (the most frequent one wins).

    Nobody writes the line down. A hand-written order had 松風町 after
    すみれ平; the vehicle's own routes go 松風公園入口 → 松風町 →
    花水小学校前 → すみれ平, and three recordings cut mid-drive were routed
    to a stop one or two further on (2026-11-11)."""
    from collections import Counter

    try:
        stops = _load_map_stop_points(resolve_stop_points_csv(map_path, stop_points_csv))
    except Exception:  # noqa: BLE001
        return {}

    def stop_at(g: RouteGoal) -> str | None:
        best = min(stops, key=lambda st: math.hypot(float(st.x) - g.x, float(st.y) - g.y), default=None)
        if best is None or math.hypot(float(best.x) - g.x, float(best.y) - g.y) > BUS_STOP_MATCH_M:
            return None
        return str(best.name)

    pairs: dict[str, Counter] = {}
    bag_dir = Path(bag_dir).expanduser()
    for bag in sorted(d for d in bag_dir.iterdir() if d.is_dir()) if bag_dir.is_dir() else []:
        names = [stop_at(g) for g in _recorded_route_goals(bag)]
        for here, there in zip(names, names[1:]):
            if here and there and here != there:
                pairs.setdefault(here, Counter())[there] += 1
    return {here: nxt.most_common(1)[0][0] for here, nxt in pairs.items()}


def next_stop_ahead(
    end_xy_yaw: tuple[float, float, float],
    *,
    map_path: Path,
    stop_points_csv: str = "stop_points.csv",
    stop_order: list[str] | None = None,
    recorded_path_xy: list[tuple[float, float]] | None = None,
    min_m: float = 10.0,
    max_m: float = 1500.0,
    passed_radius_m: float = 30.0,
    successors: dict[str, str] | None = None,
) -> RouteGoal | None:
    """The map stop the recorded bus was heading for when the recording was cut.

    A bag cut while the bus is still moving ends wherever someone pressed
    stop — on 2026-04-28 that is the stop line of a green-lit junction, 13 m
    past a car parked 0.85 m off the bus's path. Routing the replay to that
    pose asks the model to park there: the route ends on the stop line, the
    model plans a straight goal approach instead of the bus's set-up for the
    right turn, passes the parked car 0.24 m closer, and the obstacle_stop
    guard stubs its plan to a halt every few metres (measured: 9 stops over
    the last 26 m, the drive 15 % slower than the recording). The bus never
    had a goal there; its destination was the next stop on its route, past
    the turn.

    With ``successors`` (recorded_stop_successors: the stop sequence in the
    vehicle's own recorded routes) the destination is the stop after the last
    one the recording left behind. Else, with ``stop_order`` (a written
    ordered stop list) and the recorded
    path, the destination is read off the route: the last ordered stop the
    recording passed within ``passed_radius_m`` marks its progress, and the
    first stop after it that the routing graph can reach is the goal (only stops whose lanelet is missing from the map are
    skipped; see _map_lanelet_ids). Without an order the nearest
    reachable stop ahead of the end heading is used — cruder: a straight-on
    stop on another road can win over the one past the bus's turn.

    Arrival is still measured at the cut pose, drive-through: the recording
    ends there and nothing past it is in the scene. None when no stop
    qualifies; the caller keeps the raw end.
    """
    import csv

    ex, ey, eyaw = float(end_xy_yaw[0]), float(end_xy_yaw[1]), float(end_xy_yaw[2])
    known = _map_lanelet_ids(map_path)
    csv_path = resolve_stop_points_csv(map_path, stop_points_csv)
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            name = (row.get("name") or "").strip()
            if name:
                rows.setdefault(name, row)

    def routable(row: dict[str, str]) -> bool:
        lane = (row.get("lane_id") or "").strip()
        if known is None or not lane or lane in known:
            return True
        print(
            f"[info] cut-end candidate {row['name'].strip()} skipped: lanelet {lane} "
            "is not in this map"
        )
        return False

    def goal_of(row: dict[str, str]) -> RouteGoal:
        return RouteGoal(
            x=float(row["x"]), y=float(row["y"]), yaw=float(row["yaw"]),
            stamp_sec=0.0, source=f"cut_end_next_stop:{row['name'].strip()}",
        )

    # The vehicle's own sequence first (``successors``, from the routes it
    # recorded): the stop after the last one this recording left behind.
    if successors and recorded_path_xy:
        hx_, hy_ = math.cos(eyaw), math.sin(eyaw)
        last_name, last_i = None, -1
        for name, row in rows.items():
            sx, sy = float(row["x"]), float(row["y"])
            if (sx - ex) * hx_ + (sy - ey) * hy_ >= 0.0:
                continue   # still ahead of the cut: not left behind, however near
            i, d = min(((k, math.hypot(px - sx, py - sy)) for k, (px, py) in enumerate(recorded_path_xy)),
                       key=lambda kd: kd[1])
            if d <= passed_radius_m and i > last_i:
                last_name, last_i = name, i
        nxt = rows.get(successors.get(last_name or "", ""))
        if nxt is not None and routable(nxt):
            dist = math.hypot(float(nxt["x"]) - ex, float(nxt["y"]) - ey)
            if min_m <= dist <= max_m:
                return goal_of(nxt)

    order = [n for n in (stop_order or []) if n in rows]
    if order and recorded_path_xy:
        def passed(row: dict[str, str]) -> bool:
            sx, sy = float(row["x"]), float(row["y"])
            return any(math.hypot(px - sx, py - sy) <= passed_radius_m for px, py in recorded_path_xy)

        last_passed = -1
        for i, name in enumerate(order):
            if passed(rows[name]):
                last_passed = i
            elif last_passed >= 0:
                break  # the recording ends before this stop
        for name in order[last_passed + 1:]:
            row = rows[name]
            dist = math.hypot(float(row["x"]) - ex, float(row["y"]) - ey)
            if dist < min_m or dist > max_m:
                continue
            if routable(row):
                return goal_of(row)
        return None

    hx, hy = math.cos(eyaw), math.sin(eyaw)
    best: tuple[float, dict[str, str]] | None = None
    for row in rows.values():
        dx, dy = float(row["x"]) - ex, float(row["y"]) - ey
        dist = math.hypot(dx, dy)
        if dist < min_m or dist > max_m or dx * hx + dy * hy <= 0.0:
            continue
        if not routable(row):
            continue
        if best is None or dist < best[0]:
            best = (dist, row)
    return goal_of(best[1]) if best else None


BUS_STOP_MATCH_M = 6.0      # a recorded dwell this close to a map stop is a bus-stop leg
CUT_END_SPEED_MPS = 1.0     # recording ending faster than this was cut mid-drive

# Bump when goal derivation logic changes; stale caches must not resurrect old
# goal choices. v4: served bus stops are legs again (routing centres them).
_GOALS_LOGIC_VERSION = "v4"


def extract_route_goals(
    bag_path: Path,
    *,
    source: str = "auto",
    stop_speed_mps: float = 0.2,
    stop_min_sec: float = 8.0,
    min_spacing_m: float = 15.0,
    map_path: Path | None = None,
    stop_points_csv: str = "stop_points.csv",
    stop_order: list[str] | None = None,
) -> list[RouteGoal]:
    """Cached front of _extract_route_goals_uncached — the recordings run to
    79 GB and re-deriving goals re-reads all of it on every attempt."""
    from dp_multi_eval.bag_cache import cached_json

    version = (
        f"{_GOALS_LOGIC_VERSION}|{source}|{stop_speed_mps}|{stop_min_sec}|"
        f"{min_spacing_m}|{map_path}|{stop_points_csv}|{list(stop_order or [])}"
    )

    def compute() -> list[dict]:
        goals = _extract_route_goals_uncached(
            bag_path,
            source=source,
            stop_speed_mps=stop_speed_mps,
            stop_min_sec=stop_min_sec,
            min_spacing_m=min_spacing_m,
            map_path=map_path,
            stop_points_csv=stop_points_csv,
            stop_order=stop_order,
        )
        return [
            {"x": g.x, "y": g.y, "yaw": g.yaw, "stamp_sec": g.stamp_sec,
             "source": g.source}
            for g in goals
        ]

    data = cached_json(bag_path, "route_goals", version, compute)
    return [RouteGoal(**row) for row in data]


def _extract_route_goals_uncached(
    bag_path: Path,
    *,
    source: str = "auto",
    stop_speed_mps: float = 0.2,
    stop_min_sec: float = 8.0,
    min_spacing_m: float = 15.0,
    map_path: Path | None = None,
    stop_points_csv: str = "stop_points.csv",
    stop_order: list[str] | None = None,
) -> list[RouteGoal]:
    """Return ordered goals for multi-leg replay.

    ``source``:
      - ``recorded_route`` — goals from the LaneletRoute the bag recorded
      - ``goal_topic`` — only published goal PoseStampeds
      - ``stop_segments`` — ego dwell detection
      - ``stop_points`` — ordered names from ``stop_points.csv`` (web / config)
      - ``auto`` — recorded route if present, else goal topics, else stop
        segments; always ensure bag-end goal
    """
    source = (source or "auto").strip().lower()

    if source == "stop_points":
        if map_path is None:
            raise ValueError("map_path is required when multi_goal_source=stop_points")
        goals = goals_from_stop_order(
            list(stop_order or []),
            map_path=map_path,
            stop_points_csv=stop_points_csv,
        )
        return align_goals_to_bag_ego(
            goals,
            get_bag_start_xy(bag_path),
            near_stop_m=start_alignment_radius(min_spacing_m),
        )

    bag_path = bag_path.expanduser().resolve()
    ego = load_reference_ego(bag_path)
    if not ego:
        return []

    goals: list[RouteGoal] = []
    if source in ("auto", "recorded_route"):
        goals = _goals_from_recorded_route(
            bag_path, ego, reached_radius_m=start_alignment_radius(min_spacing_m)
        )
        if goals:
            print(f"[info] {len(goals)} goal(s) from the bag's recorded route")
    if source in ("goal_topic",) or (source == "auto" and not goals):
        goals = _goals_from_goal_topic(bag_path)
    if source == "stop_segments" or (source == "auto" and not goals):
        goals = _goals_from_stop_segments(
            ego,
            speed_mps=stop_speed_mps,
            min_sec=stop_min_sec,
            min_spacing_m=min_spacing_m,
        )

    # Every map bus stop the recording dwelled at is a leg, whether or not the
    # recorded route topic carried it: all three Hiratsuka references are human
    # drives whose route went 平塚駅南口 → 教会前 while the driver served
    # 商工会議所前 on the way (12.5 s on 2026-04-28). In autonomous operation
    # the operator sets each stop as a goal, so the replay does the same. The
    # mission planner refused 商工会議所前's csv pose ("footprint exceeds
    # lane"); routing now projects stop goals onto the lane centre and it is
    # accepted. The leg goal is the stop's csv pose; routing centres it.
    if source == "auto" and map_path is not None:
        try:
            stops = _load_map_stop_points(resolve_stop_points_csv(map_path, stop_points_csv))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] stop points unavailable ({exc}); bus-stop legs not added")
            stops = []
        dwells = _goals_from_stop_segments(
            ego, speed_mps=stop_speed_mps, min_sec=stop_min_sec, min_spacing_m=min_spacing_m
        )
        added = 0
        for d in dwells:
            stop = min(stops, key=lambda st: math.hypot(float(st.x) - d.x, float(st.y) - d.y), default=None)
            if stop is None or math.hypot(float(stop.x) - d.x, float(stop.y) - d.y) > BUS_STOP_MATCH_M:
                continue
            leg = RouteGoal(x=float(stop.x), y=float(stop.y), yaw=float(stop.yaw),
                            stamp_sec=d.stamp_sec, source=f"stop_points:{stop.name}")
            if any(_xy_dist(g, leg) < min_spacing_m for g in goals):
                continue
            goals.append(leg)
            added += 1
        if added:
            print(f"[info] {added} bus-stop leg(s) added from recorded dwells")
    if source == "auto" and len(goals) > 1:
        def _reached(g: RouteGoal) -> float:
            near = [e.stamp_sec for e in ego if math.hypot(e.x - g.x, e.y - g.y) <= 6.0]
            return min(near) if near else g.stamp_sec
        goals.sort(key=_reached)

    # The end of the recording. Ended at rest: that pose is the final goal
    # (snapped to a map stop it sits on). Cut while moving: a drive-through
    # leg at the cut pose — the harness routes past it to the bus's next stop
    # and ends the run when the ego passes the cut pose (see run_single_job).
    end = ego[-1]
    if end.speed_mps >= CUT_END_SPEED_MPS:
        bag_end = RouteGoal(x=end.x, y=end.y, yaw=end.yaw_rad, stamp_sec=end.stamp_sec,
                            source="bag_end:cut")
        print(f"[info] recording was cut while moving ({end.speed_mps:.1f} m/s); final leg is drive-through at the cut pose")
    else:
        bag_end = _settled_bag_end(ego)

    if not goals:
        goals = [bag_end]
    elif _xy_dist(goals[-1], bag_end) >= min_spacing_m:
        goals.append(bag_end)
    elif bag_end.source == "bag_end_settled":
        print(
            f"[info] settled bag end coincides with the '{goals[-1].source}' goal "
            f"({_xy_dist(goals[-1], bag_end):.1f}m apart) — keeping the mission goal"
        )
    elif bag_end.source != "bag_end:cut":
        # Replace last with bag_end (same place, more accurate stamp)
        goals[-1] = bag_end

    return drop_goals_at_bag_start(
        goals, (ego[0].x, ego[0].y), near_start_m=start_alignment_radius(min_spacing_m)
    )


def goals_to_jsonable(goals: list[RouteGoal]) -> list[dict[str, Any]]:
    return [
        {
            "x": g.x,
            "y": g.y,
            "yaw": g.yaw,
            "stamp_sec": g.stamp_sec,
            "source": g.source,
        }
        for g in goals
    ]
