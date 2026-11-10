"""What the reactive shield knows of the map and the recorded lights. No ROS.

The shield moves NPCs off their recording only in time. Where that leaves them
has to make sense on the road: a delayed car meets its light later than it did
in the recording, and a car held for the ego must wait before a junction or a
crosswalk, not inside one (2026-11-09 review).

| Map feature | Lanelet2 source |
|---|---|
| stop line + light | traffic_light regulatory element: ``ref_line``, referenced by a lanelet (its id is the recorded ``traffic_light_group_id``) |
| junction | lanelets with a ``turn_direction`` attribute |
| crosswalk | lanelets of subtype ``crosswalk`` |
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from pathlib import Path

# autoware_perception_msgs/TrafficLightElement
RED, AMBER, GREEN = 1, 2, 3
CIRCLE, LEFT_ARROW, RIGHT_ARROW, UP_ARROW, UP_LEFT_ARROW, UP_RIGHT_ARROW = 1, 2, 3, 4, 5, 6
SOLID_OFF = 1
ARROWS_FOR_TURN = {
    "left": {LEFT_ARROW, UP_LEFT_ARROW},
    "right": {RIGHT_ARROW, UP_RIGHT_ARROW},
    "straight": {UP_ARROW, UP_LEFT_ARROW, UP_RIGHT_ARROW},
}
GRID_M = 20.0


@dataclass
class StopLine:
    x0: float
    y0: float
    x1: float
    y1: float
    group: int    # the light's regulatory element id = recorded traffic_light_group_id
    yaw: float    # direction of the traffic it stops


@dataclass
class ShieldMap:
    stop_lines: list[StopLine] = field(default_factory=list)
    junctions: list[list[tuple[float, float]]] = field(default_factory=list)
    crosswalks: list[list[tuple[float, float]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index()

    def _index(self) -> None:
        self._grid: dict[tuple[int, int], list[tuple[str, int]]] = {}
        for kind, polys in (("j", self.junctions), ("c", self.crosswalks)):
            for i, poly in enumerate(polys):
                for cell in _cells_of([p[0] for p in poly], [p[1] for p in poly]):
                    self._grid.setdefault(cell, []).append((kind, i))
        self._line_grid: dict[tuple[int, int], list[int]] = {}
        for i, sl in enumerate(self.stop_lines):
            for cell in _cells_of([sl.x0, sl.x1], [sl.y0, sl.y1]):
                self._line_grid.setdefault(cell, []).append(i)

    def zone_at(self, x: float, y: float, kinds: str = "jc") -> bool:
        """Inside a junction (j) or a crosswalk (c)?"""
        for kind, i in self._grid.get(_cell(x, y), ()):
            if kind in kinds and _inside((self.junctions if kind == "j" else self.crosswalks)[i], x, y):
                return True
        return False

    def stop_lines_near(self, x: float, y: float) -> list[StopLine]:
        return [self.stop_lines[i] for i in self._line_grid.get(_cell(x, y), ())]


def _cell(x: float, y: float) -> tuple[int, int]:
    return int(math.floor(x / GRID_M)), int(math.floor(y / GRID_M))


def _cells_of(xs, ys):
    (cx0, cy0), (cx1, cy1) = _cell(min(xs), min(ys)), _cell(max(xs), max(ys))
    return [(cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)]


def _inside(poly, x: float, y: float) -> bool:
    hit, j = False, len(poly) - 1
    for i in range(len(poly)):
        (xi, yi), (xj, yj) = poly[i], poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit


def segment_cross(ax, ay, bx, by, sl: StopLine) -> float | None:
    """Fraction along a->b where it crosses the stop line, or None."""
    dx, dy = bx - ax, by - ay
    ex, ey = sl.x1 - sl.x0, sl.y1 - sl.y0
    den = dx * ey - dy * ex
    if abs(den) < 1e-9:
        return None
    t = ((sl.x0 - ax) * ey - (sl.y0 - ay) * ex) / den
    u = ((sl.x0 - ax) * dy - (sl.y0 - ay) * dx) / den
    return t if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else None


def load_shield_map(map_path) -> ShieldMap | None:
    """The map under ``map_path`` (a directory or an .osm), or None without one."""
    if map_path is None:
        return None
    try:
        from autoware_lanelet2_extension_python.projection import MGRSProjector
        from lanelet2.io import Origin, load
    except Exception:  # noqa: BLE001
        return None
    path = Path(map_path).expanduser()
    osm = [path] if path.suffix == ".osm" else sorted(path.glob("*.osm"))
    if not osm:
        return None
    lmap = load(str(osm[0]), MGRSProjector(Origin(0.0, 0.0)))

    def attr(e, k):
        return str(e.attributes[k]) if k in e.attributes else ""

    out = ShieldMap.__new__(ShieldMap)
    out.stop_lines, out.junctions, out.crosswalks = [], [], []
    seen = set()
    for ll in lmap.laneletLayer:
        poly = [(float(p.x), float(p.y)) for p in ll.polygon2d()]
        if attr(ll, "subtype") == "crosswalk":
            if len(poly) >= 3:
                out.crosswalks.append(poly)
            continue
        if attr(ll, "turn_direction") and len(poly) >= 3:
            out.junctions.append(poly)
        for reg in ll.regulatoryElements:
            if attr(reg, "subtype") != "traffic_light" or "ref_line" not in reg.parameters:
                continue
            for line in reg.parameters["ref_line"]:
                pts = [(float(p.x), float(p.y)) for p in line]
                if len(pts) < 2 or (reg.id, line.id) in seen:
                    continue
                seen.add((reg.id, line.id))
                mx, my = (pts[0][0] + pts[-1][0]) / 2.0, (pts[0][1] + pts[-1][1]) / 2.0
                yaw = _lane_yaw([(float(p.x), float(p.y)) for p in ll.centerline], mx, my)
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    out.stop_lines.append(StopLine(x0, y0, x1, y1, int(reg.id), yaw))
    out._index()
    return out


def _lane_yaw(center, x: float, y: float) -> float:
    best = (float("inf"), 0.0)
    for (ax, ay), (bx, by) in zip(center, center[1:]):
        d2 = ((ax + bx) / 2.0 - x) ** 2 + ((ay + by) / 2.0 - y) ** 2
        if d2 < best[0]:
            best = (d2, math.atan2(by - ay, bx - ax))
    return best[1]


class SignalTimeline:
    """The recorded lights: for each group, the lamps lit from each change on."""

    def __init__(self, data) -> None:
        self._t: dict[int, list[float]] = {}
        self._s: dict[int, list[frozenset]] = {}
        for stamp, msg in data:
            t = stamp * 1e-9 if stamp > 1e12 else float(stamp)
            for g in msg.traffic_light_groups:
                lit = frozenset(
                    (int(e.color), int(e.shape)) for e in g.elements if int(e.status) != SOLID_OFF
                )
                gid = int(g.traffic_light_group_id)
                states = self._s.setdefault(gid, [])
                if not states or states[-1] != lit:
                    self._t.setdefault(gid, []).append(t)
                    states.append(lit)

    def __bool__(self) -> bool:
        return bool(self._t)

    def state(self, group: int, t: float) -> frozenset | None:
        times = self._t.get(group)
        if not times:
            return None
        i = bisect.bisect_right(times, t) - 1
        return self._s[group][max(i, 0)] if i >= 0 else None


def permits(now: frozenset | None, recorded: frozenset | None, turn: str) -> bool:
    """May an NPC cross under ``now``, given it crossed under ``recorded``?

    A green circle, or a green arrow for its turn, lets it go; so does the very
    state it crossed under in the recording (an amber it took, an arrow the
    turn guess misses). Anything else stops it. Unknown lights stop no one.
    """
    if not now or recorded is None:
        return True
    if (GREEN, CIRCLE) in now:
        return True
    if any(c == GREEN and s in ARROWS_FOR_TURN.get(turn, ()) for c, s in now):
        return True
    return now == recorded
