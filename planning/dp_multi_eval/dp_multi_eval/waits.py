"""Why a standing ego waited: justified waits, without ROS.

A stop is the model's decision; the tool only asks whether the scene gave it a
reason. Lag and stop rules that ignored the reason failed correct driving: on
H78 (2026-11-10) the ego waited 28.9 s at a stop line while pedestrians and
cyclists crossed 13-19 m ahead, left 20.7 s behind the recording and failed
for "not keeping up".

Each STEP_SEC of a stop is justified when, at that instant:

- ``red_light``: the light ahead is red or amber (the caller's ``light_red``);
- ``vru``: a pedestrian or cyclist is on the ego's way ahead, or its
  straight-line motion enters it within ENTER_SEC;
- ``crosswalk``: a pedestrian or cyclist is on, or at the kerb of
  (CROSSWALK_NEAR_M), a crosswalk the ego's way crosses ahead — a driver
  yields to people crossing or about to cross there, whichever lane they
  are in yet;
- ``vehicle``: a vehicle stands or drives on the ego's way just ahead (a
  queue, a car still in the junction);
- ``crossing``: a moving vehicle not going the ego's way enters it within
  ENTER_SEC (traffic with priority: cross traffic, an oncoming car across a
  right turn).

An object the shield held for the ego (it waited for the ego) is no reason:
the two waited for each other, and a real road user waits for a bus in its way
too — the bus has to go (operator, 2026-11-10: a mutual wait fails).

The ego's way is the RECORDED path ahead of it (the route the ego follows),
swept by the ego's body. A vehicle the recorded bus drove round lies off it
(it must reach VEHICLE_OVERLAP_M into the swept body to count), so a parked
truck the model would not pass is no reason (H95, H63: the human
driver went round a stopped vehicle the model waited behind).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Callable

STEP_SEC = 0.5          # a stop is judged in steps of this
ENTER_SEC = 3.0         # an object moving into the ego's way this soon counts
VRU_LABELS = frozenset({"PEDESTRIAN", "BICYCLE"})
VRU_AHEAD_M = 25.0      # a walker or cyclist this far ahead of the ego's front
VRU_MARGIN_M = 1.0      # ...and this close to its swept body (about to step in)
VRU_MOVING_MPS = 0.3
VEHICLE_AHEAD_M = 15.0  # a vehicle on the way this far ahead of the front: a queue
VEHICLE_OVERLAP_M = 0.5  # ...reaching this far into the swept body: the recorded bus brushed
                         # past a parked car with ~0 m to spare, and tracker boxes jitter by
                         # tenths (H63, 2026-11-10); a queued car overlaps by a metre or more
CROSS_AHEAD_M = 30.0    # a crossing vehicle entering the way this far ahead
CROSS_MARGIN_M = 0.5
CROSS_MOVING_MPS = 1.0
CROSS_MIN_DEG = 30.0    # heading off the way's by this much: not the ego's traffic
BEHIND_FRONT_M = 1.0    # the way starts this far behind the ego's front
SNAP_M = 3.0            # the recorded path is the ego's way only this close to it
CROSSWALK_NEAR_M = 2.0  # a walker or cyclist this close to such a crosswalk is at its kerb
FRAME_TOL_SEC = 1.5     # the last object frame stays in force this long (the recorder drops some)
WAY_STEP_M = 0.5
VEL_DT_SEC = 0.6        # an object's velocity from its pose this long before


@dataclass
class EgoBody:
    front_m: float = 8.4      # base_link to the front bumper
    half_width_m: float = 1.44


def body_from_footprint(footprint) -> EgoBody:
    """The front reach and half width of a base_link footprint polygon."""
    if not footprint:
        return EgoBody()
    return EgoBody(
        front_m=max(p[0] for p in footprint),
        half_width_m=max(abs(p[1]) for p in footprint),
    )


class ObjectFrames:
    """Object samples grouped by frame stamp: ``at(t)`` is the frame in force
    at t, ``velocity(id, t)`` the object's finite-difference velocity."""

    def __init__(self, objects: list[Any]) -> None:
        frames: dict[float, dict[str, Any]] = {}
        for o in objects:
            frames.setdefault(o.stamp_sec, {})[o.object_id] = o
        self.t = sorted(frames)
        self.f = [frames[t] for t in self.t]

    def at(self, t: float, tol: float = FRAME_TOL_SEC) -> dict[str, Any]:
        k = bisect.bisect_right(self.t, t + 1e-6) - 1
        if k < 0 or t - self.t[k] > tol:
            return {}
        return self.f[k]

    def velocity(self, oid: str, t: float) -> tuple[float, float]:
        k = bisect.bisect_right(self.t, t + 1e-6) - 1
        if k < 0 or oid not in self.f[k]:
            return 0.0, 0.0
        now = self.f[k][oid]
        j = bisect.bisect_right(self.t, self.t[k] - VEL_DT_SEC + 1e-6) - 1
        while j >= 0 and self.t[k] - self.t[j] < 2.0:
            prev = self.f[j].get(oid)
            if prev is not None and self.t[k] - self.t[j] > 0.05:
                dt = self.t[k] - self.t[j]
                return (now.x - prev.x) / dt, (now.y - prev.y) / dt
            j -= 1
        return 0.0, 0.0


@dataclass
class Way:
    """The ego's way: points (x, y, yaw) every WAY_STEP_M from its position."""

    pts: list[tuple[float, float, float]] = field(default_factory=list)

    def frame(self, px: float, py: float) -> tuple[float, float, float] | None:
        """(station along the way, lateral distance, way yaw) of a point, or
        None when it lies beyond either end."""
        best = None
        for k in range(len(self.pts) - 1):
            ax, ay, ayaw = self.pts[k]
            bx, by, _ = self.pts[k + 1]
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            if seg2 <= 1e-9:
                continue
            u = ((px - ax) * dx + (py - ay) * dy) / seg2
            if (u < 0.0 and k > 0) or (u > 1.0 and k < len(self.pts) - 2):
                continue
            if u < -0.5 or u > 1.5:
                continue
            cx, cy = ax + u * dx, ay + u * dy
            d = math.hypot(px - cx, py - cy)
            if best is None or d < best[1]:
                best = (k * WAY_STEP_M + u * math.sqrt(seg2), d, math.atan2(dy, dx))
        return best


def build_way(ref: list[tuple[float, float, float]], x: float, y: float, yaw: float, length_m: float) -> Way:
    """The recorded path from the ego's spot forward, ``length_m`` long. ``ref``
    = recorded (x, y, yaw) in time order. Off the recorded path (farther than
    SNAP_M, or facing another way) the way is straight ahead."""
    best, bi = SNAP_M ** 2, None
    for i, (rx, ry, ryaw) in enumerate(ref):
        d2 = (rx - x) ** 2 + (ry - y) ** 2
        if d2 < best and abs((ryaw - yaw + math.pi) % (2 * math.pi) - math.pi) <= math.pi / 2:
            best, bi = d2, i
    pts: list[tuple[float, float, float]] = []
    acc = 0.0
    if bi is not None:
        # walk the recording forward from the spot, resampled every WAY_STEP_M
        px, py = x, y
        pts.append((x, y, ref[bi][2]))
        for rx, ry, _ in ref[bi + 1:]:
            step = math.hypot(rx - px, ry - py)
            while step >= WAY_STEP_M and acc < length_m:
                f = WAY_STEP_M / step
                qx, qy = px + f * (rx - px), py + f * (ry - py)
                pts.append((qx, qy, math.atan2(qy - py, qx - px)))
                px, py = qx, qy
                acc += WAY_STEP_M
                step = math.hypot(rx - px, ry - py)
            if acc >= length_m:
                break
    if len(pts) < 2 or acc < length_m * 0.5:
        c, s = math.cos(yaw), math.sin(yaw)
        n = int(length_m / WAY_STEP_M) + 1
        pts = [(x + c * k * WAY_STEP_M, y + s * k * WAY_STEP_M, yaw) for k in range(n)]
    return Way(pts)


def _inside(poly, x: float, y: float) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _poly_dist(poly, x: float, y: float) -> float:
    """0 inside the polygon, else the distance to its boundary."""
    if _inside(poly, x, y):
        return 0.0
    best = float("inf")
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        u = 0.0 if L2 <= 1e-12 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
        best = min(best, math.hypot(x - ax - u * dx, y - ay - u * dy))
    return best


def crosswalks_on_way(way: Way, body: EgoBody, crosswalks) -> list:
    """The crosswalk polygons the way crosses between the ego's front and
    VRU_AHEAD_M beyond it."""
    lo, hi = body.front_m - BEHIND_FRONT_M, body.front_m + VRU_AHEAD_M
    pts = [p for k, p in enumerate(way.pts) if lo <= k * WAY_STEP_M <= hi]
    out = []
    for poly in crosswalks or []:
        xs, ys = [q[0] for q in poly], [q[1] for q in poly]
        if not any(min(xs) <= px <= max(xs) and min(ys) <= py <= max(ys) for px, py, _ in pts):
            continue
        if any(_inside(poly, px, py) for px, py, _ in pts):
            out.append(poly)
    return out


def _across(o: Any, way_yaw: float) -> tuple[float, float]:
    """Half extents of an object box across and along the way."""
    a = o.yaw_rad - way_yaw
    hl, hw = 0.5 * o.length_m, 0.5 * o.width_m
    return abs(math.sin(a)) * hl + abs(math.cos(a)) * hw, abs(math.cos(a)) * hl + abs(math.sin(a)) * hw


def _on_way(way: Way, body: EgoBody, o: Any, x: float, y: float, margin: float, ahead: float) -> bool:
    fr = way.frame(x, y)
    if fr is None:
        return False
    s, d, wyaw = fr
    across, along = _across(o, wyaw)
    lo, hi = body.front_m - BEHIND_FRONT_M, body.front_m + ahead
    return d - across <= body.half_width_m + margin and s + along >= lo and s - along <= hi


def _entering(way: Way, body: EgoBody, o: Any, vx: float, vy: float, margin: float, ahead: float) -> bool:
    k = 1
    while k * STEP_SEC <= ENTER_SEC + 1e-6:
        tau = k * STEP_SEC
        if _on_way(way, body, o, o.x + vx * tau, o.y + vy * tau, margin, ahead):
            return True
        k += 1
    return False


def reasons_at(
    frames: ObjectFrames, way: Way, body: EgoBody, t: float, crosswalks=(), held_for_ego=None,
) -> set[str]:
    """The object reasons (vru, crosswalk, vehicle, crossing) a standing ego
    had at t. ``crosswalks``: the polygons the way crosses (crosswalks_on_way)."""
    out: set[str] = set()
    x0, y0 = way.pts[0][0], way.pts[0][1]
    reach = (len(way.pts) - 1) * WAY_STEP_M + 10.0
    for oid, o in frames.at(t).items():
        if math.hypot(o.x - x0, o.y - y0) > reach + 30.0:
            continue   # nothing that far can be on the way, or reach it in ENTER_SEC
        if held_for_ego is not None and held_for_ego(oid, t):
            continue   # it was waiting for the ego
        vx, vy = frames.velocity(oid, t)
        v = math.hypot(vx, vy)
        if math.hypot(o.x - x0, o.y - y0) > reach + v * ENTER_SEC:
            continue
        if o.label in VRU_LABELS:
            if _on_way(way, body, o, o.x, o.y, VRU_MARGIN_M, VRU_AHEAD_M) or (
                v >= VRU_MOVING_MPS and _entering(way, body, o, vx, vy, VRU_MARGIN_M, VRU_AHEAD_M)
            ):
                out.add("vru")
            elif any(_poly_dist(poly, o.x, o.y) <= CROSSWALK_NEAR_M for poly in crosswalks):
                out.add("crosswalk")
            continue
        if _on_way(way, body, o, o.x, o.y, -VEHICLE_OVERLAP_M, VEHICLE_AHEAD_M):
            out.add("vehicle")
            continue
        if v >= CROSS_MOVING_MPS:
            fr = way.frame(o.x, o.y)
            heading = math.atan2(vy, vx)
            ref_yaw = fr[2] if fr is not None else way.pts[0][2]
            off = abs((heading - ref_yaw + math.pi) % (2 * math.pi) - math.pi)
            if off >= math.radians(CROSS_MIN_DEG) and _entering(way, body, o, vx, vy, CROSS_MARGIN_M, CROSS_AHEAD_M):
                out.add("crossing")
    return out


def _context(x, y, yaw, ref, body, crosswalks):
    way = build_way(ref or [], x, y, yaw, body.front_m + max(VRU_AHEAD_M, CROSS_AHEAD_M) + 5.0)
    return way, crosswalks_on_way(way, body, crosswalks)


def _why(t, x, y, yaw, way, walks, frames, body, light_red, held_for_ego=None) -> set[str]:
    why: set[str] = set()
    if light_red is not None and light_red(t, x, y, yaw):
        why.add("red_light")
    if frames is not None:
        why |= reasons_at(frames, way, body, t, walks, held_for_ego)
    return why


def held_index(holds, reasons=("ego", "plan"), tol_sec: float = 0.6):
    """``held_for_ego(object_id, t)`` from shield log entries (dicts with t,
    id, reason; written every 0.5 s per held object)."""
    by: dict[str, list[float]] = {}
    for h in holds or []:
        if h.get("reason") in reasons:
            by.setdefault(str(h["id"]), []).append(float(h["t"]))
    for v in by.values():
        v.sort()

    def held_for_ego(oid: str, t: float) -> bool:
        ts = by.get(oid)
        if not ts:
            return False
        k = bisect.bisect_left(ts, t - tol_sec)
        return k < len(ts) and ts[k] <= t + tol_sec

    return held_for_ego


def explain_stop(
    t0: float, t1: float, x: float, y: float, yaw: float, *,
    frames: ObjectFrames | None, ref: list[tuple[float, float, float]] | None, body: EgoBody,
    light_red: Callable[[float, float, float, float], bool] | None = None,
    crosswalks=None,
    held_for_ego=None,
) -> dict[str, Any]:
    """How much of the stop (t0, t1) at (x, y, yaw) was justified, and by what:
    ``{"sec", "justified_sec", "reasons": {reason: sec}}``. A step counts once,
    whatever number of reasons it had."""
    way, walks = _context(x, y, yaw, ref, body, crosswalks)
    reasons: dict[str, float] = {}
    justified = 0.0
    t = t0
    while t < t1 - 1e-6:
        why = _why(t, x, y, yaw, way, walks, frames, body, light_red, held_for_ego)
        if why:
            justified += STEP_SEC
            for r in why:
                reasons[r] = reasons.get(r, 0.0) + STEP_SEC
        t += STEP_SEC
    sec = max(0.0, t1 - t0)
    return {
        "sec": round(sec, 1),
        "justified_sec": round(min(justified, sec), 1),
        "reasons": {k: round(v, 1) for k, v in sorted(reasons.items())},
    }


def unjustified_tail_sec(
    t0: float, t1: float, x: float, y: float, yaw: float, *,
    frames: ObjectFrames | None, ref: list[tuple[float, float, float]] | None, body: EgoBody,
    light_red: Callable[[float, float, float, float], bool] | None = None,
    crosswalks=None,
    held_for_ego=None,
) -> float:
    """Seconds the ego had stood without a reason when the stop (t0, t1) ended:
    walked back from t1 to the last justified step (or t0)."""
    way, walks = _context(x, y, yaw, ref, body, crosswalks)
    t = t1 - STEP_SEC
    while t >= t0 - 1e-6:
        if _why(t, x, y, yaw, way, walks, frames, body, light_red, held_for_ego):
            return round(max(0.0, t1 - (t + STEP_SEC)), 1)
        t -= STEP_SEC
    return round(max(0.0, t1 - t0), 1)


def mutual_wait_sec(
    t0: float, t1: float, x: float, y: float, yaw: float, *, held_near: Callable[[float], bool],
    frames: ObjectFrames | None, ref: list[tuple[float, float, float]] | None, body: EgoBody,
    light_red: Callable[[float, float, float, float], bool] | None = None,
    crosswalks=None, held_for_ego=None,
) -> float:
    """Seconds of the stop (t0, t1) the ego stood with no reason while an NPC
    the shield held for it stood near (``held_near(t)``): the two waited for
    each other. A real road user waits for a bus in its way too; the bus has
    to go (operator, 2026-11-10: such a mutual wait fails)."""
    way, walks = _context(x, y, yaw, ref, body, crosswalks)
    sec = 0.0
    t = t0
    while t < t1 - 1e-6:
        if held_near(t) and not _why(t, x, y, yaw, way, walks, frames, body, light_red, held_for_ego):
            sec += STEP_SEC
        t += STEP_SEC
    return round(sec, 1)


def decimate_ref(ref_ego: list[Any], step_m: float = 0.5) -> list[tuple[float, float, float]]:
    """Recorded ego samples as (x, y, yaw) every ``step_m`` of travel."""
    out: list[tuple[float, float, float]] = []
    for e in ref_ego:
        if not out or math.hypot(e.x - out[-1][0], e.y - out[-1][1]) >= step_m:
            out.append((e.x, e.y, e.yaw_rad))
    return out
