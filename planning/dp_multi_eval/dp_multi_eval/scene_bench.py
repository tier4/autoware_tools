"""Offline bench for the scene player: no planner, no simulator, no GPU.

Every recording is replayed through the very object pipeline the scene driver
uses (scene_driver.build_scene: shield, map, lights, pasts, audit), while the
ego is driven by a scripted program along the recorded path. What the bench
measures depends on the scene player only, so a change to it is judged on all
recordings at once, the same for every model:

| Program | The ego ... | What it exercises |
|---|---|---|
| recorded | drives exactly as recorded | control: the scene must stay the recording |
| lag<N> | stands N s at the start, then drives as recorded | traffic that followed the bus comes up behind |
| ahead<N> | drives N s ahead of the recording | objects the recording had not seen yet |
| zone_stops<N> | stands N s on entering every junction and crosswalk | NPCs whose way the ego blocks mid-junction |

In every program the ego also stops for what stands on its way (a generic
obstacle stop: its body swept along the recorded path for its stopping
distance plus 3 m comes within 0.5 m of it); it does not stop for what drives
away ahead of it.

| Measure | Meaning |
|---|---|
| contact_npc_frames | frames an NPC whose own step that frame brought it nearer is within 0.1 m of the ego's body (it drove into the ego) |
| contact_frames | the same with the ego moving into a standing one (the scripted ego only stops for its way ahead) |
| deadlock_sec | the ego stopped for an NPC the shield holds for the ego (or behind one held so) |
| lag_end_sec | how far behind the recording the ego ended |
| trapped_sec | NPC-seconds held for the ego inside a junction or on a crosswalk |
| off_rec_sec | NPC-seconds shown more than 2 m from their recording, within 40 m of the ego |
| audit.* | the scene audit's findings (stand_in_zone, pop_in, vanish, jump, through; blip apart) |

    ros2 run dp_multi_eval scene_bench.py --bags H78 H95 --map ~/autoware_map/hiratsuka-map
    ros2 run dp_multi_eval scene_bench.py --all --map ... --out bench.json --baseline old.json

Recordings are read once (three topics) and cached under
~/.cache/dp_multi_eval/scene_bench.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

OBJECTS_TOPIC = "/perception/object_recognition/tracking/objects"
ODOM_TOPIC = "/localization/kinematic_state"
SIGNALS_TOPIC = "/perception/traffic_light_recognition/traffic_signals"
CACHE_VERSION = "v1"
DEFAULT_PROGRAMS = ("recorded", "lag15", "ahead8", "zone_stops20")

CONTACT_M = 0.1
NEAR_M = 40.0
OFF_REC_M = 2.0
PLAN_SEC = 5.0
STOP_STANDOFF_M = 3.0     # the generic obstacle stop stands this far off
STOP_CLEAR_M = 0.5        # ... from anything its body would come this close to
STOP_DECEL = 2.5
HOLE_SEC = 2.0            # window_eval.GAP_SEC: a hole longer than this moves the scene
# Measures that must not grow (with the slack allowed before a growth counts).
GATED = {"contact_npc_frames": 2, "deadlock_sec": 2.0, "trapped_sec": 5.0,
         "audit.stand_in_zone": 0, "audit.pop_in": 2, "audit.vanish": 0, "audit.jump": 0, "audit.through": 0}


# --------------------------------------------------------------------------- recording
def _cache_path(bag: Path) -> Path:
    from dp_multi_eval.bag_cache import bag_fingerprint

    d = Path.home() / ".cache" / "dp_multi_eval" / "scene_bench"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{bag_fingerprint(bag)}_{CACHE_VERSION}.pkl"


def load_recording(bag: Path) -> SimpleNamespace:
    """objects / odom / signals as [(stamp ns, msg)], and the objects' raw
    bytes (a fresh message per frame: the shield writes into what it gets)."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    path = _cache_path(bag)
    raw = None
    if path.is_file():
        try:
            raw = pickle.loads(path.read_bytes())
        except Exception:  # noqa: BLE001
            raw = None
    if raw is None:
        from rosbag2_py import StorageFilter

        from dp_multi_eval.bag_reader import open_bag_reader

        reader = open_bag_reader(bag)
        types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        topics = [t for t in (OBJECTS_TOPIC, ODOM_TOPIC, SIGNALS_TOPIC) if t in types]
        reader.set_filter(StorageFilter(topics=topics))
        raw = {"types": {t: types[t] for t in topics}, "rows": {t: [] for t in topics}}
        while reader.has_next():
            topic, data, stamp = reader.read_next()
            raw["rows"][topic].append((int(stamp), bytes(data)))
        path.write_bytes(pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL))
    types = raw["types"]

    def decode(topic):
        if topic not in types:
            return []
        cls = get_message(types[topic])
        return [(t, deserialize_message(d, cls)) for t, d in raw["rows"][topic]]

    signals = []
    if types.get(SIGNALS_TOPIC, "").endswith("TrafficLightGroupArray"):
        signals = decode(SIGNALS_TOPIC)
    obj_cls = get_message(types[OBJECTS_TOPIC])
    return SimpleNamespace(
        objects=decode(OBJECTS_TOPIC), odom=decode(ODOM_TOPIC), signals=signals,
        raw_objects=raw["rows"][OBJECTS_TOPIC],
        fresh=lambda d: deserialize_message(d, obj_cls),
    )


# --------------------------------------------------------------------------- ego programs
class RecordedPath:
    def __init__(self, odom) -> None:
        self.t = [s * 1e-9 for s, _ in odom]
        self.x = [m.pose.pose.position.x for _, m in odom]
        self.y = [m.pose.pose.position.y for _, m in odom]
        self.q = [m.pose.pose.orientation for _, m in odom]
        self.v = [math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y) for _, m in odom]

    def index(self, tau: float) -> int:
        return min(max(bisect.bisect_left(self.t, tau), 0), len(self.t) - 1)

    def pose(self, tau: float) -> Any:
        i = self.index(tau)
        return SimpleNamespace(position=SimpleNamespace(x=self.x[i], y=self.y[i], z=0.0), orientation=self.q[i])

    @staticmethod
    def yaw(q) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Program:
    """The ego along the recorded path: its recorded time tau goes on at 1x
    unless the program holds it or something stands on its way (a generic
    obstacle stop, the same for every model)."""

    def __init__(self, name: str, path: RecordedPath, shield_map) -> None:
        self.name, self.path, self.map = name, path, shield_map
        self.kind = name.rstrip("0123456789.")
        if self.kind not in ("recorded", "lag", "ahead", "zone_stops"):
            raise ValueError(f"unknown program {name}")
        digits = name[len(self.kind):]
        self.n = float(digits) if digits else 0.0
        self.t0, self.t1 = path.t[0], path.t[-1]
        self.tau = min(self.t0 + (self.n if self.kind == "ahead" else 0.0), self.t1)
        self.hold_until = self.t0 + self.n if self.kind == "lag" else -math.inf
        self.in_zone = False

    def step(self, t: float, dt: float, blocked: bool) -> tuple[float, float]:
        """(tau, rate) at scene time t: rate 0 while the ego stands."""
        if t < self.hold_until or blocked or self.tau >= self.t1:
            return self.tau, 0.0
        self.tau = min(self.tau + dt, self.t1)
        if self.kind == "zone_stops":
            p = self.path.pose(self.tau)
            yaw = self.path.yaw(p.orientation)
            # the bus's middle (base_link + 2.1 m) inside a junction or on a crosswalk
            cx, cy = p.position.x + 2.1 * math.cos(yaw), p.position.y + 2.1 * math.sin(yaw)
            now_in = self.map is not None and self.map.zone_at(cx, cy)
            if now_in and not self.in_zone and self.path.v[self.path.index(self.tau)] > 1.0:
                self.hold_until = t + self.n
            self.in_zone = now_in
        return self.tau, 1.0

    def plan(self, tau: float, standing: bool) -> list[tuple[float, float, float, float]]:
        """The ego's next PLAN_SEC as (s from now, x, y, yaw): along the recording
        when it drives, its own pose while it stands."""
        if standing:
            p = self.path.pose(tau)
            return [(0.0, p.position.x, p.position.y, self.path.yaw(p.orientation))]
        out, i = [], self.path.index(tau)
        while i < len(self.path.t) and self.path.t[i] - tau <= PLAN_SEC:
            out.append((self.path.t[i] - tau, self.path.x[i], self.path.y[i], self.path.yaw(self.path.q[i])))
            i += 5
        return out


# --------------------------------------------------------------------------- one run
def _label(o) -> int:
    cls = getattr(o, "classification", None) or []
    return int(cls[0].label) if cls else 0


def _way(path: RecordedPath, tau: float, length: float) -> list[tuple[float, float, float]]:
    """(x, y, s) along the recorded path from the ego's rear axle."""
    i = path.index(tau)
    way, run = [(path.x[i], path.y[i], 0.0)], 0.0
    for j in range(i + 1, len(path.t)):
        d = math.hypot(path.x[j] - way[-1][0], path.y[j] - way[-1][1])
        if d < 0.3:
            continue
        run += d
        way.append((path.x[j], path.y[j], run))
        if run >= length:
            break
    return way


def _project(x: float, y: float, way) -> tuple[float, float] | None:
    """(s along the way, distance off it) of a point, or None if beside its ends."""
    best = None
    for (ax, ay, sa), (bx, by, sb) in zip(way, way[1:]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        f = 0.0 if L2 < 1e-9 else ((x - ax) * dx + (y - ay) * dy) / L2
        if f < -0.05 or f > 1.05:
            continue
        f = max(0.0, min(1.0, f))
        off = math.hypot(x - (ax + f * dx), y - (ay + f * dy))
        if best is None or off < best[1]:
            best = (sa + f * (sb - sa), off)
    return best


def _blocker(path, tau, speed, shown, prev_pos, shield):
    """The object a generic obstacle stop stands for: one the ego's body would
    come within STOP_CLEAR_M of along its way, up to its stopping distance plus
    a standoff (the body swept along the path, turns included) — not one
    driving away ahead of it."""
    look = STOP_STANDOFF_M + speed * speed / (2.0 * STOP_DECEL)
    way = _way(path, tau, look + 1.0)
    poses = []
    for (x, y, s), (nx, ny, _) in zip(way, way[1:]):
        poses.append(SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=0.0),
                                     orientation=_quat(math.atan2(ny - y, nx - x))))
    poses = poses[1::2] or poses
    for o in shown:
        p = o.kinematics.pose_with_covariance.pose.position
        if math.hypot(p.x - way[0][0], p.y - way[0][1]) > look + 20.0:
            continue
        # nearer than now along the way (something beside a standing bus that its
        # way leaves behind is no reason to stop)
        here = shield._rect_clearance(o, SimpleNamespace(
            position=SimpleNamespace(x=way[0][0], y=way[0][1], z=0.0), orientation=path.pose(tau).orientation))
        if not any((c := shield._rect_clearance(o, q)) < STOP_CLEAR_M and c < here - 0.05 for q in poses):
            continue
        key = bytes(o.object_id.uuid)
        prev = prev_pos.get(key)
        if prev is not None:
            ahead = _project(p.x, p.y, way)
            back = _project(prev[0], prev[1], way)
            if ahead and back and (ahead[0] - back[0]) / 0.1 >= 1.0 and ahead[0] > (
                    shield.p.real_center_offset_m + shield.p.real_length_m / 2.0 + STOP_STANDOFF_M):
                continue    # driving away ahead: followed, not stopped for
        return key
    return None


def _quat(yaw: float):
    return SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def run_program(rec: SimpleNamespace, map_path: str | None, program: str, overrides: dict | None = None,
                body: Any = None) -> dict:
    from dp_multi_eval.scene_driver import build_scene

    scene = build_scene(rec.objects, rec.odom, rec.signals, map_path, body=body)
    shield, gate, smap = scene.shield, scene.gate, scene.map
    for k, v in (overrides or {}).items():
        if not hasattr(shield.p, k):
            raise ValueError(f"no shield parameter {k}")
        setattr(shield.p, k, type(getattr(shield.p, k))(v) if not isinstance(getattr(shield.p, k), bool)
                else str(v).lower() in ("1", "true", "yes", "on"))
    path = RecordedPath(rec.odom)
    prog = Program(program, path, smap)
    out = {"contact_frames": 0, "contact_npc_frames": 0, "deadlock_sec": 0.0, "trapped_sec": 0.0,
           "off_rec_sec": 0.0, "lag_end_sec": 0.0, "held_sec": {}, "frames": 0,
           "contacts": [], "deadlocks": [], "findings": []}
    contact_seen: dict[bytes, dict] = {}
    dead: dict[bytes, dict] = {}
    t_first, t_end = path.t[0], path.t[-1]
    prev_t, prev_pos, shown, held = None, {}, [], {}
    tau, speed = prog.tau, 0.0
    for stamp, data in rec.raw_objects:
        t = stamp * 1e-9
        if t < t_first or t > t_end:
            continue
        if prev_t is not None and t - prev_t > HOLE_SEC:
            # A hole in the recording: the harness splits the leg there and
            # moves the scene (window_eval GAP_SEC) — object clocks start anew.
            gate.reset()
            prev_pos, shown = {}, []
        dt = 0.1 if prev_t is None else max(0.0, min(t - prev_t, 0.5))
        prev_t = t
        # the ego moves on what it saw last frame
        blocker = _blocker(path, tau, speed, shown, prev_pos, shield)
        tau, rate = prog.step(t, dt, blocker is not None)
        ego = path.pose(tau)
        speed = path.v[path.index(tau)] * rate
        if blocker is not None and held.get(blocker) in ("ego", "plan", "leader"):
            out["deadlock_sec"] += dt
            d = dead.setdefault(blocker, {"t_rel": round(t - t_first, 1), "id": blocker.hex()[:8],
                                          "why": held.get(blocker), "sec": 0.0})
            d["sec"] += dt
        shield.set_ego_plan(prog.plan(tau, rate == 0.0))
        msg = rec.fresh(data)
        raw = {bytes(o.object_id.uuid): (o.kinematics.pose_with_covariance.pose.position.x,
                                         o.kinematics.pose_with_covariance.pose.position.y) for o in msg.objects}
        shown = gate.apply(msg, ego_pose=ego, ego_speed=speed, ts_ns=stamp, at_end=False,
                           objects_on=True, now_ns=stamp).objects
        out["frames"] += 1
        held = dict(getattr(shield, "held_reason", {}) or {})
        for why in held.values():
            out["held_sec"][why] = out["held_sec"].get(why, 0.0) + dt
        touched = npc_touched = False
        pos = {}
        for o in shown:
            key = bytes(o.object_id.uuid)
            p = o.kinematics.pose_with_covariance.pose.position
            pos[key] = (p.x, p.y)
            label = _label(o)
            gap = shield._rect_clearance(o, ego)
            if label != 0 and gap <= CONTACT_M:
                touched = True
                pp = prev_pos.get(key)
                # its own step brought it nearer the ego (where the ego is now)
                moved = pp is not None and math.hypot(p.x - pp[0], p.y - pp[1]) > 0.05 and (
                    shield._rect_clearance(o, ego, obj_shift=(pp[0] - p.x, pp[1] - p.y)) > gap + 0.02)
                npc_touched |= moved
                c = contact_seen.get(key)
                if c is None:
                    rp = raw.get(key)
                    c = contact_seen[key] = {
                        "t_rel": round(t - t_first, 1), "id": key.hex()[:8], "label": label, "frames": 0,
                        "npc_frames": 0,
                        "src": "past" if key in scene.prebirth.active else "recorded" if key in raw else "kept",
                        "why": held.get(key, "-"), "ego_v": round(speed, 1),
                        "off_rec_m": round(math.hypot(p.x - rp[0], p.y - rp[1]), 1) if rp else None,
                        "at": [round(p.x, 1), round(p.y, 1)],
                    }
                c["frames"] += 1
                c["npc_frames"] += int(moved)
            if gap <= NEAR_M and key in raw and math.hypot(p.x - raw[key][0], p.y - raw[key][1]) > OFF_REC_M:
                out["off_rec_sec"] += dt
            if held.get(key) in ("ego", "plan") and smap is not None and smap.zone_at(p.x, p.y):
                out["trapped_sec"] += dt
        prev_pos = pos
        out["contact_frames"] += int(touched)
        out["contact_npc_frames"] += int(npc_touched)
    out["lag_end_sec"] = round(t_end - tau, 1)
    out["detours"] = [dict(d, t_rel=round(d["t"] - t_first, 1)) for d in getattr(shield, "detours", [])][:40]
    out["detour_count"] = {k: sum(1 for d in getattr(shield, "detours", []) if d["how"] == k) for k in ("margin", "shift", "blocked", "stuck")}
    audit = gate.audit.counts if gate.audit is not None else {}
    out["contacts"] = sorted(contact_seen.values(), key=lambda c: c["t_rel"])[:40]
    out["deadlocks"] = sorted(({**d, "sec": round(d["sec"], 1)} for d in dead.values() if d["sec"] >= 1.0),
                              key=lambda d: d["t_rel"])[:40]
    out["findings"] = [dict(f, t_rel=round(f["t"] - t_first, 1)) for f in (gate.audit.findings if gate.audit else [])
                       if f["audit"] != "blip"][:80]
    out["held_sec"] = {k: round(v, 1) for k, v in sorted(out["held_sec"].items())}
    for k in ("deadlock_sec", "trapped_sec", "off_rec_sec"):
        out[k] = round(out[k], 1)
    out["audit"] = dict(sorted(audit.items()))
    return out


# --------------------------------------------------------------------------- a GPU run's ego
def load_run(job_dir: Path) -> SimpleNamespace:
    """The ego of a finished job (output bag): its poses and speeds, its plans,
    the scene time of each (windows.json), and when objects were on."""
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
    from rosidl_runtime_py.utilities import get_message

    from dp_multi_eval.lag import scene_time_map

    job_dir = Path(job_dir)
    scene_time = scene_time_map(job_dir / "windows.json")
    if scene_time is None:
        raise SystemExit(f"{job_dir}: no windows.json (not a leg-window run)")
    r = SequentialReader()
    r.open(StorageOptions(uri=str(job_dir / "output"), storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    topics = [ODOM_TOPIC, "/planning/trajectory"]
    r.set_filter(StorageFilter(topics=[t for t in topics if t in types]))
    ego, plans = [], []
    while r.has_next():
        topic, data, stamp = r.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        wall = stamp * 1e-9
        if topic == ODOM_TOPIC:
            ego.append((wall, msg))
        else:
            pts = []
            for pt in msg.points:
                t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
                q = pt.pose.orientation
                pts.append((t, pt.pose.position.x, pt.pose.position.y,
                            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))))
            plans.append((wall, pts))
    payload = json.loads((job_dir / "windows.json").read_text(encoding="utf-8"))
    dark = []   # walls with objects off: a leg's opening until its departure reveal
    carry = []  # walls of scene jumps under an ego that was not moved (a cut it drove past)
    for leg in payload.get("legs", []):
        jumps = leg.get("jumps") or []
        if leg.get("wall_start") is None or not jumps:
            continue
        if any(sg.get("scene_jumped") for sg in leg.get("segments") or []):
            carry.append(float(jumps[0]["wall"]))
        else:
            dark.append((float(leg["wall_start"]), float(jumps[0]["wall"])))
    return SimpleNamespace(ego=ego, plans=plans, scene_time=scene_time, dark=dark, carry=carry, name=job_dir.name)


def replay_run(rec: SimpleNamespace, run: SimpleNamespace, map_path: str | None,
               trace: str | None = None, overrides: dict | None = None, body: Any = None) -> dict:
    """The scene played to the ego exactly as the model drove it in a GPU run:
    its poses, speeds and plans at 10 Hz, on the run's scene clock."""
    from dp_multi_eval.scene_driver import build_scene

    scene = build_scene(rec.objects, rec.odom, rec.signals, map_path, body=body)
    shield, gate, smap = scene.shield, scene.gate, scene.map
    for k, v in (overrides or {}).items():
        setattr(shield.p, k, type(getattr(shield.p, k))(v) if not isinstance(getattr(shield.p, k), bool)
                else str(v).lower() in ("1", "true", "yes", "on"))
    stamps = [s for s, _ in rec.raw_objects]
    plan_walls = [w for w, _ in run.plans]
    out = {"contact_npc_frames": 0, "contact_frames": 0, "trapped_sec": 0.0, "held_sec": {}, "frames": 0,
           "contacts": {}, "trace": []}
    wall0 = run.ego[0][0]
    last_wall, last_scene, prev_pos = None, None, {}
    for wall, odom in run.ego:
        if last_wall is not None and wall - last_wall < 0.1:
            continue
        dt = 0.1 if last_wall is None else min(wall - last_wall, 0.5)
        last_wall = wall
        t = run.scene_time(wall)
        if last_scene is not None and (t < last_scene - 0.05 or t - last_scene > 2.0):
            # the harness moved the scene (scene_driver._on_scene_time / _on_scene_jump)
            if any(abs(wall - c) <= 1.0 for c in getattr(run, "carry", [])):
                kept = gate.carry_jump(odom.pose.pose, t - last_scene, int(wall * 1e9))
                out.setdefault("carried", []).append(
                    {"t_rel": round(wall - wall0, 1), "jump_sec": round(t - last_scene, 1),
                     "kept": sorted(k.hex()[:8] for k in kept)})
            else:
                gate.reset()
        last_scene = t
        i = min(bisect.bisect_left(stamps, int(t * 1e9)), len(stamps) - 1)
        msg = rec.fresh(rec.raw_objects[i][1])
        j = bisect.bisect_right(plan_walls, wall) - 1
        if j >= 0 and wall - plan_walls[j] <= 1.0:
            shield.set_ego_plan(run.plans[j][1])
        ego = odom.pose.pose
        speed = math.hypot(odom.twist.twist.linear.x, odom.twist.twist.linear.y)
        dark = any(a <= wall < b for a, b in run.dark)
        shown = gate.apply(msg, ego_pose=ego, ego_speed=speed, ts_ns=int(t * 1e9), at_end=False,
                           objects_on=not dark, now_ns=int(wall * 1e9)).objects
        out["frames"] += 1
        held = dict(getattr(shield, "held_reason", {}) or {})
        for why in held.values():
            out["held_sec"][why] = out["held_sec"].get(why, 0.0) + dt
        touched = npc = False
        pos = {}
        for o in shown:
            key = bytes(o.object_id.uuid)
            p = o.kinematics.pose_with_covariance.pose.position
            pos[key] = (p.x, p.y)
            gap = shield._rect_clearance(o, ego)
            if _label(o) != 0 and gap <= CONTACT_M:
                touched = True
                pp = prev_pos.get(key)
                if pp is not None and math.hypot(p.x - pp[0], p.y - pp[1]) > 0.05 and (
                        shield._rect_clearance(o, ego, obj_shift=(pp[0] - p.x, pp[1] - p.y)) > gap + 0.02):
                    npc = True
                    out["contacts"][key.hex()[:8]] = out["contacts"].get(key.hex()[:8], 0) + 1
            if held.get(key) in ("ego", "plan") and smap is not None and smap.zone_at(p.x, p.y):
                out["trapped_sec"] += dt
            if trace and key.hex().startswith(trace):
                lin = o.kinematics.twist_with_covariance.twist.linear
                out["trace"].append((round(wall - wall0, 1), round(t - stamps[0] * 1e-9, 1), round(p.x, 1), round(p.y, 1),
                                     round(math.hypot(lin.x, lin.y), 1), round(gap, 2), held.get(key, "-"),
                                     round(speed, 1), round(shield.delays.get(key, 0.0), 1)))
        prev_pos = pos
        out["contact_frames"] += int(touched)
        out["contact_npc_frames"] += int(npc)
    out["trapped_sec"] = round(out["trapped_sec"], 1)
    out["held_sec"] = {k: round(v, 1) for k, v in sorted(out["held_sec"].items())}
    out["audit"] = dict(sorted((gate.audit.counts if gate.audit else {}).items()))
    out["findings"] = [dict(f, t_rel=round(f["t"] - wall0, 1)) for f in (gate.audit.findings if gate.audit else [])
                       if f["audit"] != "blip"][:60]
    out["detours"] = getattr(shield, "detours", [])
    return out


# --------------------------------------------------------------------------- runner
def _job(args: tuple) -> tuple[str, dict]:
    from dp_multi_eval.scene_driver import vehicle_body_for

    bag, map_path, programs, overrides, vehicle_model = args
    t0 = time.monotonic()
    rec = load_recording(Path(bag))
    load_sec = time.monotonic() - t0
    body = vehicle_body_for(vehicle_model)
    res = {}
    for prog in programs:
        t1 = time.monotonic()
        r = run_program(rec, map_path, prog, overrides, body=body)
        r["run_sec"] = round(time.monotonic() - t1, 1)
        res[prog] = r
    return Path(bag).name, {"load_sec": round(load_sec, 1), "programs": res}


def flat(r: dict) -> dict[str, float]:
    out = {k: r.get(k, 0) for k in ("contact_npc_frames", "contact_frames", "deadlock_sec", "trapped_sec",
                                    "off_rec_sec", "lag_end_sec")}
    for k, v in (r.get("audit") or {}).items():
        out[f"audit.{k}"] = v
    return out


def compare(new: dict, old: dict) -> list[str]:
    """Regressions of the gated measures, as lines; empty if none."""
    lines = []
    for bag, b in new.items():
        for prog, r in b["programs"].items():
            o = ((old.get(bag) or {}).get("programs") or {}).get(prog)
            if o is None:
                continue
            fn, fo = flat(r), flat(o)
            for k, slack in GATED.items():
                a, b_ = fo.get(k, 0), fn.get(k, 0)
                if b_ > a + slack:
                    lines.append(f"{bag:22s} {prog:14s} {k:20s} {a} -> {b_}")
    return lines


def table(results: dict) -> str:
    keys = ["contact_npc_frames", "contact_frames", "deadlock_sec", "trapped_sec", "off_rec_sec", "lag_end_sec",
            "audit.stand_in_zone", "audit.pop_in", "audit.vanish", "audit.jump", "audit.through", "audit.blip"]
    head = f"{'bag':22s} {'program':14s} " + " ".join(f"{k.replace('audit.', ''):>13s}" for k in keys)
    rows = [head, "-" * len(head)]
    for bag in sorted(results):
        for prog, r in results[bag]["programs"].items():
            f = flat(r)
            rows.append(f"{bag:22s} {prog:14s} " + " ".join(f"{f.get(k, 0):>13}" for k in keys))
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bags", nargs="*", default=[], help="bag names under --bag-dir, or paths")
    p.add_argument("--all", action="store_true", help="every bag under --bag-dir")
    p.add_argument("--bag-dir", default=None, help="folder of recordings (needed for --bags / --all)")
    p.add_argument("--map", dest="map_path", default=None)
    p.add_argument("--vehicle-model", default=None,
                   help="the ego's vehicle model (its body from <model>_description); default: the job's own, "
                   "else the pipeline's default")
    p.add_argument("--programs", nargs="*", default=list(DEFAULT_PROGRAMS))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--baseline", type=Path, default=None, help="earlier --out: exit 1 on a regression")
    p.add_argument("-j", "--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    p.add_argument("--set", nargs="*", default=[], metavar="PARAM=VALUE",
                   help="shield parameters to override (ShieldParams), e.g. spacetime=false deadlock_sec=0")
    p.add_argument("--replay-run", type=Path, default=None, metavar="JOB_DIR",
                   help="play the scene to the ego of a finished GPU job (its poses and plans), not a program")
    p.add_argument("--trace", default=None, help="with --replay-run: an object id prefix to trace")
    args = p.parse_args(argv)
    overrides = dict(kv.split("=", 1) for kv in args.set)
    from dp_multi_eval.job_config import JobConfig
    from dp_multi_eval.scene_driver import vehicle_body_for

    vehicle_model = args.vehicle_model or JobConfig.__dataclass_fields__["vehicle_model"].default
    if args.replay_run is not None:
        job = args.replay_run
        # The recording the job replayed: its own record of it, else by name under --bag-dir.
        status = job / "job_status.json"
        bag = json.loads(status.read_text(encoding="utf-8")).get("bag_path") if status.is_file() else None
        if not bag:
            if not args.bag_dir:
                p.error(f"{job} names no recording: give --bag-dir")
            bag = Path(args.bag_dir) / job.name
        rec = load_recording(Path(bag))
        res = replay_run(rec, load_run(job), args.map_path, trace=args.trace, overrides=overrides,
                         body=vehicle_body_for(vehicle_model))
        tr = res.pop("trace")
        print(json.dumps({k: v for k, v in res.items() if k not in ("findings",)}, indent=1, default=str))
        for f in res["findings"]:
            print("finding", {k: f[k] for k in f if k not in ("t",)})
        for row in tr:
            print("trace", row)
        return 0

    if args.all and not args.bag_dir:
        p.error("--all needs --bag-dir")
    bag_dir = Path(args.bag_dir or ".")
    names = sorted(d.name for d in bag_dir.iterdir() if d.is_dir()) if args.all else args.bags
    bags = [str(Path(b) if Path(b).is_dir() else bag_dir / b) for b in names]
    if not bags:
        p.error("no bags (--bags or --all)")
    jobs = [(b, args.map_path, args.programs, overrides, vehicle_model) for b in bags]
    results: dict[str, Any] = {}
    if args.jobs > 1 and len(jobs) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(args.jobs, len(jobs))) as pool:
            for name, res in pool.imap_unordered(_job, jobs):
                results[name] = res
                print(f"[bench] {name} done (load {res['load_sec']} s)", flush=True)
    else:
        for j in jobs:
            name, res = _job(j)
            results[name] = res
            print(f"[bench] {name} done (load {res['load_sec']} s)", flush=True)
    print(table(results))
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"[bench] written {args.out}")
    if args.baseline:
        reg = compare(results, json.loads(args.baseline.read_text(encoding="utf-8")))
        if reg:
            print("[bench] REGRESSIONS against the baseline:")
            print("\n".join(reg))
            return 1
        print("[bench] no regression against the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
