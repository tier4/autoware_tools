"""Scene driver: the ORIGINAL perception_reproducer replays the recording; the
driver decides which recorded instant it shows.

planning_debug_tools' perception_reproducer runs unmodified, in this process.
With ``-r 0`` it publishes, every tick, the recorded frame whose recorded ego
pose is nearest to the ego pose it subscribes to. That subscription is
remapped to FEED_TOPIC, where the driver publishes the recorded ego odometry
of the instant its clock picks (scene_clock.py). The pose is an exact
recorded one, so the nearest frame is that instant's (a standing recording
repeats a pose for at most 4 samples, 60 ms: H45 and 2026-04-28, measured
2026-10-28). Traffic lights and the recorded ego
(/perception_reproducer/rosbag_ego_odom) leave the reproducer as they are; its
objects are remapped to RAW_OBJECTS_TOPIC and finished here (objects off, the
recording's end, the reactive shield) before they reach the planner. The
recorded turn indicators are replayed by the ego's place (IndicatorClock).

| command line | clock |
|---|---|
| ``--window-clock`` | 1x while the harness drives the ego in; once the ego is on its own the scene follows its position and plays on in real time while it stands (RecordedClock rules 1-3 without the departure jump, rule 4 off) |
| no ``--window-clock`` | RecordedClock rules 1-4 (continuous runs) |

The harness talks to it as it talked to the forked reproducer (window_eval.py):
/perception_reproducer/{scene_time, hold, objects, follow_ego} in. A harness
jump always lands, follow mode or not. Out: NEAREST_TOPIC, the distance from
the ego to the nearest object it was shown (the harness keeps a lagging ego
in a scene that still has objects around it).
"""

from __future__ import annotations

import argparse
import bisect
import importlib
import math
import sys
from pathlib import Path
from typing import Any

from dp_multi_eval.prebirth import PreBirth
from dp_multi_eval.reactive_shield import ReactiveShield, ShieldParams
from dp_multi_eval.scene_audit import SceneAudit
from dp_multi_eval.scene_clock import RecordedClock, WindowClock
from dp_multi_eval.shield_map import load_shield_map

REPRODUCER_NODE = "perception_reproducer"
EGO_TOPIC = "/localization/kinematic_state"
FEED_TOPIC = "/dp_multi_eval/scene/ego_odom"
OBJECTS_TOPIC = "/perception/object_recognition/tracking/objects"
RAW_OBJECTS_TOPIC = "/dp_multi_eval/scene/tracked_objects"
SCENE_JUMP_TOPIC = "/perception_reproducer/scene_jump"   # a jump of recorded time under an ego that is not moved
PLAN_TOPIC = "/planning/trajectory"
NEAREST_TOPIC = "/perception_reproducer/nearest_object_m"
NO_OBJECT_M = 1e9   # NEAREST_TOPIC when nothing is shown
# A standing NPC does not set off into where the ego's plan takes it this far
# ahead (H78, 2026-11-09: 3 s did not yet reach the car's way, 5 s did).
EGO_PLAN_SEC = 5.0
DEADLOCK_SEC = 3.0    # ego and an NPC held for it both standing this long, it on the ego's way: resolve
INDICATOR_TOPIC = "/vehicle/status/turn_indicators_status"
# Twice the reproducer's tick (the recorded odometry's 50 Hz), so every tick
# finds the pose of the current instant, not the previous one.
FEED_PERIOD_SEC = 0.01
INDICATOR_PERIOD_SEC = 0.02
STATIONARY_MPS = 0.5      # past the recording's end only objects slower than this stay
END_STANDING_SEC = 1.0    # ... and only if their last this-many seconds of poses show it
JUMP_BLANK_SEC = 0.25     # after a scene jump under an unmoved ego, what is not kept is off this long:
                          # the planner's 3 s object histories start anew instead of showing every object leap
# Following, recorded time runs at most this times real time. 1.0: a faster
# rate fast-forwarded traffic — oncoming cars reporting 11 m/s moved at 16.7
# (H81, 2026-10-29). With the 1x floor (flow_when_behind) the scene runs at
# real time while the ego drives; it re-syncs at each leg's departure.
MAX_SCENE_RATE = 1.0
RECORDED_INDICATORS_VERSION = "v1"



def _yaw_of(q: Any) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _speed_of(obj: Any) -> float:
    tw = obj.kinematics.twist_with_covariance.twist.linear
    return math.hypot(tw.x, tw.y)


def frame_index(stamps_ns: list[int], ts_ns: int) -> int:
    """The recorded sample fed for scene instant ``ts_ns``: the first at or after
    it, the reproducer's own lookup rule for objects and lights."""
    return min(bisect.bisect_left(stamps_ns, int(ts_ns)), len(stamps_ns) - 1)


class SceneTimeline:
    """Which recorded instant the scene shows: the forked reproducer's clock
    wiring, moved out of the reproducer unchanged. No ROS."""

    def __init__(self, stamps_ns, xs, ys, speeds, yaws, *, window: bool, departure_hold: bool = True) -> None:
        self.t = [int(v) for v in stamps_ns]
        self._x = list(xs)
        self._y = list(ys)
        self._v = list(speeds)
        self._yaw = list(yaws)
        self.window = bool(window)
        self.departure_hold = bool(departure_hold)
        self.clock = (
            WindowClock(self.t)
            if self.window
            else RecordedClock(self.t, self._x, self._y, self._v, yaws=self._yaw)
        )
        self.hold = False
        self.follow = False
        self._pos_clock: RecordedClock | None = None
        self.ts: int | None = None

    @property
    def at_end(self) -> bool:
        return self.clock.at_end

    def jump(self, t_sec: float) -> None:
        """A harness jump. Following reseeds from the jumped instant (the
        departure reveal sends the jump and following-on together, in either
        order); the harness switches following off before a jump that must
        stand (a leg's opening)."""
        self.clock.jump(int(float(t_sec) * 1e9))
        self._pos_clock = None

    def set_follow(self, on: bool) -> bool:
        """The harness switches position-following per leg; True when it changed."""
        if on == self.follow:
            return False
        self.follow = on
        # On: seeded at the next tick from the window clock's instant. Off: the
        # window clock already holds the followed instant (every following
        # tick writes it there). Writing it again here overwrote a harness
        # jump that had just arrived: H81's third leg opened 24 s late, the
        # ego handed over at 6.3 m/s 100 m past its stop (2026-10-28).
        self._pos_clock = None
        return True

    def tick(
        self,
        now_ns: int,
        ego: tuple[float, float, float, float] | None,
        harness_hold: bool = False,
    ) -> int:
        """The scene instant for this tick. ``ego`` = (x, y, speed, yaw) of the
        simulated ego; until the simulator publishes it the clock does not run."""
        if ego is None:
            return self.ts if self.ts is not None else self.t[0]
        x, y, v, yaw = ego
        if self.window and self.follow and not self.hold:
            if self._pos_clock is None:
                seed = self.clock.tick(now_ns, hold=True)
                self._pos_clock = RecordedClock(
                    self.t, self._x, self._y, self._v, yaws=self._yaw,
                    real_time_floor=False, departure_jump=False, max_rate=MAX_SCENE_RATE,
                    flow_when_behind=True, hold_max_ns=None if self.departure_hold else 0,
                )
                # Never back before the instant the scene was at when the ego set off.
                self._pos_clock._floor = (int(seed), x, y)
                self._pos_clock._last = (int(seed), now_ns)
            ts = self._pos_clock.tick(x, y, v, now_ns, yaw=yaw, harness_hold=False)
            self.clock.jump(ts)   # keep the window clock at the same instant
            self.clock.tick(now_ns, hold=True)
        elif self.window:
            ts = self.clock.tick(now_ns, hold=self.hold)
        else:
            ts = self.clock.tick(x, y, v, now_ns, yaw=yaw, harness_hold=harness_hold)
        self.ts = int(ts)
        return self.ts


class ShieldLog:
    """What the shield held, and when the scene stood at the recording's end,
    as JSON lines beside the job's bag (shield_log.jsonl). The metrics read it:
    a stop the ego made facing an NPC the shield held for it, or after the
    recording ran out, is the scene's limit, not a verdict on the model."""

    PERIOD_SEC = 0.5   # each held NPC at most this often
    FLUSH_SEC = 2.0

    def __init__(self, path: str | None) -> None:
        self._f = open(path, "w", encoding="utf-8") if path else None
        self._last: dict[bytes, float] = {}
        self._at_end: bool | None = None
        self._flushed = 0.0

    def write(self, now_sec: float, objects, held_reason: dict, at_end: bool) -> None:
        if self._f is None:
            return
        import json

        if (self._at_end is None and at_end) or (self._at_end is not None and at_end != self._at_end):
            event = "recording_end" if at_end else "recording_resume"
            self._f.write(json.dumps({"t": round(now_sec, 3), "event": event}) + "\n")
        self._at_end = at_end
        for o in objects:
            key = bytes(o.object_id.uuid)
            why = held_reason.get(key)
            if not why or now_sec - self._last.get(key, -1e9) < self.PERIOD_SEC:
                continue
            self._last[key] = now_sec
            p = o.kinematics.pose_with_covariance.pose.position
            cls = getattr(o, "classification", None) or []
            self._f.write(json.dumps({
                "t": round(now_sec, 3), "id": key.hex(), "label": int(cls[0].label) if cls else 0,
                "reason": why, "x": round(p.x, 2), "y": round(p.y, 2),
                "l": round(float(o.shape.dimensions.x), 2), "w": round(float(o.shape.dimensions.y), 2),
            }) + "\n")
        if now_sec - self._flushed >= self.FLUSH_SEC:
            self._f.flush()
            self._flushed = now_sec

    def event(self, record: dict) -> None:
        """One line as given (the scene audit's findings)."""
        if self._f is not None:
            import json

            self._f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


class ObjectGate:
    """What of the reproducer's frame reaches the planner. No ROS: works on the
    message's ``objects`` list in place."""

    def __init__(self, *, shield: Any = None, log: ShieldLog | None = None, prebirth: Any = None,
                 audit: Any = None) -> None:
        self.shield = shield
        self.log = log
        self.prebirth = prebirth
        self.audit = audit
        self._end_logged = False
        self._last_ns: int | None = None
        self._kept: set[bytes] = set()          # vehicles kept over the last scene jump (carry_jump)
        self._blank_until_ns: int | None = None  # ... and until when everything else is off
        # Judged on each track's last poses, not its reported speed: the
        # tracker reports 0 for UNKNOWN objects whatever they do. A track
        # shorter than END_STANDING_SEC has no evidence that it stands: H95
        # (2026-11-10) a car born 0.2 s before the end, 44 m behind the bus,
        # reported 0.5 m/s and was frozen in the ego's lane for good.
        # Recorded poses only: a past the tool gave it (prebirth.py) is no evidence.
        tracks = getattr(shield, "tracks", None) or {}
        self._standing_at_end = None if not tracks else {
            k for k, tr in tracks.items()
            if tr.times[-1] - tr.times[getattr(tr, "ext", 0)] >= END_STANDING_SEC
            and ReactiveShield.end_speed(tr, END_STANDING_SEC) < STATIONARY_MPS
        }
        # A track the recording's end cut short inside that second is never
        # shown: it cannot be kept past the end, so all it could be is a car
        # flashing up and gone. H95 (2026-11-11): the tracker's 0.3 s double
        # of a car in the queue appeared 36 m ahead of a late ego, in the lane
        # it was about to pull out into, and vanished at the end.
        end = getattr(shield, "bag_end", None)
        self._born_at_end = set() if end is None else {
            k for k, tr in tracks.items() if end - tr.times[getattr(tr, "ext", 0)] < END_STANDING_SEC
        }

    def apply(
        self,
        msg: Any,
        *,
        ego_pose: Any,
        ego_speed: float,
        ts_ns: int,
        at_end: bool,
        objects_on: bool,
        now_ns: int,
    ) -> Any:
        if self._born_at_end:
            msg.objects = [o for o in msg.objects if bytes(o.object_id.uuid) not in self._born_at_end]
        blank = self._blank_until_ns is not None and now_ns < self._blank_until_ns
        if blank:
            msg.objects = [o for o in msg.objects if bytes(o.object_id.uuid) in self._kept]
        elif self._blank_until_ns is not None:
            self._blank_until_ns = None
        if at_end:
            # The recording has run out: what stood still is still there, what
            # was moving has no recorded future — a car frozen mid-road with
            # its last speed is not an input the vehicle ever saw (2026-04-28:
            # the ego stood beside one for 30 s).
            msg.objects = [
                o for o in msg.objects
                if _speed_of(o) < STATIONARY_MPS
                and (self._standing_at_end is None or bytes(o.object_id.uuid) in self._standing_at_end)
            ]
            if not self._end_logged:
                print(
                    f"RECORDING END: keeping {len(msg.objects)} stationary objects t={ts_ns * 1e-9:.3f}",
                    flush=True,
                )
                self._end_logged = True
        if not objects_on:
            msg.objects = []
        # Objects the ego is near enough to see before the recorded tracker
        # first reported them, on their way to where it did (prebirth.py): no
        # pop-ins in front of a lagging ego. They go through the shield like
        # the recorded ones — one path, one motion logic.
        if self.prebirth is not None and objects_on and not at_end and not blank and ego_pose is not None:
            present = {bytes(o.object_id.uuid) for o in msg.objects}
            msg.objects = list(msg.objects) + self.prebirth.objects_at(ts_ns * 1e-9, ego_pose, present)
        # NPCs the shield delayed live on to their own track's end, after the
        # recording (and so the frame) has let them go.
        if self.shield is not None and objects_on and not at_end and hasattr(self.shield, "carried"):
            present = {bytes(o.object_id.uuid) for o in msg.objects}
            msg.objects = list(msg.objects) + self.shield.carried(present, ts_ns * 1e-9)
        # The scene plays on while the ego stands, so cars that followed the
        # recorded bus come up behind a standing ego: the shield keeps every
        # NPC on its own recorded path but never inside the ego (it queues).
        # Nothing is hidden: a car the ego drives into is a contact.
        dt = max(0.01, min((now_ns - self._last_ns) * 1e-9 if self._last_ns else 0.1, 0.5))
        if self.shield is not None and msg.objects and ego_pose is not None:
            self.shield.apply(msg, ego_pose, ts_ns * 1e-9, dt, ego_speed)
        if self.log is not None:
            self.log.write(now_ns * 1e-9, msg.objects, getattr(self.shield, "held_reason", {}) or {}, at_end)
        if self.audit is not None and ego_pose is not None:
            for event in self.audit.check(
                now_ns * 1e-9, ts_ns * 1e-9, msg.objects if objects_on else None, ego_pose,
                held=getattr(self.shield, "held_reason", {}) or {},
                past=self.prebirth.active if self.prebirth is not None else (),
            ):
                if self.log is not None:
                    self.log.event(event)
        self._last_ns = now_ns
        return msg

    def reset(self) -> None:
        """The scene was moved (a leg boundary, a departure reveal)."""
        self._kept, self._blank_until_ns = set(), None
        if self.shield is not None:
            self.shield.reset()
        if self.prebirth is not None:
            self.prebirth.reset()
        if self.audit is not None:
            self.audit.reset()

    def carry_jump(self, ego_pose: Any, jump_sec: float, now_ns: int) -> set[bytes]:
        """The scene jumps ``jump_sec`` forward and the ego is not moved (it
        drove on past a recorded stand). The vehicles near it that are driving
        away stay, on their own clocks (ReactiveShield.carry_over); the rest is
        the new instant's, off for JUMP_BLANK_SEC first. Returns what stays."""
        carry = getattr(self.shield, "carry_over", None)
        if carry is None or ego_pose is None or jump_sec <= 0.0:
            self.reset()
            return set()
        self._kept = set(carry(ego_pose, jump_sec))
        self._blank_until_ns = int(now_ns + JUMP_BLANK_SEC * 1e9)
        if self.prebirth is not None:
            self.prebirth.reset()
        if self.audit is not None:
            self.audit.reset()
        return self._kept


class IndicatorReplay:
    """The recorded turn indicator at the scene's instant. No ROS.

    The bus fleet signals every departure and the models are conditioned on it
    (2026-09-01); psim's own report only mirrors the model's command. Only an
    active recorded signal is replayed — an earlier version borrowed the next
    one within 15 s and fed a LEFT for 12 s at a junction where the driver
    signalled nothing (2026-04-28). Elsewhere, with a merge topic, the
    vehicle's own report stands."""

    ACTIVE = (2, 3)   # ENABLE_LEFT, ENABLE_RIGHT

    def __init__(self, recorded: list[tuple[int, int]], *, merge: bool) -> None:
        self.t = [int(s) for s, _ in recorded]
        self.reports = [int(r) for _, r in recorded]
        self.merge = merge
        self.vehicle = 1   # DISABLE until the simulator reports a real state

    @property
    def enabled(self) -> bool:
        return len(self.t) > 1

    def on_vehicle_report(self, report: int) -> None:
        # psim mirrors the planner's command into its report, NO_COMMAND (0)
        # included, which a real vehicle never reports: 43 % of the model's
        # indicator input was 0 on 2026-04-28. Only a real state counts.
        if int(report) in (1, 2, 3):
            self.vehicle = int(report)

    def report_at(self, ts_ns: int) -> int | None:
        r = self.reports[min(bisect.bisect_left(self.t, int(ts_ns)), len(self.t) - 1)]
        if r in self.ACTIVE:
            return r
        return self.vehicle if self.merge else None


class IndicatorClock:
    """Which recorded instant's turn indicator the ego gets: the moment the
    recorded bus was where the ego is, not the scene's instant. No ROS.

    The scene plays on at 1x while the ego stands, so an ego behind the
    recording was fed the indicator of wherever the recorded bus had got to:
    waiting at a red light to turn right it got the LEFT the bus later gave
    for its next stop, for 40 s, and did not set off on the green (H78,
    2026-11-09). Where the bus stood (a stop, a queue) every instant there
    matches the place; the one nearest the scene's instant is taken, so a
    departure signal still comes when the scene reaches the departure."""

    BACK_SEC = 180.0    # how far behind the scene the ego's place is searched
    AHEAD_SEC = 30.0    # ...and ahead of it
    MATCH_M = 6.0       # farther than this from the recorded track: the scene's instant
    SAME_PLACE_M = 1.0  # instants this close to the nearest are the same place
    PERIOD_NS = 250_000_000

    def __init__(self, stamps_ns, xs, ys, yaws) -> None:
        import numpy as np

        self._np = np
        self.t = np.asarray([int(v) for v in stamps_ns], dtype=np.int64)
        self.x = np.asarray(list(xs), dtype=float)
        self.y = np.asarray(list(ys), dtype=float)
        self.yaw = np.asarray(list(yaws), dtype=float)
        self._last: tuple[int, float, float, int] | None = None   # (ts, x, y, answer)

    def at(self, ts_ns: int, ego) -> int:
        """Recorded time (ns) whose indicator belongs to the ego (x, y, v, yaw) now."""
        if ego is None or len(self.t) == 0:
            return int(ts_ns)
        ex, ey, _, eyaw = ego
        if self._last is not None:
            lts, lx, ly, ans = self._last
            if abs(ts_ns - lts) < self.PERIOD_NS and math.hypot(ex - lx, ey - ly) < 0.5:
                return ans
        np = self._np
        lo = int(np.searchsorted(self.t, int(ts_ns) - int(self.BACK_SEC * 1e9)))
        hi = int(np.searchsorted(self.t, int(ts_ns) + int(self.AHEAD_SEC * 1e9)))
        ans = int(ts_ns)
        if hi > lo:
            d = np.hypot(self.x[lo:hi] - ex, self.y[lo:hi] - ey)
            d[np.cos(self.yaw[lo:hi] - eyaw) < 0.5] = np.inf   # the same way, not the opposite lane
            dmin = float(d.min())
            if dmin <= self.MATCH_M:
                cand = np.nonzero(d <= dmin + self.SAME_PLACE_M)[0]
                k = cand[int(np.argmin(np.abs(self.t[lo + cand] - int(ts_ns))))]
                ans = int(self.t[lo + k])
        self._last = (int(ts_ns), ex, ey, ans)
        return ans


def build_scene(objects_data, odom_data, signals_data, map_path=None, *, log: ShieldLog | None = None,
                verbose: bool = False) -> Any:
    """The scene player's object pipeline for one recording: the reactive
    shield over its tracks, the map and the recorded lights, the objects'
    pasts (prebirth) and the audit, behind one ObjectGate. The scene driver
    and the offline bench (scene_bench.py) build it here, the same way.

    objects_data / odom_data / signals_data: [(stamp ns, msg)] as the
    reproducer loads them."""
    from types import SimpleNamespace

    # The shield keeps to the frame's time (frame_clock): the frame
    # follows the ego, so slower ego, slower traffic, and only an NPC
    # that would enter the ego is held back.
    # soft_stop: a blocked car brakes to a standoff instead of halting
    # at the margin on one tick. Every class is judged against the
    # ego's real body: the widened box held cars that were really
    # clear of it, away from their recording. npc_following: a car
    # also brakes for one ahead the shield has held back, instead of
    # sitting inside it in the queue (both 2026-11-09). ego_plan: a
    # standing NPC waits while the ego's plan sweeps its way — cross
    # traffic set off toward a bus about to turn into its path (H78).
    shield = ReactiveShield(
        ShieldParams(frame_clock=True, vru_real_body=True, soft_stop=True, vehicle_real_body=True,
                     npc_following=True, ego_plan_sec=EGO_PLAN_SEC, deadlock_sec=DEADLOCK_SEC)
    )
    shield.build_tracks(objects_data)
    # The map and the recorded lights: a delayed NPC stops at its red,
    # a held one waits before a junction or crosswalk, not inside it.
    shield_map = load_shield_map(map_path) if map_path else None
    if shield_map is not None:
        shield.set_map(shield_map)
        shield.set_signals(signals_data or [])
    odom = [m for _, m in odom_data]
    # The way the ego is to go: the recorded vehicle's (deadlocks are judged on it).
    shield.set_route([m.pose.pose.position.x for m in odom], [m.pose.pose.position.y for m in odom],
                     [t * 1e-9 for t, _ in odom_data], [_yaw_of(m.pose.pose.orientation) for m in odom])
    prebirth = PreBirth(
        shield.tracks, shield.parent_of, [t * 1e-9 for t, _ in odom_data],
        [m.pose.pose.position.x for m in odom], [m.pose.pose.position.y for m in odom],
        clearance=shield._rect_clearance, ego_poses=[m.pose.pose for m in odom],
    )
    # Before the tracks get their pasts: what stands at the recording's
    # end is judged on recorded poses.
    audit = SceneAudit(shield.tracks, shield_map, ego_path=(
        [t * 1e-9 for t, _ in odom_data], [m.pose.pose.position.x for m in odom], [m.pose.pose.position.y for m in odom]))
    gate = ObjectGate(shield=shield, log=log, prebirth=prebirth, audit=audit)
    prebirth.extend(shield)
    if verbose:
        print(
            f"Shield map: {len(shield_map.stop_lines)} stop-line segments, {len(shield_map.junctions)} junction "
            f"and {len(shield_map.crosswalks)} crosswalk lanelets, "
            f"lights {'on' if shield.signals else 'off'}" if shield_map is not None
            else "Shield map: none (no light, junction or crosswalk rules)",
            flush=True,
        )
        print(
            "Scene: position — the scene follows the ego after each departure; "
            f"shield over {len(shield.tracks)} recorded objects",
            flush=True,
        )
        print(f"Pre-birth: {len(prebirth)} tracks given their approach {prebirth.kinds()} — "
              "shown once the ego is as near as the bus first saw them"
              + (f"; {prebirth.cut_by_recording} cut where they met the recorded vehicle" if prebirth.cut_by_recording else ""),
              flush=True)
    return SimpleNamespace(shield=shield, gate=gate, prebirth=prebirth, map=shield_map)


def recorded_indicators(bag_path: str | Path) -> list[tuple[int, int]]:
    """The recording's turn-indicator reports as (bag time ns, report), the
    clock's time base. Read once per bag and cached like the reference ego: the
    original reproducer does not load them, and a second pass over a 37-79 GB
    recording costs as much as its own load."""
    from dp_multi_eval.bag_cache import cached_json

    bag_path = Path(bag_path).expanduser().resolve()

    def compute() -> list[list[int]]:
        from rclpy.serialization import deserialize_message
        from rosbag2_py import StorageFilter
        from rosidl_runtime_py.utilities import get_message

        from dp_multi_eval.bag_reader import open_bag_reader

        reader = open_bag_reader(bag_path)
        types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        if INDICATOR_TOPIC not in types:
            return []
        reader.set_filter(StorageFilter(topics=[INDICATOR_TOPIC]))
        msg_type = get_message(types[INDICATOR_TOPIC])
        rows = []
        while reader.has_next():
            _topic, data, stamp = reader.read_next()
            rows.append([int(stamp), int(deserialize_message(data, msg_type).report)])
        return rows

    rows = cached_json(
        bag_path, "recorded_indicators", f"{RECORDED_INDICATORS_VERSION}|{INDICATOR_TOPIC}", compute
    )
    return sorted((int(t), int(r)) for t, r in rows)


def load_original_reproducer() -> Any:
    """planning_debug_tools' perception_reproducer module, the installed script
    ``ros2 run`` starts. Refuses a fork that picks its own frames: fed poses
    would then be clocked twice."""
    from ament_index_python.packages import get_package_prefix

    lib = Path(get_package_prefix("planning_debug_tools")) / "lib" / "planning_debug_tools"
    sys.path.insert(0, str(lib))
    module = importlib.import_module("perception_reproducer")
    if hasattr(module, "WindowClock") or hasattr(module.PerceptionReproducer, "_suppress_overlapping"):
        raise SystemExit(
            f"{module.__file__} is a forked perception_reproducer that picks its own frames; "
            "the scene driver needs planning_debug_tools' original one"
        )
    return module


def original_args(bag: str, rosbag_format: str, verbose: bool) -> argparse.Namespace:
    """The original's own options: tracked objects, and ``-r 0`` — always the
    frame nearest the fed pose (no search radius, no cool-down)."""
    return argparse.Namespace(
        bag=bag,
        noise=True,
        detected_object=False,
        tracked_object=True,
        rosbag_format=rosbag_format,
        verbose=verbose,
        search_radius=0.0,
        reproduce_cool_down=0.0,
        pub_route=False,
    )


def make_scene_driver(reproducer: Any, opts: argparse.Namespace, indicators: IndicatorReplay) -> Any:
    """The ROS side: feeds the reproducer, finishes its objects, replays indicators."""
    from autoware_perception_msgs.msg import TrackedObjects
    from autoware_planning_msgs.msg import Trajectory
    from autoware_vehicle_msgs.msg import TurnIndicatorsReport
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool, Float64

    from dp_multi_eval.reactive_shield import ReactiveShield, ShieldParams

    latched = QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
    )

    class SceneDriver(Node):
        def __init__(self) -> None:
            super().__init__("dp_multi_eval_scene_driver")
            data = reproducer.rosbag_ego_odom_data
            if len(data) < 2:
                raise RuntimeError(
                    f"the bag has {len(data)} {EGO_TOPIC} message(s): no recorded track to drive the scene by"
                )
            self._odom = [m for _, m in data]
            self.timeline = SceneTimeline(
                [t for t, _ in data],
                [m.pose.pose.position.x for m in self._odom],
                [m.pose.pose.position.y for m in self._odom],
                [math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y) for m in self._odom],
                [_yaw_of(m.pose.pose.orientation) for m in self._odom],
                window=opts.window_clock,
                departure_hold=not opts.no_departure_hold,
            )
            self.shield_log = ShieldLog(opts.shield_log)
            scene = build_scene(
                reproducer.rosbag_objects_data, data,
                getattr(reproducer, "rosbag_traffic_signals_data", []) or [],
                opts.map_path, log=self.shield_log, verbose=True,
            )
            self._shield, self.gate = scene.shield, scene.gate
            self._ego: Any = None
            self._objects_on = True
            self._route_set = False
            self._autonomous = False

            self.create_subscription(Odometry, EGO_TOPIC, self._on_ego, 1)
            self.create_subscription(Trajectory, PLAN_TOPIC, self._on_plan, 1)
            self.create_subscription(TrackedObjects, RAW_OBJECTS_TOPIC, self._on_frame, 10)
            self._feed_pub = self.create_publisher(Odometry, FEED_TOPIC, 1)
            self._objects_pub = self.create_publisher(TrackedObjects, OBJECTS_TOPIC, 1)
            self._nearest_pub = self.create_publisher(Float64, NEAREST_TOPIC, 1)
            if opts.window_clock:
                # Created before scene_time: messages ready in one executor
                # cycle are taken in creation order, so a hold or a following
                # switch sent with a jump lands before it.
                self.create_subscription(Bool, "/perception_reproducer/hold", self._on_hold, latched)
                # Objects can be switched off (an empty list is published, so
                # the topic stays alive) while the ego stands at a leg's start.
                self.create_subscription(Bool, "/perception_reproducer/objects", self._on_objects, latched)
                self.create_subscription(Bool, "/perception_reproducer/follow_ego", self._on_follow, latched)
                self.create_subscription(Float64, "/perception_reproducer/scene_time", self._on_scene_time, 1)
                self.create_subscription(Float64, SCENE_JUMP_TOPIC, self._on_scene_jump, 1)
                print("Window clock: recorded time at 1x, jumped and held by the harness", flush=True)
            else:
                # Harness time is not scene time: the departure scene is held
                # while Autoware has no route or is not engaged.
                from autoware_adapi_v1_msgs.msg import OperationModeState, RouteState

                self.create_subscription(RouteState, "/api/routing/state", self._on_route, latched)
                self.create_subscription(OperationModeState, "/api/operation_mode/state", self._on_mode, latched)

            self._indicators = indicators
            self._indicator_clock = IndicatorClock(
                self.timeline.t, [m.pose.pose.position.x for m in self._odom],
                [m.pose.pose.position.y for m in self._odom], [_yaw_of(m.pose.pose.orientation) for m in self._odom],
            )
            if indicators.enabled:
                # One writer for the model's input. With a merge topic the model
                # reads it (launch overlay) and psim keeps the vehicle topic;
                # without it the replay writes the vehicle topic beside psim.
                topic = opts.indicator_merge_topic or INDICATOR_TOPIC
                self._indicator_pub = self.create_publisher(TurnIndicatorsReport, topic, 10)
                if indicators.merge:
                    self.create_subscription(
                        TurnIndicatorsReport, INDICATOR_TOPIC,
                        lambda m: self._indicators.on_vehicle_report(m.report), 10,
                    )
                self.create_timer(INDICATOR_PERIOD_SEC, self._on_indicator_tick)
                print(f"Indicator replay on: {len(indicators.t)} recorded reports -> {topic}", flush=True)
            self.create_timer(FEED_PERIOD_SEC, self._on_feed)

        # -- inputs
        def _on_ego(self, msg: Any) -> None:
            self._ego = msg

        def _on_plan(self, msg: Any) -> None:
            self._shield.set_ego_plan([
                (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9,
                 pt.pose.position.x, pt.pose.position.y, _yaw_of(pt.pose.orientation))
                for pt in msg.points
            ])

        def _on_scene_time(self, msg: Any) -> None:
            self.timeline.jump(float(msg.data))
            # The scene has been moved (a leg boundary, a departure reveal);
            # object clocks that only flow forward would keep every governed
            # car where the previous scene left it.
            self.gate.reset()

        def _on_scene_jump(self, msg: Any) -> None:
            """The scene comes to an ego that is not moved: what drives away
            from it stays (ObjectGate.carry_jump)."""
            before = self.timeline.ts
            self.timeline.jump(float(msg.data))
            jump_sec = float(msg.data) - before * 1e-9 if before is not None else 0.0
            kept = self.gate.carry_jump(
                self._ego.pose.pose if self._ego is not None else None, jump_sec,
                self.get_clock().now().nanoseconds,
            )
            print(f"SCENE_JUMP {jump_sec:+.1f}s t={float(msg.data):.3f} kept={len(kept)}", flush=True)

        def _on_hold(self, msg: Any) -> None:
            self.timeline.hold = bool(msg.data)

        def _on_follow(self, msg: Any) -> None:
            on = bool(msg.data)
            if self.timeline.set_follow(on):
                print(f"FOLLOW_EGO {'on' if on else 'off'} t={(self.timeline.ts or 0) * 1e-9:.3f}", flush=True)

        def _on_objects(self, msg: Any) -> None:
            if bool(msg.data) != self._objects_on:
                print(f"OBJECTS {'on' if msg.data else 'off'} t={(self.timeline.ts or 0) * 1e-9:.3f}", flush=True)
            self._objects_on = bool(msg.data)

        def _on_route(self, msg: Any) -> None:
            self._route_set = msg.state == msg.SET

        def _on_mode(self, msg: Any) -> None:
            self._autonomous = msg.mode == msg.AUTONOMOUS

        # -- outputs
        def _ego_state(self) -> tuple[float, float, float, float] | None:
            if self._ego is None:
                return None
            p = self._ego.pose.pose
            tw = self._ego.twist.twist.linear
            return (p.position.x, p.position.y, math.hypot(tw.x, tw.y), _yaw_of(p.orientation))

        def _on_feed(self) -> None:
            now = self.get_clock().now().nanoseconds
            ts = self.timeline.tick(
                now, self._ego_state(), harness_hold=not (self._route_set and self._autonomous)
            )
            self._feed_pub.publish(self._odom[frame_index(self.timeline.t, ts)])

        def _on_frame(self, msg: Any) -> None:
            if self._ego is None:
                return   # nothing reaches the planner before the simulator's ego exists
            tw = self._ego.twist.twist.linear
            out = self.gate.apply(
                msg,
                ego_pose=self._ego.pose.pose,
                ego_speed=math.hypot(tw.x, tw.y),
                ts_ns=self.timeline.ts if self.timeline.ts is not None else self.timeline.t[0],
                at_end=self.timeline.at_end,
                objects_on=self._objects_on and not opts.no_objects,
                now_ns=self.get_clock().now().nanoseconds,
            )
            self._objects_pub.publish(out)
            p = self._ego.pose.pose.position
            self._nearest_pub.publish(Float64(data=nearest_object_m(out.objects, p.x, p.y)))

        def _on_indicator_tick(self) -> None:
            if self._ego is None or self.timeline.ts is None:
                return
            report = self._indicators.report_at(self._indicator_clock.at(self.timeline.ts, self._ego_state()))
            if report is None:
                return
            msg = TurnIndicatorsReport()
            msg.stamp = self.get_clock().now().to_msg()
            msg.report = report
            self._indicator_pub.publish(msg)

    return SceneDriver()


def nearest_object_m(objects, x: float, y: float) -> float:
    """Centre distance from (x, y) to the nearest object; NO_OBJECT_M for none."""
    best = NO_OBJECT_M
    for o in objects:
        q = o.kinematics.pose_with_covariance.pose.position
        best = min(best, math.hypot(q.x - x, q.y - y))
    return float(best)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="The original perception_reproducer, on dp_multi_eval's scene clock"
    )
    p.add_argument("-b", "--bag", required=True, help="rosbag path, as for perception_reproducer")
    p.add_argument("-t", "--tracked-object", action="store_true", help="tracked objects (always)")
    p.add_argument("-f", "--rosbag-format", default="db3", help="split-file suffix without metadata.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--window-clock", action="store_true",
        help="the harness owns recorded time: /perception_reproducer/scene_time jumps it, "
        "/perception_reproducer/hold pauses it (leg-window evaluation)",
    )
    p.add_argument(
        "--indicator-merge-topic", default=None,
        help="publish the indicator replay on this topic as one merged stream (the planner reads it)",
    )
    p.add_argument("--no-indicator-replay", action="store_true")
    p.add_argument("--no-objects", action="store_true", help="diagnosis: replay no objects (lights stay)")
    p.add_argument(
        "--no-departure-hold", action="store_true",
        help="window clock: do not freeze the scene at a recorded departure while the ego still stands (planned cuts)",
    )
    p.add_argument("--map-path", default=None, help="lanelet2 map (dir or .osm): stop lines, junctions, crosswalks for the shield")
    p.add_argument("--shield-log", default=None, help="write held NPCs and recording-end events here (JSON lines)")
    # The forked reproducer's legacy options: accepted, ignored.
    p.add_argument("--npc-reaction", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--npc-reaction-decel", type=float, default=3.0, help=argparse.SUPPRESS)
    p.add_argument("--no-signal-position-lock", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import signal
    import threading

    opts = parse_args(argv)
    if opts.npc_reaction or opts.no_signal_position_lock:
        print("[note] --npc-reaction / --no-signal-position-lock are accepted and ignored", flush=True)

    # The harness stops the scene with SIGINT, then SIGTERM. rclpy's own handlers
    # act asynchronously and could land mid-take: 3 of 7 stops on H45 ended in
    # "Unable to convert call argument to Python object" or a segfault
    # (2026-10-28; 0 of 18 with the single-node forked reproducer).
    # Here a signal only ends the spin loop between two callbacks. While the
    # bag is still loading it stops at once.
    stop = threading.Event()
    spinning = [False]

    def on_signal(_signum: int, _frame: Any) -> None:
        stop.set()
        if not spinning[0]:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    original = load_original_reproducer()

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    # Node-scoped remaps: only the reproducer's ego input and objects output move.
    rclpy.init(
        args=[
            "scene_driver", "--ros-args",
            "-r", f"{REPRODUCER_NODE}:{EGO_TOPIC}:={FEED_TOPIC}",
            "-r", f"{REPRODUCER_NODE}:{OBJECTS_TOPIC}:={RAW_OBJECTS_TOPIC}",
        ],
        signal_handler_options=SignalHandlerOptions.NO,
    )
    reproducer = driver = executor = None
    try:
        indicators = IndicatorReplay(
            [] if opts.no_indicator_replay else recorded_indicators(opts.bag),
            merge=bool(opts.indicator_merge_topic),
        )
        print(f"Original perception_reproducer: {original.__file__} (-t -r 0, fed on {FEED_TOPIC})", flush=True)
        reproducer = original.PerceptionReproducer(original_args(opts.bag, opts.rosbag_format, opts.verbose))
        driver = make_scene_driver(reproducer, opts, indicators)
        executor = SingleThreadedExecutor()
        executor.add_node(reproducer)
        executor.add_node(driver)
        spinning[0] = True
        while rclpy.ok() and not stop.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        spinning[0] = True   # no KeyboardInterrupt inside the teardown
        if driver is not None and getattr(driver, "shield_log", None) is not None:
            driver.shield_log.close()
        if executor is not None:
            executor.shutdown()
        for node in (driver, reproducer):
            if node is not None:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
