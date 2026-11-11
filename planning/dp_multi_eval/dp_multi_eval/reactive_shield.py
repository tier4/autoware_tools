"""Keep replayed NPCs out of the ego, always.

The tool exists to compare models on given scenes, and a scene stops being
comparable the moment a replayed car drives through the ego or parks inside it.
Both happened: an earlier version held conflicting NPCs and then "released"
them after a standoff timeout, which sent them straight through a waiting ego —
and when a released car's recorded path ended where the ego stood, it stopped
there, inside it.

This version has one rule and no exceptions:

    An NPC's displayed pose is the furthest point along its own recorded path
    that keeps clear of the ego.

Blocked, it waits at that point — indefinitely, the way a queue does, published
at speed 0 — and it never reverses: only the ego can close a gap, and an object
the ego drives into stays where it is so the touch is a contact in the report
(operator, 2026-09-16). When the way opens it continues from where it actually
is, closing its accumulated delay gently rather than teleporting. Routes,
lateral motion and speed profiles stay recorded; only progress along the path
is governed. The harness resets every object clock on each scene jump it
commands (leg boundaries, the departure reveal), so a rewound scene is
followed raw instead of from clocks that only ever run forward.

What the old stopping-distance exception used to express — "a real car could
not have avoided this" — is not lost: a hold that required more than
emergency_decel to happen is logged as a forced emergency stop, a deterministic
count of how hard the ego leaned on traffic, usable to compare models without
ever corrupting the scene.
"""

from __future__ import annotations

import bisect
import copy
import math
from dataclasses import dataclass, field
from types import SimpleNamespace


BORN_MOVING_MPS = 1.0   # a track faster than this over its first second was born driving
BIRTH_SEC = 1.0         # a track first shown within this of its recorded start is being born
CARRY_SAME_WAY_RAD = math.radians(45.0)   # a vehicle heading within this of the ego's heading goes its way
PLAN_TIME_STEP_SEC = 0.25   # the ego's timed plan is kept at this spacing
SPACETIME_EGO_MPS = 0.5     # a moving ego is judged where it will be when the NPC gets there ...
SPACETIME_MARGIN_SEC = 1.0  # ... or this much later (it crosses ahead of the ego only with this to spare)
SPACETIME_HORIZON_SEC = 6.0     # space-time conflicts are judged this far into the NPC's future
CARRY_GAP_SEC = 0.5         # samples further apart than this are a gap in the recording
ROUTE_STEP_M = 0.5          # the ego's route kept this fine (a turn within its 4 m way)
DEADLOCK_CLEAR_M = 0.5      # the ego moving on would come this near it: it blocks the ego
REC_BUS_STEP_SEC = 0.1      # the recorded vehicle's poses kept this often
REC_WORSE_M = 0.3           # nearer than the recording had it by this: a conflict of the replay's making
PLAN_OFF_ROUTE_M = 1.0      # a plan pose this far off the recorded path is the replay's own way
PASS_CLEAR_M = 0.5          # a vehicle passes the ego no nearer than this (unless recorded so) ...
PASS_CLEAR_VRU_M = 0.3      # ... a pedestrian or cyclist no nearer than this ...
PASS_PARALLEL_COS = math.cos(math.radians(25.0))   # ... when it meets the ego at an angle, not alongside
ROUTE_OFF_M = 6.0           # an ego further than this from its route is off it (no route ahead)
DETOUR_RAMP_SLACK_M = 0.05  # on a detour's ramps it keeps its margin plus this (its look-ahead)


def _yaw_of(q) -> float:
    y = getattr(q, "yaw", None)     # the ego's planned poses carry theirs
    if y is not None:
        return y
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


@dataclass
class ShieldParams:
    # Must strictly COVER the collision metric's vehicle footprint
    # (10.9 x 2.87 m with overhangs), or a pose the shield holds as clear
    # can still score sub-0.1 m there — measured as 2-frame brushes against
    # a held oncoming car with the old 2.6 m width.
    ego_length_m: float = 12.5
    ego_width_m: float = 3.0
    # base_link is the REAR AXLE: the real footprint reaches 7.56 m ahead of
    # it and 3.32 m behind (overhangs + wheelbase). A rectangle centred ON
    # base_link covered only 6.25 m forward, so an oncoming car was held
    # 1.3 m inside the metric's front bumper — 19 contact frames while the
    # shield thought it clear. The rectangle's centre sits this far forward.
    ego_center_offset_m: float = 2.1
    # Clearance an NPC keeps to the ego rectangle. Its own half width is added
    # per object. This is a collision guard, not a comfort berth: at 0.7 m it
    # forbade the adjacent-lane pass entirely — an overtaking car whose track
    # began beside the recorded bus (where perception first saw it) could
    # never clear the position-locked sim ego, froze at its spawn pose, and
    # was rammed at 8.5 m/s. The recorded pass happened with the recorded bus
    # right there, so realistic geometry must allow it; the collision metric
    # fires at 0.1 m.
    # 0.35: the collision metric fires at 0.1 m and retreat moves in
    # recorded-sample steps (~0.8 m), so 0.2 left only 0.1 m of slack and a
    # creeping ego still banked 4 frames of touch against a held oncoming
    # car. With exact body-to-body distance an adjacent-lane pass keeps
    # 0.5-1.0 m, comfortably above this.
    margin_m: float = 0.35
    # NPCs further than this from the ego are replayed verbatim.
    active_radius_m: float = 60.0
    # A held NPC closes its delay at this fraction of real time once clear.
    catchup_rate: float = 0.5
    max_delay_sec: float = 120.0
    # A hold that would have needed more than this to achieve in reality is
    # logged as a forced emergency stop.
    emergency_decel_mps2: float = 3.0
    # A governed NPC's clock. False (a 1x scene): its own forward timeline, at
    # least real time and at most (1 + catchup_rate) x real time, whatever
    # the frame does. True (a position scene): the frame itself follows the
    # ego — slower ego, slower traffic — so the NPC keeps to the frame's time,
    # closes a delay it built up while blocked at catchup_rate, and never runs
    # back down its path. H81, 2026-10-28: while the ego stood where the
    # recording drove on, the scene played on and three cars that had followed
    # the recorded bus drove through the ego from behind.
    frame_clock: bool = False
    # Pedestrians and cyclists judged against the bus's own body with a small
    # margin: the car-sized box above held a cyclist still for 15 s 1-2 m
    # behind a standing bus, across whose rear its recorded path ran (H81,
    # 2026-10-29). Cars keep the box.
    vru_real_body: bool = False
    real_length_m: float = 10.88
    real_width_m: float = 2.87
    real_center_offset_m: float = 2.12
    vru_margin_m: float = 0.15
    # Vehicles judged against the same real body (with this margin) instead of
    # the widened box: closest to the recording, since a car the box held back
    # beside or behind the ego was really clear of it.
    vehicle_real_body: bool = False
    vehicle_real_margin_m: float = 0.25
    # Soft stop: a blocked NPC brakes along its own path to stand off before
    # the first pose that would touch the ego, and pulls away again at a
    # bounded acceleration. Without it the clearance gate alone stopped it on
    # one tick at the margin — a car at 10 m/s halting half a metre from the
    # bus, which the model saw as a car about to ram it (H81, 2026-10-29).
    # The clearance gate stays as the last guard; when it still has to step
    # in, that is the forced emergency stop.
    soft_stop: bool = False
    soft_decel_mps2: float = 2.5
    soft_accel_mps2: float = 1.5
    standoff_m: float = 3.0
    vru_standoff_m: float = 1.0
    # A standing NPC whose path is blocked this close ahead does not set off
    # at all: H81 (2026-10-29), a cyclist waiting at the kerb rode out into
    # the road and stopped beside the bus, whose rear covered the gap it had
    # used in the recording; a real one waits at the kerb.
    start_speed_mps: float = 0.5
    start_gate_m: float = 5.0
    # a moving ego's plan off the recorded path holds a vehicle at its line (2026-11-10)
    plan_off_route_hold: bool = True
    vru_start_gate_m: float = 8.0
    # A standing vehicle does not set off while the ego's own plan for this
    # long sweeps its way within the start gate, once the ego is inside a
    # junction (with a map; anywhere without one). Judged on the ego's pose
    # alone, cross traffic released on its green drove toward a bus that was
    # turning into its path, which then stood 1 m from it (H78, 2026-11-09).
    # Pedestrians and cyclists are not held by it: at a crosswalk the bus
    # gives way to them, not they to its plan. Moving NPCs keep the pose rule:
    # passing in front of an approaching bus is theirs to time. 0 = off;
    # set_ego_plan() supplies the plan.
    ego_plan_sec: float = 0.0
    ego_plan_step_m: float = 2.0
    ego_plan_max_age_sec: float = 1.0
    # Tracker ID switch: a track born on top of an older one of the same class
    # (boxes overlapping, within dup_lateral_m side to side, same heading) for
    # dup_share of its first dup_birth_sec is the same object under a new id —
    # two objects do not overlap. It never runs ahead of its parent's clock:
    # the shield held a cyclist behind a standing bus, the tracker then gave
    # it a new id, and that id was born at its recorded place, 1.6 m in front
    # of the bus (H78, 2026-11-09; the detections moved to the new id, the old
    # one coasted on).
    dup_lateral_m: float = 0.6
    dup_heading_deg: float = 30.0
    dup_share: float = 0.8
    dup_birth_sec: float = 1.0
    dup_min_age_sec: float = 1.0
    # With a map and the recorded lights (set_map / set_signals): an NPC the
    # shield has put behind its recording meets its light later than it did;
    # it stops at the stop line unless the light it arrives at lets it go (a
    # green circle, a green arrow for its turn, or the state it crossed under
    # in the recording). An amber it cannot stop for at red_light_amber_decel
    # it takes. Pedestrians are not judged (their lights are not recorded).
    red_light_amber_decel_mps2: float = 3.75
    # Less late than this, it takes the light it took: a car that crossed on
    # the last of an amber, 0.1 s behind, met the red, braked, fell further
    # behind and stood a whole red phase (H78, 2026-11-09).
    red_light_min_lag_sec: float = 2.0
    # ...and a vehicle held for an ego that crosses or meets its way (heading
    # more than 45 deg off) waits before a junction or a crosswalk it has not
    # entered yet, not inside it — as a driver must when the way out is
    # blocked — if it can still stop there at zone_stop_decel_mps2. A car
    # queueing behind the ego keeps its place: behind a bus waiting to turn
    # right inside a junction it queues in the junction too (H95; held 18 m
    # back at the junction's mouth, its own recording went in).
    zone_stop_decel_mps2: float = 3.75
    # Near the ego an NPC does not close its delay faster than its recording:
    # the 1.5x catch-up sent that cyclist darting into the road.
    no_catchup_radius_m: float = 15.0
    # An oncoming NPC keeps its standoff from where the ego will have stopped
    # (v^2 / 2a at this deceleration), not from where it is now: measured
    # from the current pose, cars stopped at a 5 m/s ego's nose with no room
    # left for it to stop (H81, 2026-10-29: 37 ego-caused contact frames).
    ego_stop_decel_mps2: float = 2.0
    # ...and at least where the ego's heading takes it in this long: a bus
    # pulling out into an oncoming car's lane met it within the car's own
    # stopping distance and the car halted 4.9 -> 0 m/s on one tick (H81,
    # 2026-11-09). 0 = stopping distance only.
    ego_anticipation_sec: float = 0.0
    # NPC following (with soft_stop): a governed NPC also brakes for an NPC
    # ahead of it going the same way that the shield has moved off its
    # recording (held, slowed or delayed). Judged against the ego alone, cars
    # queued behind a standing ego all braked to the same spot and sat inside
    # each other (H81: three cars for ~28 s, H35 ~10 s; 2026-11-09). NPCs that
    # follow their recording keep the recording's own gaps.
    npc_following: bool = False
    npc_standoff_m: float = 2.0
    vru_npc_standoff_m: float = 0.5
    npc_margin_m: float = 0.1
    npc_follow_heading_deg: float = 45.0
    npc_deviated_delay_sec: float = 0.2
    # Deadlock (2026-11-10): the ego has stood deadlock_sec and an NPC held
    # for it as long stands on the ego's way (its plan, or the road ahead of
    # its nose). The NPC cannot leave its path and the ego will not go
    # through it; a real one would pass it. Its way ahead is rewritten around
    # the ego: first by dropping the standoff (it passes at the margin) when
    # its recorded way does not touch the ego, else by shifting its path
    # sideways — smoothly, at most detour_max_*_m, keeping detour_clear_m off
    # the ego's body. When neither works the deadlock is reported (held
    # reason "deadlock"). 0 = off.
    deadlock_sec: float = 0.0
    # A moving ego judged where it will be (its timed plan); False: where it is.
    spacetime: bool = True
    # A delayed NPC lives on to its own track's end (carried()).
    carry_delayed: bool = True
    # Met at an angle, an NPC passes the ego no nearer than PASS_CLEAR (unless recorded so).
    pass_clear: bool = True
    deadlock_ahead_m: float = 4.0       # the ego's way: this far on (its standoff + 1 m)
    detour_max_vru_m: float = 4.0
    detour_max_vehicle_m: float = 3.5   # a lane: round a bus standing in its lane
    detour_clear_m: float = 0.5
    detour_ramp_vru_m: float = 1.5      # shortest ramp in and out (longer for a wider detour)
    detour_ramp_vehicle_m: float = 6.0
    detour_horizon_m: float = 40.0


def _between(a, b, f: float):
    """A pose a fraction f of the way from a to b (position linear, heading
    the shorter way round)."""
    f = min(max(f, 0.0), 1.0)
    ya, yb = _yaw_of(a.orientation), _yaw_of(b.orientation)
    dy = math.atan2(math.sin(yb - ya), math.cos(yb - ya))
    return _PlanPose(a.position.x + f * (b.position.x - a.position.x),
                     a.position.y + f * (b.position.y - a.position.y), ya + f * dy)


class _PlanPose:
    """A planned ego pose in the shape of a geometry_msgs Pose."""

    __slots__ = ("position", "orientation")

    def __init__(self, x: float, y: float, yaw: float) -> None:
        self.position = SimpleNamespace(x=x, y=y, z=0.0)
        self.orientation = SimpleNamespace(w=math.cos(yaw / 2.0), x=0.0, y=0.0, z=math.sin(yaw / 2.0), yaw=yaw)


@dataclass
class _ObjectTrack:
    times: list[float] = field(default_factory=list)
    samples: list = field(default_factory=list)
    # Distance travelled along the recorded path at each sample.
    s: list[float] = field(default_factory=list)
    # Index of the next sample that differs from this one (pose, size, VRU
    # body): a parked car repeats one pose for thousands of samples, and the
    # conflict scan checks each run once (20 still cars: 394 -> ~1 ms a tick).
    next_diff: list[int] = field(default_factory=list)
    # Leading samples that are not recorded: its approach before the recording
    # first saw it (prebirth.py, extend_track).
    ext: int = 0


class ReactiveShield:
    def __init__(self, params: ShieldParams | None = None):
        self.p = params or ShieldParams()
        self.tracks: dict[bytes, _ObjectTrack] = {}
        self.delays: dict[bytes, float] = {}
        # Each object's own clock. The frame selector's pose_timestamp is not
        # monotonic — repeat/cool-down switching can jump it seconds either way —
        # and an NPC keyed straight to it teleported 41 m in one tick, landing
        # 9 m in front of a 6 m/s ego whose resulting contact was billed to the
        # model. A governed NPC's effective time flows only forward, at least
        # real time (a rewound repeat frame must not freeze it in a moving
        # ego's lane) and at most (1 + catchup_rate) x real time; only the
        # clearance gate may stop it.
        self.t_eff_prev: dict[bytes, float] = {}
        # Objects that have ever come near enough to be clock-governed; they
        # stay governed (their clock flows forward, clearance-gated) until the
        # frame selector re-syncs with them or they become distant scenery.
        self.governed: set[bytes] = set()
        self.bag_end: float = 0.0
        self.held_now: set[bytes] = set()
        # The sample index last shown for each object. A displayed pose that
        # does not move must be reported as standing still: while the ego sat
        # at a bus stop, a held car kept its recorded 5.6 m/s in its twist and
        # the model was fed a stationary car claiming to drive (H81 realtime,
        # 2026-09-15).
        self.shown_idx: dict[bytes, int] = {}
        # How fast each object went where the scene first presented it near the ego.
        self._first_v: dict[bytes, float] = {}
        # The sample an object was last blocked at. It stands still for as long
        # as it is still shown there; the moment it moves on to the next sample
        # it is driving again.
        self.blocked_idx: dict[bytes, int] = {}
        self.emergency_stops: int = 0
        self._was_held: set[bytes] = set()
        # Speed of an NPC the soft stop has slowed below its recording, until
        # it is back up to the recorded speed.
        self.soft_v: dict[bytes, float] = {}
        # (sample, fraction toward the next) of a slowed NPC shown between samples.
        self.shown_f: dict[bytes, tuple[int, float]] = {}
        # Last tick's shown boxes (x, y, yaw, half length, half width, moved
        # off its recording) — the leaders NPC following brakes for.
        self._shown_boxes: dict[bytes, tuple[float, float, float, float, float, bool, float]] = {}
        # Tracker duplicate -> the track it copies (build_tracks).
        self.parent_of: dict[bytes, bytes] = {}
        self.birth_v: dict[bytes, float] = {}
        # Deadlock: how long the ego has stood, how long each NPC has been
        # held for it, which pass at the margin, and the detours made.
        self._ego_still = 0.0
        self._held_for: dict[bytes, float] = {}
        self._free: set[bytes] = set()
        self._stuck: set[bytes] = set()
        # Vehicles kept on their own clocks over a scene jump (carry_over):
        # they stay that far behind the scene for good, never catching up.
        self._carried_over: set[bytes] = set()
        self.detours: list[dict] = []
        self._route: list[tuple[float, float]] = []
        self._plan_off_route: list = []
        self._rec_t: list[float] = []
        self._rec_pose: list = []
        # Map and recorded lights (optional), and what they give each track.
        self.map = None
        self.signals = None
        self._crossings: dict[bytes, list[tuple[float, int, str, float]]] = {}
        self._zones: dict[bytes, list[tuple[float, float]]] = {}
        self._ego_in_junction = False
        # Why each NPC is standing or braking this tick: ego, plan, leader,
        # red_light (the scene driver logs it; scene-limited stops are judged on it).
        self.held_reason: dict[bytes, str] = {}
        # The ego's planned poses (ego_plan_sec ahead) and the time since they came.
        self._plan: list = []
        self._plan_timed: list = []
        self._plan_age: float = 0.0

    def set_map(self, shield_map) -> None:
        """Stop lines, junctions and crosswalks (shield_map.ShieldMap)."""
        self.map = shield_map
        self._crossings.clear()
        self._zones.clear()

    def set_signals(self, data) -> None:
        """The recorded lights: [(stamp, TrafficLightGroupArray)]."""
        from dp_multi_eval.shield_map import SignalTimeline

        self.signals = SignalTimeline(data)

    def set_ego_plan(self, points) -> None:
        """The ego's plan as (seconds from now, x, y, yaw), in order; kept every
        ego_plan_step_m up to ego_plan_sec. The ego's pose at each tick stays
        the shield's main input; the plan only holds standing NPCs back."""
        poses, timed, last, last_t = [], [], None, None
        for t, x, y, yaw in points:
            if t > self.p.ego_plan_sec:
                break
            if last_t is None or t - last_t >= PLAN_TIME_STEP_SEC:
                timed.append((t, _PlanPose(x, y, yaw)))
                last_t = t
            if last is not None and math.hypot(x - last[0], y - last[1]) < self.p.ego_plan_step_m:
                continue
            poses.append(_PlanPose(x, y, yaw))
            last = (x, y)
        self._plan = poses
        self._plan_timed = timed
        self._plan_age = 0.0
        # Where the ego plans to go and the recorded vehicle did not (a corner
        # cut, a wider turn): what the recording's traffic did not have to give
        # way to.
        offs = self._route_offsets([(q.position.x, q.position.y) for q in poses])
        self._plan_off_route = [q for q, d in zip(poses, offs) if d > PLAN_OFF_ROUTE_M]

    def _route_offsets(self, points) -> list[float]:
        """Distance from each (x, y) to the recorded path (inf without one)."""
        if not points or not self._route:
            return [math.inf] * len(points)
        try:
            import numpy as np

            if getattr(self, "_route_np", None) is None or len(self._route_np) != len(self._route):
                self._route_np = np.asarray(self._route, dtype=float)
            q = np.asarray(points, dtype=float)
            d2 = ((q[:, None, :] - self._route_np[None, :, :]) ** 2).sum(axis=2)
            return [float(v) for v in np.sqrt(d2.min(axis=1))]
        except ImportError:
            return [min(math.hypot(rx - x, ry - y) for rx, ry in self._route) for x, y in points]

    def _ego_at(self, ego_pose):
        """Where the ego will be: a function of seconds from now, from its timed
        plan; None while it stands or the plan is stale (then where it is now
        counts, as before)."""
        timed = self._plan_timed
        if not timed or self._plan_age > self.p.ego_plan_max_age_sec:
            return None
        times = [t for t, _ in timed]
        # Past its end the plan says nothing of where the ego is, unless it
        # ends standing: then it stands there.
        a, b = timed[-1][1].position, timed[max(0, len(timed) - 3)][1].position
        ends_standing = math.hypot(a.x - b.x, a.y - b.y) < 0.5

        def at(tau: float):
            # the plan was published _plan_age ago
            tau += self._plan_age
            if tau <= times[0]:
                return timed[0][1] if tau > 0.0 else ego_pose
            if tau > times[-1] + PLAN_TIME_STEP_SEC and not ends_standing:
                return None
            i = min(bisect.bisect_right(times, tau) - 1, len(timed) - 1)
            if i + 1 >= len(timed):
                return timed[i][1]
            return _between(timed[i][1], timed[i + 1][1], (tau - times[i]) / max(times[i + 1] - times[i], 1e-6))
        return at

    def reset(self) -> None:
        """Forget every object's clock and hold; the next frame is followed raw.

        The harness jumps the scene clock at each leg boundary and at each
        departure reveal — backwards by however late the ego finished the
        leg (9.9 s and 12.9 s on 2026-08-24). An object clock only flows
        forward, so after a rewind every governed car kept driving from where
        it had been: near objects were shown 7-13 s ahead of the scene and
        vanished early when their recorded track ran out (2026-09-16).
        """
        self.delays.clear()
        self.t_eff_prev.clear()
        self.governed.clear()
        self.held_now.clear()
        self._was_held.clear()
        self.shown_idx.clear()
        self._first_v.clear()
        self.blocked_idx.clear()
        self.soft_v.clear()
        self.shown_f.clear()
        self._shown_boxes.clear()
        self._ego_still = 0.0
        self._held_for.clear()
        self._free.clear()
        self._stuck.clear()
        self._carried_over.clear()

    def carry_over(self, ego_pose, jump_sec: float) -> set[bytes]:
        """The scene jumps ``jump_sec`` forward under an ego that is NOT moved
        (it drove on past a stand the recording made). Returns the vehicles
        that stay as they are.

        Everything follows the scene's new instant, as after reset() — except
        the vehicles going the ego's way ahead of it (heading within 45 deg of
        its own, within twice the active radius, where an object on its own
        clock becomes scenery). Those keep
        their own clocks: they drive on along their recorded tracks from where
        they are, at their recorded speed, and leave as they would have. A
        jump replaced every object in one frame: three cars 18-55 m ahead of
        an ego pulling out of a bus stop, going its way, were gone in the next
        frame while it kept its own place and speed (H82, 2026-11-11).

        Only that traffic is kept. A vehicle coming toward the ego, one that
        has passed it, one crossing or standing, would stay in a scene whose
        other objects are a later moment's — the second car of a queue already standing where the first
        still has to drive. When the ego itself is placed on the recording
        the scene is that recorded instant and nothing is kept: reset().
        """
        keep: set[bytes] = set()
        if jump_sec > 0.0:
            yaw_e = _yaw_of(ego_pose.orientation)
            ex = ego_pose.position.x + math.cos(yaw_e) * self.p.ego_center_offset_m
            ey = ego_pose.position.y + math.sin(yaw_e) * self.p.ego_center_offset_m
            # Where each object is on its own clock (not the last sample the
            # shield imposed: one beyond the active radius follows the frame
            # and that index is where it left the radius).
            for key, t_eff in self.t_eff_prev.items():
                tr = self.tracks.get(key)
                if tr is None or key in self.held_now or key in self.blocked_idx:
                    continue
                idx = min(max(bisect.bisect_right(tr.times, t_eff) - 1, 0), len(tr.samples) - 1)
                sample = tr.samples[idx]
                if self._is_vru(sample) or not self._label(sample):
                    continue   # vehicles only (UNKNOWN is tracker clutter)
                pose = sample.kinematics.pose_with_covariance.pose
                # Near: as far as a governed object is shown on its own clock at all.
                if self._clearance_to_ego(pose.position.x, pose.position.y, ego_pose) > 2.0 * self.p.active_radius_m:
                    continue
                if self.speed_from(tr, idx) < BORN_MOVING_MPS:
                    continue
                # Going the ego's way, ahead of it: the traffic it drives in.
                # One going the other way that has passed it also "drives
                # away" — kept, it ran 25 s late for the rest of its track and
                # stood in the road 100 m behind the ego for 8 s; cars turning
                # off and cars in side streets stayed for a second or two
                # (first carry runs, 2026-11-11).
                # Its way is where its poses go over the next second: the box
                # of a slow car is often reported back to front.
                nxt = tr.samples[bisect.bisect_right(tr.times, tr.times[idx] + 1.0) - 1].kinematics.pose_with_covariance.pose
                dx, dy = nxt.position.x - pose.position.x, nxt.position.y - pose.position.y
                yaw = math.atan2(dy, dx) if math.hypot(dx, dy) >= 0.5 else _yaw_of(pose.orientation)
                if abs((yaw - yaw_e + math.pi) % (2.0 * math.pi) - math.pi) > CARRY_SAME_WAY_RAD:
                    continue
                ahead = (pose.position.x - ex) * math.cos(yaw_e) + (pose.position.y - ey) * math.sin(yaw_e)
                if ahead > 0.0:
                    keep.add(key)
        for state in (self.delays, self.t_eff_prev, self.shown_idx, self._first_v, self.blocked_idx,
                      self.soft_v, self.shown_f, self._held_for):
            for key in [k for k in state if k not in keep]:
                del state[key]
        self._shown_boxes = {k: v for k, v in self._shown_boxes.items() if k in keep}
        self.governed &= keep
        self.held_now.clear()
        self._was_held.clear()
        self._free.clear()
        self._stuck.clear()
        self._carried_over = set(keep)
        for key in keep:
            self.delays[key] = self.delays.get(key, 0.0) + jump_sec
            self.governed.add(key)
        return keep

    def build_tracks(self, rosbag_objects_data) -> None:
        for stamp, msg in rosbag_objects_data:
            t = stamp * 1e-9 if stamp > 1e12 else float(stamp)
            for obj in msg.objects:
                key = bytes(obj.object_id.uuid)
                track = self.tracks.setdefault(key, _ObjectTrack())
                track.times.append(t)
                track.samples.append(obj)
        for track in self.tracks.values():
            self._index_track(track)
        self.bag_end = max(
            (tr.times[-1] for tr in self.tracks.values()), default=0.0
        )
        self.parent_of = self._find_duplicates(rosbag_objects_data)
        # Speed over each track's first second of poses (the tracker reports
        # 0 for UNKNOWN objects whatever they do).
        self.birth_v = {key: self.speed_from(tr, 0) for key, tr in self.tracks.items()}

    @staticmethod
    def speed_from(track: _ObjectTrack, i: int, window_sec: float = 1.0) -> float:
        """How fast the object goes from sample i on: the larger of its
        reported speed there and its next window_sec of poses."""
        j = bisect.bisect_right(track.times, track.times[i] + window_sec) - 1
        a = track.samples[i].kinematics.pose_with_covariance.pose.position
        b = track.samples[j].kinematics.pose_with_covariance.pose.position
        dt = track.times[j] - track.times[i]
        tw = track.samples[i].kinematics.twist_with_covariance.twist.linear
        v = math.hypot(tw.x, tw.y)
        return max(v, math.hypot(b.x - a.x, b.y - a.y) / dt) if dt >= 0.3 else v

    def _index_track(self, track: _ObjectTrack) -> None:
        """Path distance and runs of equal samples, from the track's samples."""
        s, prev = 0.0, None
        track.s = []
        keys = []
        for sample in track.samples:
            pose = sample.kinematics.pose_with_covariance.pose
            p, q = pose.position, pose.orientation
            if prev is not None:
                s += math.hypot(p.x - prev[0], p.y - prev[1])
            track.s.append(s)
            prev = (p.x, p.y)
            dims = sample.shape.dimensions
            keys.append((p.x, p.y, q.x, q.y, q.z, q.w, dims.x, dims.y, self._vru_body(sample)))
        n = len(keys)
        track.next_diff = [n] * n
        for i in range(n - 2, -1, -1):
            track.next_diff[i] = track.next_diff[i + 1] if keys[i + 1] == keys[i] else i + 1

    def extend_track(self, key: bytes, times: list[float], samples: list) -> None:
        """Prepend a track's approach before the recording first saw it
        (prebirth.py). From here on it is part of the object's one path: every
        rule of the shield holds on it as on the recorded part."""
        track = self.tracks[key]
        track.times = list(times) + track.times
        track.samples = list(samples) + track.samples
        track.ext += len(times)
        self._index_track(track)
        self._crossings.pop(key, None)
        self._zones.pop(key, None)

    # ------------------------------------------------------------------ deadlock
    def set_route(self, xs, ys, ts=None, yaws=None) -> None:
        """The way the ego is to go: the recorded vehicle's path (the leg's
        route), kept every ROUTE_STEP_M; with times and headings, also where
        the recorded vehicle was when (_rec_bus_at)."""
        pts = []
        for x, y in zip(xs, ys):
            if not pts or math.hypot(x - pts[-1][0], y - pts[-1][1]) >= ROUTE_STEP_M:
                pts.append((x, y))
        self._route = pts
        if ts is not None and yaws is not None:
            keep = [0]
            for i in range(1, len(ts)):
                if ts[i] - ts[keep[-1]] >= REC_BUS_STEP_SEC:
                    keep.append(i)
            self._rec_t = [ts[i] for i in keep]
            self._rec_pose = [_PlanPose(xs[i], ys[i], yaws[i]) for i in keep]

    def _rec_bus_at(self, t: float):
        """The recorded vehicle's pose at recorded time t (None outside the recording)."""
        times = self._rec_t
        if not times or t < times[0] - 0.5 or t > times[-1] + 0.5:
            return None
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return self._rec_pose[0]
        if i + 1 >= len(times):
            return self._rec_pose[-1]
        return _between(self._rec_pose[i], self._rec_pose[i + 1], (t - times[i]) / max(times[i + 1] - times[i], 1e-6))

    def _route_ahead(self, ego_pose) -> list:
        """Poses of the route ahead of the ego, deadlock_ahead_m past its nose:
        the way it is to go, whatever its plan says while it stands."""
        route = self._route
        if len(route) < 2:
            return []
        ex, ey = ego_pose.position.x, ego_pose.position.y
        eyaw = _yaw_of(ego_pose.orientation)
        best, best_d = None, math.inf
        for i, (x, y) in enumerate(route[:-1]):
            d = (x - ex) ** 2 + (y - ey) ** 2
            if d < best_d:
                nx, ny = route[i + 1]
                if math.cos(math.atan2(ny - y, nx - x) - eyaw) > 0.5:   # the same way, not the way back
                    best, best_d = i, d
        if best is None or best_d > ROUTE_OFF_M ** 2:
            return []
        out, run = [], 0.0
        limit = self.p.real_length_m / 2.0 + self.p.real_center_offset_m + self.p.deadlock_ahead_m
        for i in range(best, len(route) - 1):
            (x, y), (nx, ny) = route[i], route[i + 1]
            out.append(_PlanPose(x, y, math.atan2(ny - y, nx - x)))
            run += math.hypot(nx - x, ny - y)
            if run >= limit:
                break
        return out

    def _on_ego_way(self, sample, ego_pose) -> bool:
        """It blocks the ego: the ego's body, moved on along its route (or its
        plan, or straight on) up to deadlock_ahead_m, would come within
        DEADLOCK_CLEAR_M of it — nearer than it is now. One merely near the
        bus, which the bus could drive away from, does not deadlock it."""
        p = sample.kinematics.pose_with_covariance.pose.position
        reach = math.hypot(max(sample.shape.dimensions.x, 1.0), max(sample.shape.dimensions.y, 1.0)) / 2.0
        here = self._rect_clearance(sample, ego_pose)
        yaw = _yaw_of(ego_pose.orientation)
        plan = [q for _, q in self._plan_timed] if self._plan_age <= self.p.ego_plan_max_age_sec else []
        ahead = [_PlanPose(ego_pose.position.x + d * math.cos(yaw), ego_pose.position.y + d * math.sin(yaw), yaw)
                 for d in (1.0, self.p.deadlock_ahead_m / 2.0, self.p.deadlock_ahead_m)]
        ex, ey = ego_pose.position.x, ego_pose.position.y
        way = [q for q in plan + ahead + self._route_ahead(ego_pose)
               if math.hypot(q.position.x - ex, q.position.y - ey) <= self.p.deadlock_ahead_m + 0.5]
        for q in way:
            if self._clearance_to_ego(p.x, p.y, q) > reach + DEADLOCK_CLEAR_M:
                continue
            c = self._rect_clearance(sample, q)
            if c < DEADLOCK_CLEAR_M and c < here - 0.05:
                return True
        return here < self._margin(sample) + 0.05     # touching already

    def _resolve_deadlocks(self, ego_pose, ego_speed: float, dt: float, frame_time_sec: float) -> None:
        self._ego_still = self._ego_still + dt if ego_speed < 0.3 else 0.0
        held = {k for k, why in (self.held_reason or {}).items() if why in ("ego", "plan")}
        self._held_for = {k: self._held_for.get(k, 0.0) + dt for k in held}
        if self._ego_still < self.p.deadlock_sec:
            return
        for key, sec in list(self._held_for.items()):
            if sec < self.p.deadlock_sec or key in self._stuck or key in self._free:
                continue
            tr = self.tracks.get(key)
            i0 = self.shown_idx.get(key)
            if tr is None or i0 is None:
                continue
            if not self._on_ego_way(tr.samples[i0], ego_pose):
                # Not in the ego's way, yet waiting on a bus that has stood as
                # long: if its own way clears the bus's body it goes, at its
                # margin (a car stood 3.5 m off the flank of a bus standing
                # mid-junction for as long as the bus did; H78 july run). If
                # its way runs through the bus, it waits for the bus — the bus
                # can go.
                if self._way_clears_ego(tr, i0, ego_pose):
                    self._free.add(key)
                    self.detours.append({"t": round(frame_time_sec, 2), "id": key.hex()[:8], "how": "margin",
                                         "shift_m": 0.0})
                    self._held_for.pop(key, None)
                continue
            how = self._detour(key, tr, i0, ego_pose)
            self.detours.append({"t": round(frame_time_sec, 2), "id": key.hex()[:8], "how": how[0],
                                 "shift_m": round(how[1], 2)})
            self._held_for.pop(key, None)
            if how[0] == "stuck":
                self._stuck.add(key)

    def _way_clears_ego(self, tr: _ObjectTrack, i0: int, ego_pose) -> bool:
        """Its recorded way ahead passes the ego where it stands by its passing
        clearance: not a graze the last guard would stop hard (a car freed to
        pass 0.3 m off a bus's flank drove at it at 3 m/s and stood 0.8 m off)."""
        margin = max(self._margin(tr.samples[i0]), PASS_CLEAR_VRU_M if self._is_vru(tr.samples[i0]) else PASS_CLEAR_M)
        end = min(bisect.bisect_right(tr.s, tr.s[i0] + self.p.detour_horizon_m), len(tr.samples))
        i = i0
        nxt = tr.next_diff if len(tr.next_diff) == len(tr.samples) else None
        while i < end:
            if self._rect_clearance(tr.samples[i], ego_pose) < margin:
                return False
            i = nxt[i] if nxt is not None else i + 1
        return True

    def _detour(self, key: bytes, tr: _ObjectTrack, i0: int, ego_pose) -> tuple[str, float]:
        """("margin", 0) if its way ahead clears the ego's body by its margin
        (it may pass at that), ("shift", d) after rewriting it d metres to the
        side, or ("stuck", 0)."""
        first = tr.samples[i0]
        vru = self._is_vru(first)
        margin = self._margin(first)
        s0 = tr.s[i0]
        end = min(bisect.bisect_right(tr.s, s0 + self.p.detour_horizon_m), len(tr.samples))
        idx = list(range(i0, end))
        base = [self._rect_clearance(tr.samples[i], ego_pose) for i in idx]
        need = margin + self.p.detour_clear_m
        bad = [k for k, c in enumerate(base) if c < need]
        if not bad:
            self._free.add(key)
            return "margin", 0.0
        if base[0] < margin:
            return "stuck", 0.0       # already touching: nowhere to shift from
        sa, sb = tr.s[idx[bad[0]]], tr.s[idx[bad[-1]]]
        min_ramp = self.p.detour_ramp_vru_m if vru else self.p.detour_ramp_vehicle_m
        slope = 1.5 if vru else 0.5      # sideways per metre on: a swerve aside, a lane change
        top = self.p.detour_max_vru_m if vru else self.p.detour_max_vehicle_m

        def weight(s, ramp):
            if s <= s0:
                return 0.0
            rise = ramp   # never steeper, however near the conflict: no sideways jump
            if s < s0 + rise:
                x = (s - s0) / rise
            elif s <= max(sb, s0 + rise):
                x = 1.0
            else:
                x = 1.0 - (s - max(sb, s0 + rise)) / ramp
            x = max(0.0, min(1.0, x))
            return x * x * (3.0 - 2.0 * x)

        normals = []
        for i in idx:
            j = min(i + 1, len(tr.samples) - 1)
            k = max(i - 1, 0)
            a = tr.samples[k].kinematics.pose_with_covariance.pose.position
            b = tr.samples[j].kinematics.pose_with_covariance.pose.position
            h = math.atan2(b.y - a.y, b.x - a.x) if (b.x - a.x) ** 2 + (b.y - a.y) ** 2 > 1e-6 else \
                _yaw_of(tr.samples[i].kinematics.pose_with_covariance.pose.orientation)
            normals.append((-math.sin(h), math.cos(h)))
        def path(sign, d, ws):
            """The rewritten poses (x, y, heading) — headings along the new way."""
            xy = []
            for k, i in enumerate(idx):
                p = tr.samples[i].kinematics.pose_with_covariance.pose.position
                xy.append((p.x + sign * d * ws[k] * normals[k][0], p.y + sign * d * ws[k] * normals[k][1]))
            out = []
            for k, i in enumerate(idx):
                moved = not (ws[k] <= 0.0 and (k == 0 or ws[k - 1] <= 0.0) and (k + 1 >= len(idx) or ws[k + 1] <= 0.0))
                a, b = xy[max(k - 1, 0)], xy[min(k + 1, len(xy) - 1)]
                h = (math.atan2(b[1] - a[1], b[0] - a[0]) if moved and (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 > 1e-4
                     else _yaw_of(tr.samples[i].kinematics.pose_with_covariance.pose.orientation))
                out.append((xy[k][0], xy[k][1], h, moved))
            return out

        # Judged on the poses it will be shown at (turned along its new way):
        # within the ramps its margin (plus its look-ahead), where shifted
        # fully the detour clearance. A swerving cyclist judged unturned
        # clipped the bus's corner at 0.12 m.
        best = None
        d = 0.25
        while d <= top + 1e-6 and best is None:
            ramp = max(min_ramp * min(1.0, d / top) if vru else min_ramp, d / slope)
            ws = [weight(tr.s[i], ramp) for i in idx]
            for sign in (1.0, -1.0):
                cand = path(sign, d, ws)
                ok = True
                for k, i in enumerate(idx):
                    x, y, h, moved = cand[k]
                    if not moved:
                        if base[k] < margin and tr.s[i] > s0:
                            ok = False
                            break
                        continue
                    probe = SimpleNamespace(
                        shape=tr.samples[i].shape, classification=getattr(tr.samples[i], "classification", None),
                        kinematics=SimpleNamespace(pose_with_covariance=SimpleNamespace(pose=_PlanPose(x, y, h))))
                    c = self._rect_clearance(probe, ego_pose)
                    if c < (need if ws[k] >= 0.999 else margin + DETOUR_RAMP_SLACK_M):
                        ok = False
                        break
                if ok:
                    best, chosen = sign * d, cand
                    break
            d += 0.25
        if best is None:
            return "stuck", 0.0
        for k, i in enumerate(idx):
            x, y, h, moved = chosen[k]
            if not moved:
                continue
            smp = copy.copy(tr.samples[i])
            kin = copy.copy(smp.kinematics)
            pwc = copy.copy(kin.pose_with_covariance)
            pose = copy.deepcopy(pwc.pose)
            pose.position.x, pose.position.y = x, y
            q = pose.orientation
            q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(h / 2.0), math.cos(h / 2.0)
            pwc.pose = pose
            kin.pose_with_covariance = pwc
            smp.kinematics = kin
            tr.samples[i] = smp
        self._index_track(tr)
        self._crossings.pop(key, None)
        self._zones.pop(key, None)
        self.blocked_idx.pop(key, None)
        return "shift", best

    def carried(self, present: set[bytes], frame_time_sec: float) -> list:
        """Objects whose recording has ended but whose own clock, delayed by
        the shield, has not: they live on to their track's end. Without them a
        car waiting for the ego vanished where it waited when its recorded
        time ran out (audit `vanish held_to_its_end`). Not through the
        recording's own gaps: a pedestrian the tracker lost for 4 s stood
        frozen there, then jumped 14 m. Returns a copy of each one's sample at
        its clock, for the frame."""
        out = []
        if not self.p.carry_delayed:
            return out
        for key in self.governed:
            if key in present:
                continue
            tr = self.tracks.get(key)
            t_eff = self.t_eff_prev.get(key)
            if (tr is None or t_eff is None or frame_time_sec < tr.times[-1]
                    or self.delays.get(key, 0.0) <= self.p.npc_deviated_delay_sec
                    or t_eff >= tr.times[-1] - 0.05):
                continue
            i = min(max(bisect.bisect_right(tr.times, t_eff) - 1, tr.ext), len(tr.times) - 1)
            if i + 1 < len(tr.times) and tr.times[i + 1] - tr.times[i] > CARRY_GAP_SEC:
                continue    # inside a gap of its recording: unseen there too
            out.append(copy.deepcopy(tr.samples[i]))
        return out

    @staticmethod
    def end_speed(track: _ObjectTrack, window_sec: float = 1.0) -> float:
        """How fast the object was going when its track ended: the larger of
        its last reported speed and its last window_sec of poses. The tracker
        reports 0 for every UNKNOWN object: an oncoming one driving at 9 m/s
        when the recording ended was kept as standing, frozen mid-road in the
        ego's way (H39, 2026-11-09)."""
        tw = track.samples[-1].kinematics.twist_with_covariance.twist.linear
        v = math.hypot(tw.x, tw.y)
        i = max(track.ext, bisect.bisect_left(track.times, track.times[-1] - window_sec))   # recorded poses only
        dt = track.times[-1] - track.times[i]
        if dt >= 0.2:
            a = track.samples[i].kinematics.pose_with_covariance.pose.position
            b = track.samples[-1].kinematics.pose_with_covariance.pose.position
            v = max(v, math.hypot(b.x - a.x, b.y - a.y) / dt)
        return v

    @staticmethod
    def _label(obj):
        cls = getattr(obj, "classification", None) or []
        return int(cls[0].label) if cls else None

    def _track_crossings(self, key: bytes, track: _ObjectTrack) -> list[tuple[float, int, str, float]]:
        """(s, light group, turn, recorded time) where the track crosses a
        stop line the way the line stops traffic."""
        got = self._crossings.get(key)
        if got is not None:
            return got
        from dp_multi_eval.shield_map import segment_cross

        out: list[tuple[float, int, str, float]] = []
        n = len(track.samples)
        nxt = track.next_diff if len(track.next_diff) == n else None
        i = 0
        while i < n - 1:
            j = nxt[i] if nxt is not None else i + 1
            if j >= n:
                break
            a = track.samples[i].kinematics.pose_with_covariance.pose.position
            b = track.samples[j].kinematics.pose_with_covariance.pose.position
            if (b.x - a.x) ** 2 + (b.y - a.y) ** 2 > 1e-6:
                heading = math.atan2(b.y - a.y, b.x - a.x)
                lines = {id(sl): sl for sl in self.map.stop_lines_near(a.x, a.y) + self.map.stop_lines_near(b.x, b.y)}
                for sl in lines.values():
                    f = segment_cross(a.x, a.y, b.x, b.y, sl)
                    if f is None or math.cos(heading - sl.yaw) < 0.5:
                        continue
                    sc = track.s[i] + f * (track.s[j] - track.s[i])
                    if any(g == sl.group and abs(sc - s0) < 2.0 for s0, g, _, _ in out):
                        continue
                    tc = track.times[i] + f * (track.times[j] - track.times[i])
                    out.append((sc, sl.group, self._turn_after(track, j, sc, heading), tc))
            i = j
        self._crossings[key] = out
        return out

    @staticmethod
    def _turn_after(track: _ObjectTrack, i: int, s0: float, heading: float) -> str:
        """left / right / straight: its heading 25 m past the line."""
        k = min(bisect.bisect_left(track.s, s0 + 25.0), len(track.samples) - 1)
        if k <= i:
            return "straight"
        a = track.samples[max(i, k - 3)].kinematics.pose_with_covariance.pose.position
        b = track.samples[k].kinematics.pose_with_covariance.pose.position
        if (b.x - a.x) ** 2 + (b.y - a.y) ** 2 < 0.25:
            return "straight"
        d = math.atan2(math.sin(math.atan2(b.y - a.y, b.x - a.x) - heading),
                       math.cos(math.atan2(b.y - a.y, b.x - a.x) - heading))
        return "left" if d > math.radians(30.0) else "right" if d < -math.radians(30.0) else "straight"

    def _track_zones(self, key: bytes, track: _ObjectTrack) -> list[tuple[float, float]]:
        """(s where its front enters, s where its rear leaves) for each run of
        junctions and crosswalks along the track (runs too close to stop
        between are one)."""
        got = self._zones.get(key)
        if got is not None:
            return got
        dims = track.samples[0].shape.dimensions
        hl = max(dims.x, 1.0) / 2.0
        runs: list[list[float]] = []
        n = len(track.samples)
        nxt = track.next_diff if len(track.next_diff) == n else None
        i, inside = 0, False
        while i < n:
            p = track.samples[i].kinematics.pose_with_covariance.pose.position
            now_in = self.map.zone_at(p.x, p.y)
            if now_in and not inside:
                if runs and track.s[i] - runs[-1][1] < 2.0 * hl + 1.0:
                    runs[-1][1] = track.s[i]
                else:
                    runs.append([track.s[i], track.s[i]])
            elif now_in:
                runs[-1][1] = track.s[i]
            inside = now_in
            i = nxt[i] if nxt is not None else i + 1
        out = [(a - hl, b + hl) for a, b in runs]
        self._zones[key] = out
        return out

    def _leads_ego(self, sample, ego_pose) -> bool:
        """Ahead on the ego's way, going its way: on a pose of its plan (or the
        road just ahead of its nose) heading within 45 deg of it, in front of
        the ego's rear axle."""
        p = sample.kinematics.pose_with_covariance.pose.position
        oyaw = _yaw_of(sample.kinematics.pose_with_covariance.pose.orientation)
        eyaw = _yaw_of(ego_pose.orientation)
        dx, dy = p.x - ego_pose.position.x, p.y - ego_pose.position.y
        if dx * math.cos(eyaw) + dy * math.sin(eyaw) <= 0.0:
            return False
        plan = [q for _, q in self._plan_timed] if self._plan_age <= self.p.ego_plan_max_age_sec else []
        ahead = [_PlanPose(ego_pose.position.x + d * math.cos(eyaw), ego_pose.position.y + d * math.sin(eyaw), eyaw)
                 for d in (self.p.real_length_m / 2.0, self.p.real_length_m)]
        reach = math.hypot(max(sample.shape.dimensions.x, 1.0), max(sample.shape.dimensions.y, 1.0)) / 2.0
        cos45 = math.cos(math.radians(45.0))
        for q in plan + ahead:
            if math.cos(oyaw - _yaw_of(q.orientation)) < cos45:
                continue
            if (self._clearance_to_ego(p.x, p.y, q) <= reach + 1.0
                    and self._rect_clearance(sample, q) < self._margin(sample) + 0.5):
                return True
        return False

    def _follows_ego(self, track: _ObjectTrack, s_touch: float, ego_pose) -> bool:
        """Queueing behind the ego in its lane: where it would touch the ego it
        heads the ego's way (within 45 deg) from behind its rear axle, within
        the two widths. Judged at the touch, on the path's own direction: by
        the headings where they stood now, a car that met a turning bus's
        flank mid-junction counted as following it and queued inside the
        junction, three more behind it (H78, 2026-11-10)."""
        n = len(track.samples)
        i = min(bisect.bisect_left(track.s, s_touch), n - 1)
        pose = track.samples[i].kinematics.pose_with_covariance.pose
        j = i
        while j > 0 and track.s[i] - track.s[j] < 0.5:
            j -= 1
        if track.s[i] - track.s[j] >= 0.2:
            a = track.samples[j].kinematics.pose_with_covariance.pose.position
            heading = math.atan2(pose.position.y - a.y, pose.position.x - a.x)
        else:
            heading = _yaw_of(pose.orientation)
        eyaw = _yaw_of(ego_pose.orientation)
        if math.cos(heading - eyaw) < math.cos(math.radians(45.0)):
            return False
        dx, dy = pose.position.x - ego_pose.position.x, pose.position.y - ego_pose.position.y
        lon = dx * math.cos(eyaw) + dy * math.sin(eyaw)
        lat = -dx * math.sin(eyaw) + dy * math.cos(eyaw)
        half = self.p.real_width_m / 2.0 + max(track.samples[i].shape.dimensions.y, 1.0) / 2.0
        return lon < 0.0 and abs(lat) <= half

    def _zone_stop(self, key, track, s_now: float, s_stop: float, v: float) -> float:
        """A stop point inside a junction or crosswalk it has not entered,
        moved back to before it — when it can still stop there."""
        for z0, z1 in self._track_zones(key, track):
            if z0 > s_now + 0.1 and z0 < s_stop < z1:
                back = z0 - 0.5
                if back - s_now >= v * v / (2.0 * self.p.zone_stop_decel_mps2) - 0.1:
                    return back
                return s_stop
        return s_stop

    def _find_duplicates(self, rosbag_objects_data) -> dict[bytes, bytes]:
        """Tracks born as a new id of an older track's object (tracker ID switch)."""
        born = {k: tr.times[0] for k, tr in self.tracks.items()}
        # samples in each track's first dup_birth_sec
        frames = {k: bisect.bisect_right(tr.times, tr.times[0] + self.p.dup_birth_sec)
                  for k, tr in self.tracks.items()}
        cos_h = math.cos(math.radians(self.p.dup_heading_deg))
        hits: dict[tuple[bytes, bytes], int] = {}
        for stamp, msg in rosbag_objects_data:
            t = stamp * 1e-9 if stamp > 1e12 else float(stamp)
            objs = msg.objects
            if len(objs) < 2:
                continue
            boxes = [(bytes(o.object_id.uuid), self._label(o), self._sample_box(o)) for o in objs]
            young = [i for i, (k, _, _) in enumerate(boxes) if t <= born[k] + self.p.dup_birth_sec]
            if not young:
                continue
            cells: dict[tuple[int, int], list[int]] = {}
            for i, (_, _, b) in enumerate(boxes):
                cells.setdefault((int(b[0] // 4.0), int(b[1] // 4.0)), []).append(i)
            for i in young:
                kc, lc, bc = boxes[i]
                cx, cy = int(bc[0] // 4.0), int(bc[1] // 4.0)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for j in cells.get((cx + dx, cy + dy), ()):
                            kp, lp, bp = boxes[j]
                            # the parent is the older track, by dup_min_age_sec
                            if kp == kc or lp != lc or born[kp] > born[kc] - self.p.dup_min_age_sec:
                                continue
                            if math.cos(bc[2] - bp[2]) < cos_h or self._box_gap(*bc, *bp) > 0.0:
                                continue
                            c, sn = math.cos(-bp[2]), math.sin(-bp[2])
                            if abs(sn * (bc[0] - bp[0]) + c * (bc[1] - bp[1])) > self.p.dup_lateral_m:
                                continue
                            hits[(kc, kp)] = hits.get((kc, kp), 0) + 1
        parent: dict[bytes, bytes] = {}
        for (kc, kp), n in sorted(hits.items(), key=lambda kv: -kv[1]):
            if kc not in parent and frames[kc] >= 3 and n >= self.p.dup_share * frames[kc]:
                parent[kc] = kp
        for kc in list(parent):   # a copy of a copy follows the original
            root, seen = parent[kc], {kc}
            while root in parent and root not in seen:
                seen.add(root)
                root = parent[root]
            parent[kc] = root
        return parent

    # ------------------------------------------------------------------
    @staticmethod
    def _s_at(track: _ObjectTrack, t: float) -> float:
        """Path distance at time t, interpolated between samples."""
        i = bisect.bisect_right(track.times, t) - 1
        if i < 0:
            return track.s[0]
        if i >= len(track.times) - 1:
            return track.s[-1]
        span = track.times[i + 1] - track.times[i]
        f = (t - track.times[i]) / span if span > 0 else 0.0
        return track.s[i] + f * (track.s[i + 1] - track.s[i])

    @staticmethod
    def _t_at_s(track: _ObjectTrack, s: float, i0: int) -> float:
        """Time at which the path distance reaches s, searching from sample i0."""
        j, n = max(0, i0), len(track.times)
        while j + 1 < n and track.s[j + 1] < s:
            j += 1
        if j + 1 >= n:
            return track.times[-1]
        ds = track.s[j + 1] - track.s[j]
        f = min(1.0, max(0.0, (s - track.s[j]) / ds)) if ds > 0 else 0.0
        return track.times[j] + f * (track.times[j + 1] - track.times[j])

    def _conflict_s(
        self, track: _ObjectTrack, i0: int, s_limit: float, ego_pose, ego_reach_m: float = 0.0,
        band_m: float = 0.0, ego_at=None, clock: float | None = None, pass_clear: bool = True,
    ) -> tuple[float, float] | None:
        """(s_touch, s_enter): path distance of the first sample from i0 on
        that would touch the ego where it stands — or, for an oncoming sample,
        anywhere on the ego's way to its stop ``ego_reach_m`` ahead — and of
        where that approach first came within ``band_m`` of touching (s[i0]
        when it already is); None if nothing touches before s_limit. A
        clearance crossing between samples (~1 m apart at 10 m/s) is
        interpolated from the previous sample.

        ego_at (a moving ego with a fresh plan): each sample is judged against
        where the ego will be when the NPC gets there (+-SPACETIME_MARGIN_SEC),
        not where it is now. Judged on the ego's pose now, a cyclist the bus
        passed 2.4 m clear in the recording was held 5 s, because its way ran
        through where the turning bus stood a few seconds before it got there
        (H78 control replay, 2026-11-10)."""
        eyaw = _yaw_of(ego_pose.orientation)
        ec, es = math.cos(eyaw), math.sin(eyaw)
        last = None    # (s, clearance) of the previous sample checked exactly
        enter = None   # s where the current approach came within band_m
        # Only a pose that brings it closer to the ego than where it stands is
        # a conflict: a car the ego comes up behind, moving away, was braked
        # to a halt in the ego's lane for being inside the margin (H82).
        c_now = self._rect_clearance(track.samples[max(0, min(i0, len(track.samples) - 1))], ego_pose)
        n = len(track.samples)
        # Samples equal to the one just checked give the same answer: skip them.
        nxt = track.next_diff if len(track.next_diff) == n else None
        i = max(0, i0)
        # now on its own clock (continuous; its sample i0 may lie a tick behind)
        t0 = clock if clock is not None else track.times[max(0, min(i0, n - 1))]
        corridor = ([ego_pose] + [q for _, q in self._plan_timed]) if ego_at is not None else []
        if ego_at is not None:
            # nothing farther than this from the ego now can meet any pose of its plan
            ex0, ey0 = ego_pose.position.x, ego_pose.position.y
            sweep = max((math.hypot(q.position.x - ex0, q.position.y - ey0) for q in corridor), default=0.0) \
                + math.hypot(self.p.ego_length_m, self.p.ego_width_m) / 2.0 + self.p.ego_center_offset_m
        while i < n:
            # an approach already inside the band is followed to its end
            if track.s[i] > s_limit and enter is None:
                return None
            sample = track.samples[i]
            step = nxt[i] if nxt is not None else i + 1
            p = sample.kinematics.pose_with_covariance.pose.position
            dims = sample.shape.dimensions
            reach = math.hypot(max(dims.x, 1.0), max(dims.y, 1.0)) / 2.0 + self.p.margin_m
            vru = self._vru_body(sample)
            # It passes the ego no nearer than PASS_CLEAR (its body margin at
            # least) — unless the recording had it as near the recorded vehicle
            # then. Only a touch counted: a car released by its green grazed a
            # bus standing mid-junction at 3 m/s, 0.3 m off its flank, and the
            # last guard stopped it hard 0.8 m from it (H78 july run, 2026-11-10).
            margin = self._margin(sample)
            if pass_clear:
                # met at an angle (not alongside, as in an adjacent lane)
                rel = math.cos(_yaw_of(sample.kinematics.pose_with_covariance.pose.orientation) - eyaw)
                if abs(rel) < PASS_PARALLEL_COS:
                    margin = max(margin, PASS_CLEAR_VRU_M if self._is_vru(sample) else PASS_CLEAR_M)
            if ego_at is not None:
                tau = track.times[i] - t0
                if tau > SPACETIME_HORIZON_SEC:
                    # Further on, where the ego will be is not known (a plan
                    # ending in a wait does not say for how long); the next
                    # ticks judge it as it comes near. A car 16 s from its
                    # crossing was held for a bus whose plan ended in a wait
                    # the recording left 5 s later (H78 control); one 3 s
                    # from the side of a bus stopping mid-junction must not
                    # be let go (H78 july run, 2026-11-10).
                    return None
                if math.hypot(p.x - ex0, p.y - ey0) > sweep + reach + band_m:
                    last = enter = None
                    i = step
                    continue
                # Where the ego will be when it gets there, or up to the margin
                # later (it crosses ahead of the bus only with time to spare);
                # never where the ego was: crossing behind a bus that has
                # passed, or following it, is ordinary traffic (both were held
                # in a replay of the recording itself, 2026-11-10).
                poses = [ego_at(tau + k * SPACETIME_MARGIN_SEC / 2.0) for k in (0, 1, 2)]
                near_any = [q for q in poses if q is not None and self._clearance_to_ego(p.x, p.y, q) <= reach + band_m]
                if not near_any:
                    last = enter = None
                    i = step
                    continue
                clr = min(self._rect_clearance(sample, q) for q in near_any)
                if clr < margin and self._rec_t:
                    # The recording is the reference for how close is acceptable:
                    # the recorded vehicle met this very sample at its recorded
                    # time; no worse than that here, no conflict. With the ego
                    # driving the recording, the recording's own crossings stand
                    # (a car crossed ahead of the bus that slowed for it, held 4 s
                    # by the margin alone; H78 control, 2026-11-10).
                    rec = [self._rec_bus_at(track.times[i] + k * SPACETIME_MARGIN_SEC / 2.0) for k in (0, 1, 2)]
                    rec_clr = min((self._rect_clearance(sample, q) for q in rec if q is not None), default=None)
                    if rec_clr is not None and clr >= rec_clr - REC_WORSE_M:
                        clr = margin
                ahead = 0.0
            else:
                oyaw = _yaw_of(sample.kinematics.pose_with_covariance.pose.orientation)
                # Oncoming (heading > ~135 deg off the ego's): the ego's stopping
                # distance counts. A car ahead going the same way does not brake
                # for an ego closing in behind it.
                ahead = ego_reach_m if math.cos(oyaw - eyaw) < -0.7 else 0.0
                # The car box contains the real body, so far from it is clear of both.
                if self._clearance_to_ego(p.x, p.y, ego_pose) > reach + ahead + band_m:
                    last = enter = None
                    i = step
                    continue
                clr = self._rect_clearance(sample, ego_pose)
                if clr < margin and self._rec_t:
                    q = self._rec_bus_at(track.times[i])
                    if q is not None and clr >= self._rect_clearance(sample, q) - REC_WORSE_M:
                        clr = margin    # as near as the recording had it: its own pass
            if ahead > 0.0:
                # Shifted ego boxes no further apart than a box length cover
                # the whole way to the stop (2 steps up to 25 m, more beyond).
                box = self.p.real_length_m if vru else self.p.ego_length_m
                steps = max(2, math.ceil(ahead / box))
                for k in range(1, steps + 1):
                    d = ahead * k / steps
                    clr = min(clr, self._rect_clearance(sample, ego_pose, ego_shift=(ec * d, es * d)))
            band = margin + band_m
            if clr < band and enter is None:
                if last is None or last[1] <= clr:
                    enter = track.s[i]
                else:
                    enter = last[0] + (last[1] - band) / (last[1] - clr) * (track.s[i] - last[0])
            elif clr >= band:
                enter = None
            if clr < margin and clr < c_now - 0.01:
                if last is None or last[1] <= clr:
                    touch = track.s[i]
                else:
                    touch = last[0] + (last[1] - margin) / (last[1] - clr) * (track.s[i] - last[0])
                if ego_at is not None:
                    # Where it first comes onto the ego's way (any pose of the
                    # plan): it yields before that, not on the way.
                    corr = self._corridor_entry(track, i0, i, corridor, band_m)
                    if corr is not None and corr[0] < touch:
                        return corr[0], min(corr[1], corr[0])
                return touch, min(enter, touch)
            last = (track.s[i], clr)
            i = step
        return None

    def _corridor_entry(self, track: _ObjectTrack, i0: int, i1: int, corridor, band_m: float):
        """(s where it first comes within its margin of any pose of the ego's
        way, s where it first came within the band) from sample i0 to i1."""
        n = len(track.samples)
        nxt = track.next_diff if len(track.next_diff) == n else None
        enter = None
        i = max(0, i0)
        while i <= i1 and i < n:
            sample = track.samples[i]
            p = sample.kinematics.pose_with_covariance.pose.position
            dims = sample.shape.dimensions
            reach = math.hypot(max(dims.x, 1.0), max(dims.y, 1.0)) / 2.0 + self.p.margin_m
            margin = self._margin(sample)
            cc = min((self._rect_clearance(sample, q) for q in corridor
                      if self._clearance_to_ego(p.x, p.y, q) <= reach + band_m), default=math.inf)
            if cc < margin + band_m and enter is None:
                enter = track.s[i]
            if cc < margin:
                return track.s[i], enter if enter is not None else track.s[i]
            i = nxt[i] if nxt is not None else i + 1
        return None

    @classmethod
    def _box_gap(cls, ax, ay, ayaw, ahl, ahw, bx, by, byaw, bhl, bhw) -> float:
        """Distance between two oriented rectangles (0 when they overlap)."""
        best = float("inf")
        for px, py in cls._rect_corners(ax, ay, ayaw, ahl, ahw):
            best = min(best, cls._point_rect_clearance(px, py, bx, by, byaw, bhl, bhw))
        for px, py in cls._rect_corners(bx, by, byaw, bhl, bhw):
            best = min(best, cls._point_rect_clearance(px, py, ax, ay, ayaw, ahl, ahw))
        if best > 0.0:
            return best
        # corner-to-rectangle 0 means touching or overlapping; both count as 0
        return 0.0

    @staticmethod
    def _sample_box(sample) -> tuple[float, float, float, float, float]:
        pose = sample.kinematics.pose_with_covariance.pose
        dims = sample.shape.dimensions
        return (pose.position.x, pose.position.y, _yaw_of(pose.orientation),
                max(dims.x, 0.5) / 2.0, max(dims.y, 0.5) / 2.0)

    def _leaders(self, key: bytes, sample) -> list[tuple[float, float, float, float, float]]:
        """Moved-off-their-recording NPCs ahead of this one, going its way,
        that it does not already overlap (a recorded duplicate it never
        cleared would lock it in place)."""
        if not (self.p.soft_stop and self.p.npc_following) or not self._shown_boxes:
            return []
        fx, fy, fyaw, fhl, fhw = self._sample_box(sample)
        cos_lim = math.cos(math.radians(self.p.npc_follow_heading_deg))
        out = []
        for other, (x, y, yaw, hl, hw, moved, v) in self._shown_boxes.items():
            if other == key or not moved or math.cos(yaw - fyaw) < cos_lim:
                continue
            dx, dy = x - fx, y - fy
            if dx * math.cos(fyaw) + dy * math.sin(fyaw) <= 0.0 or math.hypot(dx, dy) > 60.0:
                continue
            if self._box_gap(fx, fy, fyaw, fhl, fhw, x, y, yaw, hl, hw) <= 0.0:
                continue
            out.append((x, y, yaw, hl, hw, v))
        return out

    def _npc_clear(self, sample, leaders) -> bool:
        if not leaders:
            return True
        fx, fy, fyaw, fhl, fhw = self._sample_box(sample)
        return all(
            self._box_gap(fx, fy, fyaw, fhl, fhw, *lead[:5]) >= self.p.npc_margin_m for lead in leaders
        )

    def _npc_conflict_s(self, track: _ObjectTrack, i0: int, s_limit: float, leaders) -> tuple[float, float] | None:
        """(path distance of the first sample from i0 on that comes within
        npc_margin of a leader, that leader's speed); None before s_limit."""
        if not leaders:
            return None
        n = len(track.samples)
        nxt = track.next_diff if len(track.next_diff) == n else None
        i, last = max(0, i0), None
        while i < n:
            if track.s[i] > s_limit:
                return None
            fx, fy, fyaw, fhl, fhw = self._sample_box(track.samples[i])
            gaps = [(self._box_gap(fx, fy, fyaw, fhl, fhw, *lead[:5]), lead[5]) for lead in leaders]
            gap, v_lead = min(gaps)
            if gap < self.p.npc_margin_m:
                if last is None or last[1] <= gap:
                    return track.s[i], v_lead
                f = (last[1] - self.p.npc_margin_m) / (last[1] - gap)
                return last[0] + f * (track.s[i] - last[0]), v_lead
            last = (track.s[i], gap)
            i = nxt[i] if nxt is not None else i + 1
        return None

    def _red_light(self, key, track, s_now: float, prev: float, look: float, v: float,
                   frame_time_sec: float) -> float | None:
        """Where the NPC must stop for a light it now meets later than it did
        in the recording (its front 0.5 m short of the stop line), or None."""
        from dp_multi_eval.shield_map import AMBER, CIRCLE, permits

        if frame_time_sec - prev < self.p.red_light_min_lag_sec:
            return None
        hl = max(track.samples[0].shape.dimensions.x, 1.0) / 2.0
        for s_cross, group, turn, t_rec in self._track_crossings(key, track):
            s_front = s_cross - hl
            if s_front < s_now - 0.3 or s_front > look:
                continue   # passed, or not yet in reach
            # The light it will meet: its recorded arrival, shifted by its delay.
            arrive = frame_time_sec + max(0.0, t_rec - prev)
            now = self.signals.state(group, arrive)
            if permits(now, self.signals.state(group, t_rec), turn):
                continue
            stop = s_front - 0.5
            if now and (AMBER, CIRCLE) in now and v * v / (2.0 * max(0.1, stop - s_now)) > self.p.red_light_amber_decel_mps2:
                continue   # too late to stop for the amber
            return stop
        return None

    def _soft_target(
        self, key: bytes, track: _ObjectTrack, cur: int, prev: float,
        target_t: float, dt_sec: float, ego_pose, ego_speed_mps: float = 0.0,
        leaders=None, frame_time_sec: float | None = None, near: bool = True,
    ) -> tuple[float, float | None]:
        """Brake the NPC ahead of a conflict on its path and limit how fast it
        pulls away again. Returns the (possibly earlier) target time and the
        speed to publish, None when the recording's own speed stands."""
        tw = track.samples[cur].kinematics.twist_with_covariance.twist.linear
        v_rec = math.hypot(tw.x, tw.y)
        s_now = self._s_at(track, prev)
        implied = max(0.0, self._s_at(track, target_t) - s_now) / dt_sec
        slowed = key in self.soft_v
        v_prev = self.soft_v.get(key, max(v_rec, implied))
        vru = self._vru_body(track.samples[cur])
        standoff = self.p.vru_standoff_m if vru else self.p.standoff_m
        gate = self.p.vru_start_gate_m if vru else self.p.start_gate_m
        standing = (self.soft_v[key] if slowed else v_rec) < self.p.start_speed_mps
        a = self.p.soft_decel_mps2
        look = s_now + max(v_prev, implied) ** 2 / (2.0 * a) + standoff + implied * dt_sec + 1.0
        if standing:
            look = max(look, s_now + gate)
        label = self._label(track.samples[cur])
        # Every class waits where a road user may wait (the motion contract,
        # docs/EVALUATION_METHOD.md): cars, cyclists, pedestrians, clutter.
        zoned = self.map is not None
        if zoned:
            # A junction or crosswalk ahead: what blocks its far side counts
            # before it goes in, not once it is inside.
            for z0, z1 in self._track_zones(key, track):
                if s_now < z0 <= look:
                    look = max(look, z1 + standoff)
                    break
        cap = math.inf
        ego_reach = max(ego_speed_mps ** 2 / (2.0 * self.p.ego_stop_decel_mps2),
                        ego_speed_mps * self.p.ego_anticipation_sec)
        # (path distance of the conflict, standoff, speed of what is there, why)
        conflicts: list[tuple[float, float, float, str]] = []
        # Out of the ego's reach only its lights count.
        # Who gives way (the motion contract): pedestrians and cyclists keep to
        # the ego's body only, never to its plan — the bus gives way to them,
        # and holding them for its plan would never test that; a vehicle ahead
        # on the ego's way, going its way, goes first (the ego follows it: one
        # was held for a bus closing in from behind and stood in its lane,
        # H78 lag replay, 2026-11-10). Other vehicles give way to where the
        # ego will be.
        # One already inside a junction or on a crosswalk clears it first: it
        # does not stop there for where the ego will be (it kept cross traffic
        # standing mid-junction, across the ego's way; all-bags bench,
        # 2026-11-10). Only the ego's body stops it.
        cp = track.samples[cur].kinematics.pose_with_covariance.pose.position
        inside = self.map is not None and self.map.zone_at(cp.x, cp.y)
        moving_ego = self.p.spacetime and ego_speed_mps >= SPACETIME_EGO_MPS
        future = self._ego_at(ego_pose) if moving_ego else None
        yields_to_plan = (future is not None and not inside and not self._is_vru(track.samples[cur])
                          and not self._leads_ego(track.samples[cur], ego_pose))
        ego_at = future if yields_to_plan else None
        if key in self._free:
            standoff = 0.0    # a deadlock let it pass the ego at the margin
        free = key in self._free
        hit = (self._conflict_s(track, cur, look + ego_reach, ego_pose, ego_reach, band_m=standoff, ego_at=ego_at,
                                clock=min(target_t, prev + dt_sec), pass_clear=self.p.pass_clear and not free)
               if near else None)
        if hit is not None and ego_at is None and future is not None:
            # Those that do not give way to the ego's plan (people, one inside
            # a junction, one ahead of it) still never go into its body — but
            # only into a body that will still be there when they get there.
            # Judged on where a moving bus stands now, a cyclist the recorded
            # bus passed 6 m clear was held 5 s while that very bus drove the
            # very recording (H78 control replay, 2026-11-10).
            still_there = self._conflict_s(track, cur, look + ego_reach, ego_pose, 0.0,
                                           band_m=standoff, ego_at=future, clock=min(target_t, prev + dt_sec),
                                           pass_clear=self.p.pass_clear and not free)
            if still_there is None:
                hit = None
        if hit is not None:
            s_touch, s_enter = hit
            # The standoff is a clearance: on a slanting way in, standing off
            # along the path still left a car 1.1 m from the bus's side (H78,
            # 2026-11-09). It stops where it first comes within the standoff —
            # unless it was well inside that band, still moving, when the
            # conflict appeared (stopping "behind" it would be an emergency brake).
            follower = zoned and self._follows_ego(track, s_touch, ego_pose)
            if standing or s_enter > s_now - 1.0:
                s_touch = min(s_touch, max(s_enter, s_now - 1.0) + standoff)
            conflicts.append((s_touch, standoff, 0.0, "queue" if follower else "ego"))
        hit = self._npc_conflict_s(track, cur, look, leaders)
        if hit is not None:
            # A leader driving on leaves room as it goes: brake to its speed by
            # the standoff, not to a standstill (otherwise a follower stop-goes
            # behind a car that only runs late).
            conflicts.append((hit[0], self.p.vru_npc_standoff_m if vru else self.p.npc_standoff_m, hit[1], "leader"))
        if self.map is not None and self.signals and frame_time_sec is not None and label != 7:
            red = self._red_light(key, track, s_now, prev, look, v_prev, frame_time_sec)
            if red is not None:
                conflicts.append((red, 0.0, 0.0, "red_light"))
        # The plan holds a standing NPC for a standing ego about to go; a
        # moving ego is judged where it will be (space-time) instead — except
        # where it plans to go off the recorded path: a vehicle standing at
        # its line, not yet in the junction, does not go in across that way.
        # The recording's own interleaving needs no hold (space-time and the
        # recording's clearances judge it); a corner the replay cuts was never
        # given way to in the recording, and a car on green drove into the
        # junction at a bus turning 3.7 m inside the recorded line, braked 3 m
        # off its side and stood there (H78 planned cuts, 2026-11-10).
        off_route = self.p.plan_off_route_hold and moving_ego and not inside and bool(self._plan_off_route)
        plan_hold = (
            near and standing and self._plan and self._plan_age <= self.p.ego_plan_max_age_sec
            and not self._is_vru(track.samples[cur])
            and (self.map is None or self._ego_in_junction)
            and (not moving_ego or off_route)
            # a plan to go is taken as intent only so long: a bus that has
            # stood longer is not about to go (H78 july run, 2026-11-10)
            and (self.p.deadlock_sec <= 0.0 or self._ego_still < self.p.deadlock_sec)
        )
        if plan_hold:
            # One already standing on the ego's planned way is not held for
            # it: the ego cannot go before it does (a car stood at the bus's
            # nose for the plan through it, the bus stood for the car; lag
            # replay of H78, 2026-11-10).
            here = track.samples[cur]
            hp = here.kinematics.pose_with_covariance.pose.position
            hreach = math.hypot(max(here.shape.dimensions.x, 1.0), max(here.shape.dimensions.y, 1.0)) / 2.0
            plan_hold = not any(
                self._clearance_to_ego(hp.x, hp.y, q) <= hreach + self.p.margin_m
                and self._rect_clearance(here, q) < self._margin(here)
                for q in self._plan
            )
        if plan_hold:
            # Where the ego is about to be counts for setting off only: it
            # holds the NPC at its line and never brakes one already moving.
            # Off the route, the NPC's whole way through the junction counts
            # (look reaches its far side), not just its first metres.
            poses = self._plan_off_route if moving_ego else self._plan
            limit = max(look, s_now + gate) if moving_ego else s_now + gate
            for pose in poses:
                h = self._conflict_s(track, cur, limit, pose, band_m=standoff)
                if h is not None:
                    conflicts.append((min(h[0], max(h[1], s_now - 1.0) + standoff), standoff, 0.0, "plan"))
                    break
        if zoned:
            # Held for the ego, it waits before a junction or a crosswalk it
            # has not entered yet, not inside it. One queueing behind the ego
            # in its lane keeps its place behind it ("queue").
            conflicts = [
                (self._zone_stop(key, track, s_now, s_c - stand, v_prev) + stand, stand, v_there, why)
                if why in ("ego", "plan") else (s_c, stand, v_there, "ego" if why == "queue" else why)
                for s_c, stand, v_there, why in conflicts
            ]
        braking = None
        for s_c, stand, v_there, why in conflicts:
            if why == "plan":
                continue   # holds a standing NPC, never brakes one
            c = math.sqrt(v_there ** 2 + 2.0 * a * max(0.0, s_c - stand - s_now))
            if c < cap:
                cap, braking = c, why
        blocking = [c for c in conflicts if c[2] < self.p.start_speed_mps]
        if braking is not None and cap < self.p.start_speed_mps:
            self.held_reason[key] = braking
        if blocking:
            s_c, _, _, why = min(blocking, key=lambda c: c[0])
            if standing and s_c - s_now <= gate:
                self.held_reason[key] = why
                # Waits where it stands until its way is clear — held even on a
                # tick whose step is zero (the first moving sample can share the
                # waiting pose), or it slips past the gate on the next one.
                # Its clock still runs through its own recorded wait: only
                # setting off is held back, not the time it would stand anyway.
                self.soft_v[key] = 0.0
                j = cur
                while (j + 1 < len(track.times) and track.times[j + 1] <= target_t
                       and track.s[j + 1] <= s_now + 0.05):
                    j += 1
                return max(prev, min(target_t, track.times[j])), 0.0
        if slowed:
            cap = min(cap, v_prev + self.p.soft_accel_mps2 * dt_sec)
        if implied > cap:
            t_s = self._t_at_s(track, s_now + cap * dt_sec, cur)
            limited_t = max(prev, min(target_t, t_s))
            if cap >= v_rec:
                # Not slower than its recording: a scene-clock step outran the
                # brake profile, or a pull-away ramp has reached the recorded
                # speed. It keeps the recorded speed — publishing the cap fed
                # the model a 17 m/s cyclist and a 14.4 m/s car recorded at
                # 12.7 (H81/H78, 2026-10-29).
                self.soft_v.pop(key, None)
                return limited_t, None
            self.soft_v[key] = cap
            return limited_t, cap
        if slowed:
            v = min(cap, max(implied, v_prev))
            if v >= v_rec - 0.5:
                self.soft_v.pop(key, None)
                return target_t, None
            self.soft_v[key] = v
            return target_t, v
        return target_t, None

    # ------------------------------------------------------------------
    def _clearance_to_ego(self, x: float, y: float, ego_pose) -> float:
        yaw = _yaw_of(ego_pose.orientation)
        off = self.p.ego_center_offset_m
        ex = ego_pose.position.x + math.cos(yaw) * off
        ey = ego_pose.position.y + math.sin(yaw) * off
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx = c * (x - ex) - s * (y - ey)
        ly = s * (x - ex) + c * (y - ey)
        dx = max(abs(lx) - self.p.ego_length_m / 2.0, 0.0)
        dy = max(abs(ly) - self.p.ego_width_m / 2.0, 0.0)
        return math.hypot(dx, dy)

    @staticmethod
    def _rect_corners(cx, cy, yaw, half_l, half_w):
        c, s = math.cos(yaw), math.sin(yaw)
        return [
            (cx + c * dx - s * dy, cy + s * dx + c * dy)
            for dx, dy in (
                (half_l, half_w), (half_l, -half_w),
                (-half_l, half_w), (-half_l, -half_w),
            )
        ]

    @staticmethod
    def _point_rect_clearance(px, py, cx, cy, yaw, half_l, half_w):
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx = c * (px - cx) - s * (py - cy)
        ly = s * (px - cx) + c * (py - cy)
        dx = max(abs(lx) - half_l, 0.0)
        dy = max(abs(ly) - half_w, 0.0)
        return math.hypot(dx, dy)

    def _away_key(self, sample, ego_pose) -> tuple[float, float]:
        """Ordering of how far a pose is from the ego: clearance (1 cm steps),
        then, overlapping, the distance between centres."""
        p = sample.kinematics.pose_with_covariance.pose.position
        yaw = _yaw_of(ego_pose.orientation)
        cx = ego_pose.position.x + math.cos(yaw) * self.p.real_center_offset_m
        cy = ego_pose.position.y + math.sin(yaw) * self.p.real_center_offset_m
        return (round(self._rect_clearance(sample, ego_pose), 2), round(math.hypot(p.x - cx, p.y - cy), 2))

    def _rect_clearance(
        self, sample, ego_pose,
        obj_shift: tuple[float, float] = (0.0, 0.0),
        ego_shift: tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """Exact clearance between the ego and object rectangles.

        The minimum distance between two convex polygons is always attained
        at a corner of one of them, so corner-vs-rectangle both ways is
        exact. The old test used only the object's half WIDTH around its
        centre: correct for a side pass, but a head-on car was "clear" with
        its centre 1.1 m from the ego's bumper — its nose a metre inside
        the ego — and a creeping ego was billed 46 contact frames against a
        held oncoming car.
        """
        pos = sample.kinematics.pose_with_covariance.pose.position
        q = sample.kinematics.pose_with_covariance.pose.orientation
        oyaw = _yaw_of(q)
        dims = sample.shape.dimensions
        ohl = max(dims.x, 1.0) / 2.0
        ohw = max(dims.y, 1.0) / 2.0
        eyaw = _yaw_of(ego_pose.orientation)
        if self._real_body(sample):
            off, ehl, ehw = self.p.real_center_offset_m, self.p.real_length_m / 2.0, self.p.real_width_m / 2.0
        else:
            off, ehl, ehw = self.p.ego_center_offset_m, self.p.ego_length_m / 2.0, self.p.ego_width_m / 2.0
        ex = ego_pose.position.x + math.cos(eyaw) * off + ego_shift[0]
        ey = ego_pose.position.y + math.sin(eyaw) * off + ego_shift[1]
        ox = pos.x + obj_shift[0]
        oy = pos.y + obj_shift[1]
        best = float("inf")
        for px, py in self._rect_corners(ox, oy, oyaw, ohl, ohw):
            best = min(best, self._point_rect_clearance(px, py, ex, ey, eyaw, ehl, ehw))
        for px, py in self._rect_corners(ex, ey, eyaw, ehl, ehw):
            best = min(best, self._point_rect_clearance(px, py, ox, oy, oyaw, ohl, ohw))
        return best

    def _clear(
        self, sample, ego_pose, half_w: float,
        ego_speed_mps: float = 0.0, dt_sec: float = 0.1,
        obj_speed_mps: float | None = None,
    ) -> bool:
        # Clear NOW and clear a short look-ahead LATER. At 12-18 m/s of
        # combined approach the gap shrinks 1.2-1.8 m within one update
        # interval, so judging only the instantaneous pose left transient
        # overlaps — two 1-2 frame contacts per fleet run, always against
        # fast oncoming traffic. Shifting both bodies along their velocities
        # and re-measuring catches that, while a parallel overtake's future
        # pose slides alongside and stays clear — a closing-speed heuristic
        # blocked those from 9 m behind.
        margin = self._margin(sample)
        if self._rect_clearance(sample, ego_pose) < margin:
            return False
        horizon = min(0.3, 3.0 * dt_sec)
        oyaw = _yaw_of(sample.kinematics.pose_with_covariance.pose.orientation)
        tw = sample.kinematics.twist_with_covariance.twist.linear
        # A car the soft stop has slowed looks ahead at its shown speed.
        ospd = math.hypot(tw.x, tw.y) if obj_speed_mps is None else obj_speed_mps
        eyaw = _yaw_of(ego_pose.orientation)
        dxo = math.cos(oyaw) * ospd * horizon
        dyo = math.sin(oyaw) * ospd * horizon
        dxe = math.cos(eyaw) * ego_speed_mps * horizon
        dye = math.sin(eyaw) * ego_speed_mps * horizon
        return (
            self._rect_clearance(sample, ego_pose, obj_shift=(dxo, dyo), ego_shift=(dxe, dye))
            >= margin
        )

    VRU_LABELS = (5, 6, 7)   # MOTORCYCLE, BICYCLE, PEDESTRIAN

    def _vru_body(self, sample) -> bool:
        if not self.p.vru_real_body:
            return False
        cls = getattr(sample, "classification", None) or []
        return bool(cls) and int(cls[0].label) in self.VRU_LABELS

    def _real_body(self, sample) -> bool:
        """Judged against the ego's real body (else the widened box)."""
        return self._vru_body(sample) or (self.p.vehicle_real_body and not self._is_vru(sample))

    def _is_vru(self, sample) -> bool:
        cls = getattr(sample, "classification", None) or []
        return bool(cls) and int(cls[0].label) in self.VRU_LABELS

    def _margin(self, sample) -> float:
        if self._vru_body(sample):
            return self.p.vru_margin_m
        if self.p.vehicle_real_body and not self._is_vru(sample):
            return self.p.vehicle_real_margin_m
        return self.p.margin_m

    def apply(
        self,
        objects_msg,
        ego_pose,
        frame_time_sec: float,
        dt_sec: float,
        ego_speed_mps: float = 0.0,
    ) -> None:
        """Impose the invariant on a *copied* frame, in place."""
        if self.p.deadlock_sec > 0.0:
            self._resolve_deadlocks(ego_pose, ego_speed_mps, dt_sec, frame_time_sec)
        self.held_now.clear()
        self.held_reason = {}
        if self.map is not None:
            yaw = _yaw_of(ego_pose.orientation)
            ex, ey = ego_pose.position.x, ego_pose.position.y
            self._ego_in_junction = any(
                self.map.zone_at(ex + d * math.cos(yaw), ey + d * math.sin(yaw), "j")
                for d in (self.p.real_center_offset_m, self.p.real_center_offset_m + self.p.real_length_m / 2.0)
            )
        kept = []
        for obj in objects_msg.objects:
            key = bytes(obj.object_id.uuid)
            track = self.tracks.get(key)
            if track is None or len(track.times) < 2:
                kept.append(obj)
                continue
            pos = obj.kinematics.pose_with_covariance.pose.position
            delay = self.delays.get(key, 0.0)
            parent = self.parent_of.get(key)
            lag = self.delays.get(parent, 0.0) if parent is not None else 0.0
            if lag > delay:
                # A tracker duplicate runs no earlier than the object it copies:
                # not born before its parent reaches the birth, then at its lag.
                if frame_time_sec - lag < track.times[0] and key not in self.shown_idx:
                    continue
                delay = lag
                self.delays[key] = lag
                self.governed.add(key)
            near = (
                self._clearance_to_ego(pos.x, pos.y, ego_pose)
                <= self.p.active_radius_m
            )
            if key not in self.governed:
                if not near and delay == 0.0:
                    # Never been near: follow the frame selector freely (scene
                    # repeats must loop distant traffic), but remember the
                    # shown time so a frame jump cannot teleport this object
                    # into the ego's path on the tick it first comes near.
                    self.t_eff_prev[key] = max(track.times[0], frame_time_sec)
                    kept.append(obj)
                    continue
                self.governed.add(key)
            # A governed NPC whose clock has run past the end of its recorded
            # track has lived its recorded life; the recording itself shows it
            # vanishing from perception there. Keeping it would freeze it at
            # its final pose with its final (often nonzero) twist — measured
            # as ghost cars parked for minutes in the departure path, one of
            # them arming the yield freeze from 12.8 m for the whole hold.
            # Objects whose track runs to the end of the bag stay only if the
            # recording left them STOPPED — a car parked at the terminus must
            # not evaporate. One left MOVING when the bag ended would have
            # driven on: freezing it mid-lane fabricated an eternal roadblock
            # that stopped a run 30 m short of its goal (two cars, 311 and
            # 289 m into their journeys, pinned across the final approach).
            prev0 = self.t_eff_prev.get(key)
            if prev0 is not None and prev0 >= track.times[-1] - 0.3:
                ended_moving = self.end_speed(track) > 0.5
                if track.times[-1] < self.bag_end - 5.0 or ended_moving:
                    self.governed.discard(key)
                    self._carried_over.discard(key)
                    self.delays.pop(key, None)
                    self.soft_v.pop(key, None)
                    self.shown_f.pop(key, None)
                    continue

            half_w = max(obj.shape.dimensions.y, 1.0) / 2.0
            # Where it stands now on its own timeline, and where the recording
            # says it should be by this frame — with its clock clamped to flow
            # forward at a bounded rate, whatever the frame selector does.
            raw_t = max(track.times[0], frame_time_sec - delay)
            prev = self.t_eff_prev.get(key, raw_t)
            cur = max(0, bisect.bisect_right(track.times, prev) - 1)
            # Nearness is judged from the pose actually shown, not the raw
            # frame pose: a car held at the ego's bumper stays governed even
            # after its recorded self has driven out of the active radius —
            # otherwise the hold "released" and sent it through the ego.
            cp = track.samples[cur].kinematics.pose_with_covariance.pose.position
            near = (
                self._clearance_to_ego(cp.x, cp.y, ego_pose)
                <= self.p.active_radius_m
            )
            catchup = self.p.catchup_rate
            if self.p.soft_stop and (
                self._clearance_to_ego(cp.x, cp.y, ego_pose) < self.p.no_catchup_radius_m
            ):
                catchup = 0.0
            if key in self._carried_over:
                catchup = 0.0   # kept over a scene jump: its recorded pace, not a faster one
            raw_target = frame_time_sec - max(0.0, delay - catchup * dt_sec)
            # A governed NPC lives its own forward timeline: at least real
            # time, at most (1 + catchup) x real time. The earlier clamp had
            # no floor, so when a long ego hold made the repeat machinery
            # rewind frame time below this clock, the NPC froze mid-lane and
            # a moving ego was billed for ramming it (3 contacts at 6.5-8.8
            # m/s in one run, each into a car frozen dead ahead). Only the
            # clearance gate may stop it, never the frame selector.
            if self.p.frame_clock:
                target_t = max(raw_target, prev)
            else:
                target_t = min(
                    max(raw_target, prev + dt_sec),
                    prev + dt_sec * (1.0 + self.p.catchup_rate),
                )
            soft_speed = None
            leaders = self._leaders(key, track.samples[cur]) if near else []
            # A delayed NPC meets its lights late wherever it is, near or not.
            late = (self.map is not None and bool(self.signals)
                    and (delay > self.p.npc_deviated_delay_sec or key in self.soft_v))
            if self.p.soft_stop and (near or late):
                target_t, soft_speed = self._soft_target(
                    key, track, cur, prev, target_t, dt_sec, ego_pose, ego_speed_mps, leaders,
                    frame_time_sec=frame_time_sec, near=near,
                )
            else:
                # Out of reach it drives at its recorded speed; a slowed speed
                # kept from an earlier hold braked it from 6.7 to 0.15 m/s on
                # the tick it came near again, with nothing in its way.
                self.soft_v.pop(key, None)
            target = max(0, bisect.bisect_right(track.times, target_t + 1e-6) - 1)

            # Advance along the path only through clear poses — and never
            # back down it. Walking a blocked NPC backwards is a car seen
            # reversing: while the ego stood at a bus stop, a held car stepped
            # 0.6 m back down its own path again and again (H81 realtime,
            # 2026-09-15). Blocked, it stays exactly where it is, the way a
            # queue does.
            reached = cur
            if near:
                c_cur = None
                stand_in = (self.p.vru_standoff_m if self._is_vru(track.samples[cur]) else self.p.standoff_m) \
                    + self._margin(track.samples[cur])
                c_last = self._rect_clearance(track.samples[cur], ego_pose)
                stopped_for_ego = key in self.blocked_idx or self.soft_v.get(key, math.inf) < self.p.start_speed_mps
                for i in range(cur, target + 1):
                    ok = self._clear(track.samples[i], ego_pose, half_w,
                                     ego_speed_mps, dt_sec, obj_speed_mps=soft_speed)
                    if ok and i > cur and c_last < stand_in and stopped_for_ego:
                        # Stopped for the ego, within its standoff, it never
                        # steps nearer the ego —
                        # unless the recording had it as near the recorded
                        # vehicle then (a car passing the bus 0.4 m off, as
                        # recorded). A car released by its green crept along
                        # the flank of a bus standing mid-junction to 0.3 m
                        # (H78 july run, 2026-11-10).
                        c_i = self._rect_clearance(track.samples[i], ego_pose)
                        if c_i < c_last - 0.02:
                            q = self._rec_bus_at(track.times[i])
                            ok = q is not None and c_i >= self._rect_clearance(track.samples[i], q) - REC_WORSE_M
                        c_last = min(c_last, c_i) if ok else c_last
                    if not ok and i > cur:
                        # A step away from the ego is always clear: a car the
                        # ego came up behind at 8.6 m/s was held where it was
                        # and the ego ran into it (H82, 2026-11-09).
                        if c_cur is None:
                            c_cur = self._away_key(track.samples[cur], ego_pose)
                        ok = self._away_key(track.samples[i], ego_pose) > c_cur
                    if i == cur or (ok and self._npc_clear(track.samples[i], leaders)):
                        reached = i   # where it stands is never a step
                    else:
                        break
            else:
                reached = target

            # An object the ego has closed in on is NOT hidden (operator,
            # 2026-09-15): a car the ego brushes past at 0.4 m vanished from
            # the frame and the video showed a phantom collision, while the
            # metric saw nothing. It keeps its pose and is published; if the
            # ego reaches it, that is a contact and belongs in the report.
            # The one exception is an object that has never been shown and
            # whose first pose already overlaps the ego: that is a car
            # materialising inside the ego (its track begins where the sim ego
            # now stands), not a car the ego drove into.
            # Likewise a track that begins MOVING within the standoff of the
            # ego: it came along the way the ego now stands on (H78,
            # 2026-11-10: an oncoming car born 0.1 m from the nose of an ego
            # standing mid-turn was held there for 30 s, 143 contact frames).
            # Either is born at its first pose, up to the scene's time, that is
            # out of the ego's way — it comes out of the ego's shadow; until
            # there is one it waits unborn, its clock at its first pose.
            # A track the scene first presents later than its first second is
            # no birth (a scene jump lands mid-track): it is judged, once, on
            # how it moves there, and a car that drove up and now stands beside
            # the ego is shown (H82, 2026-11-11: the first car of an oncoming
            # queue, standing 2 m off the ego, was withheld for 10 s and came
            # out behind it).
            if near and key not in self.shown_idx:
                first = track.samples[reached]
                v_first = self._first_v.get(key)
                if v_first is None:
                    being_born = track.times[reached] - track.times[min(track.ext, reached)] < BIRTH_SEC
                    v_first = self.birth_v.get(key, 0.0) if being_born else self.speed_from(track, reached)
                    self._first_v[key] = v_first
                moving = v_first >= BORN_MOVING_MPS
                need = (self.p.vru_standoff_m if self._is_vru(first) else self.p.standoff_m) if moving else 0.0
                if self._rect_clearance(first, ego_pose) <= need:
                    last = bisect.bisect_right(track.times, frame_time_sec + 1e-6) - 1
                    born, i = None, track.next_diff[reached]
                    while i <= last:   # distinct poses only: a parked car repeats one
                        if self._rect_clearance(track.samples[i], ego_pose) > need:
                            born = i
                            break
                        i = track.next_diff[i]
                    if born is None:
                        self.delays[key] = max(delay, frame_time_sec - track.times[reached])
                        continue
                    reached = born
                    self.delays[key] = max(0.0, frame_time_sec - track.times[born])

            held = near and reached < target
            # The clock is CONTINUOUS time: snapping it back to the sample
            # grid each tick discarded sub-sample progress, and with the
            # timer period (~0.03 s) below the track's sample spacing
            # (~0.1 s) an object whose raw frame sat at/behind its track
            # start could never accumulate a full step — it froze at its
            # spawn pose for its whole life and was struck at 8.5 m/s, in
            # two passes running, surviving every coarser fix. Only a
            # clearance block pins the clock to the blocked sample.
            self.t_eff_prev[key] = (
                target_t if reached == target else track.times[reached]
            )
            # Release from clock governance only where it cannot show: either
            # the frame selector has come back in sync with this object's
            # clock (raw-following resumes without a visible step), or the
            # object is pure scenery beyond twice the active radius.
            rp = track.samples[reached].kinematics.pose_with_covariance.pose.position
            raw_now = max(track.times[0], frame_time_sec)
            # A near object never hands back to raw: under an oscillating
            # repeat window sync is transient, and releasing on it made the
            # displayed pose ping-pong 4.7 m between the clock pose and the
            # flapped raw pose right beside a passing ego (two brush contacts
            # billed ego_caused). Sync release is for objects already outside
            # the active radius, where raw-following resumes invisibly.
            synced = (
                abs(raw_now - track.times[reached]) <= 2.0 * dt_sec
                and not held
                and not near
            )
            scenery = (
                self._clearance_to_ego(rp.x, rp.y, ego_pose)
                > 2.0 * self.p.active_radius_m
            )
            if synced or scenery:
                self.governed.discard(key)
            new_delay = min(
                self.p.max_delay_sec,
                max(0.0, frame_time_sec - self.t_eff_prev[key]),
            )
            if new_delay > 0.0:
                self.delays[key] = new_delay
            else:
                self.delays.pop(key, None)

            if held:
                self.held_now.add(key)
                self.held_reason.setdefault(key, "ego")
            if key in self._stuck and self.held_reason.get(key) in ("ego", "plan"):
                self.held_reason[key] = "deadlock"
            if key in self.held_reason:
                # Held where its recording stands anyway (a queued or parked
                # car): no hold of the shield's to report. H39 (2026-11-09):
                # a car parked in the recording was logged held for 38 s and
                # the ego's wait behind it was called the scene's.
                k_raw = min(max(0, bisect.bisect_right(track.times, frame_time_sec) - 1), len(track.times) - 1)
                pr = track.samples[k_raw].kinematics.pose_with_covariance.pose.position
                if math.hypot(rp.x - pr.x, rp.y - pr.y) <= 0.5:
                    self.held_reason.pop(key, None)
                if key not in self._was_held:
                    self._was_held.add(key)
                    tw = track.samples[reached].kinematics.twist_with_covariance.twist.linear
                    speed = math.hypot(tw.x, tw.y)
                    p0 = track.samples[reached].kinematics.pose_with_covariance.pose.position
                    gap = max(0.1, self._clearance_to_ego(p0.x, p0.y, ego_pose))
                    need = (speed * speed) / (2.0 * gap)
                    # Only a moving ego can cut traffic off; a stationary one
                    # queueing followers is their ordinary stop, not an
                    # emergency, however fast the recording had them going.
                    if need > self.p.emergency_decel_mps2 and ego_speed_mps > 1.0:
                        self.emergency_stops += 1
                        print(
                            f"SHIELD emergency stop #{self.emergency_stops}: "
                            f"{speed:.1f} m/s with {gap:.1f} m to spare "
                            f"(needed {need:.1f} m/s^2)"
                        )
            else:
                self._was_held.discard(key)

            # Always impose: the frame's raw pose may be seconds away from this
            # object's clamped clock, and letting it through is the teleport.
            # Standing still = blocked at the sample it is shown at. Not "the
            # same sample as last tick": on a 0.02 s tick a moving car stays on
            # one 0.1 s sample for several ticks, and publishing those at 0 m/s
            # fed the model a car driving 7 m/s that claimed to stand
            # (2026-08-24 realtime, 2026-09-16). Not "held" alone either: a
            # blocked car's clock creeps within its sample between blocks, and
            # the ticks it is not "held" published its recorded 5.6 m/s while
            # it stood (H81 realtime, 2026-09-15).
            self.shown_idx[key] = reached
            if held:
                self.blocked_idx[key] = reached
            elif self.blocked_idx.get(key, -1) < reached:
                self.blocked_idx.pop(key, None)
            stopped = held or self.blocked_idx.get(key) == reached
            if held and self.p.soft_stop:
                self.soft_v[key] = 0.0   # pulls away from a standstill
            # A slowed car would stand on one recorded sample (~1 m apart at
            # 10 m/s) for seconds while reporting speed: shown between its
            # samples instead, when the next one is clear too.
            # Once shown between samples it never steps back to the sample.
            moving_f = None
            if (soft_speed is not None and soft_speed >= 0.1 and not held
                    and reached + 1 < len(track.times)):
                t0, t1 = track.times[reached], track.times[reached + 1]
                f = (self.t_eff_prev[key] - t0) / (t1 - t0) if t1 > t0 else 0.0
                if 0.0 < f < 1.0 and self._clear(
                    track.samples[reached + 1], ego_pose, half_w,
                    ego_speed_mps, dt_sec, obj_speed_mps=soft_speed,
                ):
                    moving_f = f
            last = self.shown_f.get(key)
            kept_f = last[1] if last is not None and last[0] == reached else 0.0
            f = max(moving_f or 0.0, kept_f)
            if f > 0.0:
                self.shown_f[key] = (reached, f)
            else:
                self.shown_f.pop(key, None)
            self._impose(
                obj, track.samples[reached],
                stopped=(stopped and moving_f is None)
                or (soft_speed is not None and soft_speed < 0.1),
                speed=soft_speed,
                toward=(track.samples[reached + 1], f) if f > 0.0 else None,
            )
            kept.append(obj)
        objects_msg.objects = kept
        # The plan ages after the tick it arrived for (a plan set just before
        # this tick is now): aged before, a fresh plan put a 5 m/s ego 0.5 m
        # ahead of itself.
        self._plan_age += dt_sec
        if self.p.soft_stop and self.p.npc_following:
            # What each object showed this tick, and whether it has moved off
            # its recording: next tick's leaders for NPC following.
            boxes = {}
            for obj in kept:
                k = bytes(obj.object_id.uuid)
                moved = (
                    k in self.held_now or k in self.soft_v or k in self.blocked_idx
                    or self.delays.get(k, 0.0) > self.p.npc_deviated_delay_sec
                )
                cls = getattr(obj, "classification", None) or []
                if cls and int(cls[0].label) == 0:
                    continue   # UNKNOWN: tracker clutter, no one queues behind it
                x, y, yaw, hl, hw = self._sample_box(obj)
                lin = obj.kinematics.twist_with_covariance.twist.linear
                boxes[k] = (x, y, yaw, hl, hw, moved, math.hypot(lin.x, lin.y))
            self._shown_boxes = boxes

    @staticmethod
    def _impose(obj, sample, *, stopped: bool, speed: float | None = None, toward=None) -> None:
        obj.kinematics.pose_with_covariance.pose = (
            sample.kinematics.pose_with_covariance.pose
        )
        if toward is not None:
            nxt, f = toward
            pose = copy.deepcopy(sample.kinematics.pose_with_covariance.pose)
            a, b = pose.position, nxt.kinematics.pose_with_covariance.pose.position
            a.x, a.y, a.z = a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), a.z + f * (b.z - a.z)
            obj.kinematics.pose_with_covariance.pose = pose
        if stopped:
            obj.kinematics.twist_with_covariance.twist.linear.x = 0.0
            obj.kinematics.twist_with_covariance.twist.linear.y = 0.0
        else:
            obj.kinematics.twist_with_covariance.twist = (
                sample.kinematics.twist_with_covariance.twist
            )
            if speed is not None:
                # Slowed by the soft stop: the recorded direction, the slower
                # speed — on a copy, the twist is the recorded sample's own.
                twist = copy.deepcopy(sample.kinematics.twist_with_covariance.twist)
                obj.kinematics.twist_with_covariance.twist = twist
                lin = twist.linear
                v = math.hypot(lin.x, lin.y)
                if v > 1e-3:
                    lin.x, lin.y = lin.x * speed / v, lin.y * speed / v
                else:
                    lin.x, lin.y = speed, 0.0
