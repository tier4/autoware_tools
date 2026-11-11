"""Leg-window evaluation: the recording is the timeline, one leg at a time.

Why (2026-09-10): an open-loop replay tests the planner only while the ego is
where — and when — the recorded vehicle was. Every rule that bent the scene
to a diverging ego (position lock, dwell anchor, departure hold, real time
never slower, re-sync on divergence) fixed one artefact and made another,
and a continuous run let divergence accumulate: the scene at minute three
depended on what the model did at minute two, so two model versions never
saw the same scene.

Here the recording is cut where the recorded vehicle departed: one window
per leg of the route (start -> stop, stop -> stop, stop -> end), opening
PREROLL_SEC before the recorded departure so the departure itself is in the
window. Each leg starts from the recorded state — the ego is driven along
the recorded track for LEAD_SEC (the model's history) and handed over — and
the scene plays on to the leg's end: at 1x while the ego waits for its
departure, and following the ego's POSITION once it has set off (the
recorded clock's rules 1-3 — the frame in which the recorded vehicle stood
where the ego stands, real time while the ego stands, the recorded
departure at a recorded stop — with rule 4 off, so a slower ego sees
slower traffic). When the ego falls behind the recording or leaves its track (DIVERGE_*)
the segment ends and the ego is set back onto the recording's current
state, so the scene stays the recording's. Falling behind ends it only once
the scene has no recorded object left around the ego (COVER_*): a late ego
that is still among the recording's objects is in its scene (operator,
2026-11-10: H78 waited 29 s for a crosswalk group, set off and was moved
100 m on at once); each segment is a scoring
window of its own (compute_metrics: a reset is a window boundary); an ego AHEAD of the recording
drives on (the next recorded wait lets the recording catch up) and is only
noted. Divergence is not a verdict (operator, 2026-09-10): the leg goes on,
and it is judged on what the model did — reached its arrival pose, made no
stop the recording did not make.

Planned cuts (operator, 2026-11-10): a leg is also cut where the recorded
vehicle stood CUT_MIN_STOP_SEC or more, at its recorded departure (find_cuts;
``leg_cuts: off`` restores goal-only legs). The cut is no routing goal and no
arrival: the window closes when the scene reaches that departure. An ego in
step with the recording there (CUT_SYNC_SEC, on the track) goes on unmoved —
the cut is a scoring boundary only; one behind or off the track is placed
where the recording stood LEAD_SEC before the departure and handed over along
it, the objects on. An ego that drives on past the stand (CUT_PASS_M) without
waiting is ahead of the scene by the wait it did not make: the window closes
there and the SCENE jumps to the recorded departure — the ego is not touched,
as at a leg's own departure (two-model run, 2026-11-10: H82 drove a whole bag
30 s ahead of the scene, among 2 objects instead of 5). So every model drives every junction from the recorded
state, and nothing moves the ego mid-window.

The departure (operator, 2026-09-10): a leg opens with NO objects — the ego
stands where the recording stood, with the lights, and nothing else — and
the model's own set-off is the measurement of how long it takes to get
rolling. The moment it sets off, the scene is jumped to the recorded
departure instant and the objects are brought in as they were then: every
model leaves into the gap the recorded vehicle left into, with no track
jumping under it (there was none) and no touch on the ego. Nothing in the
scene ever waits for, follows or avoids the ego, and every model version
faces every leg from the same state.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LEAD_SEC = 3.5            # driven along the recording before hand-over (model history: 3 s)
PREROLL_SEC = 5.0         # a leg window opens this long before the recorded departure
GOAL_ZONE_M = 15.0        # the recorded stop that belongs to a leg goal lies within this
GAP_SEC = 2.0             # a hole in the recording longer than this splits the leg around it
DIVERGE_LAG_SEC = 8.0     # lag GROWN while driving since the ego last set off: it is losing the recording
DIVERGE_LAG_MAX_SEC = 20.0  # absolute lag past which the scene is no longer the ego's, however it got there
DIVERGE_GAP_M = 40.0      # ...and only once the recorded vehicle is this far ahead: a bus still waiting to
                          # turn 33 m on keeps the traffic the ego meets its own (H78, 2026-11-09)
DEPART_LAG_SEC = 15.0     # a standing ego gets this long after the recorded departure (measured as delay)
DIVERGE_AHEAD_SEC = 8.0   # ahead of it: the traffic belongs to a later moment
DIVERGE_LATERAL_M = 4.0   # off the recorded track (another lane, a bay the recording skipped)
DIVERGE_HOLD_SEC = 1.0    # a divergence must persist this long
COVER_M = 30.0            # a shown object within this of the ego: the scene around it is still the recording's
COVER_LOST_SEC = 3.0      # ...and none for this long: it is not (only then is a slow ego reset)
MOVING_MPS = 0.5
STANDING_SEC = 3.0        # standing this long, then moving, is a departure (its lag = the delay)
MAX_SEGMENTS = 8          # re-initialisations per leg before the leg is abandoned
CLOCK_SAMPLE_SEC = 0.5    # (wall, scene) time samples in windows.json for the metrics
FINAL_GRACE_SEC = 10.0    # the last leg may run this long past the recording's end
ARRIVAL_DWELL_SEC = 5.0   # ...longer for an ego coming to rest at its stop: it stands this long there
ARRIVAL_EXTRA_SEC = 10.0  # ...at most this much longer
ARRIVAL_NEAR_M = 15.0     # an ego this close to the arrival pose is coming to rest there
LATE_ARRIVAL_SEC = 30.0   # an intermediate leg waits this long past its end for a late ego to arrive
ARRIVE_M = 5.0            # stopped within this of the leg's arrival pose = arrived
PASS_M = 6.0              # a drive-through arrival (cut pose) counts within this
STALE_SEC = 15.0
# Planned cuts (2026-11-10): a leg is also cut where the recorded vehicle
# stood, at its recorded departure. Every model leaves each such wait from the
# scene the recording left into; nothing moves the ego mid-leg.
CUT_MIN_STOP_SEC = 3.0    # a recorded stand this long is a cut
CUT_STOP_MPS = 0.3        # recorded speed below this is standing (lag.reference_track's dwell)
CUT_MERGE_M = 15.0        # stands this close in a row (a creeping queue) are one cut, at the last departure
CUT_MIN_LEG_SEC = 10.0    # no cut leaves a leg shorter than this
CUT_SYNC_SEC = 3.0        # at a cut, an ego at most this far behind (and on the track) goes on unmoved
CUT_PASS_M = 3.0          # an ego driving on this far past a cut's stand has not waited there:
                          # the scene jumps to that recorded departure (one stopping a little
                          # further on than the recording did is still waiting)
                          # The vehicles near the ego that drive away from it stay on their own
                          # clocks; the rest is off 0.25 s around the jump (scene_driver.carry_jump)


@dataclass
class LegWindow:
    index: int                                   # 1-based
    kind: str                                    # start | departure | final
    goal: tuple[float, float, float] | None      # routing goal (None: already set)
    arrival: tuple[float, float, float] | None   # reaching this completes the leg
    t_start: float                               # recorded seconds: the window opens
    t_end: float
    depart_t: float | None                       # recorded departure the window opens before
    drive_through: bool = False                  # passing the arrival pose counts
    arrive_t: float | None = None                # recorded arrival at this leg's goal
    gaps: tuple = ()                             # holes in the recording inside the window: (start, end)
    parent: int = 0                              # the route leg (stop to stop) this window belongs to
    opens: str = "goal"                          # goal: at a recorded departure from a leg goal | cut
    closes: str = "goal"                         # goal: at the leg's arrival | cut: at a recorded departure
    objects_from_start: bool = False             # a cut opens with the objects on
    cut_pose: tuple[float, float, float] | None = None    # opens at a cut: the recorded pose there
    close_pose: tuple[float, float, float] | None = None  # closes at a cut: the recorded pose there
    cut_arrive_t: float | None = None            # opens at a cut: when the recorded vehicle came to rest there


@dataclass
class Cut:
    t_arrive: float     # the recorded vehicle came to rest (first stand of a merged run)
    t_depart: float     # ...and left (the last stand's departure): the cut instant
    x: float
    y: float
    yaw: float


def find_cuts(reference, goals, *, min_stop_sec: float = CUT_MIN_STOP_SEC) -> list[Cut]:
    """Where a leg is cut besides its goals: every recorded stand of
    ``min_stop_sec`` or more, at its departure. Not the stand the recording
    opens with, nor one it ends in, nor one within GOAL_ZONE_M of a leg goal
    (a leg boundary already), nor one touching a hole in the recording.
    Stands within CUT_MERGE_M of each other in a row are one cut (a queue
    creeping up to a light), at the last one's departure."""
    if not reference:
        return []
    stands: list[tuple[int, int]] = []        # (first index, departure index)
    i, n = 0, len(reference)
    while i < n:
        if reference[i].speed_mps >= CUT_STOP_MPS:
            i += 1
            continue
        j = i
        while j < n and reference[j].speed_mps < CUT_STOP_MPS:
            j += 1
        # j: the first sample moving again (n: the recording ends standing)
        if reference[min(j, n - 1)].stamp_sec - reference[i].stamp_sec >= min_stop_sec:
            stands.append((i, j))
        i = j
    merged: list[list[int]] = []
    for a, b in stands:
        if merged:
            pa = reference[merged[-1][0]]
            if math.hypot(reference[a].x - pa.x, reference[a].y - pa.y) <= CUT_MERGE_M or math.hypot(
                reference[a].x - reference[merged[-1][1] - 1].x, reference[a].y - reference[merged[-1][1] - 1].y
            ) <= CUT_MERGE_M:
                merged[-1][1] = b
                continue
        merged.append([a, b])
    out: list[Cut] = []
    for a, b in merged:
        if a == 0 or b >= n:
            continue                             # the recording opens or ends standing here
        group = reference[a:b]
        if any(
            math.hypot(s_.x - float(g[0]), s_.y - float(g[1])) <= GOAL_ZONE_M for g in goals for s_ in (group[0], group[-1])
        ):
            continue
        lo, hi = max(0, a - 1), min(n - 1, b)
        if any(reference[k + 1].stamp_sec - reference[k].stamp_sec > GAP_SEC for k in range(lo, hi)):
            continue                             # a hole in the recording touches it
        d = reference[b]
        out.append(Cut(reference[a].stamp_sec, d.stamp_sec, d.x, d.y, d.yaw_rad))
    return out


def _cut_times(t_start: float, t_end: float, cuts: list[Cut]) -> list[Cut]:
    """The cuts that split the window (t_start, t_end), none leaving a piece
    shorter than CUT_MIN_LEG_SEC."""
    kept: list[Cut] = []
    last = t_start
    for c in sorted(cuts, key=lambda c: c.t_depart):
        if c.t_depart - last >= CUT_MIN_LEG_SEC and t_end - c.t_depart >= CUT_MIN_LEG_SEC:
            kept.append(c)
            last = c.t_depart
    return kept


def _dwell_near(reference, goal, zone_m: float = GOAL_ZONE_M) -> tuple[float, float]:
    """(arrival, departure) of the recorded vehicle at a leg goal: the first and
    last stopped sample within the zone; a goal it drove past gives the pass."""
    gx, gy = float(goal[0]), float(goal[1])
    stopped = [
        s.stamp_sec for s in reference
        if s.speed_mps < 0.3 and math.hypot(s.x - gx, s.y - gy) <= zone_m
    ]
    if stopped:
        return stopped[0], stopped[-1]
    best = min(reference, key=lambda s: math.hypot(s.x - gx, s.y - gy))
    return best.stamp_sec, best.stamp_sec


def plan_legs(
    reference,
    goals: list[tuple[float, float, float]],
    *,
    cut_final: bool = False,
    cut_end_goal: tuple[float, float, float] | None = None,
    cuts: list[Cut] | None = None,
) -> list[LegWindow]:
    """One window per leg. ``reference``: the recording's ego (EgoSample list,
    time-ordered); ``goals``: the route's leg goals in order (the last is the
    recording's end pose). A cut recording (``cut_final``) routes its last leg
    to ``cut_end_goal`` and completes it drive-through at the cut pose.
    ``cuts`` (find_cuts) split each leg further at recorded departures: only
    the first piece routes (``goal``), only the last arrives (``arrival``)."""
    if not reference or not goals:
        return []
    t0 = reference[0].stamp_sec
    t_last = reference[-1].stamp_sec
    dwells = [_dwell_near(reference, g) for g in goals]
    # The recording usually opens standing (a terminal, a bus stop): that first
    # departure is a departure too, and the first leg measures it.
    depart0 = None
    if reference[0].speed_mps < 0.3:
        moving = next((s.stamp_sec for s in reference if s.speed_mps >= 0.3), None)
        if moving is not None and moving - t0 >= STANDING_SEC:
            depart0 = moving
    legs: list[LegWindow] = []
    for k, goal in enumerate(goals):
        final = k == len(goals) - 1
        t_start = t0 if k == 0 else max(t0, dwells[k - 1][1] - PREROLL_SEC)
        t_end = t_last if final else max(t_start + 1.0, dwells[k][1] - PREROLL_SEC)
        legs.append(
            LegWindow(
                index=k + 1,
                kind="start" if k == 0 else ("final" if final else "departure"),
                goal=(cut_end_goal if (final and cut_final and cut_end_goal is not None) else goal),
                arrival=goal,
                t_start=t_start,
                t_end=t_end,
                depart_t=depart0 if k == 0 else dwells[k - 1][1],
                drive_through=bool(final and cut_final),
                arrive_t=dwells[k][0],
                parent=k + 1,
            )
        )
    if cuts:
        legs = _split_at_cuts(legs, cuts)
    return _mark_gaps(legs, reference)


def _split_at_cuts(legs: list[LegWindow], cuts: list[Cut]) -> list[LegWindow]:
    out: list[LegWindow] = []
    for leg in legs:
        pieces = _cut_times(leg.t_start, leg.t_end, cuts)
        if not pieces:
            out.append(leg)
            continue
        bounds = [None] + pieces + [None]
        for k in range(len(pieces) + 1):
            opened, closed = bounds[k], bounds[k + 1]
            out.append(
                LegWindow(
                    index=0,
                    kind=leg.kind,
                    goal=leg.goal if opened is None else None,
                    arrival=leg.arrival if closed is None else None,
                    t_start=leg.t_start if opened is None else opened.t_depart,
                    t_end=leg.t_end if closed is None else closed.t_depart,
                    depart_t=leg.depart_t if opened is None else opened.t_depart,
                    drive_through=leg.drive_through if closed is None else False,
                    arrive_t=leg.arrive_t if closed is None else None,
                    parent=leg.parent,
                    opens="goal" if opened is None else "cut",
                    closes="goal" if closed is None else "cut",
                    objects_from_start=opened is not None,
                    cut_pose=None if opened is None else (opened.x, opened.y, opened.yaw),
                    cut_arrive_t=None if opened is None else opened.t_arrive,
                    close_pose=None if closed is None else (closed.x, closed.y, closed.yaw),
                )
            )
    return [LegWindow(**{**asdict(w), "index": i + 1}) for i, w in enumerate(out)]


def _mark_gaps(legs: list[LegWindow], reference) -> list[LegWindow]:
    """A hole in the recording (2026-08-24: 60 s and 192 m missing at 180 s)
    is not a scene, and not a reason to move the ego either (operator,
    2026-09-10): the ego drives on to the leg's goal through it with the
    objects withheld and the divergence checks suspended, and the scene
    resumes with the recording."""
    gaps = [
        (a.stamp_sec, b.stamp_sec)
        for a, b in zip(reference, reference[1:])
        if b.stamp_sec - a.stamp_sec > GAP_SEC
    ]
    if not gaps:
        return legs
    return [
        LegWindow(**{**asdict(leg), "gaps": tuple(g for g in gaps if leg.t_start < g[0] < leg.t_end)})
        for leg in legs
    ]


def passed_cut(
    x: float, y: float, yaw: float, v: float, close_pose, ts: float, t_depart: float, lateral_max_m: float,
) -> bool:
    """Has the ego driven on past a cut's stand without waiting? It is
    CUT_PASS_M or more beyond the recorded stand pose along that pose's
    heading, on the track, heading the same way and moving, while the scene
    is still more than CUT_SYNC_SEC short of the recorded departure."""
    if close_pose is None or v < MOVING_MPS or ts >= t_depart - CUT_SYNC_SEC:
        return False
    cx, cy, cyaw = float(close_pose[0]), float(close_pose[1]), float(close_pose[2])
    lon = (x - cx) * math.cos(cyaw) + (y - cy) * math.sin(cyaw)
    lat = -(x - cx) * math.sin(cyaw) + (y - cy) * math.cos(cyaw)
    return lon >= CUT_PASS_M and lon <= 60.0 and abs(lat) <= lateral_max_m and math.cos(yaw - cyaw) > 0.0


JUMP_SETTLE_SEC = 1.0     # after a jump, a fed sample of the scene before it is ignored this long at most


def before_jump(t: float, target: float, came_from: float) -> bool:
    """Is the fed recorded time ``t`` still the scene before a jump to
    ``target``? Nearer where the scene came from than where it was sent. The
    harness read such a sample 0.1 s after a jump and wrote it down as the
    scene's time: the (wall, scene) map ran 17 s back for half a second (H82,
    2026-11-11; one job in eight with a jump had one)."""
    return abs(t - came_from) <= abs(t - target)


def is_last_leg(kind: str, t_end: float, recording_end: float) -> bool:
    """The leg the recording ends in. A recording with one leg has no
    "final" one (its only leg is the "start"): H95 (2026-11-11) drove through
    the recording's last pose and 290 m on, the scene held 0.1 s short of its
    end for 40 s, until a red light stopped it."""
    return kind == "final" or t_end >= recording_end - 0.5


def final_grace_over(
    now: float, end_reached_wall: float, *, arrival_dist_m: float | None, arrived_wall: float | None,
    drive_through: bool,
) -> bool:
    """Does the last window close now? FINAL_GRACE_SEC after the recording's
    end — unless the ego is coming to rest at its stop (within ARRIVAL_NEAR_M of
    the arrival pose, not arrived ARRIVAL_DWELL_SEC ago yet): its dwell is the
    stop's service, and closing on it scored a stop 0.6 m off the sign as never
    served (H78, 2026-11-10). At most ARRIVAL_EXTRA_SEC longer."""
    over = now - end_reached_wall
    if over < FINAL_GRACE_SEC:
        return False
    if drive_through or arrival_dist_m is None or over >= FINAL_GRACE_SEC + ARRIVAL_EXTRA_SEC:
        return True
    if arrival_dist_m > ARRIVAL_NEAR_M:
        return True
    return arrived_wall is not None and now - arrived_wall >= ARRIVAL_DWELL_SEC


def scene_uncovered(nearest_m: float | None, covered_wall: float, now: float) -> bool:
    """No shown object within COVER_M of the ego for COVER_LOST_SEC. A scene
    driver that does not report it (``nearest_m`` None) counts as uncovered:
    the time rules alone then decide, as before 2026-11-10."""
    return nearest_m is None or now - covered_wall >= COVER_LOST_SEC


def falling_behind(lag_drive: float | None, lag_base: float, gap: float | None, *, uncovered: bool) -> bool:
    """A driving ego that has lost the recording: its driving lag grew
    DIVERGE_LAG_SEC since it last set off (or reached DIVERGE_LAG_MAX_SEC), the
    recorded vehicle is DIVERGE_GAP_M or more ahead, and the scene has no
    recorded object left around the ego (``uncovered``)."""
    if lag_drive is None or not uncovered:
        return False
    if gap is not None and gap < DIVERGE_GAP_M:
        return False
    return lag_drive - lag_base >= DIVERGE_LAG_SEC or lag_drive >= DIVERGE_LAG_MAX_SEC


def _lateral_offset_m(track, x: float, y: float, yaw: float) -> float:
    import numpy as np

    t, tx, ty, tyaw = track[0], track[1], track[2], track[3]
    d2 = (tx - x) ** 2 + (ty - y) ** 2
    if yaw == yaw:
        dyaw = np.abs((tyaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)
        masked = np.where(dyaw <= np.pi / 2.0, d2, np.inf)
        if np.isfinite(masked).any():
            d2 = masked
    return math.sqrt(float(d2.min()))


def run_legs(
    cfg: Any,
    env: dict[str, str],
    *,
    legs: list[LegWindow],
    reference,
    track,
    interventions: dict[str, float],
    output_dir: Path,
    on_leg=None,
) -> tuple[str, str | None]:
    """Drive every leg window; write windows.json. Returns (status, error) for
    the harness — a leg the model failed is a result, not an error.
    ``on_leg(index, total)`` is called as each leg opens (live monitor)."""
    try:
        import rclpy  # noqa: F401
        from autoware_adapi_v1_msgs.msg import RouteState
        from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool, Float64, String
    except ImportError as exc:
        return "failed", f"window_eval_unavailable: {exc}"

    from dp_multi_eval.job_waiters import (
        close_job_context,
        job_executor,
        open_job_context,
        recorded_lag_sec,
    )
    from dp_multi_eval.lag import recorded_gap_m
    from dp_multi_eval.route_control import advance_route, engage_autonomous

    stamps = [s.stamp_sec for s in reference]

    def ref_at(t: float):
        i = min(bisect.bisect_left(stamps, t), len(reference) - 1)
        return reference[i]

    context = open_job_context(env)
    old_env = os.environ.copy()
    os.environ.update(env)
    latched = QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
    )

    class Window(Node):
        def __init__(self) -> None:
            super().__init__("dp_multi_eval_window", context=context)
            self.x = self.y = self.yaw = self.v = float("nan")
            self.last_odom: float | None = None
            self.rec: Any = None
            self.rec_t: float | None = None
            self._jump: tuple[float, float, float] | None = None   # (target, came from, wall) of a jump not yet seen
            self.rec_v = float("nan")
            self.route_state: int | None = None
            self.through: tuple[float, str] | None = None   # (wall time, "id x y")
            self.nearest_m: float | None = None      # nearest shown object (scene driver)
            self.covered_wall = 0.0                  # last wall time an object was within COVER_M
            self.create_subscription(String, "/perception_reproducer/through_ego", self._cb_through, 10)
            self.create_subscription(Float64, "/perception_reproducer/nearest_object_m", self._cb_nearest, 10)
            self.create_subscription(Odometry, "/localization/kinematic_state", self._cb_ego, 1)
            self.create_subscription(
                Odometry, "/perception_reproducer/rosbag_ego_odom", self._cb_rec, 1
            )
            self.create_subscription(RouteState, "/api/routing/state", self._cb_route, latched)
            self._time_pub = self.create_publisher(Float64, "/perception_reproducer/scene_time", 1)
            self._jump_pub = self.create_publisher(Float64, "/perception_reproducer/scene_jump", 1)
            self._hold_pub = self.create_publisher(Bool, "/perception_reproducer/hold", latched)
            self._objects_pub = self.create_publisher(Bool, "/perception_reproducer/objects", latched)
            self._follow_pub = self.create_publisher(Bool, "/perception_reproducer/follow_ego", latched)
            self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose3d", 1)
            self._twist_pub = self.create_publisher(TwistStamped, "/simulation/input/initialtwist", 1)

        def _cb_ego(self, msg: Any) -> None:
            q = msg.pose.pose.orientation
            self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.x = float(msg.pose.pose.position.x)
            self.y = float(msg.pose.pose.position.y)
            tw = msg.twist.twist.linear
            self.v = math.hypot(float(tw.x), float(tw.y))
            self.last_odom = time.time()

        def _cb_rec(self, msg: Any) -> None:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if self._jump is not None:
                target, came_from, wall = self._jump
                if before_jump(t, target, came_from) and time.time() - wall < JUMP_SETTLE_SEC:
                    return   # the scene before the jump, still in flight
                self._jump = None
            self.rec = msg
            self.rec_t = t
            tw = msg.twist.twist.linear
            self.rec_v = math.hypot(float(tw.x), float(tw.y))

        def _cb_route(self, msg: Any) -> None:
            self.route_state = int(msg.state)

        def _cb_nearest(self, msg: Any) -> None:
            self.nearest_m = float(msg.data)
            if self.nearest_m <= COVER_M:
                self.covered_wall = time.time()

        def _cb_through(self, msg: Any) -> None:
            # Only a vehicle whose centre is inside the ego's body ("core"); the
            # reproducer also withholds next-lane cars brushing its wider box.
            if str(msg.data).endswith(" core"):
                self.through = (time.time(), str(msg.data))

        # -- commands
        def spin(self, sec: float) -> None:
            end = time.time() + sec
            while time.time() < end and context.ok():
                executor.spin_once(timeout_sec=0.05)

        def hold(self, on: bool) -> None:
            self._hold_pub.publish(Bool(data=bool(on)))

        def _sent(self, t: float) -> None:
            if self.rec_t is not None and abs(float(t) - self.rec_t) >= 0.5:
                self._jump = (float(t), self.rec_t, time.time())

        @property
        def jump_pending(self) -> bool:
            """The fed recorded time has not shown the last jump yet."""
            return self._jump is not None and time.time() - self._jump[2] < JUMP_SETTLE_SEC

        def jump(self, t: float) -> None:
            self._time_pub.publish(Float64(data=float(t)))
            self._sent(t)

        def scene_jump(self, t: float) -> None:
            """Recorded time to ``t`` under an ego that is not moved."""
            self._jump_pub.publish(Float64(data=float(t)))
            self._sent(t)

        def objects(self, on: bool) -> None:
            self._objects_pub.publish(Bool(data=bool(on)))

        def follow_mode(self, on: bool) -> None:
            self._follow_pub.publish(Bool(data=bool(on)))

        def place(self, x: float, y: float, z: float, yaw: float, vx: float) -> None:
            stamp = self.get_clock().now().to_msg()
            pose = PoseWithCovarianceStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = "map"
            pose.pose.pose.position.x = float(x)
            pose.pose.pose.position.y = float(y)
            pose.pose.pose.position.z = float(z)
            pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.pose.orientation.w = math.cos(yaw / 2.0)
            pose.pose.covariance = [0.0] * 36
            for i in (0, 7, 14, 21, 28, 35):
                pose.pose.covariance[i] = 0.01
            self._pose_pub.publish(pose)
            twist = TwistStamped()
            twist.header.stamp = stamp
            twist.header.frame_id = "base_link"
            twist.twist.linear.x = float(vx)
            self._twist_pub.publish(twist)

        def place_odom(self, odom: Any) -> None:
            stamp = self.get_clock().now().to_msg()
            pose = PoseWithCovarianceStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = "map"
            pose.pose.pose = odom.pose.pose
            pose.pose.covariance = [0.0] * 36
            for i in (0, 7, 14, 21, 28, 35):
                pose.pose.covariance[i] = 0.01
            self._pose_pub.publish(pose)
            twist = TwistStamped()
            twist.header.stamp = stamp
            twist.header.frame_id = "base_link"
            twist.twist = odom.twist.twist
            self._twist_pub.publish(twist)

        def wait_ego_near(self, x: float, y: float, timeout: float, tol: float = 2.0) -> bool:
            end = time.time() + timeout
            while time.time() < end and context.ok():
                executor.spin_once(timeout_sec=0.05)
                if self.x == self.x and math.hypot(self.x - x, self.y - y) <= tol:
                    return True
            return False

        def follow(self, sec: float) -> Any:
            """Drive the ego along the recorded frames for ``sec``; the last frame."""
            last = None
            end = time.time() + sec
            while time.time() < end and context.ok():
                executor.spin_once(timeout_sec=0.02)
                if self.rec is not None and self.rec is not last:
                    last = self.rec
                    self.place_odom(last)
            return last

    node = Window()
    executor = job_executor(context, node)
    results: list[dict[str, Any]] = []
    reinits = 0
    out_path = Path(output_dir) / "windows.json"

    def dump() -> None:
        out_path.write_text(
            json.dumps(
                {
                    "params": {
                        "lead_sec": LEAD_SEC, "preroll_sec": PREROLL_SEC,
                        "diverge_lag_sec": DIVERGE_LAG_SEC, "diverge_lag_max_sec": DIVERGE_LAG_MAX_SEC,
                        "depart_lag_sec": DEPART_LAG_SEC, "departure_reveal": True,
                        "late_arrival_sec": LATE_ARRIVAL_SEC,
                        "diverge_ahead_sec": DIVERGE_AHEAD_SEC,
                        "diverge_lateral_m": DIVERGE_LATERAL_M, "max_segments": MAX_SEGMENTS,
                        "cover_m": COVER_M, "cover_lost_sec": COVER_LOST_SEC,
                        "cut_min_stop_sec": CUT_MIN_STOP_SEC, "cut_merge_m": CUT_MERGE_M,
                        "cut_min_leg_sec": CUT_MIN_LEG_SEC, "cut_sync_sec": CUT_SYNC_SEC,
                        "cut_pass_m": CUT_PASS_M,
                    },
                    "legs": results,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )

    def handover(rec: dict[str, Any], t_open: float, revealed: bool = True) -> None:
        # The ego is driven along the track by the scene's own frames, so the
        # scene must run at 1x for it; after the departure the scene follows
        # the ego's position again.
        node.follow_mode(False)
        last = node.follow(LEAD_SEC)
        if revealed:
            node.follow_mode(True)
        t_h = node.rec_t if node.rec_t is not None else t_open
        rec["segments"].append(
            {
                "t_open": round(t_open, 2),
                "t_handover": round(t_h, 2),
                "wall_handover": round(time.time(), 3),
                "x": round(node.x, 2), "y": round(node.y, 2), "v": round(node.v, 2),
                "t_end": None, "ended_by": None,
            }
        )
        print(
            f"[window] leg {rec['index']}: handed over at t={t_h - stamps[0]:.1f}s "
            f"({node.x:.1f},{node.y:.1f}) v={node.v:.2f} m/s (segment {len(rec['segments'])})"
        )

    status, error = "done", None
    # What a window closed at a planned cut hands to the next: whether the ego
    # was in step with the recording there (then nothing moves it), and its
    # standing and lag bookkeeping.
    carry: dict[str, Any] | None = None
    placements = seamless_cuts = scene_jumps = 0
    try:
        for leg in legs:
            rec: dict[str, Any] = {
                **asdict(leg),
                "segments": [], "arrived_t": None, "ended_by": None,
                "wall_start": round(time.time(), 3), "wall_end": None,
            }
            results.append(rec)
            t_open = leg.t_start
            print(
                f"[window] leg {leg.index}/{len(legs)} ({leg.kind}): recorded "
                f"{leg.t_start - stamps[0]:.1f}-{leg.t_end - stamps[0]:.1f}s"
                + (f", recorded departure at {leg.depart_t - stamps[0]:.1f}s" if leg.depart_t else "")
            )
            if on_leg is not None:
                on_leg(leg.index, len(legs))
            standing_since: float | None = None
            handed, carry = carry, None
            # The first set-off of a window opened at a planned cut is that
            # departure's measurement, however briefly the ego stood.
            open_departure = leg.opens == "cut"
            lag_carry = 0.0
            if leg.opens == "cut":
                if handed is not None and handed.get("passed"):
                    # It drove on past the stand: the scene comes to it. The
                    # ego is not touched, and the vehicles driving away from
                    # it stay (three cars ahead of it vanished in one frame,
                    # H82 2026-11-11); what the new instant brings is off for
                    # a moment first, so nothing leaps in the planner's history.
                    scene_jumps += 1
                    t_from = node.rec_t if node.rec_t is not None else t_open
                    node.scene_jump(t_open)
                    node.spin(0.05)
                    node.follow_mode(True)
                    rec.setdefault("jumps", []).append(
                        {"wall": round(time.time(), 3), "from": round(t_from, 2), "to": round(t_open, 2)}
                    )
                    rec["segments"].append(
                        {
                            "t_open": round(t_open, 2), "t_handover": round(t_open, 2),
                            "wall_handover": round(time.time(), 3),
                            "x": round(node.x, 2), "y": round(node.y, 2), "v": round(node.v, 2),
                            "t_end": None, "ended_by": None, "seamless": True, "scene_jumped": True,
                            "skipped_sec": round(t_open - t_from, 1),
                        }
                    )
                    open_departure = False
                    print(
                        f"[window] leg {leg.index}: planned cut — the ego drove on past the stand; scene jumped "
                        f"{t_open - t_from:+.1f}s to the recorded departure, the ego untouched"
                    )
                elif handed is not None and handed["sync"]:
                    # In step: the window boundary is a scoring boundary only.
                    # No hold, no jump, no placement — the scene and the ego go on.
                    seamless_cuts += 1
                    rec["segments"].append(
                        {
                            "t_open": round(t_open, 2),
                            "t_handover": round(node.rec_t if node.rec_t is not None else t_open, 2),
                            "wall_handover": round(time.time(), 3),
                            "x": round(node.x, 2), "y": round(node.y, 2), "v": round(node.v, 2),
                            "t_end": None, "ended_by": None, "seamless": True,
                        }
                    )
                    standing_since = handed["standing_since"]
                    lag_carry = handed["lag_base"]
                    print(f"[window] leg {leg.index}: planned cut — ego in step with the recording, nothing moved")
                else:
                    # Behind, or off the track: placed where the recording stood
                    # LEAD_SEC before its departure, driven along it and handed
                    # over with the objects on.
                    placements += 1
                    node.follow_mode(False)
                    node.hold(True)
                    node.spin(0.1)
                    node.objects(True)
                    node.jump(t_open - LEAD_SEC)
                    node.spin(0.3)
                    s0 = ref_at(t_open - LEAD_SEC)
                    node.place(s0.x, s0.y, s0.z, s0.yaw_rad, s0.speed_mps)
                    if not node.wait_ego_near(s0.x, s0.y, 8.0):
                        print(f"[window] leg {leg.index}: the ego did not reach the cut pose; handing over where it is")
                    node.hold(False)
                    handover(rec, t_open, revealed=True)
                    rec["segments"][-1]["placed"] = True
                    if handed is not None:
                        rec["segments"][-1]["placed_lag_sec"] = handed.get("lag_sec")
                    standing_since = node.rec_t if (node.v == node.v and node.v < MOVING_MPS) else None
                    print(f"[window] leg {leg.index}: planned cut — ego placed on the recording (it was "
                          f"{'out of step' if handed is not None else 'not handed over'})")
            # -- open the window: scene held at the lead-in instant, ego placed there.
            # A leg that opens at a recorded departure opens without objects.
            if leg.opens != "cut":
                node.hold(True)
                node.follow_mode(False)
                node.objects(leg.depart_t is None)
                # Let the hold and following-off land first: a jump the scene
                # driver took while still following was overridden by the ego's
                # position (H81 leg 3 opened 24 s late, 2026-10-28).
                node.spin(0.1)
                node.jump(t_open - LEAD_SEC)
                node.spin(0.5)
                s0 = ref_at(t_open - LEAD_SEC)
                node.place(s0.x, s0.y, s0.z, s0.yaw_rad, 0.0)
                if not node.wait_ego_near(s0.x, s0.y, 8.0):
                    rec["ended_by"] = "place_failed"
                    rec["wall_end"] = round(time.time(), 3)
                    dump()
                    continue
                if leg.index > 1 and leg.goal is not None:
                    ok, msg = advance_route(
                        cfg, env, leg.goal, allow_goal_substitution=(leg.kind == "final")
                    )
                    if not ok:
                        rec["ended_by"] = f"route_failed: {msg}"
                        rec["wall_end"] = round(time.time(), 3)
                        dump()
                        continue
                    ok, msg = engage_autonomous(env, timeout_sec=cfg.auto_engage_timeout_sec)
                    if not ok:
                        rec["ended_by"] = f"engage_failed: {msg}"
                        rec["wall_end"] = round(time.time(), 3)
                        dump()
                        continue
                node.hold(False)
                handover(rec, t_open, revealed=leg.depart_t is None)
                standing_since = node.rec_t if (node.v == node.v and node.v < MOVING_MPS) else None

            budget = (leg.t_end - leg.t_start) * 1.5 + 90.0 + 15.0 * MAX_SEGMENTS
            diverge_since: float | None = None
            diverge_kind = ""
            last_check = 0.0
            last_log = 0.0
            # Lag when the ego last set off. Driving is judged on the lag it
            # ADDS from there: a departure 7 s late is the departure's fault
            # (recorded below), not a reason to fail the drive that follows.
            lag_base = lag_carry
            rec.setdefault("departures", [])
            rec.setdefault("jumps", [])
            rec.setdefault("clock", [])
            last_clock = 0.0
            # objects shown from the start: no departure to wait for, or a planned cut
            revealed = leg.depart_t is None or leg.opens == "cut"
            end_reached_wall: float | None = None   # the scene has run out (clamped at t_end)
            arrived_wall: float | None = None       # wall time the ego arrived at the leg's pose
            in_gap = False                          # the recording has a hole here: no objects, no checks
            objects_back = False                    # objects restored after the hole (the ego may still be mid-hole)
            rec["gaps"] = [[round(g0, 2), round(g1, 2)] for g0, g1 in leg.gaps]
            while context.ok():
                node.spin(0.1)
                now = time.time()
                # First, whatever else this tick does: several branches below
                # `continue`, and one of them (a late ego's arrival wait) once
                # skipped this for 20 min (H95, 2026-10-28).
                if now - rec["wall_start"] > budget:
                    rec["ended_by"] = "timeout"
                    break
                if node.last_odom is not None and now - node.last_odom > STALE_SEC:
                    rec["ended_by"] = "stale"
                    status, error = "failed", "ego pose stopped updating; the simulator is no longer publishing"
                    break
                if node.rec_t is None or node.x != node.x:
                    continue
                ts = node.rec_t
                seg = rec["segments"][-1]
                if now - last_clock >= CLOCK_SAMPLE_SEC and not node.jump_pending:
                    last_clock = now
                    rec["clock"].append([round(now, 3), round(ts, 3)])
                # a hole in the recording: the ego drives on by itself. The
                # checks stay off past the hole's end until the ego is back on
                # the recorded track — when the recording resumed, the ego was
                # still mid-hole, 46 m from any recorded sample, and the
                # lateral check moved it to the goal after all (2026-08-24).
                # ts is the stamp of the frame the reproducer shows, and inside
                # a hole that is the frame before it, then the one after: the
                # scene reads g0 for half the hole and g1 for the rest, never
                # in between, so both ends count as inside.
                gap_now = any(g0 - 0.05 <= ts <= g1 + 0.05 for g0, g1 in leg.gaps)
                if gap_now and not in_gap:
                    in_gap = True
                    node.objects(False)
                    print(f"[window] leg {leg.index}: entering a hole in the recording at t={ts - stamps[0]:.1f}s — objects withheld, checks off")
                elif in_gap and not gap_now:
                    if not objects_back:
                        objects_back = True
                        node.objects(revealed)
                        print(f"[window] leg {leg.index}: the recording resumed at t={ts - stamps[0]:.1f}s — objects back; checks once the ego is on the track")
                    if _lateral_offset_m(track, node.x, node.y, node.yaw) < DIVERGE_LATERAL_M:
                        in_gap = False
                        print(f"[window] leg {leg.index}: ego back on the recorded track at t={ts - stamps[0]:.1f}s — checks on")
                # arrival
                if rec["arrived_t"] is None and leg.arrival is not None:
                    dist = math.hypot(node.x - leg.arrival[0], node.y - leg.arrival[1])
                    arrived = (
                        (leg.drive_through and dist <= PASS_M)
                        or (dist <= ARRIVE_M and node.v <= 0.1)
                        or (node.route_state == RouteState.ARRIVED)
                    )
                    if arrived:
                        rec["arrived_t"] = round(ts, 2)
                        arrived_wall = now
                        print(f"[window] leg {leg.index}: arrived at t={ts - stamps[0]:.1f}s (dist {dist:.1f} m, v={node.v:.2f})")
                        if is_last_leg(leg.kind, leg.t_end, stamps[-1]) and leg.drive_through:
                            # A cut pose: nothing to dwell for. A final bus stop
                            # is served like any other — the window runs on to
                            # the recording's end so the dwell is measured
                            # (2026-08-24: ended 0 s after arriving, "passed").
                            rec["ended_by"] = "arrived"
                            break
                # window end. The scene clamps at the recording's end, so the
                # final leg's grace is wall time from the moment it got there.
                # A leg that ends where the recording ends is the last one,
                # whatever its kind: the scene clamps at the recording's end, so
                # an intermediate leg's grace (recorded time past t_end) never
                # ran out (H63, H95: single-leg bags, 2026-10-28).
                if leg.closes == "cut":
                    # The ego drove on past the stand without waiting: the
                    # window closes here, the next opens with the scene at
                    # the recorded departure.
                    if not in_gap and passed_cut(
                        node.x, node.y, node.yaw, node.v, leg.close_pose, ts, leg.t_end, DIVERGE_LATERAL_M
                    ):
                        seg.update(
                            {
                                "at": [round(node.x, 2), round(node.y, 2)], "ego_v": round(node.v, 2),
                                "passed_cut": True,
                            }
                        )
                        rec["cut_passed"] = {
                            "t": round(ts, 2), "wall": round(now, 3), "skipped_sec": round(leg.t_end - ts, 1),
                            "at": [round(node.x, 2), round(node.y, 2)], "ego_v": round(node.v, 2),
                        }
                        carry = {"sync": True, "passed": True, "standing_since": None, "lag_base": 0.0, "lag_sec": None}
                        rec["ended_by"] = "cut"
                        print(
                            f"[window] leg {leg.index}: ego drove on past the cut's stand at t={ts - stamps[0]:.1f}s "
                            f"(v={node.v:.2f}), {leg.t_end - ts:.1f}s before the recorded departure"
                        )
                        break
                    # A planned cut closes the window at the recorded departure,
                    # wherever the ego is: no arrival to wait for.
                    if ts >= leg.t_end - 0.05:
                        lag_c = recorded_lag_sec(track, node.x, node.y, node.yaw, ts)
                        lateral_c = _lateral_offset_m(track, node.x, node.y, node.yaw)
                        sync = (
                            not in_gap
                            and lateral_c < DIVERGE_LATERAL_M
                            and (lag_c is None or lag_c <= CUT_SYNC_SEC)
                        )
                        seg.update(
                            {
                                "lag_sec": None if lag_c is None else round(lag_c, 1),
                                "lateral_m": round(lateral_c, 1),
                                "at": [round(node.x, 2), round(node.y, 2)],
                                "ego_v": round(node.v, 2),
                            }
                        )
                        carry = {
                            "sync": sync, "standing_since": standing_since, "lag_base": lag_base,
                            "lag_sec": None if lag_c is None else round(lag_c, 1),
                        }
                        rec["ended_by"] = "cut"
                        print(
                            f"[window] leg {leg.index}: planned cut at t={ts - stamps[0]:.1f}s — ego "
                            f"{'in step' if sync else 'out of step'} (lag={'?' if lag_c is None else round(lag_c, 1)}s, "
                            f"lateral={lateral_c:.1f}m, v={node.v:.2f})"
                        )
                        break
                elif is_last_leg(leg.kind, leg.t_end, stamps[-1]):
                    if ts >= leg.t_end - 0.05 and end_reached_wall is None:
                        end_reached_wall = now
                    if end_reached_wall is not None and final_grace_over(
                        now, end_reached_wall,
                        arrival_dist_m=(
                            math.hypot(node.x - leg.arrival[0], node.y - leg.arrival[1]) if leg.arrival is not None else None
                        ),
                        arrived_wall=arrived_wall, drive_through=leg.drive_through,
                    ):
                        rec["ended_by"] = "window_end"
                        break
                elif ts >= leg.t_end - 0.5:
                    # A late ego is let arrive (operator, 2026-09-10: arriving
                    # late is not a failure): the window runs on for it, up to
                    # LATE_ARRIVAL_SEC past its end. Arriving counts: within
                    # ARRIVE_M and rolling in.
                    if rec["arrived_t"] is None and leg.arrival is not None:
                        dist = math.hypot(node.x - leg.arrival[0], node.y - leg.arrival[1])
                        if dist <= ARRIVE_M and node.v <= 1.0:
                            rec["arrived_t"] = round(ts, 2)
                            print(f"[window] leg {leg.index}: arriving at the window end (dist {dist:.1f} m, v={node.v:.2f})")
                        elif ts < leg.t_end + LATE_ARRIVAL_SEC:
                            if now - last_log >= 15.0:
                                last_log = now
                                print(f"[window] leg {leg.index}: past the window end, {dist:.0f} m from the arrival pose — letting it arrive")
                            continue
                        else:
                            rec["end_dist_m"] = round(dist, 1)
                    rec["ended_by"] = "window_end"
                    break
                # divergence (not inside a hole: there is nothing to diverge from)
                if in_gap:
                    node.through = None
                elif now - last_check >= 0.25:
                    last_check = now
                    # Behind: the recording's LAST pass over the ego's spot, and a
                    # recorded wait within 12 m counts from its departure (dwell
                    # rule). Ahead: the recording's FIRST pass — the dwell rule made
                    # an ego braking 8 m short of a red-light stop "56 s ahead" of
                    # a recording about to wait there, and the last pass made an
                    # ego pulling into a bay "12 s ahead" of a recording dwelling
                    # in it.
                    lag = recorded_lag_sec(track, node.x, node.y, node.yaw, ts)
                    # Falling behind is judged on the recording's DRIVING time:
                    # a wait it made further on is no lead over the ego.
                    lag_drive = recorded_lag_sec(track, node.x, node.y, node.yaw, ts, driving=True)
                    gap = recorded_gap_m(track, node.x, node.y, node.yaw, ts)
                    lag_raw = recorded_lag_sec(
                        track, node.x, node.y, node.yaw, ts, dwell_rule=False, earliest=True
                    )
                    lateral = _lateral_offset_m(track, node.x, node.y, node.yaw)
                    kind = ""
                    standing = node.v < MOVING_MPS
                    if standing:
                        if standing_since is None:
                            standing_since = ts
                    else:
                        first_setoff = not revealed and leg.depart_t is not None
                        if first_setoff or (
                            standing_since is not None and (open_departure or ts - standing_since >= STANDING_SEC)
                        ):
                            delay = None if lag is None else round(max(lag, 0.0), 1)
                            dep = {"t": round(ts, 2), "at": [round(node.x, 2), round(node.y, 2)], "delay_sec": delay}
                            if open_departure and leg.depart_t is not None:
                                # against the recorded departure the window opened at
                                dep["delay_sec"] = round(ts - float(leg.depart_t), 1)
                                dep["cut"] = True
                            if first_setoff:
                                # The leg's departure: the ego set off on its own
                                # in an empty scene; it leaves into the scene the
                                # recording left into, objects and all.
                                revealed = True
                                depart = float(leg.depart_t)
                                node.jump(depart)
                                # Let the jump land first: objects switched on
                                # with it showed one frame of the scene before
                                # the jump (a car 66 m off, then 16 m: H95,
                                # 2026-10-29).
                                node.spin(0.05)
                                node.objects(True)
                                node.follow_mode(True)
                                dep["delay_sec"] = round(ts - depart, 1)
                                dep["revealed_at"] = round(depart, 2)
                                rec["jumps"].append({"wall": round(time.time(), 3), "from": round(ts, 2), "to": round(depart, 2)})
                                print(
                                    f"[window] leg {leg.index}: ego set off {ts - depart:+.1f}s against the recorded "
                                    f"departure — scene jumped to it, objects revealed"
                                )
                                lag = 0.0
                            rec["departures"].append(dep)
                            print(f"[window] leg {leg.index}: set off at t={ts - stamps[0]:.1f}s, {dep['delay_sec']}s after the recording did here")
                        if standing_since is not None or lag_base == 0.0:
                            lag_base = 0.0 if lag_drive is None else max(lag_drive, 0.0)
                        standing_since = None
                        open_departure = False
                    if lag is not None and node.rec_v >= MOVING_MPS:
                        if standing and lag >= DEPART_LAG_SEC:
                            kind = "stall"
                        elif not standing and falling_behind(
                            lag_drive, lag_base, gap, uncovered=scene_uncovered(node.nearest_m, node.covered_wall, now)
                        ):
                            kind = "slow"
                    elif lag_raw is not None and lag_raw <= -DIVERGE_AHEAD_SEC and node.v >= MOVING_MPS:
                        kind = "ahead"
                    elif lateral >= DIVERGE_LATERAL_M and node.v >= MOVING_MPS:
                        kind = "lateral"
                    # A recorded object drove into the ego's body: the scene is no
                    # longer the ego's. Set it back onto the recording at once.
                    if (
                        node.through is not None
                        and node.through[0] > float(seg["wall_handover"]) + 1.0
                    ):
                        kind = "through"
                        node.through = None
                        diverge_since = now - DIVERGE_HOLD_SEC
                        diverge_kind = kind
                    if kind == "stall":
                        # The scene plays on while the ego stands (scene_clock
                        # rule 2): a standing ego is no frozen scene to repair,
                        # and setting it back onto the recording hid whether it
                        # was stuck — H95 stood 17.5 s behind a parked truck and
                        # "arrived" 0.6 s after being moved 54 m (2026-10-28).
                        # Noted once per leg; the leg's stuck rule judges it.
                        if not rec.get("stall"):
                            rec["stall"] = {
                                "t": round(ts, 2), "at": [round(node.x, 2), round(node.y, 2)],
                                "lag_sec": round(lag, 1), "ego_v": round(node.v, 2),
                            }
                            print(
                                f"[window] leg {leg.index}: ego standing {lag:.1f}s behind the recording at "
                                f"t={ts - stamps[0]:.1f}s ({node.x:.1f},{node.y:.1f}) — left to the stuck rule"
                            )
                        kind = ""
                    if kind == "ahead":
                        # Noted once per leg; the drive goes on.
                        if not rec.get("ahead"):
                            rec["ahead"] = {
                                "t": round(ts, 2), "at": [round(node.x, 2), round(node.y, 2)],
                                "ahead_sec": round(-lag_raw, 1), "ego_v": round(node.v, 2),
                            }
                            print(
                                f"[window] leg {leg.index}: ego {-lag_raw:.1f}s ahead of the recording at "
                                f"t={ts - stamps[0]:.1f}s ({node.x:.1f},{node.y:.1f}) — driving on"
                            )
                        kind = ""
                    if kind:
                        if diverge_since is None:
                            diverge_since = now
                            diverge_kind = kind
                    else:
                        diverge_since = None
                    if diverge_since is not None and now - diverge_since >= DIVERGE_HOLD_SEC:
                        seg.update(
                            {
                                "t_end": round(ts, 2), "ended_by": f"diverged:{diverge_kind}",
                                "wall_end": round(time.time(), 3),
                                "lag_sec": None if lag is None else round(lag, 1),
                                "lag_base_sec": round(lag_base, 1),
                                "lateral_m": round(lateral, 1),
                                "nearest_object_m": None if node.nearest_m is None else round(min(node.nearest_m, 9999.0), 1),
                                "at": [round(node.x, 2), round(node.y, 2)],
                                "ego_v": round(node.v, 2), "rec_v": round(node.rec_v, 2),
                            }
                        )
                        reinits += 1
                        print(
                            f"[window] leg {leg.index}: diverged ({diverge_kind}) at "
                            f"t={ts - stamps[0]:.1f}s ({node.x:.1f},{node.y:.1f}) v={node.v:.2f}, "
                            f"lag={lag if lag is None else round(lag, 1)}s "
                            f"raw={lag_raw if lag_raw is None else round(lag_raw, 1)}s lateral={lateral:.1f}m"
                        )
                        diverge_since = None
                        if len(rec["segments"]) >= MAX_SEGMENTS:
                            rec["ended_by"] = "abandoned"
                            break
                        if not revealed:
                            node.objects(True)
                            revealed = True
                        handover(rec, ts, revealed=True)
                        lag_base = 0.0
                        standing_since = None
                        continue
                    if now - last_log >= 15.0:
                        last_log = now
                        print(
                            f"[window] leg {leg.index}: t={ts - stamps[0]:.1f}s v={node.v:.2f} "
                            f"lag={'?' if lag is None else round(lag, 1)}s "
                            f"raw={'?' if lag_raw is None else round(lag_raw, 1)}s "
                            f"lateral={lateral:.1f}m segments={len(rec['segments'])}"
                        )
            seg = rec["segments"][-1] if rec["segments"] else None
            if seg is not None and seg["ended_by"] is None:
                seg["t_end"] = round(node.rec_t, 2) if node.rec_t is not None else None
                seg["ended_by"] = rec["ended_by"]
                seg["wall_end"] = round(time.time(), 3)
            rec["wall_end"] = round(time.time(), 3)
            print(f"[window] leg {leg.index} ended: {rec['ended_by']} (segments={len(rec['segments'])}, arrived={rec['arrived_t'] is not None})")
            dump()
            if status == "failed":
                break
        node.follow_mode(False)
        node.hold(True)
    finally:
        try:
            dump()
        except Exception:  # noqa: BLE001
            pass
        executor.shutdown()
        node.destroy_node()
        close_job_context(context)
        os.environ.clear()
        os.environ.update(old_env)
    interventions["reinits"] = reinits
    interventions["cut_placements"] = placements
    interventions["cuts_seamless"] = seamless_cuts
    interventions["cut_scene_jumps"] = scene_jumps
    return status, error
