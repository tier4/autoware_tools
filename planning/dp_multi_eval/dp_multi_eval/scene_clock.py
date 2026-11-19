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

"""One recorded-time clock for the whole replayed scene (objects, lights,
turn indicators).

The scene driver (scene_driver.py) runs it and feeds the instant it picks to
the original perception_reproducer; until 2026-10-28 it lived inside a forked
reproducer (planning_debug_tools/scripts/perception_replayer/recorded_clock.py).

The recording is a timeline the ego is not on: it is slower here, earlier
there, and it stops for its own reasons. Which recorded instant the scene
should show is decided by three rules, stated by the operator on 2026-09-03:

1. Moving: the scene flows with the ego's POSITION — the recorded frame in
   which the recorded vehicle stood where the ego stands now.
2. Stopped: the recording plays on in REAL TIME from the instant the ego
   arrived, so the surroundings do exactly what they did while the recorded
   vehicle waited here.
3. Departing: the ego leaves into the scene the recorded vehicle left into.
   If the ego waits longer than the recording did, the scene holds at that
   departure instant (bounded, then the recording flows on so a car that
   pulled out in the recording is not frozen ahead forever); if it leaves
   earlier, the clock jumps forward to it.

Recorded time never runs backwards inside a stop-and-go episode (a floor
that follows the clock up); the floor is released once the ego has driven
25 m from the last stop, so the next junction is decided afresh.

4. (2026-09-10) While the ego drives, recorded time never runs slower than
   real time. Rule 1 alone plays every agent at ego speed / recorded speed:
   measured on 2026-04-28, an oncoming car approached at 5-8 m/s (recorded
   12) while the ego braked below the recording, then jumped to 14 m/s at
   the stop threshold, and the model stopped in that slow motion. With the
   ego on the recording's pace the position and the clock agree and the
   scene is the recorded one; only a slower ego now sees the traffic run on
   at its recorded speed. Past the recording's end (``at_end``) the
   reproducer keeps the objects that stood still in the last frame.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class _Dwell:
    anchor_ns: int          # recorded time at which the ego came to rest
    wall0_ns: int           # wall time at which it came to rest
    depart_ns: int | None   # recorded departure to align with (None: no recorded stop here)
    x: float
    y: float
    held_since_ns: int | None = None


class RecordedClock:
    MOVING_MPS = 0.3          # below this the ego counts as stopped
    STOP_MPS = 0.3            # recorded speed that counts as a recorded stop
    RELEASE_M = 25.0          # distance from the last stop that releases the floor
    NEXT_STOP_M = 30.0        # how far ahead a recorded stop may be to count as "this stop"
    SUSTAINED_MPS = 1.5       # recorded speed that counts as having got away
    SUSTAINED_NS = int(3e9)   # for this long
    HOLD_MAX_NS = int(6e9)    # longest freeze at the departure scene

    def __init__(self, stamps_ns, xs, ys, speeds, yaws=None, real_time_floor: bool = True,
                 departure_jump: bool = True, max_rate: float | None = None,
                 flow_when_behind: bool = False, hold_max_ns: int | None = None):
        """Recorded ego samples, time-ordered; ``yaws`` (rad) enable the heading check.
        ``real_time_floor`` False drops rule 4: the scene then follows the ego's
        position alone while it drives (slower ego, slower traffic).
        ``departure_jump`` False drops rule 3's jump forward to the recorded
        departure when the ego leaves earlier: H81 (2026-10-28) stood 15 m short
        of a 55 s recorded wait, and creeping off jumped the scene 40 s ahead —
        a bicycle landed on the ego's nose. The scene then goes on from where
        it stands (never back).
        ``max_rate`` caps how fast recorded time may run while the ego drives,
        as a multiple of real time: an ego twice the recording's speed played
        the traffic at up to 3.3x, and an oncoming car came at ~36 m/s (H63,
        2026-10-29). The scene then trails a much faster ego.
        ``flow_when_behind`` keeps recorded time running at real time while the
        ego drives behind the scene: after a stand the scene is ahead of the
        ego's position, and the floor froze it under a creeping ego — walkers
        stood still while reporting 1.5 m/s and the model crept for 24 s
        (H81, 2026-10-29). Once the ego catches up, its position rules again.
        ``hold_max_ns`` bounds rule 3's freeze at the departure scene
        (HOLD_MAX_NS by default); 0 lets the recording flow on at once: with
        planned cuts the window closes at that departure (2026-11-10)."""
        self.real_time_floor = real_time_floor
        self.departure_jump = departure_jump
        self.max_rate = max_rate
        self.flow_when_behind = flow_when_behind
        self.hold_max_ns = self.HOLD_MAX_NS if hold_max_ns is None else int(hold_max_ns)
        self.t = list(stamps_ns)
        self.x = list(xs)
        self.y = list(ys)
        self.v = list(speeds)
        self._xy = np.column_stack([np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float)])
        self._yaw = None if yaws is None else np.asarray(list(yaws), dtype=float)
        self._dwell: _Dwell | None = None
        self._floor: tuple[int, float, float] | None = None
        self._last: tuple[int, int] | None = None   # (ts, now_ns) of the previous tick

    @property
    def at_end(self) -> bool:
        """True once the scene has reached the recording's last frame."""
        return self._last is not None and self._last[0] >= self.t[-1]

    # -- geometry -----------------------------------------------------------
    NEAR_TOL_M = 0.5          # poses this close to the nearest count as the same spot

    def nearest_index(self, x: float, y: float, yaw: float | None = None) -> int:
        """Recorded instant for this spot: the EARLIEST pose within NEAR_TOL_M of
        the nearest one, heading within 90 deg of the ego's when known.

        Where the recorded vehicle stood still, hundreds of poses share one
        spot and a plain argmin lands on whichever jittered closest — arrival
        or departure at random. An ego rolling up to a bus stop then saw the
        pull-OUT blinker 5 m before the stop (2026-04-28). The earliest pose is
        the arrival; the dwell then plays forward in real time (tick). The
        heading check keeps a route that returns along the same street from
        locking onto the return trip's poses on the opposite lane.
        """
        d = self._xy - np.array([x, y])
        dist2 = np.einsum("ij,ij->i", d, d)
        if yaw is not None and self._yaw is not None:
            dyaw = np.abs((self._yaw - yaw + np.pi) % (2 * np.pi) - np.pi)
            masked = np.where(dyaw <= np.pi / 2, dist2, np.inf)
            if np.isfinite(masked).any():
                dist2 = masked
        best = float(np.min(dist2))
        tol = (math.sqrt(best) + self.NEAR_TOL_M) ** 2
        return int(np.argmax(dist2 <= tol))  # first index within tolerance = earliest

    def _departure_after(self, idx: int, x: float, y: float) -> int | None:
        """Recorded departure from the recorded stop this ego is queued for.

        Walk forward from ``idx`` while the recording is within NEXT_STOP_M of
        the ego; the first recorded sample below STOP_MPS marks that stop. From
        there the first SUSTAINED_NS run above SUSTAINED_MPS is the getaway;
        the departure instant is where the vehicle first started rolling into it. None when the
        recording did not stop within reach (it flowed through here).
        """
        n = len(self.t)
        k = idx
        stop = None
        while k < n and math.hypot(self.x[k] - x, self.y[k] - y) <= self.NEXT_STOP_M:
            if self.v[k] < self.STOP_MPS:
                stop = k
                break
            k += 1
        if stop is None:
            return None
        run_start = None
        j = stop
        while j < n:
            if self.v[j] > self.SUSTAINED_MPS:
                if run_start is None:
                    run_start = j
                if self.t[j] - self.t[run_start] >= self.SUSTAINED_NS:
                    # The departure is when the vehicle first started rolling
                    # into that getaway, not when it was already fast: walk
                    # back while it kept moving. That is the instant whose
                    # signals (the pull-out blinker) and traffic it left with.
                    k = run_start
                    while k > stop and self.v[k - 1] >= self.STOP_MPS:
                        k -= 1
                    return self.t[k]
            else:
                run_start = None
            j += 1
        return self.t[-1]

    # -- the clock ------------------------------------------------------------
    def tick(
        self, x: float, y: float, ego_v: float, now_ns: int, yaw: float | None = None,
        harness_hold: bool = False,
    ) -> int:
        ts = self._tick(x, y, ego_v, now_ns, yaw, harness_hold)
        ts = min(ts, self.t[-1])
        self._last = (ts, now_ns)
        return ts

    def _tick(
        self, x: float, y: float, ego_v: float, now_ns: int, yaw: float | None,
        harness_hold: bool,
    ) -> int:
        idx = self.nearest_index(x, y, yaw)
        pos_ts = self.t[idx]
        if self._floor is not None and math.hypot(x - self._floor[1], y - self._floor[2]) > self.RELEASE_M:
            self._floor = None
        floor_ts = self._floor[0] if self._floor is not None else None

        if ego_v >= self.MOVING_MPS:
            if self._dwell is not None:
                # Rule 3: leave into the recorded departure scene.
                if self._dwell.depart_ns is not None and self.departure_jump:
                    floor_ts = max(floor_ts or 0, self._dwell.depart_ns)
                self._dwell = None
            ts = pos_ts if floor_ts is None else max(pos_ts, floor_ts)
            if self.flow_when_behind and self._last is not None and pos_ts < self._last[0]:
                ts = max(ts, self._last[0] + max(0, now_ns - self._last[1]))
            if self._last is not None and self.real_time_floor:
                # Rule 4: not slower than real time.
                ts = max(ts, self._last[0] + max(0, now_ns - self._last[1]))
            if self._last is not None and self.max_rate is not None:
                ts = min(ts, self._last[0] + int(self.max_rate * max(0, now_ns - self._last[1])))
                ts = max(ts, self._last[0])   # the cap never runs it backwards
            fx, fy = (self._floor[1], self._floor[2]) if self._floor is not None else (x, y)
            self._floor = (ts, fx, fy)
            return ts

        if self._dwell is None:
            spot = pos_ts if floor_ts is None else max(pos_ts, floor_ts)
            anchor = spot
            if self._last is not None and self.max_rate is not None:
                # A capped clock trails an ego that drove faster than the
                # recording; coming to rest must not jump it to the ego's spot.
                # H81 (2026-10-29): it jumped 18.8 s on stopping and a
                # pedestrian crossing in front of the ego vanished mid-crossing.
                anchor = min(anchor, self._last[0] + int(self.max_rate * max(0, now_ns - self._last[1])))
                anchor = max(anchor, self._last[0])
            # The departure to hold at is the recording's from where the ego
            # stands: searched from the trailing anchor, 30 m+ back, it found
            # none and the hold was silently lost (2026-11-05 review).
            sidx = bisect.bisect_left(self.t, spot)
            self._dwell = _Dwell(anchor, now_ns, self._departure_after(min(sidx, len(self.t) - 1), x, y), x, y)
        d = self._dwell
        ts = d.anchor_ns + (now_ns - d.wall0_ns)          # Rule 2: real time
        if d.depart_ns is not None and ts > d.depart_ns:  # Rule 3: hold at departure, bounded
            if d.held_since_ns is None or harness_hold:
                # The bound is for the model's hesitation, not for the harness:
                # while Autoware has no route or is not engaged the ego cannot
                # leave, so the departure scene waits and the budget starts
                # when the ego is free to go. Measured 2026-04-28 (map v206):
                # 190 s of route setup at 教会前 ran the scene 184 s past the
                # departure and into the recording's end.
                d.held_since_ns = now_ns
            over = now_ns - d.held_since_ns
            ts = d.depart_ns + max(0, over - self.hold_max_ns)
        if floor_ts is not None:
            ts = max(ts, floor_ts)
        self._floor = (ts, d.x, d.y)
        return ts


class WindowClock:
    """Recorded time at 1x, nothing else: the scene is the recording.

    The harness opens each evaluation window by jumping the clock to the
    window's recorded start (``jump``) and pauses it while it sets the ego up
    (``hold``). No position, no dwell, no departure rule: where the ego is has
    no bearing on what the scene shows, so the world the model sees is exactly
    the recorded one and the ego's lag behind it is the harness's measurement,
    not the clock's problem. Held until the first jump, so nothing plays while
    the stack boots.
    """

    def __init__(self, stamps_ns):
        self.t = [int(v) for v in stamps_ns]
        self._ts = self.t[0]
        self._last_now: int | None = None
        self._pending: int | None = None
        self._started = False

    @property
    def at_end(self) -> bool:
        return self._ts >= self.t[-1]

    def jump(self, ts_ns: int) -> None:
        self._pending = int(ts_ns)

    MAX_STEP_NS = int(0.5e9)  # a starved timer must not fast-forward the scene

    def tick(self, now_ns: int, hold: bool = False) -> int:
        if self._pending is not None:
            self._ts = min(max(self._pending, self.t[0]), self.t[-1])
            self._pending = None
            self._started = True
        elif self._started and not hold and self._last_now is not None:
            # 2026-08-24 (two jobs on one machine): the reproducer's timer did
            # not run for a minute and then added the whole minute in one
            # tick — the scene jumped 60 s under an ego that had driven on.
            # Time it did not tick is time the scene did not see.
            step = min(max(0, now_ns - self._last_now), self.MAX_STEP_NS)
            self._ts = min(self._ts + step, self.t[-1])
        self._last_now = now_ns
        return self._ts
