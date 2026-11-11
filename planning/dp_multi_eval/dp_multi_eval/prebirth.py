"""An object's approach before the recording first saw it.

The recorded tracker reports an object from the moment the bus first saw it,
typically 20-60 m away. An ego that lags the bus is much nearer to that spot,
and the object appeared out of nothing a few metres in front of it (H78: a
wave of pedestrians 10-16 m ahead of the standing ego, born 32-54 m from the
bus).

What is added is the part of its past that physics gives, nothing more:

| Recorded at birth | Its past |
|---|---|
| standing (within STOOD_M of its first pose for its first STOOD_SEC) | it stood there: its first pose, as long before as it stands after (up to PREBIRTH_STAND_SEC) |
| moving, classified (>= MOVER_MPS over its first second of poses) | it came from there: back along that motion, up to PREBIRTH_MOVE_SEC / PREBIRTH_BACK_M |
| anything else (UNKNOWN on the move, creeping, too short to tell) | none: it appears as recorded |

Never a moving object frozen at its first pose: pedestrians walking at birth
stood 30 s in the middle of a crosswalk, in the ego's lane, and the ego waited
for them (H78, 2026-11-10).

The past is prepended to the object's track in the shield (extend_track), so it
has one path and one motion logic: an approaching pre-born car brakes for the
ego, a pre-born pedestrian waits at the kerb, like any other NPC. This module
only decides when the object becomes visible — once the ego is as near to it as
the bus was when the bus first saw it — and feeds its raw pose to the shield.
"""

from __future__ import annotations

import bisect
import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

PREBIRTH_STAND_SEC = 30.0      # a standing object is shown at most this long before its birth
PREBIRTH_MOVE_SEC = 8.0        # a moving one is traced back at most this long ...
PREBIRTH_BACK_M = 30.0         # ... and this far (a straight line; roads bend)
STOOD_M = 0.5                  # within this of its first pose ...
STOOD_SEC = 2.0                # ... for its first this long (at least MIN_LIFE_SEC of it seen): it stood
MOVER_MPS = 0.5                # at least this fast over its first second of poses: it was on its way
MOVER_MIN_SEC = 0.5            # ... judged on at least this much of its own track
NO_PAST_LABELS = (0,)          # UNKNOWN on the move is tracker clutter: no past invented
MIN_LIFE_SEC = 1.0             # shorter tracks are tracker noise, not objects to give a past
START_MARGIN_SEC = 1.0         # tracks born in the recording's first second were already there
HANDOVER_SEC = 2.0             # a track ending this shortly before ...
HANDOVER_M = 3.0               # ... this close to a birth: the same object under a new id
EGO_CLEAR_M = 1.0              # it becomes visible only if its way to its birth pose keeps this off the ego
REC_CLEAR_M = 0.3              # a past this close to the recorded vehicle, when it was there, is not one it had
THROUGH_M = 0.3                # ... nor one this deep in another object that was there then
THROUGH_GAP_SEC = 0.3          # "there then": a recorded sample of it within this of the instant
VRU_LABELS = (5, 6, 7)         # MOTORCYCLE, BICYCLE, PEDESTRIAN: they share space among themselves
GRID_M = 20.0                  # cell of the index of where each object ever is
VISIBLE_MIN_M = 10.0           # the ego sees at least this far whatever the bus's first range
STEP_SEC = 0.1                 # sample spacing of a traced-back approach
STAND_STEP_SEC = 0.5


def _yaw(q: Any) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _label(obj: Any) -> int | None:
    cls = getattr(obj, "classification", None) or []
    return int(cls[0].label) if cls else None


def _xy(sample: Any) -> tuple[float, float]:
    p = sample.kinematics.pose_with_covariance.pose.position
    return p.x, p.y


def _box(sample: Any) -> tuple[float, float, float, float, float]:
    pose = sample.kinematics.pose_with_covariance.pose
    d = sample.shape.dimensions
    return pose.position.x, pose.position.y, _yaw(pose.orientation), max(d.x, 0.5) / 2.0, max(d.y, 0.5) / 2.0


class _There:
    """Where an object is, when: a recorded track, or a past already given."""

    __slots__ = ("root", "times", "samples", "vru", "step", "cells")

    def __init__(self, root: bytes, times: list, samples: list, label: int | None, step: float) -> None:
        self.root, self.times, self.samples = root, times, samples
        self.vru = label in VRU_LABELS
        self.step = step          # how long one sample stands for
        self.cells = {(int(x // GRID_M), int(y // GRID_M)) for x, y in map(_xy, samples)}

    def box_at(self, t: float) -> tuple | None:
        if not self.times or t < self.times[0] - 1e-6 or t > self.times[-1] + self.step:
            return None
        i = max(bisect.bisect_right(self.times, t) - 1, 0)
        if i + 1 < len(self.times) and self.times[i + 1] - t < t - self.times[i]:
            i += 1
        if abs(self.times[i] - t) > max(self.step, THROUGH_GAP_SEC):
            return None           # the recording lost it here
        return _box(self.samples[i])


@dataclass
class _Past:
    key: bytes
    kind: str            # "stood" | "moving"
    t_birth: float
    times: list          # ascending, all before t_birth
    samples: list        # light samples: the first recorded one at another place and speed
    first: Any           # the first recorded sample (a real message, copied to inject)
    v: float
    seen_m: float = 0.0  # how far the bus was from it when the bus first saw it


class PreBirth:
    """Pasts for recorded tracks (the shield's, before they are extended)."""

    def __init__(self, tracks: dict, parent_of: dict, ego_t: list[float], ego_x: list[float],
                 ego_y: list[float], clearance=None, ego_poses: list | None = None) -> None:
        """tracks: key -> .times (s) / .samples (TrackedObject); ego_t in
        seconds; clearance(sample, ego_pose) -> metres between the sample's box
        and the ego's body; ego_poses: the recorded vehicle's poses at ego_t
        (with both, a past is kept clear of where the recorded vehicle was)."""
        self._clearance = clearance
        self._active: set[bytes] = set()
        self.cut_by_recording = 0   # pasts shortened or dropped: they ran through the recorded vehicle
        self.cut_by_others = 0      # ... or through another object that was there
        first = min((tr.times[0] for tr in tracks.values() if tr.times), default=0.0)
        ends = sorted((tr.times[-1], k) for k, tr in tracks.items() if tr.times)
        end_t = [e[0] for e in ends]
        # One object under several ids — a tracker ID switch (parent_of) or a
        # track that ends where a new one starts (H78: a pedestrian group
        # re-born ten times in 4 s, most ids under 1 s) — is one chain, judged
        # by its whole life and given a past once, at its first id.
        pred: dict[bytes, bytes] = {}
        for key, tr in tracks.items():
            if not tr.times:
                continue
            if key in parent_of:
                pred[key] = parent_of[key]
                continue
            x, y = _xy(tr.samples[0])
            k = self._handover(tracks, ends, end_t, key, tr.times[0], x, y, _label(tr.samples[0]))
            if k is not None:
                pred[key] = k
        chain_end: dict[bytes, float] = {}
        members: dict[bytes, list[bytes]] = {}
        self.root_of: dict[bytes, bytes] = {}
        for key, tr in tracks.items():
            if not tr.times:
                continue
            root, seen = key, {key}
            while root in pred and pred[root] not in seen:
                root = pred[root]
                seen.add(root)
            chain_end[root] = max(chain_end.get(root, 0.0), tr.times[-1])
            members.setdefault(root, []).append(key)
            self.root_of[key] = root
        pasts: list[_Past] = []
        for key, tr in tracks.items():
            if not tr.times or key in pred:
                continue
            t0 = tr.times[0]
            if t0 < first + START_MARGIN_SEC or chain_end.get(key, t0) - t0 < MIN_LIFE_SEC:
                continue
            past = self._past(key, tr, [tracks[k] for k in members.get(key, [key])])
            if past is not None and clearance is not None and ego_poses:
                past = self._clear_of_recording(past, ego_t, ego_poses)
            if past is None:
                continue
            i = min(max(bisect.bisect_left(ego_t, t0), 0), len(ego_t) - 1)
            x0, y0 = _xy(tr.samples[0])
            past.seen_m = max(math.hypot(x0 - ego_x[i], y0 - ego_y[i]), VISIBLE_MIN_M)
            pasts.append(past)
        pasts.sort(key=lambda p: p.t_birth)
        pasts = self._clear_of_others(pasts, tracks)
        self._pasts = pasts
        self._t = [p.t_birth for p in pasts]

    @staticmethod
    def _handover(tracks, ends, end_t, key, t0, x, y, label) -> bytes | None:
        """The track this one continues: same class, ended shortly before
        (or just after) this birth, within HANDOVER_M of it."""
        lo = bisect.bisect_left(end_t, t0 - HANDOVER_SEC)
        hi = bisect.bisect_right(end_t, t0 + 0.5)
        for _, k in ends[lo:hi]:
            if k == key:
                continue
            tr = tracks[k]
            if tr.times[0] >= t0:
                continue
            last = tr.samples[-1]
            if _label(last) != label:
                continue
            px, py = _xy(last)
            if math.hypot(px - x, py - y) <= HANDOVER_M:
                return k
        return None

    def _clear_of_recording(self, past: _Past, ego_t: list[float], ego_poses: list) -> _Past | None:
        """The past from after the last instant it would have been within
        REC_CLEAR_M of the recorded vehicle; None if nothing is left. The
        recorded vehicle drove there and met nobody: a straight line traced
        back from 1.9 s of a pedestrian's track ran through the bus itself,
        and the pedestrian was shown standing in the ego's lane 8 m ahead of
        it — hit at 4.3 m/s, billed to the model (H39, 2026-11-10)."""
        last_hit = -1
        for i, (t, s) in enumerate(zip(past.times, past.samples)):
            if t < ego_t[0] or t > ego_t[-1]:
                continue
            k = min(max(bisect.bisect_left(ego_t, t), 0), len(ego_t) - 1)
            if self._clearance(s, ego_poses[k]) < REC_CLEAR_M:
                last_hit = i
        if last_hit < 0:
            return past
        self.cut_by_recording += 1
        if last_hit + 1 >= len(past.times):
            return None
        past.times = past.times[last_hit + 1:]
        past.samples = past.samples[last_hit + 1:]
        return past

    def _clear_of_others(self, pasts: list[_Past], tracks: dict) -> list[_Past]:
        """Each past from after the last instant it would have been THROUGH_M
        deep in another object that was there then; dropped if nothing is left.

        "There then": another object's recorded track at that recorded time, or
        the past already given to one born earlier. Two things are not in one
        place: a line traced back from a car's first second ran through a truck
        parked at the kerb, and a car first seen standing "stood" where the
        same car was still shown under its older tracker id, or where others
        drove through meanwhile (two-model run, 2026-11-11: 51 such
        pass-throughs within 40 m of the ego in 12 jobs). Usually the object
        was first seen where it was because the other one hid it: it comes
        out from behind it. Not judged: UNKNOWN others (tracker clutter),
        pedestrians and cyclists among themselves (a group), and the ids of
        its own chain."""
        from dp_multi_eval.geometry import box_overlap_m

        grid: dict[tuple[int, int], list[_There]] = {}

        def add(th: _There) -> None:
            for c in th.cells:
                grid.setdefault(c, []).append(th)

        for key, tr in tracks.items():
            if tr.times and _label(tr.samples[0]) not in (None, *NO_PAST_LABELS):
                add(_There(self.root_of.get(key, key), tr.times, tr.samples, _label(tr.samples[0]), 0.0))
        out = []
        for p in pasts:
            vru = _label(p.first) in VRU_LABELS
            root = self.root_of.get(p.key, p.key)
            step = STAND_STEP_SEC if p.kind == "stood" else STEP_SEC
            mine = _There(root, p.times, p.samples, _label(p.first), step)
            seen: set[int] = set()
            near = []
            for c in mine.cells:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for th in grid.get((c[0] + dx, c[1] + dy), ()):
                            if id(th) not in seen:
                                seen.add(id(th))
                                if th.root != root and not (vru and th.vru):
                                    near.append(th)
            t_a, t_b = p.times[0], p.t_birth
            last_hit = None
            for th in near:
                lo = max(t_a, th.times[0]) if last_hit is None else max(last_hit, th.times[0])
                hi = min(t_b, th.times[-1] + th.step)
                if hi <= lo:
                    continue
                # Judged at the other's samples and at the past's own (a past
                # that stands is one box; a car passing through it is seen at
                # the car's rate).
                i0, i1 = bisect.bisect_left(th.times, lo), bisect.bisect_right(th.times, hi)
                j0, j1 = bisect.bisect_left(p.times, lo), bisect.bisect_right(p.times, hi)
                for t in sorted(set(th.times[i0:i1]) | set(p.times[j0:j1]), reverse=True):
                    if not lo <= t < t_b:
                        continue
                    a, b = mine.box_at(t), th.box_at(t)
                    if a is not None and b is not None and box_overlap_m(a, b) > THROUGH_M:
                        last_hit = t if last_hit is None else max(last_hit, t)
                        break
            if last_hit is not None:
                self.cut_by_others += 1
                k = bisect.bisect_right(p.times, last_hit + 1e-6)
                if k >= len(p.times):
                    continue
                p.times, p.samples = p.times[k:], p.samples[k:]
                mine = _There(root, p.times, p.samples, _label(p.first), step)
            if _label(p.first) not in (None, *NO_PAST_LABELS):
                add(mine)
            out.append(p)
        return out

    @staticmethod
    def _past(key: bytes, tr: Any, chain: list) -> _Past | None:
        """Stood, moving or none — from the poses only: the tracker's first
        speeds are guesses (a car reporting 5 m/s barely moved in its first
        second) and it reports 0 for every UNKNOWN object."""
        t0, s0 = tr.times[0], tr.samples[0]
        x0, y0 = _xy(s0)
        # Stood: everything the chain shows of its first STOOD_SEC is at its
        # first pose. It is met mid-wait: it stood as long before as it is
        # seen to stand after (a pedestrian pausing 2 s on a crosswalk did not
        # stand there for 30).
        seen = sorted((t, *_xy(s)) for c in chain for t, s in zip(c.times, c.samples) if t <= t0 + PREBIRTH_STAND_SEC)
        left = next((t for t, x, y in seen if math.hypot(x - x0, y - y0) > STOOD_M), None)
        stood = (left if left is not None else seen[-1][0]) - t0
        if stood >= min(STOOD_SEC, max(MIN_LIFE_SEC, seen[-1][0] - t0)):
            n = int(min(PREBIRTH_STAND_SEC, stood) / STAND_STEP_SEC)
            times = [t0 - k * STAND_STEP_SEC for k in range(n, 0, -1)]
            return _Past(key, "stood", t0, times, [_light(s0, x0, y0, 0.0, 0.0)] * n, s0, 0.0)
        # Moving: its own first second of poses.
        j = bisect.bisect_right(tr.times, t0 + 1.0) - 1
        x1, y1 = _xy(tr.samples[j])
        dt, d = tr.times[j] - t0, math.hypot(x1 - x0, y1 - y0)
        if dt < MOVER_MIN_SEC or d < MOVER_MPS * dt or _label(s0) in NO_PAST_LABELS:
            return None
        v, heading = d / dt, math.atan2(y1 - y0, x1 - x0)
        n = int(min(PREBIRTH_MOVE_SEC, PREBIRTH_BACK_M / v) / STEP_SEC)
        if n < 1:
            return None
        rel = heading - _yaw(s0.kinematics.pose_with_covariance.pose.orientation)
        vx, vy = v * math.cos(rel), v * math.sin(rel)    # the twist is in the object's frame
        times, samples = [], []
        for k in range(n, 0, -1):
            back = v * k * STEP_SEC
            times.append(t0 - k * STEP_SEC)
            samples.append(_light(s0, x0 - back * math.cos(heading), y0 - back * math.sin(heading), vx, vy))
        return _Past(key, "moving", t0, times, samples, s0, v)

    def __len__(self) -> int:
        return len(self._pasts)

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self._pasts:
            out[p.kind] = out.get(p.kind, 0) + 1
        return out

    def extend(self, shield: Any) -> None:
        """Make each past part of its track in the shield."""
        for p in self._pasts:
            shield.extend_track(p.key, p.times, p.samples)

    def reset(self) -> None:
        """The scene was moved (a leg boundary, a reveal): visibility is judged anew."""
        self._active.clear()

    @property
    def active(self) -> set[bytes]:
        """Objects now shown on their past."""
        return self._active

    def objects_at(self, frame_t: float, ego_pose: Any, present: set[bytes]) -> list[Any]:
        """Raw objects on their past at scene time frame_t (seconds), for the
        shield to move like any other: those the ego is near enough to see,
        and those it has seen (an ego creeping back over the edge of sight
        flickered a pedestrian for 11 s, H78)."""
        if ego_pose is None:
            return []
        ex, ey = ego_pose.position.x, ego_pose.position.y
        lo = bisect.bisect_right(self._t, frame_t)
        hi = bisect.bisect_right(self._t, frame_t + PREBIRTH_STAND_SEC)
        out, live = [], set()
        for p in self._pasts[lo:hi]:
            if p.key in present or frame_t < p.times[0]:
                continue
            i = max(0, bisect.bisect_right(p.times, frame_t) - 1)
            sample = p.samples[i]
            if p.key not in self._active and not self._comes_into_sight(p, i, ex, ey, ego_pose):
                continue
            live.add(p.key)
            obj = copy.deepcopy(p.first)
            pose, src = obj.kinematics.pose_with_covariance.pose, sample.kinematics.pose_with_covariance.pose
            pose.position.x, pose.position.y = src.position.x, src.position.y
            lin = obj.kinematics.twist_with_covariance.twist.linear
            tw = sample.kinematics.twist_with_covariance.twist.linear
            lin.x, lin.y = tw.x, tw.y
            out.append(obj)
        self._active = live
        return out

    def _comes_into_sight(self, p: _Past, i: int, ex: float, ey: float, ego_pose: Any) -> bool:
        x, y = _xy(p.samples[i])
        if math.hypot(x - ex, y - ey) > p.seen_m:
            return False
        if self._clearance is None:
            return True
        # Its way from here to its birth pose is clear of the ego: a past
        # through a standing ego is not one it had (H78 97d686e4 was shown,
        # dropped inside the ego for 4 s, then born beside it).
        way = [p.first] if p.kind == "stood" else p.samples[i::max(1, int(1.0 / (p.v * STEP_SEC)))] + [p.first]
        return all(self._clearance(s, ego_pose) >= EGO_CLEAR_M for s in way)


def _light(first: Any, x: float, y: float, vx: float, vy: float) -> Any:
    """A sample of a past: the first recorded sample's identity, size and
    heading at another place and speed. Its pose and twist are its own real
    messages (the shield hands them on to the published object)."""
    pose = copy.deepcopy(first.kinematics.pose_with_covariance.pose)
    pose.position.x, pose.position.y = x, y
    twist = copy.deepcopy(first.kinematics.twist_with_covariance.twist)
    twist.linear.x, twist.linear.y = vx, vy
    return SimpleNamespace(
        object_id=first.object_id,
        classification=getattr(first, "classification", None),
        shape=first.shape,
        kinematics=SimpleNamespace(
            pose_with_covariance=SimpleNamespace(pose=pose),
            twist_with_covariance=SimpleNamespace(twist=twist),
        ),
    )
