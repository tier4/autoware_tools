"""The NPC motion contract, checked on every frame that is published. No ROS.

The scene player moves recorded objects only in time, for a few stated reasons
(reactive_shield.py, prebirth.py). Whatever those rules do, what the planner is
shown must still look like traffic. This is the check, on the output itself:

| An NPC ... | Finding when it does not |
|---|---|
| stands in a junction or on a crosswalk only as long as its recording stood there | `stand_in_zone` |
| comes into view no nearer than the recording first saw it (a new tracker id where one just was is the same object) | `pop_in` (a recorded track under 1 s: `blip`) |
| leaves where its recorded track ends, not while it waits beside the ego | `vanish` |
| moves continuously | `jump` |

Findings go to shield_log.jsonl ({"audit": kind, ...}); the metrics list them
per run (`scene_audit`). They do not change what is published: a finding is a
defect of the scene player to fix, found by the tool instead of in a video.
"""

from __future__ import annotations

import math
from typing import Any

NEAR_M = 30.0        # findings are looked for this close to the ego's body
POP_M = 15.0         # appearing or vanishing closer than this is a finding
STAND_M = 0.3        # moved less than this: standing
STAND_SEC = 3.0      # standing this much longer than allowed is a finding ...
STAND_AGAIN_SEC = 5.0    # ... reported again every this long while it lasts
DWELL_M = 0.5        # the recording "stood there": its poses within this of the place
JUMP_M = 2.0         # one frame's step beyond this (plus 2x its speed x dt) is a jump
GONE_SEC = 1.0       # absent this long: gone (shorter is the frame selector's flicker)
GRACE_SEC = 1.0      # after the scene was moved or switched on, nothing "appears" or "jumps"
BLIP_SEC = 1.0       # a recorded track shorter than this is tracker noise
CLOSER_M = 5.0       # appearing this much nearer than the bus first saw it is the player's pop-in
HANDOVER_SEC = 2.0   # a new id within this long ...
HANDOVER_M = 3.0     # ... and this close to where one of its class left is the same object
GAP_SEC = 0.5        # recorded samples further apart than this: the recording lost it between
EGO_L, EGO_W, EGO_OFF = 10.88, 2.87, 2.12


def _yaw(q: Any) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _ego_frame(x: float, y: float, ego_pose: Any) -> tuple[float, float, float]:
    """(distance to the ego's body, longitudinal, lateral) of a point."""
    yaw = _yaw(ego_pose.orientation)
    dx, dy = x - ego_pose.position.x, y - ego_pose.position.y
    lon = dx * math.cos(yaw) + dy * math.sin(yaw)
    lat = -dx * math.sin(yaw) + dy * math.cos(yaw)
    gx = max(abs(lon - EGO_OFF) - EGO_L / 2.0, 0.0)
    gy = max(abs(lat) - EGO_W / 2.0, 0.0)
    return math.hypot(gx, gy), lon, lat


class _Seen:
    __slots__ = ("x", "y", "t", "d", "ax", "ay", "stand", "told", "gone", "held", "label", "rec", "ft")

    def __init__(self, x: float, y: float, t: float, d: float, label: int) -> None:
        self.x, self.y, self.t, self.d = x, y, t, d
        self.ax, self.ay, self.stand, self.told = x, y, 0.0, 0.0
        self.gone: float | None = None
        self.held = False
        self.label = label
        self.rec: float | None = None
        self.ft = 0.0


class SceneAudit:
    def __init__(self, tracks: dict | None = None, shield_map: Any = None, ego_path=None) -> None:
        """ego_path: the recorded ego as (times s, xs, ys) — where the bus was
        when it first saw each object."""
        self.tracks = tracks or {}
        self.map = shield_map
        self.ego_path = ego_path
        self._first_seen: dict[tuple, float | None] = {}
        self._left: list[tuple[float, float, float, int]] = []   # (t, x, y, label) of recent disappearances
        self._seen: dict[bytes, _Seen] = {}
        self._now: float | None = None
        self._frame_t: float | None = None
        self._grace_until = -math.inf
        self._on = False
        self.counts: dict[str, int] = {}
        self.findings: list[dict] = []

    def reset(self) -> None:
        """The scene was moved: what is there next is not an appearance."""
        self._seen.clear()
        self._frame_t = None
        self._grace_until = (self._now or 0.0) + GRACE_SEC

    # -- what the recording says
    def _bus_first_m(self, key: bytes, frame_t: float | None = None) -> float | None:
        """How far the recorded bus was from the object when the recording
        (re)gained it: the start of the unbroken run of its track that holds
        frame_t (its birth, or the end of a gap), else its first recorded pose."""
        import bisect

        tr = self.tracks.get(key)
        if tr is None or not tr.times or self.ego_path is None:
            return None
        i0 = min(getattr(tr, "ext", 0), len(tr.times) - 1)
        i = i0
        if frame_t is not None:
            j = bisect.bisect_right(tr.times, frame_t, lo=i0) - 1
            if j >= i0:
                while j > i0 and tr.times[j] - tr.times[j - 1] <= GAP_SEC:
                    j -= 1
                i = j
        cache = (key, i)
        if cache in self._first_seen:
            return self._first_seen[cache]
        et, ex, ey = self.ego_path
        k = min(max(bisect.bisect_left(et, tr.times[i]), 0), len(et) - 1)
        p = tr.samples[i].kinematics.pose_with_covariance.pose.position
        d = math.hypot(p.x - ex[k], p.y - ey[k])
        self._first_seen[cache] = d
        return d

    def _handover(self, now: float, x: float, y: float, label: int, objects, key: bytes) -> bool:
        """A new id where the same object just was: one that left within
        HANDOVER_SEC, or one shown on top of it now (a tracker ID switch)."""
        self._left = [e for e in self._left if now - e[0] <= HANDOVER_SEC]
        if any(lab == label and math.hypot(x - lx, y - ly) <= HANDOVER_M for _, lx, ly, lab in self._left):
            return True
        present = {bytes(o.object_id.uuid) for o in objects}
        for k, st in self._seen.items():    # gone from this frame, not yet counted as left
            if (k != key and k not in present and st.label == label and now - st.t <= HANDOVER_SEC
                    and math.hypot(x - st.x, y - st.y) <= HANDOVER_M):
                return True
        for o in objects:
            k = bytes(o.object_id.uuid)
            if k == key or k not in self._seen:
                continue
            q = o.kinematics.pose_with_covariance.pose.position
            cls = getattr(o, "classification", None) or []
            if (int(cls[0].label) if cls else 0) == label and math.hypot(q.x - x, q.y - y) <= 1.0:
                return True
        return False

    def _recorded(self, key: bytes) -> tuple[float, float, float] | None:
        """(first recorded time, last time, recorded life) of a track."""
        tr = self.tracks.get(key)
        if tr is None or not tr.times:
            return None
        t0 = tr.times[min(getattr(tr, "ext", 0), len(tr.times) - 1)]
        return t0, tr.times[-1], tr.times[-1] - t0

    def _rec_at(self, key: bytes, t: float, tol: float = 0.3) -> tuple[float, float] | None:
        """The recorded pose at scene time t, if the recording has the object then."""
        import bisect

        tr = self.tracks.get(key)
        if tr is None or not tr.times:
            return None
        i0 = getattr(tr, "ext", 0)
        j = bisect.bisect_left(tr.times, t, lo=i0)
        best = None
        for k in (j - 1, j):
            if i0 <= k < len(tr.times) and abs(tr.times[k] - t) <= tol:
                if best is None or abs(tr.times[k] - t) < abs(tr.times[best] - t):
                    best = k
        if best is None:
            return None
        p = tr.samples[best].kinematics.pose_with_covariance.pose.position
        return p.x, p.y

    def _dwell(self, key: bytes, x: float, y: float) -> float:
        """Seconds the recorded track spends within DWELL_M of a place."""
        tr = self.tracks.get(key)
        if tr is None:
            return 0.0
        total, prev_t, prev_in = 0.0, None, False
        for t, s in zip(tr.times[getattr(tr, "ext", 0):], tr.samples[getattr(tr, "ext", 0):]):
            p = s.kinematics.pose_with_covariance.pose.position
            now_in = math.hypot(p.x - x, p.y - y) <= DWELL_M
            if now_in and prev_in:
                total += t - prev_t
            prev_t, prev_in = t, now_in
        return total

    # -- the check
    def check(self, now: float, frame_t: float, objects, ego_pose: Any, *, held=None, past=()) -> list[dict]:
        """Findings of this frame. ``objects`` None: objects are switched off.
        ``held``: key -> the shield's reason; ``past``: keys shown on the
        approach prebirth gave them."""
        held = held or {}
        dt = 0.1 if self._now is None else max(0.0, now - self._now)
        self._now = now
        if objects is None:
            if self._on:
                self.reset()
            self._on = False
            return []
        if not self._on:
            self._on = True
            self.reset()
        step = None if self._frame_t is None else frame_t - self._frame_t
        self._frame_t = frame_t
        if step is not None and (step < -0.05 or step > max(1.0, 3.0 * dt)):
            self.reset()                                   # the scene clock jumped
            self._frame_t = frame_t
        advancing = step is not None and step > 0.2 * dt   # held or ended: nobody "stands too long"
        grace = now < self._grace_until
        out: list[dict] = []
        present = set()
        for o in objects:
            key = bytes(o.object_id.uuid)
            present.add(key)
            p = o.kinematics.pose_with_covariance.pose.position
            lin = o.kinematics.twist_with_covariance.twist.linear
            d, lon, _ = _ego_frame(p.x, p.y, ego_pose)
            cls = getattr(o, "classification", None) or []
            label = int(cls[0].label) if cls else 0
            st = self._seen.get(key)
            if st is None:
                st = self._seen[key] = _Seen(p.x, p.y, now, d, label)
                st.ft = frame_t
                if not grace and d <= POP_M and not self._handover(now, p.x, p.y, label, objects, key):
                    rec = self._recorded(key)
                    bus_m = self._bus_first_m(key, frame_t)
                    centre_m = math.hypot(p.x - ego_pose.position.x, p.y - ego_pose.position.y)
                    if rec is not None and rec[2] < BLIP_SEC:
                        out.append(self._finding(now, "blip", key, label, p, d, src="recorded"))
                    elif bus_m is None or centre_m < bus_m - CLOSER_M:
                        # nearer than the recording first had it: the player's
                        src = "past" if key in past else "birth" if rec and frame_t - rec[0] < 1.0 else "late"
                        out.append(self._finding(now, "pop_in", key, label, p, d, lon=round(lon, 1), src=src,
                                                 bus_m=None if bus_m is None else round(bus_m, 1)))
                continue
            was_gone, st.gone = st.gone, None
            move = math.hypot(p.x - st.x, p.y - st.y)
            gap = now - st.t
            # Within the grace after the scene was moved, a step is the move's: a
            # frame of the old instant still in flight lands after the reset
            # (a 227 m "jump" when the harness moved the scene back, H78 2026-11-10).
            if not grace and gap <= GONE_SEC and move > JUMP_M + 2.0 * math.hypot(lin.x, lin.y) * gap and min(d, st.d) <= NEAR_M:
                # The recording's own step over the same scene time is the tracker's, not ours.
                a, b = self._rec_at(key, st.ft), self._rec_at(key, frame_t)
                own = math.hypot(b[0] - a[0], b[1] - a[1]) if a and b else 0.0
                if move - own > JUMP_M:
                    out.append(self._finding(now, "jump", key, label, p, d, m=round(move, 1), rec_m=round(own, 1)))
            st.x, st.y, st.t, st.d, st.held, st.ft = p.x, p.y, now, d, key in held, frame_t
            if math.hypot(p.x - st.ax, p.y - st.ay) > STAND_M:
                st.ax, st.ay, st.stand, st.told, st.rec = p.x, p.y, 0.0, 0.0, None
            elif advancing and was_gone is None:
                st.stand += min(dt, step, 0.5)
                if st.stand >= STAND_SEC and d <= NEAR_M and self.map is not None and self.map.zone_at(st.ax, st.ay):
                    if st.rec is None:
                        st.rec = self._dwell(key, st.ax, st.ay)
                    # First seen standing, it may have stood as long before (prebirth).
                    over = st.stand - 2.0 * st.rec
                    # Waiting at a red light it now meets late is traffic, not a fault.
                    red = held.get(key) == "red_light"
                    if over >= STAND_SEC and not red and (st.told == 0.0 or st.stand - st.told >= STAND_AGAIN_SEC):
                        why = held.get(key) or ("past" if key in past else "scene")
                        out.append(self._finding(now, "stand_in_zone", key, label, p, d,
                                                 sec=round(st.stand, 1), rec_sec=round(st.rec, 1), why=why))
                        st.told = st.stand
        for key, st in list(self._seen.items()):
            if key in present:
                continue
            if st.gone is None:
                st.gone = now
            if now - st.gone < GONE_SEC:
                continue
            del self._seen[key]
            self._left.append((now, st.x, st.y, st.label))
            if grace or st.d > POP_M:
                continue
            rec = self._recorded(key)
            if rec is not None and rec[2] < BLIP_SEC:
                continue                                   # a blip ends as it came
            if rec is not None and frame_t < rec[1] - 1.0:
                if self._rec_at(key, frame_t, tol=0.5) is None and not st.held:
                    continue                               # the recording loses it here too
                why = "dropped"                            # its recording goes on; the player took it out
            elif st.held:
                why = "held_to_its_end"                    # waited for the ego until its recorded time ran out
            else:
                continue                                   # its recorded track ends here: the recording's own
            out.append({"t": round(now, 3), "audit": "vanish", "id": key.hex(), "label": st.label,
                        "x": round(st.x, 2), "y": round(st.y, 2), "d": round(st.d, 1), "why": why})
        for f in out:
            self.counts[f["audit"]] = self.counts.get(f["audit"], 0) + 1
        self.findings.extend(out)
        return out

    @staticmethod
    def _finding(now: float, kind: str, key: bytes, label: int, p: Any, d: float, **more) -> dict:
        return {"t": round(now, 3), "audit": kind, "id": key.hex(), "label": label,
                "x": round(p.x, 2), "y": round(p.y, 2), "d": round(d, 1)} | more


# Kinds that are the scene player's to answer for (a blip is the recording's).
DEFECTS = ("stand_in_zone", "pop_in", "vanish", "jump")


def summarize(records: list[dict], t0: float = 0.0) -> dict:
    """Findings of one run, for the metrics: counts by kind (one per object and
    kind, the longest standing kept) and the findings themselves."""
    best: dict[tuple[str, str], dict] = {}
    for r in records:
        k = (r["audit"], r["id"])
        if k not in best or r.get("sec", 0.0) > best[k].get("sec", 0.0):
            first = best[k]["t"] if k in best else r["t"]
            best[k] = dict(r, t=first)
    events = sorted(best.values(), key=lambda r: r["t"])
    counts: dict[str, int] = {}
    for r in events:
        counts[r["audit"]] = counts.get(r["audit"], 0) + 1
    return {
        "counts": counts,
        "defects": sum(n for k, n in counts.items() if k in DEFECTS),
        "events": [
            {"t_rel": round(r["t"] - t0, 1), "kind": r["audit"], "object_id": r["id"][:8], "label": r.get("label"),
             "at": [r.get("x"), r.get("y")], "ego_dist_m": r.get("d")}
            | {k: r[k] for k in ("sec", "rec_sec", "why", "src", "lon", "m") if k in r}
            for r in events if r["audit"] in DEFECTS
        ][:60],
    }
