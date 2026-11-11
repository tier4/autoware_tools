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

"""Stop-go: unplanned stops, creep time and start-brake cycles of a run, against
the recording at the same places.

The control that was missing on 2026-09-09: a change was judged on one video
window while the run as a whole had more start-brake cycles than before. A
stop is planned when it lies within GOAL_ZONE_M of a leg goal or the spawn, or
within WAIT_ZONE_M of a place the recording itself stood WAIT_MIN_S; everything
else is unplanned. Creep is time between 0.05 and CREEP_V m/s outside those
zones; a start-brake cycle is a rise of RISE_V followed by a fall below
STOP_GO_V outside them. The recording is scored the same way; the run is
flagged when it stops, creeps or cycles clearly more than the recording did.
Gating."""
import bisect, math

GOAL_ZONE_M = 12.0     # around a leg goal / bus stop: planned stop
WAIT_ZONE_M = 12.0     # around a place the recording stood >= WAIT_MIN_S: planned wait (12 m: a bay exit 15 m on is not the bay)
WAIT_MIN_S = 3.0
STOP_V = 0.1
CREEP_V = 1.0
RISE_V = 1.0           # a start-brake cycle: speed rose by this, fell below STOP_GO_V, rose again
STOP_GO_V = 0.5
BLIP_S = 1.0           # a speed blip shorter than this and below STOP_GO_V does not end a stop


def _stops(ego, v_stop=STOP_V, min_s=0.5):
    """Standstills of at least min_s. A localisation blip inside one (0.12 m/s
    for 0.15 s in a 14 s bus-stop dwell, h95) must not split it in two: the
    second half then read as a stop the zone had not excused, and that one
    'recorded stop' offset a real unplanned stop of the run (2026-09-14)."""
    out, cur, blip = [], None, None
    for e in ego:
        if e.speed_mps < v_stop:
            cur = cur or [e, e]; cur[1] = e; blip = None
        elif cur and e.speed_mps < STOP_GO_V and (blip is None or e.stamp_sec - blip < BLIP_S):
            blip = e.stamp_sec if blip is None else blip
        else:
            if cur and cur[1].stamp_sec - cur[0].stamp_sec >= min_s: out.append(cur)
            cur = None; blip = None
    if cur and cur[1].stamp_sec - cur[0].stamp_sec >= min_s: out.append(cur)
    return out


def _zones(points, radius):
    return [(x, y, radius) for x, y in points]


def _in_zones(x, y, zones):
    return any(math.hypot(x - zx, y - zy) <= r for zx, zy, r in zones)


def stop_go(ego, ref_ego, goals_xy):
    """ego/ref_ego: EgoSample lists. goals_xy: leg goals (x, y). Returns a dict."""
    wait_pts = [(a.x, a.y) for a, b in _stops(ref_ego, min_s=WAIT_MIN_S)] if ref_ego else []
    zones = _zones(goals_xy, GOAL_ZONE_M) + _zones(wait_pts, WAIT_ZONE_M)
    if ego:
        zones.append((ego[0].x, ego[0].y, GOAL_ZONE_M))  # spawn

    def analyse(samples):
        # A zone excuses one stop: the planned one. A second stop inside the
        # same zone after the ego moved on (the stall 8 m past a bay exit) is
        # unplanned — it masked a 60 s stall on 2026-04-28.
        used = set()
        stops = []
        for a, b in _stops(samples):
            zone = next((i for i, (zx, zy, r) in enumerate(zones) if math.hypot(a.x - zx, a.y - zy) <= r), None)
            if zone is None or zone in used:
                stops.append((a, b))
            else:
                used.add(zone)
        creep = 0.0; prev = None
        for e in samples:
            if prev is not None and 0.05 <= e.speed_mps < CREEP_V and not _in_zones(e.x, e.y, zones):
                creep += max(0.0, e.stamp_sec - prev.stamp_sec)
            prev = e
        # start-brake cycles on a lightly smoothed speed
        v = [s.speed_mps for s in samples]
        sm = [sum(v[max(0, i - 2):i + 3]) / len(v[max(0, i - 2):i + 3]) for i in range(len(v))]
        cycles = []; state = "low"; peak = 0.0; peak_i = 0; low = sm[0] if sm else 0.0
        for i, s in enumerate(sm):
            if state == "low":
                if s - low >= RISE_V:
                    state = "high"; peak = s; peak_i = i
                low = min(low, s)
            else:
                if s > peak:
                    peak = s; peak_i = i
                if s < STOP_GO_V and peak - s >= RISE_V:
                    if not _in_zones(samples[i].x, samples[i].y, zones):
                        cycles.append((peak_i, i))
                    state = "low"; low = s
        t0 = samples[0].stamp_sec if samples else 0.0
        return {
            "unplanned_stops": len(stops),
            "unplanned_stop_sec": round(sum(b.stamp_sec - a.stamp_sec for a, b in stops), 1),
            "creep_sec": round(creep, 1),
            "start_brake_cycles": len(cycles),
            "stops_at": [(round(a.stamp_sec - t0), round(b.stamp_sec - a.stamp_sec, 1)) for a, b in stops][:12],
            # each cycle's braking phase, peak to standstill: (t_peak, t_end) from the first sample, and where it ended
            "cycles_at": [
                (round(samples[p].stamp_sec - t0, 1), round(samples[i].stamp_sec - t0, 1), round(samples[i].x, 2), round(samples[i].y, 2), round(getattr(samples[i], "yaw_rad", 0.0), 3))
                for p, i in cycles
            ][:12],
        }

    out = {"run": analyse(ego), "recording": analyse(ref_ego) if ref_ego else None}
    rec = out["recording"] or {"unplanned_stops": 0, "creep_sec": 0.0, "start_brake_cycles": 0}
    out["flagged"] = (
        out["run"]["unplanned_stops"] > rec["unplanned_stops"] + 1
        or out["run"]["start_brake_cycles"] > rec["start_brake_cycles"] + 1
        or out["run"]["creep_sec"] > rec["creep_sec"] + 15.0
    )
    return out
