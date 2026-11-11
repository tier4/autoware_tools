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

"""The ego against the recording's timeline, without ROS: the recorded track,
the lag behind it, and the wall-to-scene time map of a leg-window run."""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any

RESYNC_TOL_M = 0.5        # poses this close to the nearest count as the same spot
RESYNC_FUTURE_S = 30.0    # recorded samples further ahead than this are another pass
RESYNC_DWELL_M = 12.0     # a recorded wait this close is the ego's wait too (stop_go's zone)
RESYNC_REC_MOVING_MPS = 0.5  # re-sync only onto a recorded vehicle that is moving


def reference_track(ego: list[Any]):
    """(t, x, y, yaw, dwells) of the recording's ego at 10 Hz, for the lag.

    ``dwells`` are the places the recorded vehicle stood (< 0.3 m/s for >= 3 s)
    as (x, y, arrival_t, departure_t): an ego waiting near one of them is
    waiting with the recording, and its lag counts from the recorded departure.
    """
    import numpy as np

    rows = []
    last = -1e9
    for e in ego:
        if e.stamp_sec - last >= 0.1:
            rows.append((e.stamp_sec, e.x, e.y, e.yaw_rad))
            last = e.stamp_sec
    if not rows:
        return None
    dwells = []
    run: list[Any] = []
    for e in list(ego) + [None]:
        if e is not None and e.speed_mps < 0.3:
            run.append(e)
            continue
        if run and run[-1].stamp_sec - run[0].stamp_sec >= 3.0:
            dwells.append((run[0].x, run[0].y, run[0].stamp_sec, run[-1].stamp_sec))
        run = []
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], np.asarray(dwells, dtype=float).reshape(-1, 4)


def recorded_lag_sec(
    track, x: float, y: float, yaw: float, scene_t: float, *, dwell_rule: bool = True,
    earliest: bool = False, driving: bool = False,
) -> float | None:
    """Seconds the scene has run past the ego's own position.

    ``track``: (t, x, y, yaw, dwells) of the recording's ego. The ego's spot is
    the recorded sample nearest to it (heading within 90 deg, so the return
    trip on the same street does not count); where the recording stood still
    hundreds of samples share the spot, and the LATEST of them is the instant
    the recorded vehicle left it, which is what a waiting ego is measured
    against (``earliest`` takes the first instead: the instant the recorded
    vehicle got here, which is what an ego that may be AHEAD is measured
    against — at a bay the latest sample is the departure, and an ego that
    had just pulled in read as 12 s ahead of a recording still dwelling
    there, 2026-08-25). With ``dwell_rule`` a recorded wait (``dwells``: x, y,
    arrival, departure) within RESYNC_DWELL_M is the ego's wait as well — an
    ego 4.5 m short of the spot where the recording sat 56 s at a red light
    was measured 7 s behind it while both stood (2026-04-28) — so the lag
    then counts from the recorded departure, which lies in the future for as
    long as the recording still stands. Samples (and waits) beginning more
    than RESYNC_FUTURE_S ahead of the scene belong to a later pass over the
    same street and are ignored. None when nothing qualifies.

    ``driving``: only the time the recording spent DRIVING since it left the
    ego's spot counts — its waits after that (``dwells``) are taken out. The
    recorded bus passed a stop line, then waited 35 s inside the junction,
    30 m on, for its right turn; the ego, stopped at that line by the red,
    read 35 s behind and was set 36 m forward the moment it moved off on the
    green, its turn never driven (H78, 2026-11-09). In driving terms it was a
    few seconds behind a bus still waiting in the junction.
    """
    import numpy as np

    t, tx, ty, tyaw, dwells = track
    ok = t <= scene_t + RESYNC_FUTURE_S
    if yaw == yaw:
        dyaw = np.abs((tyaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)
        ok &= dyaw <= np.pi / 2.0
    if not ok.any():
        return None
    d2 = np.where(ok, (tx - x) ** 2 + (ty - y) ** 2, np.inf)
    best = math.sqrt(float(d2.min()))
    here = d2 <= (best + RESYNC_TOL_M) ** 2
    lag = float(scene_t - (t[here].min() if earliest else t[here].max()))
    if dwell_rule:
        for dx, dy, arrive, depart in dwells:
            if math.hypot(dx - x, dy - y) <= RESYNC_DWELL_M and arrive <= scene_t + RESYNC_FUTURE_S:
                lag = min(lag, float(scene_t - depart))
    if driving and lag > 0.0:
        left = scene_t - lag
        stood = sum(max(0.0, min(float(depart), scene_t) - max(float(arrive), left)) for _, _, arrive, depart in dwells)
        lag = max(0.0, lag - stood)
    return lag


def recorded_gap_m(track, x: float, y: float, yaw: float, scene_t: float) -> float | None:
    """How far along its track the recorded vehicle is ahead of the ego at
    ``scene_t`` (metres; negative behind). The ego's spot is matched as in
    recorded_lag_sec (heading within 90 deg, not a later pass)."""
    import numpy as np

    t, tx, ty, tyaw, _ = track
    if len(t) == 0:
        return None
    ok = t <= scene_t + RESYNC_FUTURE_S
    if yaw == yaw:
        ok &= np.abs((tyaw - yaw + np.pi) % (2.0 * np.pi) - np.pi) <= np.pi / 2.0
    if not ok.any():
        return None
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))])
    d2 = np.where(ok, (tx - x) ** 2 + (ty - y) ** 2, np.inf)
    k_ego = int(np.argmin(d2))
    k_rec = min(int(np.searchsorted(t, scene_t)), len(t) - 1)
    return float(s[k_rec] - s[k_ego])


def scene_time_map(windows_json: Path):
    """wall time -> recorded (scene) time of a leg-window run, from windows.json:
    from LEAD_SEC before each hand-over the scene ran at 1x from that segment's
    recorded instant; between segments it was held at the next lead-in; the
    departure alignments are jumps. Runs from 2026-11-10 on also
    sample the scene time every 0.5 s (legs' ``clock``): those samples are
    points too, so a freeze at a recorded departure and the recording's end
    are followed (without them the map ran 1x through both: +7 s after a
    6 s freeze, +17 s at H78's end). Never past the recording's last window.
    None when the run was not a leg-window run."""
    try:
        payload = json.loads(Path(windows_json).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    lead = float((payload.get("params") or {}).get("lead_sec", 3.5))
    points = []
    for leg in payload.get("legs", []):
        for seg in leg.get("segments") or []:
            if seg.get("wall_handover") is not None and seg.get("t_handover") is not None:
                if seg.get("seamless"):
                    # a planned cut the ego was in step at: no lead-in, the scene ran on
                    points.append((float(seg["wall_handover"]), float(seg["t_handover"]), False))
                else:
                    points.append((float(seg["wall_handover"]) - lead, float(seg["t_handover"]) - lead, True))
        for jump in leg.get("jumps") or []:
            points.append((float(jump["wall"]), float(jump["to"]), False))
        for wall, t in leg.get("clock") or []:
            points.append((float(wall), float(t), False))
    t_last = max((float(leg["t_end"]) for leg in payload.get("legs", []) if leg.get("t_end") is not None), default=None)
    points.sort()
    if not points:
        return None
    starts = [w for w, _, _ in points]

    def scene_time(wall: float) -> float:
        i = bisect.bisect_right(starts, wall) - 1
        if i < 0:
            return points[0][1]
        w0, t0, _ = points[i]
        t = t0 + (wall - w0)
        if i + 1 < len(points) and points[i + 1][2]:
            t = min(t, points[i + 1][1])  # held at the next lead-in until it starts
        if t_last is not None:
            t = min(t, t_last)
        return t

    return scene_time
