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

"""How long the recorded vehicle stood still at a place.

The give-up budget for a stalled ego used to be one number for the whole run, which
has to be wrong somewhere: too short and the replay is abandoned mid-way through a
wait the recording itself contains, too long and a scenario that can never move
burns the clock. The recording knows the answer per place — it drove this road once
and its dwell there is on record.

Measured on a bag where the ego was abandoned after 90 s: at that exact spot the
recorded vehicle had stopped for 15 s and then gone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DwellProfile:
    """Recorded stop durations, queryable by position."""

    samples: list[Any]
    stop_speed_mps: float = 0.05

    def expected_dwell_sec(self, x: float, y: float, radius_m: float = 6.0) -> float:
        """Longest unbroken stop the recording made within radius_m of (x, y).

        Zero when the recording never stopped there — which is itself the answer:
        a place the recorded vehicle drove through is not a place to wait.
        """
        r2 = radius_m * radius_m
        longest = 0.0
        run_start: float | None = None
        prev_in = False
        for s in self.samples:
            inside = (s.x - x) ** 2 + (s.y - y) ** 2 <= r2
            stopped = s.speed_mps <= self.stop_speed_mps
            if inside and stopped:
                if run_start is None:
                    run_start = s.stamp_sec
                longest = max(longest, s.stamp_sec - run_start)
            else:
                run_start = None
            prev_in = inside
        return longest


class _Sample:
    __slots__ = ("stamp_sec", "x", "y", "speed_mps")

    def __init__(self, stamp_sec: float, x: float, y: float, speed_mps: float):
        self.stamp_sec = stamp_sec
        self.x = x
        self.y = y
        self.speed_mps = speed_mps


def load_dwell_profile(bag_path: Path, stop_speed_mps: float = 0.05) -> DwellProfile | None:
    """Ego track of the recording being replayed, or None when it is unavailable.

    Cached: the profile needs four numbers per sample and the recordings run to
    79 GB, which every attempt was re-reading in full to get them.
    """
    try:
        from dp_multi_eval.bag_cache import cached_json

        def compute() -> list[list[float]]:
            profile = _load_dwell_profile_uncached(bag_path, stop_speed_mps)
            if profile is None:
                raise RuntimeError("dwell profile unavailable")
            return [
                [s.stamp_sec, s.x, s.y, s.speed_mps] for s in profile.samples
            ]

        rows = cached_json(bag_path, "dwell_profile", "v1", compute)
        return DwellProfile(
            samples=[_Sample(*row) for row in rows],
            stop_speed_mps=stop_speed_mps,
        )
    except Exception:  # noqa: BLE001
        return None


def _load_dwell_profile_uncached(
    bag_path: Path, stop_speed_mps: float = 0.05
) -> DwellProfile | None:
    try:
        from dp_multi_eval.bag_reader import load_bag_series

        series = load_bag_series(
            bag_path, objects_topic=None, trajectory_topic=None
        )
    except Exception:  # noqa: BLE001
        return None
    if not series.ego:
        return None
    return DwellProfile(samples=list(series.ego), stop_speed_mps=stop_speed_mps)


def stall_budget_sec(
    profile: DwellProfile | None,
    x: float,
    y: float,
    *,
    floor_sec: float,
    factor: float = 2.0,
    margin_sec: float = 20.0,
    radius_m: float = 6.0,
) -> tuple[float, float]:
    """Seconds to allow a stall here, and the recorded dwell it came from.

    Generous by construction: the replay meets the scene at a different moment than
    the recording did, so the same wait can legitimately run longer. Never below
    floor_sec, so a place the recording rolled through still gets a fair chance.
    """
    if profile is None:
        return floor_sec, float("nan")
    dwell = profile.expected_dwell_sec(x, y, radius_m)
    return max(floor_sec, dwell * factor + margin_sec), dwell
