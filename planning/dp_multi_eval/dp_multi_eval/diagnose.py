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

"""Turn a job's failure string into an explanation a person can act on.

The harness records what happened precisely — ``leg_2:wait_not_started;
leg_wait_exhausted forward_shifts=0`` says exactly which leg died and how — but
it says nothing about what that means or what to change, and reading it takes
knowledge of the drive loop's internals. Every failure in this package has been
diagnosed at least once already; the diagnosis is what belongs in the results,
next to the evidence that supports it.

Each rule matches on the recorded error plus whatever the job already measured,
and returns three lines: what happened, why, and what to look at. Nothing here
re-reads bags or re-runs anything — it only phrases what is already known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Diagnosis:
    headline: str
    because: str
    look_at: str

    def lines(self) -> list[str]:
        return [
            f"DIAGNOSIS — {self.headline}",
            f"  Why: {self.because}",
            f"  Look at: {self.look_at}",
        ]

    def as_dict(self) -> dict[str, str]:
        return {"headline": self.headline, "why": self.because, "look_at": self.look_at}


def _num(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def diagnose(
    error: str | None,
    *,
    status: str | None = None,
    interventions: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    reproducer_log_tail: str | None = None,
) -> Diagnosis | None:
    """Best explanation for how this job ended, or None when nothing matches."""
    err = (error or "").strip()
    iv = interventions or {}
    met = metrics or {}
    shifts = int(iv.get("forward_shifts") or 0)

    # --- the run never got to drive ---------------------------------------
    if "gpu_busy" in err:
        return Diagnosis(
            "there was no room on the GPU for this job's planner",
            f"The job waited and the card stayed full: {err.strip()[:200]}",
            "dead processes still holding CUDA memory — nvidia-smi lists them with no "
            "process name. They are already gone and cannot be killed; only a driver "
            "reset returns that memory. Until then, run fewer workers.",
        )

    if "planner inference disabled" in err:
        reason = err.split(":", 1)[-1].strip()
        if "major_version" in reason:
            return Diagnosis(
                "the model and the planner build are different versions, so nothing was loaded",
                f"The planner refused the weights it was given: {reason}",
                "the model's args.json major_version against WEIGHT_MINIMUM_VERSION "
                "(WEIGHT_MAJOR_VERSION in older builds) in the planner's constants.hpp; "
                "job_status.json build records both. Either the checkpoint or the build has moved; forcing "
                "them together would score a model the planner cannot really run.",
            )
        if "out of memory" in reason.lower() or "cuda" in reason.lower():
            return Diagnosis(
                "the planner could not build its engine on the GPU",
                f"TensorRT failed while the card was under pressure: {reason}",
                "nvidia-smi during a run, and whether a previous job left a node holding "
                "its CUDA context.",
            )
        return Diagnosis(
            "the planner never loaded its model, so no trajectory was ever published",
            f"It stopped with: {reason}",
            "the planner's own ERROR line in psim_launch.log — it names the cause.",
        )

    if "perception_ready_timeout" in err:
        objects = _num(r"objects=(\d+)", err)
        tail = (reproducer_log_tail or "").strip().splitlines()
        last = tail[-1].strip() if tail else ""
        if objects == 0 and last:
            return Diagnosis(
                "the scene replay never published anything, so the run refused to engage",
                f"perception_reproducer stopped early — its last words were: {last}",
                "reproducer.log in the job directory. A crash on startup is usually a "
                "missing Python dependency in whichever interpreter ros2 run picked.",
            )
        return Diagnosis(
            "the scene replay never published objects in time",
            "The reproducer loads the whole bag before it publishes; a long recording "
            "can outlast the budget, and a crashed one never publishes at all.",
            "reproducer.log, then bag_load_base_sec / bag_load_duration_factor if the "
            "load was merely slow.",
        )

    # --- routing -----------------------------------------------------------
    if "stopped updating" in err or "no longer publishing" in err:
        return Diagnosis(
            "the simulator stopped publishing mid-run, so nothing was driving",
            "The ego pose stopped arriving while the leg was still being waited on. "
            "This is not a stalled vehicle — there was no vehicle to stall.",
            "the tail of psim_launch.log for a node that died, and the output bag: "
            "every recorded topic ends at the same moment when the stack goes down.",
        )

    if "route_budget_exhausted" in err or "route_total_timeout" in err:
        return Diagnosis(
            "route setup spent its whole budget without getting a goal accepted",
            "The mission planner rejected every candidate — the goal itself, the "
            "pulled-back variants and the map-stop substitutes.",
            "the rejection reason in the log. 'Goal's footprint exceeds lane' means the "
            "pose cannot host the vehicle; 'planned route is empty' means no route "
            "exists from where the ego stands. Neither is fixed by retrying.",
        )

    if "route_setup_failed" in err or "advance_route" in err:
        return Diagnosis(
            "the route could not be set for this leg",
            "The goal was refused by the mission planner. Dwell-derived goals land "
            "wherever the recorded vehicle happened to halt, including junctions where "
            "no goal is valid.",
            "route_goals.json for where the leg goals came from. A bag carrying "
            "/planning/mission_planning/route gives goals its own planner already "
            "accepted; inference is the fallback.",
        )

    # --- the drive ---------------------------------------------------------
    if "wait_not_started" in err and "leg_wait_exhausted" in err:
        leg = re.match(r"leg_(\d+)", err)
        which = f"Leg {leg.group(1)}" if leg else "A later leg"
        return Diagnosis(
            f"{which.lower()} never got any drive time",
            "The earlier legs consumed the whole drive budget, so this one started "
            "already past the deadline.",
            "how long the previous leg took — most often it is not driving but route "
            "advance retrying a rejected goal. multi_goal_leg_timeout_sec sets the per-leg "
            "cap; the overall budget scales with the number of legs.",
        )

    if "stuck_abort" in err or "stuck" in (status or ""):
        secs = _num(r"for (\d+)s", err) or _num(r"cumulative=(\d+)s", err)
        held = f" for {secs:.0f}s" if secs else ""
        # When the recording shows the stack replacing the model's plans with stop
        # stubs for most of the run, the stall is not the model's decision and no
        # amount of repositioning can break it: StopPointFixer only fires on a
        # stopped ego, its 3-point stub is one the mode manager cannot engage on
        # ("closest point not found"), and an ego that cannot engage stays stopped.
        stub = met.get("stack_intervention") or {}
        if stub.get("status") == "ok" and stub.get("flagged"):
            ratio = float(stub.get("stop_point_ratio") or 0.0)
            longest = float(stub.get("longest_forced_stop_sec") or 0.0)
            return Diagnosis(
                "this model cannot pull away from a stop: its departure plans fail "
                "the stack's long_stop gate",
                f"StopPointFixer replaced the model's trajectory with a stop point in "
                f"{100 * ratio:.0f}% of cycles (longest unbroken stretch {longest:.0f}s). "
                "The gate itself is passable — the source recording, driven by the "
                "production pipeline, shows zero such replacements across four real "
                "from-rest departures, one after 56 s stopped. A model whose pull-away "
                "exceeds 0.25 m/s within its first second clears it; this one's does "
                "not, so every stop risks becoming permanent.",
                "green_latched_sec (seconds of green light spent latched) and "
                "attribution in stack_intervention, plus the recorded "
                "candidate_trajectories, which hold the plans that were replaced.",
            )
        if shifts == 0:
            return Diagnosis(
                f"the ego stood still{held} and nothing moved it",
                "The scene plays on while the ego stands (scene_clock rule 2), so what "
                "held it did not clear in the recording either: a parked vehicle, or a "
                "wait the model would not end.",
                "what stood in front of the ego in the video, and the planning factors "
                "at the ego's position.",
            )
        return Diagnosis(
            f"the ego kept stalling{held} even after {shifts} forward shift(s)",
            "Repositioning broke the stall each time but the ego stopped again, so the "
            "give-up budget ran out.",
            "the preview video around the stall, and the planning factors: a STOP factor "
            "at the ego's own position means something in the scene is holding it.",
        )

    if "route_not_arrived" in err:
        goal = met.get("goal_stop_precision") or {}
        err_m = goal.get("position_error_m")
        near = f" It stopped {err_m:.1f} m from the goal." if isinstance(err_m, (int, float)) else ""
        return Diagnosis(
            "the leg ran out of time without Autoware reporting ARRIVED",
            f"The route stayed SET to the end.{near} Arrival is only registered when the "
            "ego stops near the goal it was actually routed to — which is not always the "
            "goal it was handed, since a rejected goal gets relocated.",
            "effective_goal_xy in the route setup message, and the distance the ego "
            "actually reached in the preview video.",
        )

    if "interrupted" in err:
        return Diagnosis(
            "the run was stopped by hand",
            "Ctrl+C cancelled the queue; this job's simulators were shut down and the "
            "job left for a rerun.",
            "nothing — rerun it. Interrupted jobs are re-queued, not counted as failures.",
        )

    # --- completed, but the metrics object --------------------------------
    if (status or "").lower() == "done":
        ref = met.get("reference_speed") or {}
        col = met.get("collision_rate") or {}
        if col.get("flagged") and ref.get("flagged"):
            p90 = ref.get("p90_diff_mps")
            ratio = ref.get("time_ratio")
            detail = ""
            if isinstance(p90, (int, float)) and isinstance(ratio, (int, float)):
                detail = (
                    f" It ran up to {p90:+.1f} m/s faster than the recording and covered "
                    f"the route in {ratio * 100:.0f}% of the recorded time."
                )
            return Diagnosis(
                "flagged for contact, but the speed difference is the finding underneath it",
                "The replayed traffic does not react to this ego: NPCs repeat what they "
                f"did around a vehicle that was driving at a different speed.{detail}",
                "reference_speed before collision_rate. Check whether the contact was an "
                "NPC cutting in front of a faster ego rather than the ego driving into "
                "something.",
            )
        if col.get("flagged"):
            events = col.get("event_count")
            n = f" ({events} frame(s))" if events else ""
            return Diagnosis(
                f"the drive completed but the ego footprint touched a vehicle{n}",
                "Contact is measured between footprints, so a close pass by a long "
                "vehicle counts even when neither driver would call it one.",
                "the preview video at the flagged timestamp, and whether the other "
                "vehicle was overtaking — the replay is open-loop and cannot yield.",
            )
        if ref.get("flagged"):
            ratio = ref.get("time_ratio")
            detail = f" — the route took {ratio * 100:.0f}% of the recorded time" if isinstance(ratio, (int, float)) else ""
            return Diagnosis(
                f"the drive completed but not at the recorded pace{detail}",
                "The ego's speed differs from the same vehicle on the same road in the "
                "recording, which is a property of the planner rather than the scenario.",
                "reference_speed: median is the typical difference, p90 where it runs "
                "fastest relative to the recording.",
            )
    return None
