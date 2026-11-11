"""Phase 5 — single-file HTML dashboard from manifest + metrics."""

from __future__ import annotations

import argparse
import html
import re
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dp_multi_eval.job_status import preview_video_path


def load_metrics(job: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(job["output_dir"]) / "metrics.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_job_status(job: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(job["output_dir"]) / "job_status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _status_counts(jobs: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0, "dry_run": 0}
    for job in jobs:
        status = str(job.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _running_detail(job: dict[str, Any]) -> str:
    js = load_job_status(job)
    parts: list[str] = ["RUNNING"]
    if js:
        phase = js.get("phase")
        if phase:
            parts.append(str(phase).replace("_", " "))
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        start = js.get("start_time")
        if start:
            try:
                t0 = datetime.fromisoformat(str(start))
                total = (now - t0.astimezone(timezone.utc)).total_seconds()
                parts.append(f"total {total:.0f}s")
            except ValueError:
                pass
        phase_start = js.get("phase_start_time")
        if phase_start:
            try:
                t1 = datetime.fromisoformat(str(phase_start))
                phase_sec = (now - t1.astimezone(timezone.utc)).total_seconds()
                parts.append(f"phase {phase_sec:.0f}s")
            except ValueError:
                pass
    domain = (js or {}).get("domain_id", job.get("domain_id"))
    if domain is not None:
        parts.append(f"domain {domain}")
    return " · ".join(parts)


def _pending_label(job: dict[str, Any]) -> str:
    attempts = int(job.get("attempts") or 0)
    if attempts:
        return f"PENDING (retry {attempts})"
    return "PENDING"


def _fmt_rate(value: Any) -> str:
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return f"{100.0 * float(value):.1f}%"
    return "—"


def _fmt_m(value: Any) -> str:
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return f"{float(value):.2f}"
    return "—"


def _model_key(job: dict[str, Any]) -> str:
    """The column a run belongs to: its model."""
    return str(job.get("model_name") or "")


def _model_label(key: str) -> str:
    """A model column key as HTML."""
    return f"<span class='mname'>{html.escape(key)}</span>"


def _card_head(bag: str, model: str, models: list[str], verdict: str, cls: str) -> str:
    """Scenario name, verdict, and the model."""
    sub = f"<span class='model'>{_model_label(model)}</span>"
    return (
        f"<header><div class='title'><h3>{html.escape(bag)}</h3>{sub}</div>"
        f"<span class='pill pill-{cls}'>{html.escape(verdict)}</span></header>"
    )


_CHIP_LABELS = (
    ("collision_rate", "collision"),
    ("stuck_rate", "stuck"),
    ("out_of_boundary", "off-lane"),
)


def _metric_chips(metrics: dict[str, Any] | None) -> str:
    """One chip per rule, reading pass or the value that broke it.

    A rate alone does not say whether it mattered; the chip carries the verdict and
    the number together, so a glance answers "did anything trip, and by how much".
    """
    if not metrics:
        return ""
    chips = []
    for key, label in _CHIP_LABELS:
        block = metrics.get(key) or {}
        if block.get("status") not in (None, "ok"):
            continue
        flagged = bool(block.get("flagged"))
        rate = block.get("rate")
        val = _fmt_rate(rate) if isinstance(rate, (int, float)) else "—"
        state = "bad" if flagged else "good"
        tip = ""
        if "ego_caused_episodes" in block:
            # Leg-window run: only a border the ego drove over fails it; one the
            # harness placed it on is shown, not held against the planner.
            ego_n, placed_n = len(block["ego_caused_episodes"]), len(block.get("placement_episodes") or [])
            state = "bad" if ego_n else ("warn" if placed_n else "good")
            tip = html.escape(f"{ego_n} crossing(s) by the ego, {placed_n} at a harness placement")
        chips.append(
            f"<span class='chip chip-{state}' title='{tip}'><span class='k'>{label}</span>"
            f"<span class='v'>{val}</span></span>"
        )
    ref = metrics.get("reference_speed") or {}
    pace = _pace(metrics)
    if ref.get("status") == "ok" and pace is not None:
        state = "warn" if ref.get("flagged") else "good"
        chips.append(
            f"<span class='chip chip-{state}'><span class='k'>pace</span>"
            f"<span class='v'>{pace * 100:.0f}% of recorded</span></span>"
        )
    # Bus stops the recording served: did the ego pull in and stop at the sign?
    bus = metrics.get("bus_stop_service") or {}
    if bus.get("status") == "ok" and bus.get("recorded"):
        state = "bad" if bus.get("flagged") else "good"
        tip = html.escape(
            "; ".join(
                f"{st.get('name')}: {st.get('verdict')}"
                + (f" (lat {st.get('lat_m')} m, lon {st.get('lon_m')} m, {st.get('ego_dwell_sec')}s)" if st.get('lat_m') is not None else "")
                for st in bus.get("stops", [])
            )
        )
        chips.append(
            f"<span class='chip chip-{state}' title='{tip}'><span class='k'>bus stops</span>"
            f"<span class='v'>{bus['served']}/{bus['recorded']} served</span></span>"
        )
    # Start-brake cycles: the check that a fix made the drive smoother, not
    # just one window of it.
    sg = metrics.get("stop_go") or {}
    if sg.get("status") in ("ok", "no_reference") and sg.get("run"):
        run_ = sg["run"]; rec_ = sg.get("recording") or {}
        state = "bad" if sg.get("flagged") else "good"
        tip = html.escape(
            f"unplanned stops {run_['unplanned_stops']} ({run_['unplanned_stop_sec']} s), creep {run_['creep_sec']} s, "
            f"start-brake cycles {run_['start_brake_cycles']}; recording: stops {rec_.get('unplanned_stops', '?')}, "
            f"creep {rec_.get('creep_sec', '?')} s, cycles {rec_.get('start_brake_cycles', '?')}; stops at (t, s): {run_.get('stops_at')}"
        )
        chips.append(
            f"<span class='chip chip-{state}' title='{tip}'><span class='k'>stop-go</span>"
            f"<span class='v'>{run_['unplanned_stops']} stops · {run_['creep_sec']:.0f} s creep · {run_['start_brake_cycles']} cycles</span></span>"
        )
    # Recorded objects that drove through the ego (withheld frames): where the
    # scene and a slower ego parted ways. Informational, never a verdict.
    thr_ego = metrics.get("objects_through_ego") or {}
    if thr_ego.get("status") == "ok" and (thr_ego.get("count") or thr_ego.get("yielding")):
        tip = html.escape(
            "; ".join(
                f"{ev.get('object')} at ({ev.get('x')}, {ev.get('y')})"
                + (f" t={ev.get('t_rel')}s" if ev.get("t_rel") is not None else "")
                + (f" {ev.get('duration_sec')}s" if ev.get("duration_sec") is not None else "")
                for ev in thr_ego.get("events", [])
            )
        )
        chips.append(
            f"<span class='chip chip-warn' title='{tip}'><span class='k'>through ego</span>"
            f"<span class='v'>{thr_ego['count']}"
            + (f" · yield {thr_ego['yielding']}" if thr_ego.get("yielding") else "")
            + "</span></span>"
        )
    # Leg windows: the recording's legs, each started from the recorded state.
    win = metrics.get("windows") or {}
    if win.get("status") == "ok" and win.get("legs_total"):
        state = "bad" if win.get("flagged") else "good"
        parts = []
        for lg in win.get("legs", []):
            cut = ""
            if lg.get("opens") == "cut":
                cut = ", cut" + (
                    " — placed on the recording" if lg.get("placed")
                    else f" — scene jumped {lg['scene_jumped']:+.0f}s to the ego" if lg.get("scene_jumped") is not None
                    else " — ego in step"
                )
            bits = [f"leg {lg.get('index')} ({lg.get('kind')}{cut}, {lg.get('t_start_rel')}-{lg.get('t_end_rel')}s): {'passed' if lg.get('passed') else 'FAILED'}"]
            if lg.get("diverged"):
                bits.append("re-init " + ", ".join(f"{d.get('kind')} t={d.get('t_rel')}s at {d.get('at')}" for d in lg["diverged"]))
            if lg.get("recording_ran_out"):
                ro = lg["recording_ran_out"]
                bits.append(f"the recording ran out while it stood ({ro.get('stood')}, {ro.get('idle_sec')}s without a reason) — inconclusive")
            if lg.get("ahead"):
                bits.append(f"ran {lg['ahead'].get('ahead_sec')}s ahead of the recording")
            if lg.get("gaps_sec"):
                bits.append("drove through a " + "+".join(f"{g}s" for g in lg["gaps_sec"]) + " hole in the recording")
            if lg.get("stalled"):
                bits.append("stalled 15 s+ where the recording drove")
            for d in lg.get("did_not_depart") or []:
                bits.append(f"did not depart: {d.get('delay_sec')}s late at t={d.get('t_rel')}s")
            if lg.get("departure_delay_sec") is not None:
                bits.append(f"departure {lg['departure_delay_sec']:+.1f}s")
            elif lg.get("kind") in ("departure", "final") and lg.get("opens") != "cut":
                bits.append("did not depart")
            late = [d for d in (lg.get("departures") or []) if (d.get("delay_sec") or 0) >= 3.0]
            if late:
                bits.append("set off late " + ", ".join(f"{d['delay_sec']}s at t={d.get('t_rel')}s" for d in late))
            if not lg.get("reached"):
                bits.append("not reached" + (f" ({lg['end_dist_m']} m short)" if lg.get("end_dist_m") is not None else ""))
            elif lg.get("arrival_delay_sec") is not None:
                bits.append(f"arrival {lg['arrival_delay_sec']:+.1f}s")
            gs_ = lg.get("goal_stop") or {}
            if gs_.get("status") == "ok":
                bits.append(
                    f"goal-stop lat {gs_.get('lateral_m')} m / lon {gs_.get('longitudinal_m')} m"
                    + (
                        (" (off the goal beyond tolerance)" if gs_.get("gating", True)
                         else " (off where the recording waited in traffic — reported)")
                        if gs_.get("flagged") else ""
                    )
                )
            elif gs_.get("status") in ("excluded_stuck", "no_stop_near_goal"):
                bits.append(f"goal-stop: {gs_['status']}")
            sg_ = lg.get("stop_go") or {}
            if sg_.get("long_stops"):
                bits.append("stops " + ", ".join(f"{st[1]}s at t={st[0]}s" for st in sg_["long_stops"]))
            if sg_.get("red_light_waits"):
                bits.append("waited at red " + ", ".join(f"{st[1]}s at t={st[0]}s" for st in sg_["red_light_waits"]))
            if sg_.get("hiccups"):
                bits.append(f"{len(sg_['hiccups'])} hiccup(s) <3s")
            justified = [w for w in lg.get("waits") or [] if w.get("justified_sec")]
            if justified:
                bits.append("justified waits " + ", ".join(
                    f"{w['justified_sec']}/{w['sec']}s at t={w['t_rel']}s ("
                    + ", ".join(f"{k} {v}s" for k, v in (w.get("reasons") or {}).items()) + ")"
                    for w in justified
                ))
            cp_ = lg.get("cut_pass")
            if cp_:
                bits.append(
                    f"drove on past the cut's stand ({cp_.get('skipped_sec')}s of the recorded wait skipped"
                    + (", UNDER A RED LIGHT" if cp_.get("light") == "red" else "") + ")"
                )
            if lg.get("scene_jump_contact"):
                sj = lg["scene_jump_contact"]
                bits.append(f"contact with {str(sj.get('object_id'))[:8]} at t={sj.get('t_rel')}s right after the scene jump — not judged from there (inconclusive)")
            for sd in lg.get("scene_defects") or []:
                bits.append(f"scene defect at t={sd.get('t_rel')}s: " + " inside ".join(str(o) for o in sd.get("object_ids") or [])
                            + f" ({sd.get('depth_m')} m deep, {sd.get('ego_dist_m')} m from the ego) — inconclusive")
            for mw in lg.get("mutual_waits") or []:
                bits.append(f"mutual wait {mw.get('mutual_sec')}s of {mw.get('sec')}s at t={mw.get('t_rel')}s with "
                            + ", ".join(str(o)[:8] for o in (mw.get('object_ids') or [])) + " (waiting for the ego"
                            + ("; no way round it for " + ", ".join(mw["no_way_round"]) if mw.get("no_way_round") else "") + ")")
            if lg.get("goal_missed"):
                inc = lg.get("incomplete") or {}
                bits.append(f"goal missed: {lg['goal_missed']}" + (
                    f" (window {inc.get('window')}, unexplained lag {inc.get('unexplained_lag_sec')}s of {inc.get('lag_sec')}s, v={inc.get('ego_v')})"
                    if inc else ""
                ))
            parts.append("; ".join(bits))
        tip = html.escape(" | ".join(parts))
        chips.append(
            f"<span class='chip chip-{state}' title='{tip}'><span class='k'>legs</span>"
            f"<span class='v'>{win['legs_passed']}/{win['legs_total']} passed"
            + (f" · {win['cut_placements']} placed" if win.get("cut_placements") else "")
            + "</span></span>"
        )
    cf = metrics.get("comfort") or {}
    if cf.get("status") in ("ok", "no_reference") and cf.get("run"):
        run_ = cf["run"]; rec_ = cf.get("recording") or {}
        state = "bad" if cf.get("flagged") else "good"
        tip = html.escape(
            f"harsh brakes {run_['harsh_brake_count']} ({run_['harsh_brake_sec']} s, max decel {run_['max_decel_mps2']} m/s^2), "
            f"harsh jerk events {run_['harsh_jerk_count']}, |jerk| p95 {run_['jerk_p95_mps3']} m/s^3; recording: brakes "
            f"{rec_.get('harsh_brake_count', '?')}, max decel {rec_.get('max_decel_mps2', '?')}, jerk p95 {rec_.get('jerk_p95_mps3', '?')}; "
            f"brakes at (t, s, m/s^2): {run_.get('harsh_brakes')}"
        )
        chips.append(
            f"<span class='chip chip-{state}' title='{tip}'><span class='k'>comfort</span>"
            f"<span class='v'>{run_['harsh_brake_count']} harsh brakes · max brake {run_['max_decel_mps2']:.1f} m/s² · jerk p95 {run_['jerk_p95_mps3']:.1f} m/s³</span></span>"
        )
    px = metrics.get("proximity") or {}
    if px.get("status") == "ok" and px.get("min_distance_moving_m") is not None:
        c_ = px.get("closest") or {}
        tip = html.escape(
            f"closest {px['min_distance_moving_m']} m to {c_.get('label')} {str(c_.get('object_id'))[:8]} at t={c_.get('t_rel')}s "
            f"({c_.get('ego_speed_mps')} m/s); {px.get('close_sec')} s within 1 m while moving"
        )
        nm = px.get("near_misses") or []
        if nm:
            tip += " | " + "; ".join(
                f"{n.get('distance_m')} m to {n.get('label')} at t={n.get('t_rel')}s ({n.get('ego_speed_mps')} m/s)"
                for n in nm[:5]
            )
        state = "warn" if (nm or float(px["min_distance_moving_m"]) < 0.5) else "good"
        chips.append(
            f"<span class='chip chip-{state}' title='{html.escape(tip) if nm else tip}'><span class='k'>proximity</span>"
            f"<span class='v'>{px['min_distance_moving_m']:.2f} m · {px.get('close_sec')} s &lt;1 m"
            + (f" · {len(nm)} near miss{'es' if len(nm) != 1 else ''} &lt;{px.get('near_miss_m', 0.5)} m" if nm else "")
            + "</span></span>"
        )
    # Segments the model did not drive the way the recording did: the ego fell
    # behind the recording and was set back onto it.
    rs = metrics.get("resync") or {}
    if rs.get("status") == "ok" and rs.get("count"):
        tip = html.escape(
            "; ".join(
                f"#{i + 1} t={ev.get('t_rel')}s {ev.get('kind')} at ({ev.get('x')}, {ev.get('y')})"
                f" lag {ev.get('lag_sec')}s, moved {ev.get('jump_m')} m, handed back at {ev.get('handed_back_v')} m/s"
                for i, ev in enumerate(rs.get("events", []))
            )
        )
        chips.append(
            f"<span class='chip chip-bad' title='{tip}'><span class='k'>re-sync</span>"
            f"<span class='v'>{rs['count']}</span></span>"
        )
    # Who was driving: the model's plans, or stop stubs the stack put in their
    # place. On a stalled run this chip is the difference between blaming the
    # model and blaming the stack.
    stub = metrics.get("stack_intervention") or {}
    if stub.get("status") == "ok" and isinstance(
        stub.get("stop_point_ratio"), (int, float)
    ):
        state = "warn" if stub.get("flagged") else "good"
        longest = stub.get("longest_forced_stop_sec")
        extra = (
            f", {longest:.0f}s unbroken"
            if isinstance(longest, (int, float)) and longest
            else ""
        )
        # "% of cycles" meant nothing to the first person who read it. The
        # trajectory publishes at a steady 10 Hz, so the share of messages is the
        # share of the run — say that instead, and put the mechanism in a tooltip
        # rather than in a chip that has room for six words.
        tip = html.escape(
            "Share of the run where the controller received a StopPointFixer stop "
            "point (3 points) instead of the model's plan (80 points), and the "
            "longest unbroken stretch of it. High values mean the ego was held by "
            "the stack, not stopped by the model."
        )
        chips.append(
            f"<span class='chip chip-{state}' title='{tip}'>"
            f"<span class='k'>plan replaced</span>"
            f"<span class='v'>{stub['stop_point_ratio'] * 100:.0f}% of run"
            f"{extra}</span></span>"
        )
    return f"<div class='chips'>{''.join(chips)}</div>" if chips else ""


def _video_block(video: Path, results_root: Path | None) -> str:
    """An inline player for the run, not a path to go and find.

    The preview is the fastest way to understand a result, and it was reachable only
    by copying a path out of the page and opening it by hand. The source is relative
    to the dashboard so the pair stays portable — copy or mount the results directory
    anywhere and the players still resolve.

    preload="none" matters: a grid can hold dozens of runs, and nothing should be
    fetched until someone presses play.
    """
    if not video.is_file():
        return ""
    src = str(video)
    if results_root is not None:
        try:
            src = str(video.resolve().relative_to(Path(results_root).resolve()))
        except ValueError:
            src = str(video)
    # Cache-buster: reruns overwrite the same filename, and the browser kept
    # serving the bytes of a superseded run — the user debugged a lurching,
    # goal-missing drive that no current output contained. The mtime in the
    # query makes every rerun a fresh URL.
    try:
        src = f"{src}?v={int(video.stat().st_mtime)}"
    except OSError:
        pass
    src = html.escape(src, quote=True)
    # A new tab: inside the GUI the page sits in an iframe, and following the
    # link replaced the dashboard with the bare video — with no way back.
    return (
        f"<div class='vid' data-src='{src}'>"
        f"<div class='vid-path'><a href='{src}' target='_blank' rel='noopener'>open preview</a> "
        f"<code>{html.escape(video.name)}</code></div></div>"
    )


def _diagnosis_block(job: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    """The plain-language reading of this outcome, if one was recorded.

    Without it the cell shows the raw error, which is exact and unreadable. The
    diagnosis is computed once when the job finishes; the dashboard only has to
    stop hiding it.
    """
    found = None
    for source in (metrics or {}), (load_job_status(job) or {}):
        if isinstance(source.get("diagnosis"), dict):
            found = source["diagnosis"]
            break
    if not found:
        return ""
    head = html.escape(str(found.get("headline") or ""))
    why = html.escape(str(found.get("why") or ""))
    look = html.escape(str(found.get("look_at") or ""))
    return (
        f"<div class='diag'><b>{head}</b>"
        f"<div class='diag-why'>{why}</div>"
        f"<div class='diag-look'>Look at: {look}</div></div>"
    )


def _pace(metrics: dict[str, Any] | None) -> float | None:
    """Run time over recorded time. In window mode the legs' drive time from
    set-off to arrival (dwells excluded); otherwise reference_speed's whole-bag
    span, which also counts start-up, holds and the final grace."""
    wd = ((metrics or {}).get("windows") or {}).get("drive_ratio")
    if isinstance(wd, (int, float)):
        return float(wd)
    tr = ((metrics or {}).get("reference_speed") or {}).get("time_ratio")
    return float(tr) if isinstance(tr, (int, float)) else None


def _reference_speed_face(metrics: dict[str, Any] | None) -> str:
    """How this drive compared with the recording it replayed."""
    ref = (metrics or {}).get("reference_speed") or {}
    if ref.get("status") != "ok":
        return ""
    ratio = _pace(metrics)
    p90 = ref.get("p90_diff_mps")
    if not isinstance(ratio, (int, float)) or not isinstance(p90, (int, float)):
        return ""
    cls = "ref-flag" if ref.get("flagged") else "ref-ok"
    return (
        f"<div class='{cls}'>ref {ratio * 100:.0f}% of recorded time"
        f" · p90 {p90:+.1f} m/s</div>"
    )


def _intervention_note(job: dict[str, Any]) -> str:
    """Short badge naming what the harness did to this run, or nothing if clean."""
    counts = job.get("interventions") or {}
    parts = []
    if float(counts.get("forward_shifts", 0)) > 0:
        shifts = int(counts["forward_shifts"])
        metres = float(counts.get("reposition_total_m", 0.0))
        parts.append(f"{shifts}× forward shift ({metres:.0f} m)")
    if float(counts.get("blinker_nudges", 0)) > 0:
        parts.append(f"{int(counts['blinker_nudges'])}× blinker nudge")
    if float(counts.get("resyncs", 0)) > 0:
        parts.append(f"{int(counts['resyncs'])}× re-sync")
    if not parts:
        return ""
    return f" <span class='assisted'>assisted: {html.escape(', '.join(parts))}</span>"


def _sample_note(used: float, attempted: float) -> str:
    """Sample size behind an average, so a subset average cannot read as a full one."""
    used_i = int(used)
    attempted_i = int(attempted)
    if attempted_i <= 0:
        return ""
    cls = "nsub" if used_i == attempted_i else "nsub partial"
    return f" <span class='{cls}'>n={used_i}/{attempted_i}</span>"


def _safe_rate(block: dict[str, Any] | None) -> float | None:
    if not block or "rate" not in block:
        return None
    rate = block.get("rate")
    if isinstance(rate, (int, float)) and not math.isnan(float(rate)):
        return float(rate)
    return None


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return None


def collect_stop_scatter_points(
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One point per completed job with valid goal-frame lat/lon."""
    points: list[dict[str, Any]] = []
    for job in jobs:
        metrics = load_metrics(job)
        if metrics is None:
            continue
        goal = metrics.get("goal_stop_precision", {}) or {}
        lat = _finite(goal.get("lateral_m"))
        lon = _finite(goal.get("longitudinal_m"))
        if lat is None or lon is None:
            continue
        bag = str(job.get("bag_key") or "")
        points.append(
            {
                "id": bag,
                "bag": bag,
                "model": _model_key(job),
                "lateral_m": lat,
                "longitudinal_m": lon,
                "heading_error_deg": _finite(goal.get("heading_error_deg")) or 0.0,
                "position_error_m": _finite(goal.get("position_error_m")),
                "passed": bool(metrics.get("pass")),
                "flagged": bool(goal.get("flagged")),
                "status": str(goal.get("status") or ""),
            }
        )
    return points


def _pose_arrow_svg(
    cx: float,
    cy: float,
    heading_deg: float,
    *,
    length_px: float,
    cls: str,
) -> str:
    """RViz-style 2D pose arrow in SVG (X forward right, Y left up on plot).

    ``heading_deg`` is ROS yaw in the goal frame: 0 = +X (forward/right on plot),
    positive = CCW toward +Y (left/up on plot). SVG Y grows down, so rotate by -yaw.
    """
    # Convert plot-frame ROS yaw to SVG rotation (degrees, CW-positive in SVG).
    svg_rot = -float(heading_deg)
    # Arrow in local SVG coords with tip along +x before rotation.
    shaft = length_px * 0.55
    head = length_px * 0.45
    half_w = max(3.5, length_px * 0.12)
    half_head = max(6.0, length_px * 0.22)
    # Body from -shaft*0.2 to tip so the pose sits near the arrow center/back.
    pts = (
        f"{-shaft * 0.15:.1f},{half_w:.1f} "
        f"{shaft:.1f},{half_w:.1f} "
        f"{shaft:.1f},{half_head:.1f} "
        f"{shaft + head:.1f},0 "
        f"{shaft:.1f},{-half_head:.1f} "
        f"{shaft:.1f},{-half_w:.1f} "
        f"{-shaft * 0.15:.1f},{-half_w:.1f}"
    )
    return (
        f'<g transform="translate({cx:.1f} {cy:.1f}) rotate({svg_rot:.2f})">'
        f'<polygon points="{pts}" class="{cls}"/>'
        f"</g>"
    )


def _stop_scatter_svg(
    points: list[dict[str, Any]],
    *,
    title: str,
    lat_tol_m: float = 2.0,
    lon_tol_m: float = 2.0,
) -> str:
    """Goal-frame scatter like RViz top-down: X=forward (right), Y=left (up)."""
    if not points:
        return (
            f"<div class='scatter-card'><h3>{html.escape(title)}</h3>"
            "<p class='hint'>No stop samples with valid lat/lon yet.</p></div>"
        )

    lats = [p["lateral_m"] for p in points]
    lons = [p["longitudinal_m"] for p in points]
    hdgs = [float(p.get("heading_error_deg") or 0.0) for p in points]
    avg_lat = sum(lats) / len(lats)
    avg_lon = sum(lons) / len(lons)
    avg_hdg = sum(hdgs) / len(hdgs)

    pad = 0.45
    span = max(
        max(abs(v) for v in lats + [avg_lat, lat_tol_m]) + pad,
        max(abs(v) for v in lons + [avg_lon, lon_tol_m]) + pad,
        1.0,
    )
    x_min = y_min = -span
    x_max = y_max = span

    width, height = 540, 540
    ml, mr, mt, mb = 58, 28, 36, 52
    pw, ph = width - ml - mr, height - mt - mb

    def sx(x_fwd: float) -> float:
        # Plot X = longitudinal / ROS +X → right on screen
        return ml + (x_fwd - x_min) / (x_max - x_min) * pw

    def sy(y_left: float) -> float:
        # Plot Y = lateral / ROS +Y → up on screen (SVG y flips)
        return mt + (y_max - y_left) / (y_max - y_min) * ph

    ox, oy = sx(0.0), sy(0.0)
    # Arrow length ~ 12% of plot span in pixels
    arrow_len = max(28.0, min(48.0, pw * 0.10))

    tick = 0.5 if span <= 3 else (1.0 if span <= 8 else 2.0)
    grid_parts: list[str] = []
    t = -math.floor(span / tick) * tick
    while t <= span + 1e-9:
        if abs(t) > 1e-9:
            gx, gy = sx(t), sy(t)
            grid_parts.append(
                f'<line x1="{sx(x_min):.1f}" y1="{gy:.1f}" x2="{sx(x_max):.1f}" '
                f'y2="{gy:.1f}" class="grid"/>'
            )
            grid_parts.append(
                f'<line x1="{gx:.1f}" y1="{sy(y_min):.1f}" x2="{gx:.1f}" '
                f'y2="{sy(y_max):.1f}" class="grid"/>'
            )
            grid_parts.append(
                f'<text x="{gx:.1f}" y="{height - 20:.1f}" class="tick">{t:g}</text>'
            )
            grid_parts.append(
                f'<text x="12" y="{gy + 4:.1f}" class="tick">{t:g}</text>'
            )
        t += tick

    tol_rect = (
        f'<rect x="{sx(-lon_tol_m):.1f}" y="{sy(lat_tol_m):.1f}" '
        f'width="{sx(lon_tol_m) - sx(-lon_tol_m):.1f}" '
        f'height="{sy(-lat_tol_m) - sy(lat_tol_m):.1f}" '
        f'class="tol-box"/>'
    )

    # Axes + RViz-style triad at goal (X red-ish forward, Y green-ish left)
    axes = (
        f'<line x1="{sx(x_min):.1f}" y1="{oy:.1f}" x2="{sx(x_max):.1f}" '
        f'y2="{oy:.1f}" class="axis"/>'
        f'<line x1="{ox:.1f}" y1="{sy(y_min):.1f}" x2="{ox:.1f}" '
        f'y2="{sy(y_max):.1f}" class="axis"/>'
        # Corner triad (fixed screen inset) — X right, Y up
        f'<g class="rviz-triad" transform="translate({ml + 28:.0f} {mt + 28:.0f})">'
        f'<line x1="0" y1="0" x2="36" y2="0" class="triad-x"/>'
        f'<polygon points="36,0 28,-5 28,5" class="triad-x"/>'
        f'<line x1="0" y1="0" x2="0" y2="-36" class="triad-y"/>'
        f'<polygon points="0,-36 -5,-28 5,-28" class="triad-y"/>'
        f'<circle cx="0" cy="0" r="3" class="triad-origin"/>'
        f'<text x="40" y="4" class="triad-label">X fwd</text>'
        f'<text x="4" y="-40" class="triad-label">Y left</text>'
        f"</g>"
    )

    goal_pose = (
        _pose_arrow_svg(ox, oy, 0.0, length_px=arrow_len * 1.15, cls="goal-arrow")
        + f'<text x="{ox + 10:.1f}" y="{oy - 14:.1f}" class="goal-label">GOAL</text>'
    )

    point_parts: list[str] = []
    legend_rows: list[str] = []
    for p in points:
        px, py = sx(p["longitudinal_m"]), sy(p["lateral_m"])
        hdg = float(p.get("heading_error_deg") or 0.0)
        cls = "pt-pass" if p["passed"] and not p.get("flagged") else "pt-fail"
        pid = html.escape(str(p["id"]))
        point_parts.append(
            _pose_arrow_svg(px, py, hdg, length_px=arrow_len, cls=cls)
            + f'<text x="{px + 12:.1f}" y="{py - 10:.1f}" class="pt-id">{pid}</text>'
        )
        legend_rows.append(
            "<tr>"
            f"<td><span class='swatch {cls}'></span><b>{pid}</b></td>"
            f"<td>{html.escape(str(p['bag']))}</td>"
            f"<td>{_fmt_m(p['longitudinal_m'])}</td>"
            f"<td>{_fmt_m(p['lateral_m'])}</td>"
            f"<td>{_fmt_m(hdg)}</td>"
            f"<td>{'PASS' if p['passed'] else 'FAIL'}</td>"
            "</tr>"
        )

    ax, ay = sx(avg_lon), sy(avg_lat)
    avg_mark = (
        f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="14" class="avg-ring"/>'
        + _pose_arrow_svg(ax, ay, avg_hdg, length_px=arrow_len, cls="avg-arrow")
        + f'<text x="{ax + 14:.1f}" y="{ay + 5:.1f}" class="avg-label">AVG</text>'
    )
    legend_rows.append(
        "<tr class='avg-row'>"
        "<td><span class='swatch avg'></span><b>AVG</b></td>"
        f"<td>mean of {len(points)} stops</td>"
        f"<td>{_fmt_m(avg_lon)}</td>"
        f"<td>{_fmt_m(avg_lat)}</td>"
        f"<td>{_fmt_m(avg_hdg)}</td>"
        "<td>—</td>"
        "</tr>"
    )

    svg = f"""
<svg class="stop-scatter" viewBox="0 0 {width} {height}" width="{width}" height="{height}"
     role="img" aria-label="{html.escape(title)}">
  <rect x="0" y="0" width="{width}" height="{height}" class="plot-bg"/>
  {''.join(grid_parts)}
  {tol_rect}
  {axes}
  {goal_pose}
  {''.join(point_parts)}
  {avg_mark}
  <text x="{width / 2:.0f}" y="{height - 8}" class="axis-title">
    X / longitudinal [m]  (forward →)
  </text>
  <text x="14" y="{height / 2:.0f}" class="axis-title-y"
        transform="rotate(-90 14 {height / 2:.0f})">
    Y / lateral [m]  (left ↑)
  </text>
</svg>
"""
    return f"""
<div class="scatter-card">
  <h3>{html.escape(title)}</h3>
  <p class="hint">RViz-style goal frame: <b>X forward (right)</b>, <b>Y left (up)</b>.
     Arrows show stop pose (position + heading). Dashed box = lat/lon tolerance
     (±{_fmt_m(lat_tol_m)} / ±{_fmt_m(lon_tol_m)} m). Yellow ring = average pose.</p>
  <div class="scatter-row">
    {svg}
    <table class="scatter-legend summary">
      <tr><th>ID</th><th>Rosbag</th><th>X lon [m]</th><th>Y lat [m]</th>
      <th>yaw [deg]</th><th>Result</th></tr>
      {''.join(legend_rows)}
    </table>
  </div>
</div>
"""


def render_stop_scatter_section(
    points: list[dict[str, Any]],
    *,
    lat_tol_m: float = 2.0,
    lon_tol_m: float = 2.0,
    precision_table_html: str = "",
) -> str:
    """One cohesive goal-stop panel: map + optional per-rosbag table."""
    if not points:
        cards_html = (
            "<p class='hint'>No map markers yet — need status=ok with lat/lon. "
            "Bags that never reached the goal still appear in the table below.</p>"
        )
    else:
        models = sorted({p["model"] for p in points})
        cards: list[str] = []
        for model in models:
            subset = [p for p in points if p["model"] == model]
            title = (
                f"Stop poses — {model}" if len(models) > 1 else "Stop poses (goal frame)"
            )
            cards.append(
                _stop_scatter_svg(
                    subset,
                    title=title,
                    lat_tol_m=lat_tol_m,
                    lon_tol_m=lon_tol_m,
                )
            )
        cards_html = f"<div class='scatter-grid'>{''.join(cards)}</div>"

    table_block = ""
    if precision_table_html:
        table_block = (
            "<div class='precision-table-wrap'>"
            "<h3>Per-rosbag detail</h3>"
            "<p class='hint'>IDs match arrow labels. Rows with status ≠ ok have no marker.</p>"
            f"{precision_table_html}"
            "</div>"
        )

    return (
        "<section class='goal-stop-panel' id='goal-stop'>"
        "<h2>Goal stop precision</h2>"
        "<p>Where each rosbag stopped vs the goal "
        "(RViz: <b>X forward →</b>, <b>Y left ↑</b>). "
        "Arrow = stop pose; label = scenario ID.</p>"
        f"{cards_html}"
        f"{table_block}"
        "</section>"
    )


def build_dashboard(manifest: dict[str, Any], output_html: Path) -> Path:
    jobs = manifest.get("jobs", [])
    models = sorted({_model_key(j) for j in jobs})
    bags = sorted({j["bag_key"] for j in jobs})
    counts = _status_counts(jobs)
    total = len(jobs) or 1
    done_frac = counts["done"] / total
    in_progress = counts["pending"] + counts["running"] > 0
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    results_root = Path(manifest.get("results_root") or output_html.parent)

    # Index job by (bag, model)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for j in jobs:
        by_key[(j["bag_key"], _model_key(j))] = j

    model_stats: dict[str, dict[str, float]] = {
        m: {
            # attempted = every job that reached a terminal state (done or failed).
            # A failed job produces no metrics, so counting only `n` (completed)
            # would drop it from both sides of the pass rate and report a perfect
            # score for a run that half failed.
            "attempted": 0,
            "failed": 0,
            "assisted": 0,
            "n": 0,
            "pass": 0,
            "collision_rate_sum": 0.0,
            "collision_rate_n": 0,
            "oob_rate_sum": 0.0,
            "oob_rate_n": 0,
            "stuck_rate_sum": 0.0,
            "stuck_rate_n": 0,
            "goal_pos_sum": 0.0,
            "legs_passed": 0,
            "legs_total": 0,
            "harsh_brakes": 0,
            "harsh_brakes_rec": 0,
            "comfort_n": 0,
            "max_decel": 0.0,
            "max_decel_rec": 0.0,
            "jerk_p95_sum": 0.0,
            "jerk_p95_rec_sum": 0.0,
            "prox_min": float("inf"),
            "prox_close_sec": 0.0,
            "near_misses": 0,
            "bus_served": 0,
            "bus_recorded": 0,
            "dep_delay_sum": 0.0,
            "dep_delay_n": 0,
            "long_stops": 0,
            "stalls": 0,
            "late_departures": 0,
            "goal_lat_sum": 0.0,
            "goal_lon_sum": 0.0,
            "goal_n": 0,
            # The axes that actually separate candidate models in this fleet:
            # how often the stack had to replace their plans (departure hold),
            # how much green light they sat through latched, how their pace
            # compares to the recording, and how many contacts were their own.
            "stub_sum": 0.0,
            "stub_n": 0,
            "green_sum": 0.0,
            "pace_sum": 0.0,
            "pace_n": 0,
            "ego_caused": 0,
        }
        for m in models
    }

    precision_rows: list[str] = []
    stop_points = collect_stop_scatter_points(jobs)
    # Prefer threshold values from the first available metrics payload if present
    lat_tol = 2.0
    lon_tol = 2.0
    for job in jobs:
        metrics = load_metrics(job)
        if not metrics:
            continue
        thr = metrics.get("thresholds") or {}
        if isinstance(thr.get("goal_lateral_tolerance_m"), (int, float)):
            lat_tol = float(thr["goal_lateral_tolerance_m"])
        if isinstance(thr.get("goal_longitudinal_tolerance_m"), (int, float)):
            lon_tol = float(thr["goal_longitudinal_tolerance_m"])
        break
    rows_html = []
    for bag in bags:
        cells = []
        for model in models:
            job = by_key.get((bag, model))
            if not job:
                cells.append(
                    f"<article class='card na'><header><h3>{html.escape(bag)}</h3>"
                    f"<span class='pill pill-na'>no job</span></header></article>"
                )
                continue

            status = str(job.get("status") or "pending")
            detail_id = html.escape(job["job_id"])

            if status == "pending":
                tip = html.escape(_pending_label(job))
                cells.append(
                    f"<article class='card pending'>{_card_head(bag, model, models, 'QUEUED', 'pending')}"
                    f"<p class='muted'>{tip}</p></article>"
                )
                continue

            if status == "running":
                # The manifest marks every dispatched job "running", including those
                # still queued behind a busy worker. Only a job that has written its
                # own status file has actually started.
                js = load_job_status(job)
                started = bool(js and js.get("phase"))
                phase = html.escape(str((js or {}).get("phase") or "waiting for a worker"))
                verdict, cls = ("RUNNING", "running") if started else ("QUEUED", "pending")
                cells.append(
                    f"<article class='card {cls}'>{_card_head(bag, model, models, verdict, cls)}"
                    f"<p class='muted'>{phase}</p>"
                    f"<p class='path'><code>{html.escape(job['output_dir'])}</code></p>"
                    f"</article>"
                )
                continue

            if status == "skipped":
                cells.append(
                    f"<article class='card na'>{_card_head(bag, model, models, 'SKIPPED', 'na')}"
                    f"<p class='muted'>excluded from this run</p></article>"
                )
                continue

            if status == "dry_run":
                cells.append(
                    f"<article class='card pending'>{_card_head(bag, model, models, 'DRY RUN', 'pending')}"
                    f"<p class='muted'>validated, not simulated</p></article>"
                )
                continue

            if status == "failed":
                tip = html.escape(str(job.get("error") or "failed"))
                model_stats[model]["attempted"] += 1
                model_stats[model]["failed"] += 1
                # A failed run still records a bag, and now a preview video too.
                # Link them: these are the runs that most need looking at.
                fail_video = preview_video_path(job, results_root=results_root)
                fail_links = _diagnosis_block(job, load_metrics(job))
                fail_links += f"<div class='raw-err'>{tip}</div>"
                fail_links += _video_block(fail_video, results_root)
                fail_links += (
                    f"<div>output: <code>{html.escape(job['output_dir'])}</code></div>"
                )
                cells.append(
                    f"<article class='card fail'>"
                    f"{_card_head(bag, model, models, 'FAILED', 'fail')}"
                    f"{fail_links}</article>"
                )
                continue

            metrics = load_metrics(job)
            model_stats[model]["attempted"] += 1
            model_stats[model]["n"] += 1
            if metrics is None:
                cells.append(
                    f"<article class='card processing'>"
                    f"{_card_head(bag, model, models, 'SCORING', 'processing')}"
                    f"<p class='muted'>drive finished; metrics still being computed</p></article>"
                )
                continue

            passed = bool(metrics.get("pass"))
            # An arrival the harness had to assist is reported separately: it is not
            # a failure, but it is not evidence the planner drove the scenario either.
            assisted = not bool(job.get("clean_run", True))
            if passed:
                model_stats[model]["pass"] += 1
                if assisted:
                    model_stats[model]["assisted"] += 1

            goal = metrics.get("goal_stop_precision", {}) or {}
            stuck = metrics.get("stuck_rate", {}) or {}
            col = metrics.get("collision_rate", {}) or {}
            oob = metrics.get("out_of_boundary", {}) or {}

            si = metrics.get("stack_intervention", {}) or {}
            rs = metrics.get("reference_speed", {}) or {}
            if isinstance(si.get("stop_point_ratio"), (int, float)):
                model_stats[model]["stub_sum"] += float(si["stop_point_ratio"])
                model_stats[model]["stub_n"] += 1
            if isinstance(si.get("green_latched_sec"), (int, float)):
                model_stats[model]["green_sum"] += float(si["green_latched_sec"])
            if _pace(metrics) is not None:
                model_stats[model]["pace_sum"] += _pace(metrics)
                model_stats[model]["pace_n"] += 1
            model_stats[model]["ego_caused"] += int(
                (col.get("by_attribution") or {}).get("ego_caused", 0)
            )
            win_ = metrics.get("windows", {}) or {}
            if win_.get("status") == "ok" and win_.get("legs_total"):
                model_stats[model]["legs_passed"] += int(win_.get("legs_passed") or 0)
                model_stats[model]["legs_total"] += int(win_.get("legs_total") or 0)
            cf_ = metrics.get("comfort", {}) or {}
            if cf_.get("status") in ("ok", "no_reference") and cf_.get("run"):
                run_c = cf_["run"]; rec_c = cf_.get("recording") or {}
                model_stats[model]["harsh_brakes"] += int(run_c.get("harsh_brake_count") or 0)
                model_stats[model]["harsh_brakes_rec"] += int(rec_c.get("harsh_brake_count") or 0)
                model_stats[model]["comfort_n"] += 1
                model_stats[model]["max_decel"] = max(model_stats[model]["max_decel"], float(run_c.get("max_decel_mps2") or 0.0))
                model_stats[model]["max_decel_rec"] = max(model_stats[model]["max_decel_rec"], float(rec_c.get("max_decel_mps2") or 0.0))
                model_stats[model]["jerk_p95_sum"] += float(run_c.get("jerk_p95_mps3") or 0.0)
                model_stats[model]["jerk_p95_rec_sum"] += float(rec_c.get("jerk_p95_mps3") or 0.0)
            px_ = metrics.get("proximity", {}) or {}
            if px_.get("status") == "ok" and px_.get("min_distance_moving_m") is not None:
                model_stats[model]["prox_min"] = min(model_stats[model]["prox_min"], float(px_["min_distance_moving_m"]))
                model_stats[model]["prox_close_sec"] += float(px_.get("close_sec") or 0.0)
                model_stats[model]["near_misses"] += len(px_.get("near_misses") or [])
            bs_ = metrics.get("bus_stop_service", {}) or {}
            if bs_.get("status") == "ok":
                model_stats[model]["bus_served"] += int(bs_.get("served") or 0)
                model_stats[model]["bus_recorded"] += int(bs_.get("recorded") or 0)
            for lg_ in (win_.get("legs") or []):
                if lg_.get("departure_delay_sec") is not None:
                    model_stats[model]["dep_delay_sum"] += float(lg_["departure_delay_sec"])
                    model_stats[model]["dep_delay_n"] += 1
                model_stats[model]["long_stops"] += len(((lg_.get("stop_go") or {}).get("long_stops")) or [])
                model_stats[model]["stalls"] += 1 if lg_.get("stalled") else 0
                model_stats[model]["late_departures"] += len(lg_.get("did_not_depart") or [])

            for key, block in (
                ("collision_rate", col),
                ("oob_rate", oob),
                ("stuck_rate", stuck),
            ):
                r = _safe_rate(block)
                if r is not None:
                    model_stats[model][f"{key}_sum"] += r
                    model_stats[model][f"{key}_n"] += 1

            goal_status = goal.get("status")
            if goal_status == "ok":
                pos = goal.get("position_error_m")
                lat = goal.get("abs_lateral_m", goal.get("lateral_m"))
                lon = goal.get("abs_longitudinal_m", goal.get("longitudinal_m"))
                if isinstance(pos, (int, float)) and not math.isnan(float(pos)):
                    model_stats[model]["goal_pos_sum"] += float(pos)
                    model_stats[model]["goal_n"] += 1
                    if isinstance(lat, (int, float)) and not math.isnan(float(lat)):
                        model_stats[model]["goal_lat_sum"] += abs(float(lat))
                    if isinstance(lon, (int, float)) and not math.isnan(float(lon)):
                        model_stats[model]["goal_lon_sum"] += abs(float(lon))

            video = preview_video_path(job, results_root=results_root)
            # The card shows the diagnosis and the player at top level; the details
            # block is for the raw numbers only, so they are not built twice.
            video_link = ""

            # Always-visible stop-precision summary (lat / lon) on the cell face
            if goal_status == "ok":
                goal_face = (
                    f"<div class='goal-face'>"
                    f"pos {_fmt_m(goal.get('position_error_m'))} m<br/>"
                    f"lat {_fmt_m(goal.get('lateral_m'))} m "
                    f"(|lat| {_fmt_m(goal.get('abs_lateral_m'))})<br/>"
                    f"lon {_fmt_m(goal.get('longitudinal_m'))} m "
                    f"(|lon| {_fmt_m(goal.get('abs_longitudinal_m'))})<br/>"
                    f"hdg {_fmt_m(goal.get('heading_error_deg'))}°"
                    f"</div>"
                )
            else:
                goal_face = (
                    f"<div class='goal-face muted'>goal: "
                    f"{html.escape(str(goal_status or 'n/a'))}</div>"
                )
            # A run can meet every rule and still not have driven the road the way it
            # was driven, so this sits on the cell face rather than inside the detail.
            goal_face += _reference_speed_face(metrics)
            # Say a preview exists without making someone open the cell to find out.
            if video.is_file():
                goal_face += "<div class='has-vid'>▶ preview</div>"

            detail = (
                f"<div class='detail' id='d-{detail_id}'>"
                f"<div class='metric-block'><b>Goal stop precision</b> "
                f"<span class='hint'>(goal frame; low-speed samples near goal)</span></div>"
                f"<table class='mini'>"
                f"<tr><th>position [m]</th><td>{_fmt_m(goal.get('position_error_m'))}</td></tr>"
                f"<tr><th>lateral [m]</th><td>{_fmt_m(goal.get('lateral_m'))} "
                f"(abs {_fmt_m(goal.get('abs_lateral_m'))})</td></tr>"
                f"<tr><th>longitudinal [m]</th><td>{_fmt_m(goal.get('longitudinal_m'))} "
                f"(abs {_fmt_m(goal.get('abs_longitudinal_m'))})</td></tr>"
                f"<tr><th>heading [deg]</th><td>{_fmt_m(goal.get('heading_error_deg'))}</td></tr>"
                f"<tr><th>samples / status</th><td>{goal.get('sample_count', '—')} / "
                f"{html.escape(str(goal.get('status')))}</td></tr>"
                f"<tr><th>flagged</th><td>{goal.get('flagged')}</td></tr>"
                f"</table>"
                f"<div class='metric-block'><b>Rates</b></div>"
                f"<div><b>stuck_rate</b>={_fmt_rate(stuck.get('rate'))} "
                f"(flagged={stuck.get('flagged')}, dur={_fmt_m(stuck.get('duration_sec'))}s)</div>"
                f"<div><b>collision_rate</b>={_fmt_rate(col.get('rate'))} "
                f"(flagged={col.get('flagged')}, min_d={_fmt_m(col.get('min_distance_m'))} m)</div>"
                f"<div><b>oob_rate</b>={_fmt_rate(oob.get('rate'))} "
                f"(flagged={oob.get('flagged')}, min_d={_fmt_m(oob.get('min_distance_m'))} m)</div>"
                f"{_intervention_note(job)}"
                f"{video_link}"
                f"</div>"
            )
            cls = "pass" if passed else "fail"
            # The verdict reads as legs where the run had them: "2/3 legs passed"
            # says more than FAIL (operator, 2026-09-10).
            win_v = metrics.get("windows") or {}
            if win_v.get("status") == "ok" and win_v.get("legs_total"):
                lp, lt = int(win_v.get("legs_passed") or 0), int(win_v["legs_total"])
                label = f"{lp}/{lt} legs passed"
                if passed and lp == lt:
                    label = f"PASS · {label}"
                elif not passed and lt >= 2 and lt - lp == 1:
                    cls = "partial"   # one leg short of a pass: orange, not red
            else:
                label = "PASS" if passed else "FAIL"
            if passed and assisted:
                cls, label = "assist", f"{label} (assisted)"
            cells.append(
                f"<article class='card {cls}'>"
                f"{_card_head(bag, model, models, label, cls)}"
                f"{_diagnosis_block(job, metrics)}"
                f"{_metric_chips(metrics)}"
                f"{goal_face}"
                f"{_intervention_note(job)}"
                f"{_video_block(video, results_root)}"
                f"<details class='more'><summary>all numbers</summary>{detail}</details>"
                f"</article>"
            )

            precision_rows.append(
                "<tr>"
                f"<td><b>{html.escape(bag)}</b></td>"
                f"<td class='bag'>{html.escape(bag)}</td>"
                f"<td>{html.escape(model)}</td>"
                f"<td class='{cls}'>{label}</td>"
                f"<td>{html.escape(str(goal.get('status')))}</td>"
                f"<td>{_fmt_m(goal.get('position_error_m'))}</td>"
                f"<td>{_fmt_m(goal.get('lateral_m'))}</td>"
                f"<td>{_fmt_m(goal.get('longitudinal_m'))}</td>"
                f"<td>{_fmt_m(goal.get('abs_lateral_m'))}</td>"
                f"<td>{_fmt_m(goal.get('abs_longitudinal_m'))}</td>"
                f"<td>{_fmt_m(goal.get('heading_error_deg'))}</td>"
                f"<td>{goal.get('sample_count', '—')}</td>"
                "</tr>"
            )
        # One block per bag; each model's run folds under its own line and
        # opens on click (operator, 2026-09-10).
        entries = []
        for model, cell in zip(models, cells):
            m_cls = re.search(r"class='card ([a-z]+)'", cell)
            m_pill = re.search(r"pill pill-([a-z]+)'>([^<]*)", cell)
            cls_ = m_cls.group(1) if m_cls else "na"
            pill_cls = m_pill.group(1) if m_pill else cls_
            pill_txt = html.unescape(m_pill.group(2)) if m_pill else "—"
            entries.append(
                f"<details class='model-run {cls_}'>"
                f"<summary><span class='model-name'>{_model_label(model)}</span>"
                f"<span class='pill pill-{pill_cls}'>{html.escape(pill_txt)}</span></summary>"
                f"{cell}</details>"
            )
        rows_html.append(
            f"<section class='bag-block'><h3 class='bag-title'>{html.escape(bag)}</h3>{''.join(entries)}</section>"
        )

    summary_rows = []
    for model in models:
        s = model_stats[model]
        # Pass rate is over every attempted job: a job that never finished is not a
        # pass, and hiding it inflates the score exactly on the hard scenarios.
        attempted = int(s["attempted"])
        pass_rate = (s["pass"] / attempted * 100.0) if attempted else 0.0
        completed_rate = (s["n"] / attempted * 100.0) if attempted else 0.0
        avg_pos = (s["goal_pos_sum"] / s["goal_n"]) if s["goal_n"] else float("nan")
        avg_lat = (s["goal_lat_sum"] / s["goal_n"]) if s["goal_n"] else float("nan")
        avg_lon = (s["goal_lon_sum"] / s["goal_n"]) if s["goal_n"] else float("nan")
        avg_col = (
            s["collision_rate_sum"] / s["collision_rate_n"]
            if s["collision_rate_n"]
            else float("nan")
        )
        avg_oob = (
            s["oob_rate_sum"] / s["oob_rate_n"] if s["oob_rate_n"] else float("nan")
        )
        avg_stuck = (
            s["stuck_rate_sum"] / s["stuck_rate_n"] if s["stuck_rate_n"] else float("nan")
        )
        fail_note = (
            f" <span class='nsub partial'>{int(s['failed'])} failed</span>"
            if s["failed"]
            else ""
        )
        clean_passes = int(s["pass"]) - int(s["assisted"])
        clean_rate = (clean_passes / attempted * 100.0) if attempted else 0.0
        assist_note = (
            f" <span class='nsub partial'>{int(s['assisted'])} assisted</span>"
            if s["assisted"]
            else ""
        )
        stub_cell = (
            f"{s['stub_sum'] / s['stub_n'] * 100:.0f}%"
            f"{_sample_note(s['stub_n'], attempted)}"
            if s["stub_n"]
            else "—"
        )
        pace_cell = (
            f"{s['pace_sum'] / s['pace_n']:.2f}x"
            f"{_sample_note(s['pace_n'], attempted)}"
            if s["pace_n"]
            else "—"
        )
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{clean_passes}/{attempted} ({clean_rate:.0f}%){assist_note}</td>"
            f"<td>{int(s['pass'])}/{attempted} ({pass_rate:.0f}%){fail_note}</td>"
            f"<td>{int(s['n'])}/{attempted} ({completed_rate:.0f}%)</td>"
            f"<td>{_fmt_m(avg_pos)} / {_fmt_m(avg_lat)} / {_fmt_m(avg_lon)}"
            f"{_sample_note(s['goal_n'], attempted)}</td>"
            f"<td>{_fmt_rate(avg_col)}{_sample_note(s['collision_rate_n'], attempted)}</td>"
            f"<td>{_fmt_rate(avg_oob)}{_sample_note(s['oob_rate_n'], attempted)}</td>"
            f"<td>{_fmt_rate(avg_stuck)}{_sample_note(s['stuck_rate_n'], attempted)}</td>"
            f"<td>{stub_cell}</td>"
            f"<td>{s['green_sum']:.0f}s</td>"
            f"<td>{pace_cell}</td>"
            f"<td>{int(s['ego_caused'])}</td>"
            f"<td>{int(s['legs_passed'])}/{int(s['legs_total'])}"
            f"{(' (' + format(s['legs_passed'] / s['legs_total'] * 100.0, '.0f') + '%)') if s['legs_total'] else ''}</td>"
            f"<td>{int(s['harsh_brakes'])} vs rec. {int(s['harsh_brakes_rec'])}{_sample_note(s['comfort_n'], attempted)}</td>"
            "</tr>"
        )

    # The page's purpose: several models side by side, one metric per row.
    def _cmp_cell(model: str, key: str) -> str:
        st = model_stats[model]
        att = int(st["attempted"])
        if key == "success":
            return f"{int(st['pass'])}/{att} ({st['pass'] / att * 100:.0f}%)" if att else "—"
        if key == "legs":
            return f"{int(st['legs_passed'])}/{int(st['legs_total'])} ({st['legs_passed'] / st['legs_total'] * 100:.0f}%)" if st["legs_total"] else "—"
        if key == "contacts":
            return str(int(st["ego_caused"]))
        if key == "bus":
            return f"{int(st['bus_served'])}/{int(st['bus_recorded'])}" if st["bus_recorded"] else "—"
        if key == "dep":
            return f"{st['dep_delay_sum'] / st['dep_delay_n']:+.1f} s (n={int(st['dep_delay_n'])})" if st["dep_delay_n"] else "—"
        if key == "late":
            return str(int(st["late_departures"]))
        if key == "stops":
            return str(int(st["long_stops"]))
        if key == "stalls":
            return str(int(st["stalls"]))
        if key == "brakes":
            return f"{int(st['harsh_brakes'])} (rec. {int(st['harsh_brakes_rec'])})" if st["comfort_n"] else "—"
        if key == "decel":
            return f"{st['max_decel']:.2f} (rec. {st['max_decel_rec']:.2f})" if st["comfort_n"] else "—"
        if key == "jerk":
            return f"{st['jerk_p95_sum'] / st['comfort_n']:.2f} (rec. {st['jerk_p95_rec_sum'] / st['comfort_n']:.2f})" if st["comfort_n"] else "—"
        if key == "prox":
            return f"{st['prox_min']:.2f} m · {st['prox_close_sec']:.0f} s <1 m" if st["prox_min"] != float("inf") else "—"
        if key == "oob":
            return _fmt_rate(st["oob_rate_sum"] / st["oob_rate_n"]) if st["oob_rate_n"] else "—"
        if key == "near_miss":
            return str(st["near_misses"])
        if key == "pace":
            return f"{st['pace_sum'] / st['pace_n']:.2f}x" if st["pace_n"] else "—"
        if key == "stub":
            return f"{st['stub_sum'] / st['stub_n'] * 100:.0f}%" if st["stub_n"] else "—"
        return "—"

    comparison_rows = (
        ("success", "Success rate", "runs passed / attempted (all gating metrics clear)"),
        ("legs", "Legs passed", "route legs passed / total, over all bags"),
        ("contacts", "Ego-caused contacts", "contacts the ego drove into (gating)"),
        ("bus", "Bus stops served", "recorded bus stops the ego served / recorded"),
        ("dep", "Departure delay", "mean of the ego's set-off against the recorded departure, per leg"),
        ("late", "Did not depart", "legs where the ego stood 15 s past a recorded departure"),
        ("stops", "Unplanned stops ≥3 s", "stops the recording did not make (gating)"),
        ("stalls", "Stalls", "legs where the ego stood 15 s where the recording drove"),
        ("brakes", "Harsh brakes", "brakes beyond 2.0 m/s² for 0.3 s, hand-overs excluded (gating vs recording+1)"),
        ("decel", "Max brake [m/s²]", "strongest deceleration over the bags"),
        ("jerk", "Jerk p95 [m/s³]", "mean over bags of the 95th percentile of |jerk|"),
        ("prox", "Proximity", "closest a vehicle came while moving, and seconds within 1 m"),
        ("oob", "Off-lane rate", "share of frames outside lanelet borders / curbs"),
        ("near_miss", "Near misses &lt;0.5 m", "approaches closer than 0.5 m while moving, one per approach"),
        ("pace", "Pace vs recording", "drive time from set-off to arrival / the recording's, per leg (dwells excluded); whole-bag span for non-window runs"),
        ("stub", "Plan replaced", "share of trajectories the stack replaced with stop stubs"),
    )
    comparison_html = ""
    if models:
        head_c = "".join(f"<th>{_model_label(m)}</th>" for m in models)
        body_c = "".join(
            f"<tr><th class='bagname' title='{html.escape(tip)}'>{html.escape(label)}</th>"
            + "".join(f"<td>{html.escape(_cmp_cell(m, key))}</td>" for m in models)
            + "</tr>"
            for key, label, tip in comparison_rows
        )
        comparison_html = (
            "<section><h2>Models side by side</h2>"
            "<p class='muted'>One metric per row over every completed bag; hover a row name for its definition. "
            "Gating metrics decide a run's verdict; the rest are reported.</p>"
            f"<div class='tablewrap'><table class='matrix'><tr><th>Metric</th>{head_c}</tr>{body_c}</table></div>"
            "</section>"
        )

    # One-glance verdict matrix: bag rows, model columns. With several models
    # the cards answer "what happened in this run"; this table answers the
    # question the tool exists for — "which model handled which scene".
    matrix_html = ""
    if len(models) > 1:
        head = "".join(f"<th>{_model_label(m)}</th>" for m in models)
        body = []
        for bag in bags:
            tds = []
            for m in models:
                job = by_key.get((bag, m))
                if not job:
                    tds.append("<td class='na'>—</td>")
                    continue
                st = str(job.get("status") or "pending")
                if st == "done":
                    met = load_metrics(job)
                    if met is None:
                        tds.append("<td class='processing'>scoring</td>")
                    else:
                        ok = bool(met.get("pass"))
                        si = met.get("stack_intervention", {}) or {}
                        bits = []
                        if isinstance(si.get("stop_point_ratio"), (int, float)):
                            bits.append(f"stub {si['stop_point_ratio'] * 100:.0f}%")
                        if _pace(met) is not None:
                            bits.append(f"{_pace(met):.2f}x")
                        ego_c = int(
                            (met.get("collision_rate", {}).get("by_attribution") or {}).get(
                                "ego_caused", 0
                            )
                        )
                        if ego_c:
                            bits.append(f"{ego_c} contact")
                        sub = html.escape(" · ".join(bits))
                        cls = "pass" if ok else "fail"
                        win_m = met.get("windows", {}) or {}
                        if win_m.get("status") == "ok" and win_m.get("legs_total"):
                            lp, lt = int(win_m.get("legs_passed") or 0), int(win_m["legs_total"])
                            label = f"{lp}/{lt} legs"
                            if not ok and lt >= 2 and lt - lp == 1:
                                cls = "partial"
                        else:
                            label = "PASS" if ok else "FAIL"
                        tds.append(
                            f"<td class='{cls}'><b>{label}</b>"
                            f"<div class='nsub'>{sub}</div></td>"
                        )
                elif st == "failed":
                    # A failed job usually still drove and recorded a bag; "NO
                    # RUN" told a viewer the opposite of what happened. Show the
                    # drive's own numbers when metrics exist.
                    met = load_metrics(job)
                    if met:
                        si = met.get("stack_intervention", {}) or {}
                        bits = []
                        if isinstance(si.get("stop_point_ratio"), (int, float)):
                            bits.append(f"stub {si['stop_point_ratio'] * 100:.0f}%")
                        if _pace(met) is not None:
                            bits.append(f"{_pace(met):.2f}x")
                        err = str(job.get("error") or "")
                        why = "did not arrive" if "not_arrived" in err else "run ended early"
                        sub = html.escape(" · ".join(bits + [why]))
                        tds.append(
                            f"<td class='fail'><b>INCOMPLETE</b>"
                            f"<div class='nsub'>{sub}</div></td>"
                        )
                    else:
                        tds.append(
                            "<td class='fail'><b>NO RUN</b>"
                            "<div class='nsub'>failed before driving</div></td>"
                        )
                elif st == "skipped":
                    tds.append("<td class='na'>skipped</td>")
                else:
                    tds.append(f"<td class='pending'>{html.escape(st)}</td>")
            body.append(f"<tr><th class='bagname'>{html.escape(bag)}</th>{''.join(tds)}</tr>")
        matrix_html = (
            "<section><h2>Model comparison</h2>"
            "<div class='tablewrap'><table class='matrix'>"
            f"<tr><th>Bags</th>{head}</tr>{''.join(body)}"
            "</table></div></section>"
        )

    precision_table = (
        "<table class='summary'>"
        "<tr>"
        "<th>ID</th><th>Scenario</th><th>Model</th><th>Result</th><th>Status</th>"
        "<th>pos [m]</th><th>lat [m]</th><th>lon [m]</th>"
        "<th>|lat| [m]</th><th>|lon| [m]</th><th>heading [deg]</th><th>n</th>"
        "</tr>"
        + (
            "".join(precision_rows)
            if precision_rows
            else '<tr><td colspan="12">No completed metrics yet.</td></tr>'
        )
        + "</table>"
    )
    scatter_html = render_stop_scatter_section(
        stop_points,
        lat_tol_m=lat_tol,
        lon_tol_m=lon_tol,
        precision_table_html=precision_table,
    )

    header = "".join(f"<th>{_model_label(m)}</th>" for m in models)

    # Signature changes when any job status (or done count) changes — used to reload
    # the page only then, not on a fixed timer (so open PASS/FAIL details stay open).
    # A dispatched job reads QUEUED until its worker writes its status file; the
    # signature carries that, so an open page reloads when the job has really
    # started instead of saying QUEUED for the whole drive (2026-09-16).
    progress_sig = "|".join(
        f"{j.get('job_id')}:{j.get('status')}:{j.get('attempts')}:"
        f"{bool(j.get('status') == 'running' and (load_job_status(j) or {}).get('phase'))}"
        for j in jobs
    )
    progress_sig = f"{counts['done']}-{counts['running']}-{counts['failed']}-{counts['pending']}:{hash(progress_sig) & 0xFFFFFFFF:08x}"

    if in_progress:
        refresh_block = (
            '<p class="live">Live — page reloads only when a job finishes or status changes '
            "(open PASS/FAIL details stay open).</p>"
            '<p class="live">To stop the run: <strong>Ctrl-C</strong> in the run_all.py terminal '
            "(or <code>kill -INT &lt;pid&gt;</code>). Running jobs are stopped and left pending; "
            "rerunning the same command resumes them.</p>"
        )
        poll_js = f"""
var PROGRESS_SIG = {json.dumps(progress_sig)};
var POLL_MS = 5000;
var OPEN_KEY = "dp_multi_eval_open_details";

function saveOpenDetails() {{
  var openIds = [];
  document.querySelectorAll(".detail.open").forEach(function(el) {{
    if (el.id) openIds.push(el.id);
  }});
  try {{ sessionStorage.setItem(OPEN_KEY, JSON.stringify(openIds)); }} catch (e) {{}}
}}

function restoreOpenDetails() {{
  try {{
    var raw = sessionStorage.getItem(OPEN_KEY);
    if (!raw) return;
    JSON.parse(raw).forEach(function(id) {{
      var el = document.getElementById(id);
      if (el) el.classList.add("open");
    }});
  }} catch (e) {{}}
}}

function toggle(id) {{
  var el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle("open");
  if (el.classList.contains("open")) mountVideos(el);
  saveOpenDetails();
}}

// Players are built when a cell is first opened. A grid can hold a hundred runs,
// and a hundred media elements sitting in the DOM cost memory even with
// preload="none" and nothing playing.
function mountVideos(root) {{
  var slots = root.querySelectorAll(".vid[data-src]");
  for (var i = 0; i < slots.length; i++) {{
    var slot = slots[i];
    var src = slot.getAttribute("data-src");
    slot.removeAttribute("data-src");
    var v = document.createElement("video");
    v.controls = true;
    v.preload = "none";
    v.playsInline = true;
    v.src = src;
    slot.insertBefore(v, slot.firstChild);
  }}
}}

function pollProgress() {{
  fetch("dashboard_live.json", {{ cache: "no-store" }})
    .then(function(r) {{ return r.ok ? r.json() : null; }})
    .then(function(data) {{
      if (!data) return;
      if (data.in_progress === false) {{
        if (data.sig !== PROGRESS_SIG) {{
          saveOpenDetails();
          location.reload();
        }}
        return;
      }}
      if (data.sig && data.sig !== PROGRESS_SIG) {{
        saveOpenDetails();
        location.reload();
      }}
    }})
    .catch(function() {{}});
}}

document.addEventListener("DOMContentLoaded", function() {{
  restoreOpenDetails();
  setInterval(pollProgress, POLL_MS);
  startLiveDrive();
}});
"""
    else:
        refresh_block = (
            '<p class="done">All jobs finished. Open details stay open; use browser refresh '
            "only if you start a new run.</p>"
        )
        poll_js = """
var OPEN_KEY = "dp_multi_eval_open_details";

function saveOpenDetails() {
  var openIds = [];
  document.querySelectorAll(".detail.open").forEach(function(el) {
    if (el.id) openIds.push(el.id);
  });
  try { sessionStorage.setItem(OPEN_KEY, JSON.stringify(openIds)); } catch (e) {}
}

function restoreOpenDetails() {
  try {
    var raw = sessionStorage.getItem(OPEN_KEY);
    if (!raw) return;
    JSON.parse(raw).forEach(function(id) {
      var el = document.getElementById(id);
      if (el) el.classList.add("open");
    });
  } catch (e) {}
}

function toggle(id) {
  var el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle("open");
  if (el.classList.contains("open")) mountVideos(el);
  saveOpenDetails();
}

// Players are built when a cell is first opened. A grid can hold a hundred runs,
// and a hundred media elements sitting in the DOM cost memory even with
// preload="none" and nothing playing.
function mountVideos(root) {
  var slots = root.querySelectorAll(".vid[data-src]");
  for (var i = 0; i < slots.length; i++) {
    var slot = slots[i];
    var src = slot.getAttribute("data-src");
    slot.removeAttribute("data-src");
    var v = document.createElement("video");
    v.controls = true;
    v.preload = "none";
    v.playsInline = true;
    v.src = src;
    slot.insertBefore(v, slot.firstChild);
  }
}

document.addEventListener("DOMContentLoaded", function() {
  restoreOpenDetails();
  var open = document.querySelectorAll(".detail.open");
  for (var i = 0; i < open.length; i++) mountVideos(open[i]);
  startLiveDrive();
});
"""

    live_drive_js = r"""
function startLiveDrive() {
  var badge = document.getElementById("live-drive-badge");
  var meta = document.getElementById("live-drive-meta");
  var trailEl = document.getElementById("live-drive-trail");
  var goalsEl = document.getElementById("live-drive-goals");
  var egoEl = document.getElementById("live-drive-ego");
  var gridEl = document.getElementById("live-drive-grid");
  if (!badge || !meta || !trailEl) return;
  var W = 640, H = 420, PAD = 36;

  function project(points) {
    var xs = points.map(function(p) { return p.x; });
    var ys = points.map(function(p) { return p.y; });
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
    var dx = Math.max(20, maxX - minX), dy = Math.max(20, maxY - minY);
    var scale = Math.min((W - 2 * PAD) / dx, (H - 2 * PAD) / dy);
    return function(x, y) {
      return [
        PAD + (x - minX) * scale,
        H - PAD - (y - minY) * scale
      ];
    };
  }

  function render(data) {
    // The panel exists only while a job is driving; there is no idle state.
    var section = document.getElementById("live-drive");
    if (!data || !data.active) {
      section.hidden = true;
      return;
    }
    section.hidden = false;
    badge.textContent = "LIVE";
    badge.className = "pill-run pulse";
    var speed = (data.speed_mps == null) ? "?" : Number(data.speed_mps).toFixed(2);
    var xy = (data.x == null) ? "x=?, y=?" :
      ("x=" + Number(data.x).toFixed(1) + ", y=" + Number(data.y).toFixed(1));
    var leg = "";
    if (data.leg_idx != null) {
      leg = " · leg " + data.leg_idx + "/" + (data.leg_total || "?");
    }
    meta.textContent =
      (data.bag_key || data.job_id || "?") + " · " + (data.phase || "") + leg +
      " · " + xy + " · v=" + speed + " m/s · " + (data.updated_iso || "") +
      (data.note ? (" · " + data.note) : "");

    var pts = (data.trail || []).slice();
    (data.goals || []).forEach(function(g) {
      if (g && g.x != null && g.y != null) pts.push({x: g.x, y: g.y});
    });
    if (data.x != null && data.y != null) pts.push({x: data.x, y: data.y});
    if (pts.length < 1) {
      trailEl.setAttribute("points", "");
      goalsEl.innerHTML = "";
      egoEl.innerHTML = "";
      return;
    }
    var toXY = project(pts);
    var trail = (data.trail || []).map(function(p) {
      var xy2 = toXY(p.x, p.y); return xy2[0].toFixed(1) + "," + xy2[1].toFixed(1);
    }).join(" ");
    trailEl.setAttribute("points", trail);

    var ghtml = "";
    (data.goals || []).forEach(function(g, i) {
      if (!g || g.x == null) return;
      var p = toXY(g.x, g.y);
      ghtml += '<circle class="goal-dot" cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="4"/>';
      ghtml += '<text class="goal-label" x="' + (p[0] + 6).toFixed(1) + '" y="' + (p[1] - 6).toFixed(1) + '">' +
        (g.label || ("G" + (i + 1))) + "</text>";
    });
    goalsEl.innerHTML = ghtml;

    if (data.x != null && data.y != null) {
      var ep = toXY(data.x, data.y);
      var yaw = (data.yaw == null) ? 0 : data.yaw;
      // map yaw (CCW from +X) to SVG rotate (CW from +X), and Y is flipped in project.
      var deg = -yaw * 180 / Math.PI;
      egoEl.innerHTML =
        '<g transform="translate(' + ep[0].toFixed(1) + " " + ep[1].toFixed(1) +
        ") rotate(" + deg.toFixed(1) + ')">' +
        '<polygon class="ego-arrow" points="14,0 -8,-7 -8,7"/>' +
        "</g>";
    } else {
      egoEl.innerHTML = "";
    }

    // light grid
    var grid = "";
    for (var i = 1; i < 4; i++) {
      var x = (W * i / 4); var y = (H * i / 4);
      grid += '<line class="grid-line" x1="' + x + '" y1="0" x2="' + x + '" y2="' + H + '"/>';
      grid += '<line class="grid-line" x1="0" y1="' + y + '" x2="' + W + '" y2="' + y + '"/>';
    }
    if (gridEl) gridEl.innerHTML = grid;
  }

  function pollLive() {
    fetch("live_drive.json", { cache: "no-store" })
      .then(function(r) { return r.ok ? r.json() : null; })
      .then(render)
      .catch(function() { render(null); });
  }
  pollLive();
  setInterval(pollLive, 2000);
}
"""

    # Always include live-drive poller (cheap; only reads a small JSON file).
    poll_js = live_drive_js + "\n" + poll_js

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>DP Multi-Eval Dashboard</title>
<meta name="dp-progress-sig" content="{html.escape(progress_sig)}"/>
<script>
// ?host=streamlit&theme=light|dark: the GUI embeds the page and passes its own theme,
// which can differ from the OS setting that prefers-color-scheme follows.
(function () {{
  var q = new URLSearchParams(location.search), root = document.documentElement;
  var theme = q.get("theme");
  if (q.get("host") === "streamlit") {{
    root.dataset.host = "streamlit";
    if (theme !== "light" && theme !== "dark")
      theme = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }}
  if (theme === "light" || theme === "dark") root.dataset.theme = theme;
  // Embedded: report the content height so the GUI grows the frame to fit it
  // and the page scrolls as one, with no scrollbar of its own.
  if (root.dataset.host === "streamlit" && window.parent !== window) {{
    var last = 0;
    var send = function () {{
      if (!document.body) return;
      var h = Math.ceil(document.body.getBoundingClientRect().height);
      if (h > 0 && h !== last) {{
        last = h;
        window.parent.postMessage({{type: "dp-dashboard:height", height: h}}, "*");
      }}
    }};
    addEventListener("DOMContentLoaded", function () {{
      send();
      if (typeof ResizeObserver !== "undefined") new ResizeObserver(send).observe(document.body);
    }});
    addEventListener("load", send);
  }}
}})();
</script>
<style>
:root {{
  --bg:#f6f7f9; --panel:#ffffff; --panel-2:#f0f2f5; --line:#dfe3e8; --line-2:#eceff3;
  --ink:#151a21; --ink-2:#48525f; --ink-3:#6f7b8a;
  --brand:#2f6f5e; --brand-soft:#e2efea;
  --good:#1f7a4d; --good-soft:#e3f3ea;
  --warn:#9a5b08; --warn-soft:#faeddb;
 --partial:#e07b00; --partial-soft:#fff1e0;
  --bad:#a3312a;  --bad-soft:#f9e5e3;
  --busy:#2b5f9e; --busy-soft:#e5eefa;
  --radius:10px;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#0e1216; --panel:#161c22; --panel-2:#1c242c; --line:#262f38; --line-2:#1e262e;
  --ink:#e7ecf1; --ink-2:#aab6c2; --ink-3:#7d8994;
  --brand:#5cc0a4; --brand-soft:#16302a;
  --good:#59c98d; --good-soft:#142c20;
  --warn:#d9a05b; --warn-soft:#2e2415;
  --partial:#f0932b; --partial-soft:#33210d;
  --bad:#e58b80;  --bad-soft:#31201d;
  --busy:#79aee8; --busy-soft:#152233;
}} }}
:root[data-theme="dark"] {{
  --bg:#0e1216; --panel:#161c22; --panel-2:#1c242c; --line:#262f38; --line-2:#1e262e;
  --ink:#e7ecf1; --ink-2:#aab6c2; --ink-3:#7d8994;
  --brand:#5cc0a4; --brand-soft:#16302a;
  --good:#59c98d; --good-soft:#142c20;
  --warn:#d9a05b; --warn-soft:#2e2415;
  --partial:#f0932b; --partial-soft:#33210d;
  --bad:#e58b80;  --bad-soft:#31201d;
  --busy:#79aee8; --busy-soft:#152233;
}}
/* Inside the GUI: Streamlit's default surfaces, text and font; status colours stay. */
:root[data-host="streamlit"] {{
  --bg:#ffffff; --panel:#ffffff; --panel-2:#f0f2f6; --line:#d5dae5; --line-2:#f0f2f6;
  --ink:#31333f; --ink-2:#555867; --ink-3:#808495; --brand-soft:#f0f2f6; --radius:8px;
}}
:root[data-host="streamlit"][data-theme="dark"] {{
  --bg:#0e1117; --panel:#0e1117; --panel-2:#262730; --line:#3d4044; --line-2:#262730;
  --ink:#fafafa; --ink-2:#d5dae5; --ink-3:#a3a8b8; --brand-soft:#262730;
}}
:root[data-host="streamlit"] body {{
  font-family:"Source Sans","Source Sans Pro",ui-sans-serif,system-ui,sans-serif;
  padding:0 0 8px;
}}
* {{ box-sizing: border-box; }}
body {{
  margin:0; padding:28px clamp(14px,4vw,40px) 72px; background:var(--bg); color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
}}
.wrap {{ max-width:1400px; margin:0 auto; display:flex; flex-direction:column; gap:28px; }}
h1 {{ font-size:1.65rem; font-weight:650; letter-spacing:-.02em; margin:0; }}
h2 {{ font-size:1.05rem; font-weight:650; letter-spacing:-.01em; margin:0 0 12px; }}
h3 {{ font-size:.95rem; font-weight:650; margin:0; letter-spacing:-.01em; }}
p {{ margin:0; }}
code {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:.85em; }}
.muted {{ color:var(--ink-3); font-size:.88rem; }}
.path code {{ color:var(--ink-3); font-size:.78rem; word-break:break-all; }}

header.top {{ display:flex; flex-direction:column; gap:6px; padding-bottom:18px; border-bottom:1px solid var(--line); }}
header.top .meta {{ color:var(--ink-3); font-size:.85rem; display:flex; gap:16px; flex-wrap:wrap; }}

/* KPI tiles */
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:14px 16px; }}
.tile .n {{ font-size:1.7rem; font-weight:650; font-variant-numeric:tabular-nums; line-height:1.1; }}
.tile .l {{ font-size:.76rem; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-3); margin-top:4px; }}
.tile.good .n {{ color:var(--good); }} .tile.bad .n {{ color:var(--bad); }}
.tile.busy .n {{ color:var(--busy); }} .tile.warn .n {{ color:var(--warn); }}
.bar {{ height:6px; background:var(--line-2); border-radius:99px; overflow:hidden; }}
.bar i {{ display:block; height:100%; background:var(--good); }}

/* job cards */
.cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:14px; align-items:start; }}
.bags {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(480px,1fr)); gap:14px; align-items:start; }}
.bag-block {{ background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:12px 14px; display:flex; flex-direction:column; gap:6px; }}
.bag-title {{ margin:0 0 4px 0; word-break:break-word; }}
details.model-run {{ border-left:4px solid var(--line); border-radius:6px; background:var(--panel-2, transparent); }}
details.model-run.pass {{ border-left-color:var(--good); }}
details.model-run.fail {{ border-left-color:var(--bad); }}
details.model-run.partial {{ border-left-color:var(--partial); }}
details.model-run.assist {{ border-left-color:var(--warn); }}
details.model-run > summary {{ cursor:pointer; list-style:none; display:flex; justify-content:space-between; align-items:center; gap:10px; padding:8px 10px; }}
details.model-run > summary::-webkit-details-marker {{ display:none; }}
details.model-run > summary .model-name {{ font-weight:600; word-break:break-word; }}
details.model-run[open] > summary {{ border-bottom:1px solid var(--line); }}
details.model-run > .card {{ border:none; border-left:none; border-radius:0; }}
.card {{
  background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--line);
  border-radius:var(--radius); padding:14px 16px; display:flex; flex-direction:column; gap:10px;
}}
.card.pass {{ border-left-color:var(--good); }}
.card.fail {{ border-left-color:var(--bad); }}
.card.assist {{ border-left-color:var(--warn); }}
.card.partial {{ border-left-color:var(--partial); }}
.card.running {{ border-left-color:var(--busy); }}
.card.pending, .card.na, .card.processing {{ border-left-color:var(--line); }}
.card header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }}
.card .title {{ display:flex; flex-direction:column; gap:2px; min-width:0; }}
.card h3 {{ word-break:break-word; }}
.model {{ font-size:.76rem; color:var(--ink-3); }}
.pill {{
  font-size:.7rem; font-weight:650; letter-spacing:.05em; text-transform:uppercase;
  padding:3px 9px; border-radius:99px; white-space:nowrap; flex:none;
}}
.pill-pass {{ background:var(--good-soft); color:var(--good); }}
.mname {{ margin-right:6px; }}
.pill-fail {{ background:var(--bad-soft); color:var(--bad); }}
.pill-assist {{ background:var(--warn-soft); color:var(--warn); }}
.pill-partial {{ background:var(--partial-soft); color:var(--partial); }}
.pill-running {{ background:var(--busy-soft); color:var(--busy); }}
.pill-pending, .pill-na, .pill-processing {{ background:var(--panel-2); color:var(--ink-3); }}

/* metric chips */
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{ display:inline-flex; align-items:baseline; gap:6px; padding:3px 9px; border-radius:99px;
        font-size:.76rem; background:var(--panel-2); border:1px solid var(--line-2); }}
.chip .k {{ color:var(--ink-3); }}
.chip .v {{ font-variant-numeric:tabular-nums; font-weight:600; }}
.chip-good .v {{ color:var(--good); }}
.chip-bad {{ background:var(--bad-soft); border-color:transparent; }} .chip-bad .v {{ color:var(--bad); }}
.chip-warn {{ background:var(--warn-soft); border-color:transparent; }} .chip-warn .v {{ color:var(--warn); }}

/* diagnosis */
.diag {{ background:var(--brand-soft); border-radius:8px; padding:10px 12px; }}
.diag b {{ display:block; margin-bottom:4px; font-size:.9rem; }}
.diag-why, .diag-look {{ font-size:.82rem; color:var(--ink-2); margin-top:4px; }}
.raw-err {{ font-family:ui-monospace,monospace; font-size:.76rem; color:var(--ink-3); word-break:break-word; }}
.goal-face {{ font-size:.8rem; color:var(--ink-2); font-variant-numeric:tabular-nums; line-height:1.5; }}
.goal-face.muted {{ color:var(--ink-3); font-style:italic; }}
.ref-ok, .ref-flag {{ font-size:.78rem; font-variant-numeric:tabular-nums; }}
.ref-ok {{ color:var(--ink-3); }} .ref-flag {{ color:var(--warn); font-weight:600; }}
.assisted {{ font-size:.8rem; color:var(--warn); }}
.has-vid {{ font-size:.78rem; color:var(--busy); }}

/* video */
.vid video {{ width:100%; border-radius:8px; border:1px solid var(--line); background:#000; display:block; }}
.vid-path {{ font-size:.75rem; color:var(--ink-3); margin-top:5px; }}
.vid-path a {{ color:var(--busy); }}

/* progressive disclosure */
details.more, details.defs {{ border-top:1px solid var(--line-2); padding-top:8px; }}
details.more > summary, details.defs > summary {{
  cursor:pointer; font-size:.8rem; color:var(--ink-3); list-style:none; user-select:none;
}}
details.more > summary::before, details.defs > summary::before {{ content:"▸ "; }}
details[open].more > summary::before, details[open].defs > summary::before {{ content:"▾ "; }}
.detail {{ display:block; margin-top:8px; font-size:.82rem; color:var(--ink-2); }}
.metric-block {{ margin-top:8px; }}
.hint {{ color:var(--ink-3); font-weight:400; font-size:.8rem; }}

/* tables kept for summary / precision */
.tablewrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:var(--radius); background:var(--panel); }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem; }}
th {{ text-align:left; font-size:.72rem; text-transform:uppercase; letter-spacing:.06em;
     color:var(--ink-3); font-weight:600; padding:10px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }}
td {{ padding:9px 12px; border-bottom:1px solid var(--line-2); color:var(--ink-2); font-variant-numeric:tabular-nums; }}
tr:last-child td {{ border-bottom:none; }}
table.mini {{ width:100%; }} table.mini th {{ width:42%; text-transform:none; letter-spacing:0; font-size:.78rem; }}
table.mini th, table.mini td {{ padding:4px 8px; border-bottom:1px solid var(--line-2); }}
.nsub {{ font-size:.75rem; color:var(--ink-3); }} .nsub.partial {{ color:var(--bad); }}

/* live drive */
.live-drive {{ background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:16px; }}
.live-meta {{ color:var(--ink-3); font-size:.82rem; font-family:ui-monospace,monospace; margin:6px 0 10px; white-space:pre-wrap; }}
svg.live-drive-svg {{ background:var(--panel-2); border-radius:8px; max-width:100%; height:auto; display:block; }}
svg.live-drive-svg .plot-bg {{ fill:var(--panel-2); }}
svg.live-drive-svg .grid-line {{ stroke:var(--line); stroke-width:1; }}
svg.live-drive-svg .goal-dot {{ fill:var(--warn); stroke:var(--panel); stroke-width:1; }}
svg.live-drive-svg .goal-label {{ fill:var(--ink-2); font-size:11px; }}
svg.live-drive-svg .ego-arrow {{ fill:var(--good); stroke:var(--panel); stroke-width:1.2; }}
.pulse {{ animation:pulse 1.6s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}

/* scatter (kept, retinted) */
svg.stop-scatter {{ background:var(--panel-2); border-radius:8px; max-width:100%; height:auto; }}
svg.stop-scatter .plot-bg {{ fill:var(--panel-2); }}
svg.stop-scatter .grid {{ stroke:var(--line); }} svg.stop-scatter .axis {{ stroke:var(--ink-3); }}
svg.stop-scatter .tol-box {{ fill:color-mix(in srgb,var(--busy) 10%,transparent); stroke:var(--busy); stroke-dasharray:6 4; }}
svg.stop-scatter .pt-pass {{ fill:var(--good); }} svg.stop-scatter .pt-fail {{ fill:var(--bad); }}
svg.stop-scatter .pt-id {{ fill:var(--ink); font-size:11px; font-weight:600; font-family:ui-monospace,monospace; }}
svg.stop-scatter .avg-ring {{ fill:none; stroke:var(--warn); stroke-width:2.5; }}
svg.stop-scatter .avg-arrow, svg.stop-scatter .goal-arrow {{ fill:var(--warn); }}
svg.stop-scatter .avg-label, svg.stop-scatter .goal-label {{ fill:var(--ink-2); font-size:11px; font-weight:600; }}
svg.stop-scatter .triad-x {{ stroke:var(--bad); fill:var(--bad); }}
svg.stop-scatter .triad-y {{ stroke:var(--good); fill:var(--good); }}
svg.stop-scatter .triad-origin {{ fill:var(--ink); }}
svg.stop-scatter .triad-label, svg.stop-scatter .axis-title, svg.stop-scatter .axis-title-y {{ fill:var(--ink-3); font-size:11px; }}
svg.stop-scatter .tick {{ fill:var(--ink-3); font-size:10px; }}
.scatter-grid {{ display:flex; flex-direction:column; gap:14px; }}
.goal-stop-panel {{ background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:14px; }}
.scatter-row {{ display:flex; flex-wrap:wrap; gap:16px; align-items:flex-start; }}
.scatter-legend {{ min-width:280px; flex:1; }}
.scatter-legend .swatch {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
.scatter-legend .swatch.pt-pass {{ background:var(--good); }}
.scatter-legend .swatch.pt-fail {{ background:var(--bad); }}
.scatter-legend .swatch.avg {{ background:var(--warn); }}
.scatter-legend .avg-row td {{ color:var(--warn); }}

table.matrix {{ border-collapse: collapse; width: 100%; }}
table.matrix th, table.matrix td {{ border: 1px solid var(--line, #ddd); padding: 8px 12px; text-align: center; }}
table.matrix th.bagname {{ text-align: left; font-weight: 600; }}
table.matrix td.pass {{ background: color-mix(in srgb, #2e9e44 14%, transparent); }}
table.matrix td.fail {{ background: color-mix(in srgb, #d64545 14%, transparent); }}
table.matrix td.partial {{ background: color-mix(in srgb, #e07b00 18%, transparent); }}
table.matrix td.na, table.matrix td.pending {{ color: var(--muted, #888); }}
table.matrix .nsub {{ font-size: 11px; opacity: .75; margin-top: 2px; }}
</style>
<script>
{poll_js}
</script>
</head>
<body>
<div class="wrap">

<header class="top">
  <h1>Diffusion Planner evaluation</h1>
  <div class="meta">
    <span>results <code>{html.escape(str(manifest.get('results_root','')))}</code></span>
    <span>updated {generated_at}</span>
  </div>
  {refresh_block}
</header>

<section>
  <div class="tiles">
    <div class="tile good"><div class="n">{counts['done']}</div><div class="l">completed</div></div>
    <div class="tile bad"><div class="n">{counts['failed']}</div><div class="l">failed</div></div>
    <div class="tile busy"><div class="n">{counts['running']}</div><div class="l">in flight</div></div>
    <div class="tile"><div class="n">{counts['pending']}</div><div class="l">pending</div></div>
    <div class="tile"><div class="n">{total}</div><div class="l">total jobs</div></div>
  </div>
  <div class="bar" style="margin-top:12px"><i style="width:{done_frac * 100:.1f}%"></i></div>
</section>

{comparison_html}
{matrix_html}

<section>
  <h2>Runs</h2>
  <p class="muted">One block per bag; click a model's line to open that run's results.</p>
  <div class="bags">
{''.join(rows_html)}
  </div>
</section>

<section class="live-drive" id="live-drive" hidden>
  <h2>Live drive <span id="live-drive-badge" class="pill pill-run pulse">LIVE</span></h2>
  <p class="muted">Ego trail sampled about every 2 s, straight from <code>live_drive.json</code> — no RViz, no rosbridge.</p>
  <div id="live-drive-meta" class="live-meta"></div>
  <svg id="live-drive-svg" class="live-drive-svg" viewBox="0 0 640 420" width="640" height="420">
    <rect class="plot-bg" x="0" y="0" width="640" height="420"/>
    <g id="live-drive-grid"></g>
    <polyline id="live-drive-trail" fill="none" stroke="#2b5f9e" stroke-width="2.5"/>
    <g id="live-drive-goals"></g>
    <g id="live-drive-ego"></g>
  </svg>
</section>

<section>
  <details class="defs">
    <summary>Per-model summary (older columns)</summary>
    <div class="tablewrap" style="margin-top:10px"><table>
    <tr>
      <th>Model</th><th>Clean pass</th><th>Pass incl. assisted</th><th>Completed</th>
      <th>Goal pos / |lat| / |lon| [m]</th><th>Collision</th><th>Off-lane</th><th>Stuck</th>
      <th>Plan replaced</th><th>Green latched</th><th>Pace vs rec.</th><th>Ego-caused contacts</th>
      <th>Legs passed</th><th>Harsh brakes</th>
    </tr>
    {''.join(summary_rows)}
    </table></div>
  </details>
</section>

{scatter_html}

<section>
  <details class="defs">
    <summary>What each metric measures</summary>
    <div class="tablewrap" style="margin-top:10px"><table>
      <tr><th>Metric</th><th>Definition</th></tr>
      <tr><td>collision_rate</td><td>Share of sampled frames where the ego footprint comes within
        <code>collision_distance_m</code> of a vehicle-class NPC.</td></tr>
      <tr><td>stuck_rate</td><td>Share of scenario time the ego is nearly stopped away from its goal,
        in stretches of at least <code>stuck_duration_sec</code>.</td></tr>
      <tr><td>success_rate</td><td>The run's verdict (all gating metrics clear) and legs passed / total; the summary table aggregates per model.</td></tr>
      <tr><td>comfort</td><td>Harsh brakes (deceleration beyond 2.0 m/s² for 0.3 s) and harsh jerk (|jerk| beyond 2.5 m/s³) from the smoothed speed, hand-overs excluded, recording scored the same way. Flags when the run brakes harshly more often than the recording plus one. Gating.</td></tr>
      <tr><td>proximity</td><td>Closest a replayed vehicle came to the ego's footprint while moving (withheld objects excluded) and time within 1 m. Reported.</td></tr>
      <tr><td>goal-stop per leg</td><td>Lateral / longitudinal error of the ego's rest position nearest each leg's arrival pose; excluded_stuck when the leg stalled or never got there.</td></tr>
      <tr><td>out_of_boundary</td><td>Share of sampled frames where the footprint crosses an uncrossable
        Lanelet2 border (road_border, curbstone) or comes within <code>oob_margin_m</code> of one.</td></tr>
      <tr><td>goal_stop_precision</td><td>Mean stop error in the goal frame — position, lateral,
        longitudinal, heading — over low-speed samples near the goal.</td></tr>
      <tr><td>reference_speed</td><td>Ego speed against the source recording at the same place.
        Reported, never gating: a replay legitimately differs from what it replays.</td></tr>
    </table></div>
  </details>
</section>

</div>
</body>
</html>
"""
    output_html = output_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page, encoding="utf-8")

    live = {
        "sig": progress_sig,
        "in_progress": in_progress,
        "done": counts["done"],
        "running": counts["running"],
        "pending": counts["pending"],
        "failed": counts["failed"],
        "updated_at": generated_at,
    }
    (output_html.parent / "dashboard_live.json").write_text(
        json.dumps(live, indent=2) + "\n", encoding="utf-8"
    )
    return output_html


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 5: build HTML dashboard")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args(argv)
    manifest = json.loads(args.manifest.expanduser().read_text(encoding="utf-8"))
    out = args.output or Path(manifest["results_root"]) / "dashboard.html"
    path = build_dashboard(manifest, out)
    print(f"[done] dashboard → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
