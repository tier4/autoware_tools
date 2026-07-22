"""Phase 5 — single-file HTML dashboard from manifest + metrics."""

from __future__ import annotations

import argparse
import html
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
                "model": str(job.get("model_name") or ""),
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
    models = sorted({j["model_name"] for j in jobs})
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
        by_key[(j["bag_key"], j["model_name"])] = j

    model_stats: dict[str, dict[str, float]] = {
        m: {
            "n": 0,
            "pass": 0,
            "collision_rate_sum": 0.0,
            "collision_rate_n": 0,
            "oob_rate_sum": 0.0,
            "oob_rate_n": 0,
            "stuck_rate_sum": 0.0,
            "stuck_rate_n": 0,
            "goal_pos_sum": 0.0,
            "goal_lat_sum": 0.0,
            "goal_lon_sum": 0.0,
            "goal_n": 0,
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
        cells = [f"<td class='bag'>{html.escape(bag)}</td>"]
        for model in models:
            job = by_key.get((bag, model))
            if not job:
                cells.append("<td class='na'>—</td>")
                continue

            status = str(job.get("status") or "pending")
            detail_id = html.escape(job["job_id"])

            if status == "pending":
                tip = html.escape(_pending_label(job))
                cells.append(f"<td class='pending' title='{tip}'>{tip}</td>")
                continue

            if status == "running":
                label = html.escape(_running_detail(job))
                js = load_job_status(job)
                phase = html.escape(str((js or {}).get("phase") or "simulation"))
                tip = html.escape(str(job.get("error") or label))
                detail = (
                    f"<div class='detail' id='d-{detail_id}'>"
                    f"<div>phase={phase}</div>"
                    f"<div>output: <code>{html.escape(job['output_dir'])}</code></div>"
                    f"</div>"
                )
                cells.append(
                    f"<td class='running' onclick=\"toggle('d-{detail_id}')\">"
                    f"<span class='pulse'>{label}</span>{detail}</td>"
                )
                continue

            if status == "dry_run":
                tip = html.escape(str(job.get("error") or "dry run"))
                cells.append(f"<td class='pending' title='{tip}'>DRY-RUN</td>")
                continue

            if status == "failed":
                tip = html.escape(str(job.get("error") or "failed"))
                cells.append(f"<td class='fail' title='{tip}'>FAILED</td>")
                continue

            metrics = load_metrics(job)
            model_stats[model]["n"] += 1
            if metrics is None:
                cells.append("<td class='processing'>DONE (metrics pending)</td>")
                continue

            passed = bool(metrics.get("pass"))
            if passed:
                model_stats[model]["pass"] += 1

            goal = metrics.get("goal_stop_precision", {}) or {}
            stuck = metrics.get("stuck_rate", {}) or {}
            col = metrics.get("collision_rate", {}) or {}
            oob = metrics.get("out_of_boundary", {}) or {}

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
            video_link = ""
            if video.is_file():
                video_link = f"<div>video: <code>{html.escape(str(video))}</code></div>"

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
                f"{video_link}"
                f"</div>"
            )
            cls = "pass" if passed else "fail"
            label = "PASS" if passed else "FAIL"
            cells.append(
                f"<td class='{cls}' onclick=\"toggle('d-{detail_id}')\">"
                f"<div class='verdict'>{label}</div>{goal_face}{detail}</td>"
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
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    summary_rows = []
    for model in models:
        s = model_stats[model]
        pass_rate = (s["pass"] / s["n"] * 100.0) if s["n"] else 0.0
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
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{int(s['pass'])}/{int(s['n'])} ({pass_rate:.0f}%)</td>"
            f"<td>{_fmt_m(avg_pos)} / {_fmt_m(avg_lat)} / {_fmt_m(avg_lon)}</td>"
            f"<td>{_fmt_rate(avg_col)}</td>"
            f"<td>{_fmt_rate(avg_oob)}</td>"
            f"<td>{_fmt_rate(avg_stuck)}</td>"
            "</tr>"
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

    header = "".join(f"<th>{html.escape(m)}</th>" for m in models)

    # Signature changes when any job status (or done count) changes — used to reload
    # the page only then, not on a fixed timer (so open PASS/FAIL details stay open).
    progress_sig = "|".join(
        f"{j.get('job_id')}:{j.get('status')}:{j.get('attempts')}" for j in jobs
    )
    progress_sig = f"{counts['done']}-{counts['running']}-{counts['failed']}-{counts['pending']}:{hash(progress_sig) & 0xFFFFFFFF:08x}"

    if in_progress:
        refresh_block = (
            '<p class="live">Live — page reloads only when a job finishes or status changes '
            "(open PASS/FAIL details stay open).</p>"
            '<p class="live">To stop the run: open the Streamlit GUI → <strong>Live Progress</strong> '
            "→ check <em>Confirm stop</em> → <strong>Stop evaluation</strong>.</p>"
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
  saveOpenDetails();
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
  saveOpenDetails();
}

document.addEventListener("DOMContentLoaded", function() {
  restoreOpenDetails();
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
    if (!data) {
      badge.textContent = "idle";
      badge.className = "pill-pend";
      meta.textContent = "Waiting for live_drive.json…";
      return;
    }
    if (data.active) {
      badge.textContent = "LIVE";
      badge.className = "pill-run pulse";
    } else {
      badge.textContent = "idle";
      badge.className = "pill-pend";
    }
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
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; background: #0f1419; color: #e7ecf1; }}
h1,h2 {{ font-weight: 600; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
th, td {{ border: 1px solid #2a3440; padding: 8px; text-align: center; vertical-align: top; }}
th {{ background: #1a2330; position: sticky; top: 0; }}
td.bag {{ text-align: left; font-family: ui-monospace, monospace; white-space: nowrap; }}
td.pass {{ background: #143d2a; color: #7dffa6; cursor: pointer; }}
td.fail {{ background: #4a1c1c; color: #ffb4b4; cursor: pointer; }}
td.pending {{ background: #1c2430; color: #9eb4c8; }}
td.running {{ background: #3d3414; color: #ffd866; cursor: pointer; }}
td.processing {{ background: #2a3040; color: #c8d6ff; }}
td.na {{ background: #1c1f24; color: #8899aa; }}
.detail {{ display: none; margin-top: 8px; text-align: left; font-size: 12px; color: #c9d4e0; }}
.detail.open {{ display: block; }}
code {{ font-size: 11px; }}
.summary td {{ text-align: left; }}
.progress-wrap {{ background: #1a2330; border-radius: 6px; height: 18px; margin: 8px 0 16px; overflow: hidden; }}
.progress-bar {{ background: linear-gradient(90deg, #2d7dd2, #7dffa6); height: 100%; }}
.status-pills span {{ display: inline-block; margin-right: 12px; padding: 4px 10px; border-radius: 999px; font-size: 12px; }}
.pill-done {{ background: #143d2a; color: #7dffa6; }}
.pill-run {{ background: #3d3414; color: #ffd866; }}
.pill-pend {{ background: #1c2430; color: #9eb4c8; }}
.pill-fail {{ background: #4a1c1c; color: #ffb4b4; }}
.live-drive {{ margin: 18px 0 28px; padding: 14px 16px; border: 1px solid #2a3440; border-radius: 8px; background: #121820; }}
.live-drive h2 {{ margin: 0 0 6px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
.live-meta {{ color: #9bb0c7; font-size: 13px; margin: 0 0 10px; font-family: ui-monospace, monospace; white-space: pre-wrap; }}
svg.live-drive-svg {{ background: #0f1419; border-radius: 6px; border: 1px solid #2a3440; max-width: 100%; height: auto; display: block; }}
svg.live-drive-svg .plot-bg {{ fill: #0f1419; }}
svg.live-drive-svg .grid-line {{ stroke: #1e2a38; stroke-width: 1; }}
svg.live-drive-svg .goal-dot {{ fill: #ffd866; stroke: #0f1419; stroke-width: 1; }}
svg.live-drive-svg .goal-label {{ fill: #c9d4e0; font-size: 11px; }}
svg.live-drive-svg .ego-arrow {{ fill: #3dffa6; stroke: #0f1419; stroke-width: 1.2; }}
.live {{ color: #ffd866; }}
.done {{ color: #9eb4c8; }}
.pulse {{ animation: pulse 1.5s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.55; }} }}
.legend {{ background: #1a2330; border: 1px solid #2a3440; border-radius: 8px; padding: 12px 16px; margin: 12px 0 20px; font-size: 12px; color: #c9d4e0; }}
.legend dt {{ font-weight: 600; color: #e7ecf1; margin-top: 8px; }}
.legend dd {{ margin: 2px 0 0 12px; }}
.verdict {{ font-weight: 700; letter-spacing: 0.04em; margin-bottom: 4px; }}
.goal-face {{ font-size: 11px; line-height: 1.35; color: #d7e0ea; margin: 2px 0 4px; font-family: ui-monospace, monospace; }}
.goal-face.muted {{ color: #9eb4c8; font-style: italic; }}
.metric-block {{ margin-top: 8px; margin-bottom: 4px; }}
.hint {{ color: #9eb4c8; font-weight: 400; font-size: 11px; }}
table.mini {{ width: 100%; border-collapse: collapse; margin: 4px 0 8px; font-size: 11px; }}
table.mini th, table.mini td {{ border: 1px solid #2a3440; padding: 3px 6px; text-align: left; }}
table.mini th {{ background: #121820; width: 40%; color: #c9d4e0; font-weight: 600; }}
.scatter-grid {{ display: flex; flex-direction: column; gap: 16px; margin: 8px 0 12px; }}
.goal-stop-panel {{
  background: #141c26; border: 1px solid #2a3440; border-radius: 10px;
  padding: 16px 18px; margin: 16px 0 24px;
}}
.goal-stop-panel > h2 {{ margin-top: 0; }}
.precision-table-wrap {{ margin-top: 12px; }}
.precision-table-wrap h3 {{ margin: 8px 0 4px; font-size: 14px; color: #e7ecf1; }}
.scatter-card {{ background: #1a2330; border: 1px solid #2a3440; border-radius: 8px; padding: 12px 16px; }}
.scatter-card h3 {{ margin: 0 0 6px; font-size: 15px; color: #e7ecf1; }}
.scatter-row {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }}
.scatter-legend {{ min-width: 280px; flex: 1; }}
.scatter-legend .swatch {{
  display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin-right: 6px; vertical-align: middle;
}}
.scatter-legend .swatch.pt-pass {{ background: #3dffa6; }}
.scatter-legend .swatch.pt-fail {{ background: #ff6b6b; }}
.scatter-legend .swatch.avg {{ background: #ffd866; box-shadow: inset 0 0 0 2px #1a2330; }}
.scatter-legend .avg-row td {{ color: #ffd866; }}
svg.stop-scatter {{ background: #0f1419; border-radius: 6px; border: 1px solid #2a3440; }}
svg.stop-scatter .plot-bg {{ fill: #0f1419; }}
svg.stop-scatter .grid {{ stroke: #243040; stroke-width: 1; }}
svg.stop-scatter .axis {{ stroke: #6a7f96; stroke-width: 1.5; }}
svg.stop-scatter .tol-box {{
  fill: rgba(125, 170, 255, 0.08); stroke: #7daaff; stroke-width: 1.5;
  stroke-dasharray: 6 4;
}}
svg.stop-scatter .goal-arrow {{ fill: #c8d6e8; stroke: #7daaff; stroke-width: 1.2; }}
svg.stop-scatter .goal-label {{ fill: #c9d4e0; font-size: 11px; font-weight: 700; }}
svg.stop-scatter .pt-pass {{ fill: #3dffa6; stroke: #0f1419; stroke-width: 1.2; }}
svg.stop-scatter .pt-fail {{ fill: #ff6b6b; stroke: #0f1419; stroke-width: 1.2; }}
svg.stop-scatter .pt-id {{
  fill: #e7ecf1; font-size: 12px; font-weight: 700; font-family: ui-monospace, monospace;
}}
svg.stop-scatter .avg-ring {{ fill: none; stroke: #ffd866; stroke-width: 2.5; }}
svg.stop-scatter .avg-arrow {{ fill: #ffd866; stroke: #0f1419; stroke-width: 1.2; }}
svg.stop-scatter .avg-label {{ fill: #ffd866; font-size: 12px; font-weight: 700; }}
svg.stop-scatter .triad-x {{ stroke: #ff6b6b; fill: #ff6b6b; stroke-width: 2.5; }}
svg.stop-scatter .triad-y {{ stroke: #3dffa6; fill: #3dffa6; stroke-width: 2.5; }}
svg.stop-scatter .triad-origin {{ fill: #e7ecf1; }}
svg.stop-scatter .triad-label {{ fill: #c9d4e0; font-size: 11px; font-weight: 600; }}
svg.stop-scatter .tick {{ fill: #6a7f96; font-size: 10px; }}
svg.stop-scatter .axis-title, svg.stop-scatter .axis-title-y {{
  fill: #9eb4c8; font-size: 11px; text-anchor: middle;
}}
</style>
<script>
{poll_js}
</script>
</head>
<body>
<h1>Diffusion Planner Evaluation</h1>
<p>Results root: <code>{html.escape(str(manifest.get('results_root','')))}</code></p>
<p>Updated: <code>{generated_at}</code></p>
{refresh_block}

<h2>Metric definitions</h2>
<dl class="legend">
  <dt>stuck_rate</dt>
  <dd>Fraction of scenario time ego is nearly stopped away from the goal
      (contiguous stretches ≥ stuck_duration_sec). rate = stuck_duration / bag_duration.</dd>
  <dt>collision_rate</dt>
  <dd>Fraction of ego frames within collision_distance_m of a vehicle-class NPC.
      rate = colliding_frames / ego_frames.</dd>
  <dt>oob_rate (out_of_boundary)</dt>
  <dd>Fraction of ego frames where the footprint crosses Lanelet2
      <b>road_border</b> / <b>curbstone</b> LineStrings (or is within oob_margin_m).
      rate = oob_frames / ego_frames.</dd>
  <dt>goal_stop_precision</dt>
  <dd>Average stop error in the goal frame (position, <b>lateral</b>, <b>longitudinal</b>, heading)
      over low-speed samples near the goal. Fails if |lat|, |lon|, or position exceeds tolerances.</dd>
</dl>

<h2>Overall progress</h2>
<div class="status-pills">
  <span class="pill-done">done {counts['done']}/{total}</span>
  <span class="pill-run">running {counts['running']}</span>
  <span class="pill-pend">pending {counts['pending']}</span>
  <span class="pill-fail">failed {counts['failed']}</span>
</div>
<div class="progress-wrap"><div class="progress-bar" style="width:{done_frac * 100:.1f}%"></div></div>

<section class="live-drive" id="live-drive">
  <h2>Live drive <span id="live-drive-badge" class="pill-pend">idle</span></h2>
  <p class="hint">Low-CPU ego trail (samples ~every 2s). Open this <code>dashboard.html</code> while a job is driving —
  no RViz / rosbridge. Polls <code>live_drive.json</code>.</p>
  <div id="live-drive-meta" class="live-meta">Waiting for live_drive.json…</div>
  <svg id="live-drive-svg" class="live-drive-svg" viewBox="0 0 640 420" width="640" height="420">
    <rect class="plot-bg" x="0" y="0" width="640" height="420"/>
    <g id="live-drive-grid"></g>
    <polyline id="live-drive-trail" fill="none" stroke="#2d7dd2" stroke-width="2.5"/>
    <g id="live-drive-goals"></g>
    <g id="live-drive-ego"></g>
  </svg>
</section>

<h2>Per-model summary</h2>
<table class="summary">
<tr>
  <th>Model</th>
  <th>Pass rate</th>
  <th>Avg goal pos / |lat| / |lon| [m]</th>
  <th>Avg collision rate</th>
  <th>Avg OOB rate</th>
  <th>Avg stuck rate</th>
</tr>
{''.join(summary_rows)}
</table>

{scatter_html}

<h2>Scenario × model</h2>
<p>Each cell shows PASS/FAIL plus <b>stop precision</b> (pos / lat / lon). Click to expand full rates.</p>
<table>
<tr><th>Scenario</th>{header}</tr>
{''.join(rows_html)}
</table>
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
