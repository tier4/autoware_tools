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


def build_dashboard(manifest: dict[str, Any], output_html: Path) -> Path:
    jobs = manifest.get("jobs", [])
    models = sorted({j["model_name"] for j in jobs})
    bags = sorted({j["bag_key"] for j in jobs})
    counts = _status_counts(jobs)
    total = len(jobs) or 1
    done_frac = counts["done"] / total
    in_progress = counts["pending"] + counts["running"] > 0
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Index job by (bag, model)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for j in jobs:
        by_key[(j["bag_key"], j["model_name"])] = j

    model_stats: dict[str, dict[str, float]] = {
        m: {"n": 0, "pass": 0, "collision": 0, "oob": 0, "stuck": 0, "goal_err_sum": 0.0, "goal_n": 0}
        for m in models
    }

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
            if metrics.get("collision_rate", {}).get("flagged"):
                model_stats[model]["collision"] += 1
            if metrics.get("out_of_boundary", {}).get("flagged"):
                model_stats[model]["oob"] += 1
            if metrics.get("stuck_rate", {}).get("flagged"):
                model_stats[model]["stuck"] += 1
            gerr = metrics.get("goal_stop_precision", {}).get("position_error_m")
            goal_status = metrics.get("goal_stop_precision", {}).get("status")
            if goal_status == "ok" and isinstance(gerr, (int, float)) and not math.isnan(gerr):
                model_stats[model]["goal_err_sum"] += float(gerr)
                model_stats[model]["goal_n"] += 1

            goal = metrics.get("goal_stop_precision", {})
            stuck = metrics.get("stuck_rate", {})
            col = metrics.get("collision_rate", {})
            oob = metrics.get("out_of_boundary", {})
            video = Path(job["output_dir"]) / "preview.mp4"
            video_link = ""
            if video.is_file():
                video_link = f"<div>video: <code>{html.escape(str(video))}</code></div>"

            detail = (
                f"<div class='detail' id='d-{detail_id}'>"
                f"<div>stuck={stuck.get('flagged')} dur={stuck.get('duration_sec')}</div>"
                f"<div>collision={col.get('flagged')} min_d={col.get('min_distance_m')}</div>"
                f"<div>oob={oob.get('flagged')} min_d={oob.get('min_distance_m')}</div>"
                f"<div>goal_err={goal.get('position_error_m')} m, "
                f"heading={goal.get('heading_error_deg')}° "
                f"(n={goal.get('sample_count', '—')}, status={goal.get('status')})</div>"
                f"{video_link}"
                f"</div>"
            )
            cls = "pass" if passed else "fail"
            label = "PASS" if passed else "FAIL"
            cells.append(
                f"<td class='{cls}' onclick=\"toggle('d-{detail_id}')\">{label}{detail}</td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    summary_rows = []
    for model in models:
        s = model_stats[model]
        rate = (s["pass"] / s["n"] * 100.0) if s["n"] else 0.0
        avg_goal = (s["goal_err_sum"] / s["goal_n"]) if s["goal_n"] else float("nan")
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{s['pass']}/{s['n']} ({rate:.0f}%)</td>"
            f"<td>{avg_goal:.2f}</td>"
            f"<td>{int(s['collision'])}</td>"
            f"<td>{int(s['oob'])}</td>"
            f"<td>{int(s['stuck'])}</td>"
            "</tr>"
        )

    header = "".join(f"<th>{html.escape(m)}</th>" for m in models)
    refresh_block = ""
    if in_progress:
        refresh_block = (
            '<p class="live">Live — auto-refreshing every 5s while jobs are pending/running.</p>'
            '<p class="live">To stop the run: open the Streamlit GUI → <strong>Live Progress</strong> '
            "→ check <em>Confirm stop</em> → <strong>Stop evaluation</strong>.</p>"
            '<meta http-equiv="refresh" content="5"/>'
        )
    else:
        refresh_block = '<p class="done">All jobs finished. Reload the page after starting a new run.</p>'

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>DP Multi-Eval Dashboard</title>
{refresh_block}
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
.live {{ color: #ffd866; }}
.done {{ color: #9eb4c8; }}
.pulse {{ animation: pulse 1.5s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.55; }} }}
</style>
<script>
function toggle(id) {{
  var el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('open');
}}
</script>
</head>
<body>
<h1>Diffusion Planner Evaluation</h1>
<p>Results root: <code>{html.escape(str(manifest.get('results_root','')))}</code></p>
<p>Updated: <code>{generated_at}</code></p>
{refresh_block}

<h2>Overall progress</h2>
<div class="status-pills">
  <span class="pill-done">done {counts['done']}/{total}</span>
  <span class="pill-run">running {counts['running']}</span>
  <span class="pill-pend">pending {counts['pending']}</span>
  <span class="pill-fail">failed {counts['failed']}</span>
</div>
<div class="progress-wrap"><div class="progress-bar" style="width:{done_frac * 100:.1f}%"></div></div>

<h2>Per-model summary</h2>
<table class="summary">
<tr><th>Model</th><th>Pass rate</th><th>Avg goal error [m]</th><th>Collisions</th><th>OOB</th><th>Stuck</th></tr>
{''.join(summary_rows)}
</table>

<h2>Scenario × model</h2>
<p>Click a completed or running cell to expand details.</p>
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
