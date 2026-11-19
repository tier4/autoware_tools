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

"""Re-score finished jobs with the current metric code — no re-simulation.

Every verdict is computed offline from what a job left on disk (the recorded
output/ bag, goal_pose.json, windows.json, route_goals.json) plus the source
recording and the map. So when a metric rule changes — a stop excused as a
red-light wait, a contact re-attributed — the existing runs can be re-judged
in minutes instead of re-driven in hours, and the dashboard rebuilt from the
new metrics.json files.

    ros2 run dp_multi_eval rescore.py -c ~/dp_multi_eval_pipeline.yaml
    ros2 run dp_multi_eval rescore.py -c ~/dp_multi_eval_pipeline.yaml --only H95 --dry-run

The pipeline config names results_root (where manifest.json is), map_path and
the optional thresholds / topics_yaml / vehicle_info_yaml overrides, exactly as
run_all.py reads them. Each done job's bag_path is taken from the manifest.
With --dry-run the new metrics go to metrics_rescored.json beside the old file
and nothing else changes; without it metrics.json is replaced (the
orchestrator's run_status / interventions / clean_run stamps are kept) and
dashboard.html is rebuilt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dp_multi_eval.build_dashboard import build_dashboard
from dp_multi_eval.compute_metrics import compute_metrics, load_thresholds
from dp_multi_eval.geometry import resolve_vehicle_info_yaml
from dp_multi_eval.generate_manifest import load_pipeline_config
from dp_multi_eval.run_orchestrator import load_manifest, resolve_goal_stop_required
from dp_multi_eval.topics import load_topics

# Fields the orchestrator adds to metrics.json after compute_metrics; they describe
# the run, not the metrics, and survive a rescore.
RUN_STAMPS = ("run_status", "run_error", "interventions", "clean_run", "diagnosis")


def _verdict(metrics: dict[str, Any]) -> str:
    w = metrics.get("windows") or {}
    legs = f" {w['legs_passed']}/{w['legs_total']}" if w.get("legs_total") else ""
    return ("PASS" if metrics.get("pass") else "FAIL") + legs


def rescore_job(
    job: dict[str, Any],
    *,
    map_path: Path,
    thresholds: Path | None,
    topics_yaml: Path | None,
    vehicle_info_yaml: Path | None,
    write: bool,
) -> tuple[str, str] | None:
    """Recompute one job's metrics. Returns (old verdict, new verdict), or None
    when the job has nothing to score."""
    out_dir = Path(job["output_dir"])
    bag = out_dir / "output"
    goal_file = out_dir / "goal_pose.json"
    old_file = out_dir / "metrics.json"
    if not (bag.is_dir() and goal_file.is_file() and old_file.is_file()):
        return None
    goal = json.loads(goal_file.read_text(encoding="utf-8"))
    old = json.loads(old_file.read_text(encoding="utf-8"))
    target = old_file if write else out_dir / "metrics_rescored.json"
    new = compute_metrics(
        bag,
        goal_x=float(goal["x"]),
        goal_y=float(goal["y"]),
        goal_yaw=float(goal.get("yaw", 0.0)),
        map_path=map_path,
        thresholds=load_thresholds(thresholds),
        topics=load_topics(topics_yaml),
        vehicle_info_yaml=vehicle_info_yaml,
        output_json=target,
        reference_bag=Path(job["bag_path"]) if job.get("bag_path") else None,
        goal_stop_required=resolve_goal_stop_required(job),
    )
    new.update({k: old[k] for k in RUN_STAMPS if k in old})
    target.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    return _verdict(old), _verdict(new)


def rescore(config_path: Path, *, only: str | None = None, dry_run: bool = False) -> int:
    cfg = load_pipeline_config(config_path)
    results_root = Path(cfg["results_root"]).expanduser().resolve()
    manifest_path = results_root / "manifest.json"
    manifest = load_manifest(manifest_path)
    opt = lambda key: Path(cfg[key]).expanduser() if cfg.get(key) else None  # noqa: E731
    jobs = [j for j in manifest["jobs"] if j.get("status") == "done" and (not only or only in j["job_id"])]
    print(f"[info] {len(jobs)} done job(s) to rescore" + (f" matching --only {only}" if only else ""))
    changed = 0
    for job in jobs:
        res = rescore_job(
            job,
            map_path=Path(cfg.get("map_path", "/opt/autoware/maps")).expanduser(),
            thresholds=opt("thresholds"),
            topics_yaml=opt("topics_yaml"),
            vehicle_info_yaml=resolve_vehicle_info_yaml(
                cfg.get("vehicle_info_yaml"), cfg.get("vehicle_model", "lv828l")
            ),
            write=not dry_run,
        )
        if res is None:
            print(f"  {job['job_id']}: no output to score")
            continue
        before, after = res
        changed += before != after
        print(f"  {job['job_id']}: {before} -> {after}" + ("  CHANGED" if before != after else ""))
    if dry_run:
        print(f"[info] dry run: {changed} verdict(s) would change; new metrics in metrics_rescored.json")
    else:
        dashboard = build_dashboard(manifest, results_root / "dashboard.html")
        print(f"[info] {changed} verdict(s) changed; dashboard rebuilt: {dashboard}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Re-score finished jobs with the current metric code")
    p.add_argument("-c", "--config", type=Path, required=True, help="pipeline YAML (results_root, map_path, ...)")
    p.add_argument("--only", default=None, help="rescore only the jobs whose id contains this string")
    p.add_argument("--dry-run", action="store_true", help="write metrics_rescored.json beside metrics.json; touch nothing else")
    args = p.parse_args(argv)
    return rescore(args.config, only=args.only, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
