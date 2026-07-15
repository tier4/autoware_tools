"""Phase 6 — one-command full pipeline with retry + dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.build_dashboard import build_dashboard
from dp_multi_eval.generate_manifest import generate_manifest, load_pipeline_config
from dp_multi_eval.run_orchestrator import load_manifest, run_orchestrator, save_manifest


def job_has_real_output(job: dict[str, Any]) -> bool:
    """True only when a recorded output bag exists (not dry-run placeholders)."""
    return (Path(job["output_dir"]) / "output").exists()


def reconcile_manifest(manifest: dict[str, Any]) -> int:
    """Reset bogus done/dry_run entries so real runs are not skipped."""
    reset = 0
    for job in manifest.get("jobs", []):
        if job.get("status") not in ("done", "dry_run"):
            continue
        if job_has_real_output(job):
            continue
        job["status"] = "pending"
        job["error"] = None
        reset += 1
    return reset


def merge_manifests(existing: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Keep done jobs; refresh pending list with any new model/bag combos."""
    by_id = {j["job_id"]: j for j in existing.get("jobs", [])}
    merged_jobs = []
    for job in fresh["jobs"]:
        old = by_id.get(job["job_id"])
        if old and old.get("status") == "done" and job_has_real_output(old):
            merged_jobs.append(old)
        elif old:
            job["attempts"] = old.get("attempts", 0)
            job["status"] = old.get("status", "pending")
            job["error"] = old.get("error")
            if (
                old.get("status") == "failed"
                and old.get("model_config_path") != job.get("model_config_path")
            ):
                job["status"] = "pending"
                job["error"] = None
            merged_jobs.append(job)
        else:
            merged_jobs.append(job)
    fresh["jobs"] = merged_jobs
    return fresh


def run_all(
    config_path: Path,
    *,
    max_workers: int | None = None,
    dry_run: bool = False,
    max_retries: int = 1,
) -> int:
    cfg = load_pipeline_config(config_path)
    results_root = Path(cfg["results_root"]).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    manifest_path = results_root / "manifest.json"

    fresh = generate_manifest(
        models=cfg["models"],
        results_root=results_root,
        mode=str(cfg.get("mode", "reproducer")),
        rosbag_dir=Path(cfg["rosbag_dir"]) if cfg.get("rosbag_dir") else None,
        scenario_dir=Path(cfg["scenario_dir"]) if cfg.get("scenario_dir") else None,
        param_template=(
            Path(cfg["param_template_path"]).expanduser()
            if cfg.get("param_template_path")
            else None
        ),
    )
    if manifest_path.is_file():
        existing = load_manifest(manifest_path)
        manifest = merge_manifests(existing, fresh)
    else:
        manifest = fresh
    n_reset = reconcile_manifest(manifest)
    if n_reset:
        print(f"[info] Re-queued {n_reset} jobs without recorded output (e.g. prior dry-run)")
    save_manifest(manifest_path, manifest)
    print(f"[info] Manifest: {manifest_path} ({len(manifest['jobs'])} jobs)")
    build_dashboard(manifest, results_root / "dashboard.html")

    workers = int(max_workers if max_workers is not None else cfg.get("max_workers", 2))
    domain_start = int(cfg.get("domain_id_start", 10))
    domain_ids = list(range(domain_start, domain_start + max(workers, 1)))

    extra = {
        k: cfg[k]
        for k in (
            "mode",
            "vehicle_model",
            "sensor_model",
            "vehicle_id",
            "end_condition",
            "bag_duration_margin_sec",
            "route_timeout_sec",
            "psim_startup_sec",
            "route_setup_timeout_sec",
            "auto_engage_timeout_sec",
            "perception_warmup_sec",
            "perception_ready_timeout_sec",
            "perception_ready_min_objects",
            "perception_ready_stable_sec",
            "stuck_blinker_nudge",
            "stuck_blinker_speed_mps",
            "stuck_blinker_trigger_sec",
            "stuck_blinker_hold_sec",
            "stuck_blinker_cooldown_sec",
            "architecture_type",
            "scenario_timeout_sec",
            "scenario_record_warmup_sec",
            "planning_setting",
            "video_fps",
            "video_sample_dt",
            "video_view_frame",
            "video_view_range_m",
            "show_planning_factors",
            "rviz",
            "thresholds",
            "topics_yaml",
            "vehicle_info_yaml",
        )
        if k in cfg
    }
    extra.setdefault("mode", str(cfg.get("mode", "reproducer")))

    # Pass 1 + optional retries for failed
    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Re-queue failed (once)
            changed = False
            for job in manifest["jobs"]:
                if job["status"] == "failed" and int(job.get("attempts") or 0) <= max_retries:
                    job["status"] = "pending"
                    changed = True
            if not changed:
                break
            save_manifest(manifest_path, manifest)
            print(f"[info] Retry pass {attempt}/{max_retries}")

        run_orchestrator(
            manifest_path,
            max_workers=workers,
            domain_ids=domain_ids,
            map_path=Path(cfg.get("map_path", "/opt/autoware/maps")),
            dry_run=dry_run,
            retry_failed=True,
            render_video_flag=bool(cfg.get("render_video", True)),
            extra_opts=extra,
        )
        manifest = load_manifest(manifest_path)

    dashboard = build_dashboard(manifest, results_root / "dashboard.html")
    n = len(manifest["jobs"])
    n_done = sum(1 for j in manifest["jobs"] if j["status"] == "done")
    n_fail = sum(1 for j in manifest["jobs"] if j["status"] == "failed")
    # Pass among done with metrics
    n_pass = 0
    for j in manifest["jobs"]:
        if j["status"] != "done":
            continue
        mpath = Path(j["output_dir"]) / "metrics.json"
        if mpath.is_file():
            try:
                if json.loads(mpath.read_text(encoding="utf-8")).get("pass"):
                    n_pass += 1
            except json.JSONDecodeError:
                pass
        else:
            n_pass += 1  # completed but metrics not computed yet

    if dry_run:
        n_dry = sum(1 for j in manifest["jobs"] if j["status"] == "dry_run")
        print(
            f"DRY-RUN: {n_dry}/{n} validated, {n_fail} failed — "
            f"no simulations launched; see {manifest_path}"
        )
    else:
        print(
            f"DONE: {n_done}/{n} completed ({n_pass} metric-pass), {n_fail} failed — "
            f"see {manifest_path}"
        )
    print(f"Dashboard: {dashboard}")
    return 0 if dry_run or n_fail == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 6: full evaluation pipeline")
    p.add_argument("-c", "--config", type=Path, required=True)
    p.add_argument("--max_workers", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_retries", type=int, default=1)
    args = p.parse_args(argv)
    return run_all(
        args.config,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    sys.exit(main())
