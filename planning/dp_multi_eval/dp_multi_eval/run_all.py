"""Phase 6 — one-command full pipeline with retry + dashboard."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.build_dashboard import build_dashboard
from dp_multi_eval.generate_manifest import generate_manifest, load_pipeline_config
from dp_multi_eval.job_config import PIPELINE_KNOBS
from dp_multi_eval.run_orchestrator import load_manifest, run_orchestrator, save_manifest


def job_has_real_output(job: dict[str, Any]) -> bool:
    """True only when a recorded output bag exists (not dry-run placeholders)."""
    return (Path(job["output_dir"]) / "output").exists()


def reconcile_manifest(manifest: dict[str, Any]) -> int:
    """Re-queue entries that cannot be what they say: done/dry_run without a
    recorded output bag, and "running" — run_all is the only orchestrator of a
    results root, so at its start a running job is one whose run_all died
    (terminal closed, 2026-09-16: two orphaned stacks and two jobs no later
    run would ever pick up)."""
    reset = 0
    for job in manifest.get("jobs", []):
        if job.get("status") == "running":
            job["status"] = "pending"
            job["error"] = "interrupted (previous run_all died)"
            reset += 1
        elif job.get("status") in ("done", "dry_run") and not job_has_real_output(job):
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
    # Jobs the fresh manifest does not mention — a run of a subset of the
    # bags, another model, the other playback mode — are still results;
    # they stay, status and all. Rebuilding from a subset config dropped 30
    # of 38 finished jobs from the manifest (2026-09-15).
    fresh_ids = {j["job_id"] for j in fresh["jobs"]}
    merged_jobs.extend(j for j in existing.get("jobs", []) if j["job_id"] not in fresh_ids)
    fresh["jobs"] = merged_jobs
    return fresh


def run_all(
    config_path: Path,
    *,
    max_workers: int | None = None,
    dry_run: bool = False,
    max_retries: int = 1,
    only: str | None = None,
) -> int:
    t_start = time.monotonic()
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
        # A subset of the folder (the GUI's selection); without it every bag runs.
        bags=[Path(b) for b in cfg["bag_whitelist"]] if cfg.get("bag_whitelist") else None,
        scenarios=(
            [Path(s) for s in cfg["scenario_whitelist"]] if cfg.get("scenario_whitelist") else None
        ),
        param_template=(
            Path(cfg["param_template_path"]).expanduser()
            if cfg.get("param_template_path")
            else None
        ),
        scenario_repeat_count=int(cfg.get("scenario_repeat_count", 1)),
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

    # A key missing from the config map is skipped, not an error; a key missing
    # from PIPELINE_KNOBS is silently ignored — keep that list complete.
    extra = {k: cfg[k] for k in PIPELINE_KNOBS if k in cfg}
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

        try:
            run_orchestrator(
                manifest_path,
                max_workers=workers,
                domain_ids=domain_ids,
                map_path=Path(cfg.get("map_path", "/opt/autoware/maps")),
                dry_run=dry_run,
                retry_failed=True,
                render_video_flag=bool(cfg.get("render_video", True)),
                extra_opts=extra,
                only=only,
            )
        except KeyboardInterrupt:
            # The orchestrator has already cancelled the queue and stopped the
            # simulators; a stack trace here would only obscure that.
            manifest = load_manifest(manifest_path)
            build_dashboard(manifest, results_root / "dashboard.html")
            print("[warn] Run interrupted; interrupted jobs left pending for a rerun")
            return 130
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
        elapsed_min = (time.monotonic() - t_start) / 60.0
        print(
            f"DONE: {n_done}/{n} completed ({n_pass} metric-pass), {n_fail} failed "
            f"in {elapsed_min:.1f} min — see {manifest_path}"
        )
    print(f"Dashboard: {dashboard}")
    return 0 if dry_run or n_fail == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 6: full evaluation pipeline")
    p.add_argument("-c", "--config", type=Path, required=True)
    p.add_argument("--max_workers", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument(
        "--only",
        default=None,
        help="run only the jobs whose id contains this string (e.g. a bag key); "
        "every other job keeps its status and its results",
    )
    args = p.parse_args(argv)
    return run_all(
        args.config,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        only=args.only,
    )


if __name__ == "__main__":
    sys.exit(main())
