"""Phase 4b — resumable multi-job orchestrator with domain-id pool."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.build_dashboard import build_dashboard
from dp_multi_eval.bag_reader import load_bag_series
from dp_multi_eval.compute_metrics import Thresholds, compute_metrics, load_thresholds
from dp_multi_eval.generate_manifest import bag_key
from dp_multi_eval.render_video import render_video
from dp_multi_eval.run_single_job import JobConfig, run_single_job, update_job_progress
from dp_multi_eval.topics import load_topics


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def extract_goal_from_input_bag(bag_path: Path, min_move_m: float = 0.1) -> tuple[float, float, float]:
    """Best-effort goal from source bag poses (reuses batch-eval helper if present)."""
    try:
        from rosbag_utils import get_poses_from_bag

        _start, goal = get_poses_from_bag(bag_path, min_move_m=min_move_m)
        import math

        q = goal.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return float(goal.position.x), float(goal.position.y), float(yaw)
    except Exception:  # noqa: BLE001
        # Fallback: last ego pose in bag if any
        try:
            from dp_multi_eval.bag_reader import load_bag_series

            series = load_bag_series(bag_path)
            if series.ego:
                e = series.ego[-1]
                return e.x, e.y, e.yaw_rad
        except Exception:  # noqa: BLE001
            pass
    return 0.0, 0.0, 0.0


def _process_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry — must be picklable (top-level)."""
    job = payload["job"]
    opts = payload["opts"]
    domain_id = payload["domain_id"]
    result: dict[str, Any] = {
        "job_id": job["job_id"],
        "status": "failed",
        "error": None,
        "domain_id": domain_id,
    }
    try:
        out_dir = Path(job["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        if opts.get("dry_run"):
            cfg = JobConfig(
                model_config=Path(job["model_config_path"]),
                bag_path=Path(job["bag_path"]),
                output_dir=out_dir,
                domain_id=domain_id,
                map_path=Path(opts["map_path"]),
                vehicle_model=opts.get("vehicle_model", "lv828l"),
                sensor_model=opts.get("sensor_model", "aip_x2_gen2"),
                vehicle_id=opts.get("vehicle_id"),
                dry_run=True,
                skip_route_setup=True,
            )
            jr = run_single_job(cfg)
            result["status"] = jr.status
            result["error"] = jr.error
            return result

        cfg = JobConfig(
            model_config=Path(job["model_config_path"]),
            bag_path=Path(job["bag_path"]),
            output_dir=out_dir,
            domain_id=domain_id,
            map_path=Path(opts["map_path"]),
            vehicle_model=opts.get("vehicle_model", "lv828l"),
            sensor_model=opts.get("sensor_model", "aip_x2_gen2"),
            vehicle_id=opts.get("vehicle_id"),
            topics=load_topics(Path(opts["topics_yaml"]) if opts.get("topics_yaml") else None),
            end_condition=opts.get("end_condition", "route_arrived"),
            bag_duration_margin_sec=float(opts.get("bag_duration_margin_sec", 30.0)),
            route_timeout_sec=float(opts.get("route_timeout_sec", 900.0)),
            psim_startup_sec=float(opts.get("psim_startup_sec", 120.0)),
            route_setup_timeout_sec=float(opts.get("route_setup_timeout_sec", 90.0)),
            auto_engage_timeout_sec=float(opts.get("auto_engage_timeout_sec", 60.0)),
            perception_warmup_sec=float(opts.get("perception_warmup_sec", 5.0)),
            perception_ready_timeout_sec=float(opts.get("perception_ready_timeout_sec", 45.0)),
            perception_ready_min_objects=int(opts.get("perception_ready_min_objects", 1)),
            perception_ready_stable_sec=float(opts.get("perception_ready_stable_sec", 2.0)),
            dry_run=False,
            rviz=bool(opts.get("rviz", False)),
        )
        jr = run_single_job(cfg)
        if jr.status != "done":
            result["status"] = "failed"
            result["error"] = jr.error or "job_failed"
            return result

        bag_out = out_dir / "output"
        if not bag_out.exists():
            result["status"] = "failed"
            result["error"] = "output_bag_missing"
            return result

        goal = extract_goal_from_input_bag(Path(job["bag_path"]))
        (out_dir / "goal_pose.json").write_text(
            json.dumps({"x": goal[0], "y": goal[1], "yaw": goal[2]}, indent=2) + "\n",
            encoding="utf-8",
        )

        topics = load_topics(Path(opts["topics_yaml"]) if opts.get("topics_yaml") else None)
        thresholds = load_thresholds(
            Path(opts["thresholds"]) if opts.get("thresholds") else None
        )
        sample_dt = float(thresholds.sample_dt)
        video_sample_dt = float(opts.get("video_sample_dt", sample_dt))

        update_job_progress(
            out_dir,
            phase="post_processing",
            start_time=jr.start_time or "",
            domain_id=domain_id,
            bag_path=str(job["bag_path"]),
            model_config=str(job["model_config_path"]),
        )

        show_factors = bool(opts.get("show_planning_factors", True))
        # Single bag read for metrics + video (object downsampling cuts load time ~10x).
        series = load_bag_series(
            bag_out,
            ego_topic=topics.ego_pose,
            objects_topic=topics.predicted_objects,
            velocity_topic=topics.vehicle_status_velocity,
            trajectory_topic=topics.trajectory,
            planning_factor_topics=topics.planning_factors if show_factors else [],
            object_sample_dt=sample_dt,
            trajectory_sample_dt=video_sample_dt,
            factor_sample_dt=video_sample_dt,
            skip_zero_size_objects=True,
        )

        metrics_path = out_dir / "metrics.json"
        compute_metrics(
            bag_out,
            goal_x=goal[0],
            goal_y=goal[1],
            goal_yaw=goal[2],
            map_path=Path(opts["map_path"]) if opts.get("map_path") else None,
            thresholds=thresholds,
            topics=topics,
            vehicle_info_yaml=(
                Path(opts["vehicle_info_yaml"]) if opts.get("vehicle_info_yaml") else None
            ),
            output_json=metrics_path,
            series=series,
        )

        if opts.get("render_video", True):
            try:
                update_job_progress(
                    out_dir,
                    phase="rendering_video",
                    start_time=jr.start_time or "",
                    domain_id=domain_id,
                    bag_path=str(job["bag_path"]),
                    model_config=str(job["model_config_path"]),
                )
                render_video(
                    bag_out,
                    out_dir / "preview.mp4",
                    metrics_json=metrics_path,
                    map_path=Path(opts["map_path"]) if opts.get("map_path") else None,
                    goal=goal,
                    fps=float(opts.get("video_fps", 8.0)),
                    sample_dt=video_sample_dt,
                    topics=topics,
                    vehicle_info_yaml=(
                        Path(opts["vehicle_info_yaml"]) if opts.get("vehicle_info_yaml") else None
                    ),
                    series=series,
                    view_frame=str(opts.get("video_view_frame", "map")),
                    view_range_m=float(opts.get("video_view_range_m", 40.0)),
                    show_planning_factors=show_factors,
                )
            except Exception as exc:  # noqa: BLE001
                # Video is nice-to-have; don't fail the job solely on render
                (out_dir / "render_error.txt").write_text(str(exc), encoding="utf-8")

        result["status"] = "done"
        result["error"] = None
        return result
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = f"{exc}\n{traceback.format_exc()[-500:]}"
        return result


def _refresh_dashboard(manifest: dict[str, Any], manifest_path: Path) -> None:
    results_root = Path(manifest.get("results_root") or manifest_path.parent)
    try:
        build_dashboard(manifest, results_root / "dashboard.html")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] dashboard refresh failed: {exc}")


def run_orchestrator(
    manifest_path: Path,
    *,
    max_workers: int = 2,
    domain_ids: list[int] | None = None,
    map_path: Path = Path("/opt/autoware/maps"),
    dry_run: bool = False,
    retry_failed: bool = True,
    render_video_flag: bool = True,
    extra_opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    domain_ids = domain_ids or list(range(10, 10 + max(max_workers, 1)))
    opts: dict[str, Any] = {
        "map_path": str(map_path),
        "dry_run": dry_run,
        "render_video": render_video_flag,
        **(extra_opts or {}),
    }

    pending = [
        j
        for j in manifest["jobs"]
        if j["status"] in ("pending", "failed", "dry_run")
        and (retry_failed or j["status"] == "pending")
    ]
    total = len(manifest["jobs"])
    done0 = sum(1 for j in manifest["jobs"] if j["status"] == "done")
    print(f"[info] Orchestrator: {len(pending)} to run, {done0}/{total} already done")

    if not pending:
        return manifest

    # ProcessPool needs spawn-safe; limit concurrency to len(domain_ids)
    workers = min(max_workers, len(domain_ids), len(pending))
    domain_pool = list(domain_ids[:workers])

    # Sequential assignment of domain via round-robin in submission order
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for idx, job in enumerate(pending):
            job["status"] = "running"
            job["attempts"] = int(job.get("attempts") or 0) + 1
            domain_id = domain_pool[idx % len(domain_pool)]
            fut = pool.submit(_process_job, {"job": job, "opts": opts, "domain_id": domain_id})
            futures[fut] = job["job_id"]
        save_manifest(manifest_path, manifest)
        _refresh_dashboard(manifest, manifest_path)

        finished = 0
        for fut in as_completed(futures):
            finished += 1
            job_id = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {"job_id": job_id, "status": "failed", "error": str(exc)}
            for job in manifest["jobs"]:
                if job["job_id"] == res["job_id"]:
                    job["status"] = res["status"]
                    job["error"] = res.get("error")
                    job["domain_id"] = res.get("domain_id")
                    break
            save_manifest(manifest_path, manifest)
            _refresh_dashboard(manifest, manifest_path)
            n_done = sum(1 for j in manifest["jobs"] if j["status"] == "done")
            n_fail = sum(1 for j in manifest["jobs"] if j["status"] == "failed")
            print(
                f"[progress] {finished}/{len(pending)} batch | "
                f"overall {n_done}/{total} done, {n_fail} failed | last={job_id}→{res['status']}"
            )

    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 4: run pending/failed jobs from manifest")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--max_workers", type=int, default=2)
    p.add_argument("--domain_ids", default="10-13", help="e.g. 10-13 or 10,11,12,13")
    p.add_argument("--map_path", type=Path, default=Path("/opt/autoware/maps"))
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--no_render", action="store_true")
    p.add_argument("--thresholds", type=Path, default=None)
    p.add_argument("--topics-yaml", type=Path, default=None)
    p.add_argument("--vehicle_info_yaml", type=Path, default=None)
    p.add_argument("-c", "--config", type=Path, default=None, help="Optional pipeline YAML overrides")
    args = p.parse_args(argv)

    extra: dict[str, Any] = {}
    if args.config and args.config.expanduser().is_file():
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        for key in (
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
            "video_fps",
            "video_sample_dt",
            "video_view_frame",
            "video_view_range_m",
            "show_planning_factors",
            "rviz",
        ):
            if key in raw:
                extra[key] = raw[key]
        if "map_path" in raw and args.map_path == Path("/opt/autoware/maps"):
            args.map_path = Path(raw["map_path"])
    if args.thresholds:
        extra["thresholds"] = str(args.thresholds)
    if args.topics_yaml:
        extra["topics_yaml"] = str(args.topics_yaml)
    if args.vehicle_info_yaml:
        extra["vehicle_info_yaml"] = str(args.vehicle_info_yaml)

    text = args.domain_ids.strip()
    if "-" in text and "," not in text:
        a, b = text.split("-", 1)
        domain_ids = list(range(int(a), int(b) + 1))
    else:
        domain_ids = [int(x) for x in text.split(",") if x.strip()]

    run_orchestrator(
        args.manifest,
        max_workers=args.max_workers,
        domain_ids=domain_ids,
        map_path=args.map_path,
        dry_run=args.dry_run,
        render_video_flag=not args.no_render,
        extra_opts=extra,
    )
    return 0


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on some platforms
    sys.exit(main())
