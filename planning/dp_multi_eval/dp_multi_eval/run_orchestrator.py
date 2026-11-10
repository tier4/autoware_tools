"""Phase 4b — resumable multi-job orchestrator with domain-id pool."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.geometry import resolve_vehicle_info_yaml
from dp_multi_eval.build_dashboard import build_dashboard
from dp_multi_eval.bag_reader import load_bag_series
from dp_multi_eval.compute_metrics import Thresholds, compute_metrics, load_thresholds
from dp_multi_eval.diagnose import diagnose
from dp_multi_eval.generate_manifest import bag_key
from dp_multi_eval.job_config import PIPELINE_KNOBS
from dp_multi_eval.job_status import finalize_job_status, preview_video_path
from dp_multi_eval.process_utils import cleanup_evaluation_processes
from dp_multi_eval.render_video import render_video
from dp_multi_eval.run_single_job import JobConfig, run_single_job, update_job_progress
from dp_multi_eval.scenario_sim import (
    extract_goal_from_output_bag,
    extract_goal_from_scenario,
)
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


def resolve_job_goal(job: dict[str, Any], bag_out: Path) -> tuple[float, float, float]:
    """Goal for metrics: scenario file (SS2) → route_goals.json → input bag → output bag."""
    mode = str(job.get("mode") or "reproducer")
    if mode == "scenario_simulator" and job.get("scenario_path"):
        goal = extract_goal_from_scenario(Path(job["scenario_path"]))
        if goal is not None:
            return goal
        return extract_goal_from_output_bag(bag_out)

    # Multi-goal runs write route_goals.json; use the last leg for stop-precision metrics.
    goals_path = Path(job["output_dir"]) / "route_goals.json"
    if goals_path.is_file():
        try:
            payload = json.loads(goals_path.read_text(encoding="utf-8"))
            goals = payload.get("goals") or []
            if goals:
                g = goals[-1]
                return float(g["x"]), float(g["y"]), float(g.get("yaw", 0.0))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    if job.get("bag_path"):
        return extract_goal_from_input_bag(Path(job["bag_path"]))
    return extract_goal_from_output_bag(bag_out)


def resolve_goal_stop_required(job: dict[str, Any]) -> bool:
    """False when route_goals.json says the run ends drive-through at a cut pose
    (the recording was cut while moving; there is no stop to score): the
    single-goal path writes ``arrival``, a multi-goal run's last leg carries
    the source ``bag_end:cut`` (the harness ends it drive-through, 6 m; a run
    that had reached every leg was flagged goal_not_reached for not parking
    at the cut pose, 2026-04-28)."""
    goals_path = Path(job["output_dir"]) / "route_goals.json"
    if goals_path.is_file():
        try:
            payload = json.loads(goals_path.read_text(encoding="utf-8"))
            if payload.get("arrival") == "drive_through_at_cut_pose":
                return False
            goals = payload.get("goals") or []
            return not (goals and str(goals[-1].get("source", "")).startswith("bag_end:cut"))
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass
    return True


def _job_config_from_payload(job: dict[str, Any], opts: dict[str, Any], domain_id: int) -> JobConfig:
    mode = str(job.get("mode") or opts.get("mode") or "reproducer")
    return JobConfig(
        model_config=Path(job["model_config_path"]),
        bag_path=Path(job["bag_path"]) if job.get("bag_path") else None,
        scenario_path=Path(job["scenario_path"]) if job.get("scenario_path") else None,
        mode=mode,
        output_dir=Path(job["output_dir"]),
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
        require_perception_ready=bool(opts.get("require_perception_ready", True)),
        bag_load_base_sec=float(opts.get("bag_load_base_sec", 120.0)),
        bag_load_duration_factor=float(opts.get("bag_load_duration_factor", 1.0)),
        bag_load_timeout_cap_sec=float(opts.get("bag_load_timeout_cap_sec", 3600.0)),
        gpu_min_free_mib=float(opts.get("gpu_min_free_mib", JobConfig.gpu_min_free_mib)),
        gpu_wait_timeout_sec=float(
            opts.get("gpu_wait_timeout_sec", JobConfig.gpu_wait_timeout_sec)
        ),
        # Not bool(): that would turn the string "auto" into True and force
        # multi-goal on every bag. JobConfig resolves the value itself.
        multi_goal=opts.get("multi_goal", JobConfig.multi_goal),
        multi_goal_source=str(opts.get("multi_goal_source", "auto")),
        multi_goal_stop_speed_mps=float(opts.get("multi_goal_stop_speed_mps", 0.2)),
        multi_goal_stop_min_sec=float(opts.get("multi_goal_stop_min_sec", 8.0)),
        multi_goal_min_spacing_m=float(opts.get("multi_goal_min_spacing_m", 15.0)),
        multi_goal_stop_order=[
            str(n).strip()
            for n in (opts.get("multi_goal_stop_order") or [])
            if str(n).strip()
        ]
        or None,
        stop_points_csv=str(opts.get("stop_points_csv", "stop_points.csv")),
        # Defaults come from JobConfig, never repeated literals: a stale copy here
        # silently overrides the dataclass whenever the pipeline config omits the key.
        goal_backoff_steps_m=opts.get("goal_backoff_steps_m", JobConfig.goal_backoff_steps_m),
        stop_fallback_max_dist_m=float(
            opts.get("stop_fallback_max_dist_m", JobConfig.stop_fallback_max_dist_m)
        ),
        route_total_timeout_sec=float(
            opts.get("route_total_timeout_sec", JobConfig.route_total_timeout_sec)
        ),
        stuck_blinker_nudge=bool(
            opts.get("stuck_blinker_nudge", JobConfig.stuck_blinker_nudge)
        ),
        stall_speed_mps=float(opts.get("stall_speed_mps", JobConfig.stall_speed_mps)),
        stuck_blinker_speed_mps=float(opts.get("stuck_blinker_speed_mps", 0.15)),
        stuck_blinker_trigger_sec=float(opts.get("stuck_blinker_trigger_sec", 12.0)),
        stuck_blinker_hold_sec=float(opts.get("stuck_blinker_hold_sec", 3.0)),
        stuck_blinker_cooldown_sec=float(opts.get("stuck_blinker_cooldown_sec", 25.0)),
        stuck_abort_sec=float(opts.get("stuck_abort_sec", JobConfig.stuck_abort_sec)),
        stack_hold_detect_sec=float(
            opts.get("stack_hold_detect_sec", JobConfig.stack_hold_detect_sec)
        ),
        resync_lag_sec=float(opts.get("resync_lag_sec", JobConfig.resync_lag_sec)),
        window_eval=bool(opts.get("window_eval", JobConfig.window_eval)),
        leg_cuts=str(opts.get("leg_cuts", JobConfig.leg_cuts)),
        scene_objects=bool(opts.get("scene_objects", JobConfig.scene_objects)),
        planner_warmup_timeout_sec=float(
            opts.get("planner_warmup_timeout_sec", JobConfig.planner_warmup_timeout_sec)
        ),
        disable_long_stop_fixer=bool(
            opts.get("disable_long_stop_fixer", JobConfig.disable_long_stop_fixer)
        ),
        match_vehicle_dynamics=bool(
            opts.get("match_vehicle_dynamics", JobConfig.match_vehicle_dynamics)
        ),
        indicator_merge=bool(opts.get("indicator_merge", JobConfig.indicator_merge)),
        final_drive_through_arrival=bool(
            opts.get("final_drive_through_arrival", JobConfig.final_drive_through_arrival)
        ),
        reposition_nudge_mps=float(
            opts.get("reposition_nudge_mps", JobConfig.reposition_nudge_mps)
        ),
        npc_reaction=bool(opts.get("npc_reaction", JobConfig.npc_reaction)),
        npc_reaction_decel=float(
            opts.get("npc_reaction_decel", JobConfig.npc_reaction_decel)
        ),
        stuck_reposition_trigger_sec=float(opts.get("stuck_reposition_trigger_sec", 30.0)),
        stuck_reposition_forward_m=float(
            opts.get("stuck_reposition_forward_m", JobConfig.stuck_reposition_forward_m)
        ),
        stuck_reposition_max_count=int(
            opts.get("stuck_reposition_max_count", JobConfig.stuck_reposition_max_count)
        ),
        multi_goal_arrival_tolerance_m=float(
            opts.get("multi_goal_arrival_tolerance_m", 5.0)
        ),
        multi_goal_leg_timeout_sec=float(opts.get("multi_goal_leg_timeout_sec", 300.0)),
        live_web_monitor=bool(opts.get("live_web_monitor", True)),
        live_web_sample_sec=float(opts.get("live_web_sample_sec", 2.0)),
        live_drive_root=(
            Path(opts["results_root"])
            if opts.get("results_root")
            else (Path(job["results_root"]) if job.get("results_root") else None)
        ),
        architecture_type=str(opts.get("architecture_type", "awf/universe/20250130")),
        scenario_timeout_sec=float(opts.get("scenario_timeout_sec", 300.0)),
        scenario_record_warmup_sec=float(opts.get("scenario_record_warmup_sec", 15.0)),
        dry_run=bool(opts.get("dry_run", False)),
        rviz=bool(opts.get("rviz", False)),
        # A model's own planner (an ML planner model) wins over the run's setting.
        planning_setting=str(
            job.get("planning_setting") or opts.get("planning_setting", "diffusion_planner")
        ),
    )


def _attach_diagnosis(out_dir: Path, result: dict[str, Any]) -> None:
    """Say what the outcome means, next to the evidence for it.

    The recorded error is precise and unreadable — understanding
    ``leg_2:wait_not_started; leg_wait_exhausted`` takes knowledge of the drive
    loop. Every one of these has been diagnosed before; the diagnosis belongs in
    the results rather than in someone's memory.
    """
    metrics_path = out_dir / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metrics = {}

    tail = None
    repro_log = out_dir / "reproducer.log"
    if repro_log.is_file():
        try:
            tail = "\n".join(repro_log.read_text(errors="replace").splitlines()[-6:])
        except OSError:
            tail = None

    found = diagnose(
        result.get("error"),
        status=result.get("status"),
        interventions=result.get("interventions"),
        metrics=metrics,
        reproducer_log_tail=tail,
    )
    if found is None:
        return
    for line in found.lines():
        print(f"[info] {line}")
    result["diagnosis"] = found.as_dict()
    if metrics:
        metrics["diagnosis"] = found.as_dict()
        try:
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
    status_path = out_dir / "job_status.json"
    if status_path.is_file():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            payload["diagnosis"] = found.as_dict()
            status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except (json.JSONDecodeError, OSError):
            pass


def classify_run_outcome(error: str | None, failed: bool) -> str:
    """done / stuck / timeout / setup_failed / failed, from how the drive ended.

    The ticket asks for stuck runs to be marked as stuck rather than lumped in with
    everything else, and the distinction is the one that matters when reading a
    batch: a job that could not be set up is a scenario or configuration problem,
    a job that timed out may just need a longer budget, and a job that stalled is
    the planner or the scene — only the last is evidence about the model.
    """
    if not failed:
        return "done"
    text = (error or "").lower()
    if "stopped updating" in text or "no longer publishing" in text:
        return "sim_died"
    if "stuck" in text or "reposition" in text:
        return "stuck"
    if "route setup" in text or "localization_failed" in text or "route_setup" in text:
        return "setup_failed"
    if "perception_ready_timeout" in text or "planner inference disabled" in text:
        return "setup_failed"
    if "timeout" in text or "exhausted" in text or "not_arrived" in text:
        return "timeout"
    if "interrupted" in text:
        return "interrupted"
    return "failed"


def _stamp_run_status(
    metrics_path: Path,
    *,
    run_status: str,
    run_error: str | None,
    interventions: dict[str, float] | None = None,
) -> None:
    """Record how the drive ended, and what the harness did to it, in metrics.json.

    Failed runs now get metrics too (so a failure can be reviewed rather than only
    guessed at), which means every consumer needs a way to tell a partial drive from
    a completed one. Without this marker a failed run's numbers look like a result.

    ``interventions`` records blinker nudges and forward shifts. A run that only
    arrived because it was teleported is not the same result as one that drove
    there, and ``clean_run`` is what separates them.
    """
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    payload["run_status"] = run_status
    if run_error:
        payload["run_error"] = run_error
    counts = dict(interventions or {})
    payload["interventions"] = counts
    # stall_x / stall_y are the last stall POSITION (map coordinates, ~50000),
    # not intervention counts; counting them marked every run "assisted".
    payload["clean_run"] = not any(
        float(v) > 0 for k, v in counts.items() if not str(k).startswith("stall_")
    )
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _post_inputs(opts: dict[str, Any]) -> tuple[Any, Any, float, float, bool]:
    topics = load_topics(Path(opts["topics_yaml"]) if opts.get("topics_yaml") else None)
    thresholds = load_thresholds(Path(opts["thresholds"]) if opts.get("thresholds") else None)
    sample_dt = float(thresholds.sample_dt)
    video_sample_dt = float(opts.get("video_sample_dt", sample_dt))
    show_factors = bool(opts.get("show_planning_factors", True))
    return topics, thresholds, sample_dt, video_sample_dt, show_factors


def _load_series(
    bag_out: Path, topics: Any, sample_dt: float, video_sample_dt: float, show_factors: bool
) -> Any:
    """One read of the output bag (object downsampling cuts load time ~10x)."""
    return load_bag_series(
        bag_out,
        ego_topic=topics.ego_pose,
        objects_topic=topics.predicted_objects,
        velocity_topic=topics.vehicle_status_velocity,
        trajectory_topic=topics.trajectory,
        planning_factor_topics=topics.planning_factors if show_factors else [],
        turn_indicators_status_topic=topics.turn_indicators_status,
        turn_indicators_cmd_topic=topics.turn_indicators_cmd,
        object_sample_dt=sample_dt,
        trajectory_sample_dt=video_sample_dt,
        factor_sample_dt=video_sample_dt,
        skip_zero_size_objects=True,
    )


def _render_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Render-pool entry: draw one job's preview after its metrics exist.

    Off the simulator worker on purpose. Measured on 2026-04-28 the render was
    264 s of a 6.8 min post-processing step during which the worker's psim slot
    sat idle. The output bag is read again here (~23 s) rather than pickled
    across processes.
    """
    job, opts = payload["job"], payload["opts"]
    out_dir = Path(job["output_dir"])
    bag_out = out_dir / "output"
    results_root = opts.get("results_root") or job.get("results_root")
    try:
        topics, _thresholds, sample_dt, video_sample_dt, show_factors = _post_inputs(opts)
        series = _load_series(bag_out, topics, sample_dt, video_sample_dt, show_factors)
        render_video(
            bag_out,
            preview_video_path(job, results_root=results_root, for_write=True),
            metrics_json=out_dir / "metrics.json",
            map_path=Path(opts["map_path"]) if opts.get("map_path") else None,
            goal=tuple(payload["goal"]),
            fps=float(opts.get("video_fps", 8.0)),
            sample_dt=video_sample_dt,
            topics=topics,
            vehicle_info_yaml=resolve_vehicle_info_yaml(
                opts.get("vehicle_info_yaml"), opts.get("vehicle_model", "lv828l")
            ),
            series=series,
            view_frame=str(opts.get("video_view_frame", "map")),
            view_range_m=float(opts.get("video_view_range_m", 40.0)),
            show_planning_factors=show_factors,
            # Same reference the speed metric uses: the bag being replayed.
            reference_bag=Path(job["bag_path"]) if job.get("bag_path") else None,
        )
        return {"job_id": job["job_id"], "ok": True}
    except Exception as exc:  # noqa: BLE001
        # Video is nice-to-have; the job's result stands without it.
        (out_dir / "render_error.txt").write_text(str(exc), encoding="utf-8")
        return {"job_id": job["job_id"], "ok": False, "error": str(exc)}


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
        results_root = opts.get("results_root") or job.get("results_root")

        if opts.get("dry_run"):
            cfg = _job_config_from_payload(job, {**opts, "dry_run": True}, domain_id)
            cfg.skip_route_setup = True
            jr = run_single_job(cfg)
            result["status"] = jr.status
            result["error"] = jr.error
            return result

        cfg = _job_config_from_payload(job, opts, domain_id)
        jr = run_single_job(cfg)
        # Carry interventions on the result even when the job fails — a failure that
        # took seven teleports to reach is a different failure from an immediate one.
        result["interventions"] = dict(jr.interventions or {})
        result["clean_run"] = jr.clean_run
        run_failed = jr.status != "done"
        if run_failed:
            # Keep the drive-phase error (route_not_arrived, stuck, …) — it is the
            # only description of what went wrong. Post-processing below still runs
            # so the failure is reviewable, but it must not overwrite this.
            result["status"] = "failed"
            result["error"] = jr.error or "job_failed"

        bag_out = out_dir / "output"
        if not bag_out.exists():
            if not run_failed:
                result["status"] = "failed"
                result["error"] = "output_bag_missing"
            return result

        try:
            goal = resolve_job_goal(job, bag_out)
        except Exception as exc:  # noqa: BLE001
            # Without a goal there is nothing to measure against and nothing to draw.
            (out_dir / "post_process_error.txt").write_text(
                f"goal_resolution_failed: {exc}\n", encoding="utf-8"
            )
            if not run_failed:
                result["status"] = "failed"
                result["error"] = f"goal_resolution_failed: {exc}"
            return result

        (out_dir / "goal_pose.json").write_text(
            json.dumps({"x": goal[0], "y": goal[1], "yaw": goal[2]}, indent=2) + "\n",
            encoding="utf-8",
        )

        topics, thresholds, sample_dt, video_sample_dt, show_factors = _post_inputs(opts)

        update_job_progress(
            out_dir,
            phase="post_processing",
            start_time=jr.start_time or "",
            domain_id=domain_id,
            bag_path=job.get("bag_path"),
            scenario_path=job.get("scenario_path"),
            model_config=str(job["model_config_path"]),
        )

        series = _load_series(bag_out, topics, sample_dt, video_sample_dt, show_factors)

        metrics_path = out_dir / "metrics.json"
        try:
            compute_metrics(
                bag_out,
                goal_x=goal[0],
                goal_y=goal[1],
                goal_yaw=goal[2],
                map_path=Path(opts["map_path"]) if opts.get("map_path") else None,
                thresholds=thresholds,
                topics=topics,
                vehicle_info_yaml=resolve_vehicle_info_yaml(
                    opts.get("vehicle_info_yaml"), opts.get("vehicle_model", "lv828l")
                ),
                output_json=metrics_path,
                series=series,
                # The bag being replayed is the reference the drive is compared to.
                reference_bag=Path(job["bag_path"]) if job.get("bag_path") else None,
                goal_stop_required=resolve_goal_stop_required(job),
            )
            # Stamp how the drive ended. A failed run's metrics describe a partial
            # drive: useful for triage, never comparable to a completed run.
            outcome = classify_run_outcome(result.get("error"), run_failed)
            _stamp_run_status(
                metrics_path,
                run_status=outcome,
                run_error=result["error"] if run_failed else None,
                interventions=jr.interventions,
            )
            result["outcome"] = outcome
        except Exception as exc:  # noqa: BLE001
            (out_dir / "post_process_error.txt").write_text(
                f"compute_metrics_failed: {exc}\n{traceback.format_exc()[-500:]}\n",
                encoding="utf-8",
            )
            if not run_failed:
                result["status"] = "failed"
                result["error"] = f"compute_metrics_failed: {exc}"
                run_failed = True

        render_on_failure = bool(opts.get("render_video_on_failure", True))
        if opts.get("render_video", True) and (not run_failed or render_on_failure):
            # Drawn by the render pool (_render_job) once this worker has
            # returned, so the simulator slot goes to the next bag meanwhile.
            result["render_goal"] = [float(goal[0]), float(goal[1]), float(goal[2])]

        if not run_failed:
            result["status"] = "done"
            result["error"] = None
        return result
    except KeyboardInterrupt:
        # Ctrl+C reaches the pool workers too. Record it as a real outcome so the
        # finally block below still stops the simulator and finalizes job_status —
        # letting it escape bare leaves psim running and the status file mid-phase.
        result["status"] = "failed"
        result["error"] = "interrupted"
        return result
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = f"{exc}\n{traceback.format_exc()[-500:]}"
        return result
    finally:
        # Every exit path leaves a terminal job_status.json. Post-processing writes
        # `running` progress updates, so without this the file stays stuck on the
        # last phase and cannot be trusted by resume or monitoring.
        if not opts.get("dry_run"):
            try:
                finalize_job_status(
                    Path(job["output_dir"]),
                    status=result["status"],
                    error=result.get("error"),
                    extra={
                        "interventions": result.get("interventions") or {},
                        "clean_run": bool(result.get("clean_run", True)),
                    },
                )
            except Exception:  # noqa: BLE001
                pass
            # After the status file is terminal, so the diagnosis sees the final
            # outcome rather than the phase the job was in when it stopped.
            try:
                _attach_diagnosis(Path(job["output_dir"]), result)
            except Exception:  # noqa: BLE001
                pass


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
    only: str | None = None,
) -> dict[str, Any]:
    """``only``: run just the jobs whose id contains this string (one bag, one
    model). The others keep their status and their results; nothing else is
    touched. Without it every pending and failed job runs."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    domain_ids = domain_ids or list(range(10, 10 + max(max_workers, 1)))
    opts: dict[str, Any] = {
        "map_path": str(map_path),
        "dry_run": dry_run,
        "render_video": render_video_flag,
        **(extra_opts or {}),
        "results_root": str(manifest.get("results_root") or manifest_path.parent),
    }

    pending = [
        j
        for j in manifest["jobs"]
        if j["status"] in ("pending", "failed", "dry_run")
        and (retry_failed or j["status"] == "pending")
        and (only is None or only in j["job_id"])
    ]
    if only is not None:
        print(f"[info] --only {only}: {len(pending)} job(s) selected")
    total = len(manifest["jobs"])
    done0 = sum(1 for j in manifest["jobs"] if j["status"] == "done")
    print(f"[info] Orchestrator: {len(pending)} to run, {done0}/{total} already done")

    if not pending:
        return manifest

    # ProcessPool needs spawn-safe; limit concurrency to len(domain_ids)
    workers = min(max_workers, len(domain_ids), len(pending))
    domain_pool = list(domain_ids[:workers])

    # A job takes a free domain when it starts and hands it back when it ends.
    # Round-robin by submission order let a job start on a domain another job
    # was still driving on; its start-up cleanup then killed that job's simulator.
    futures = {}
    fut_domain: dict[Any, int] = {}
    free_domains = list(domain_pool)
    queue = list(pending)
    interrupted = False
    render_pool = (
        ProcessPoolExecutor(max_workers=max(1, int(opts.get("render_workers", 1))))
        if opts.get("render_video", True)
        else None
    )
    render_futures: dict[Any, str] = {}
    pool = ProcessPoolExecutor(max_workers=workers)

    def _submit_ready() -> None:
        while queue and free_domains:
            job = queue.pop(0)
            domain_id = free_domains.pop(0)
            job["status"] = "running"
            job["attempts"] = int(job.get("attempts") or 0) + 1
            fut = pool.submit(_process_job, {"job": job, "opts": opts, "domain_id": domain_id})
            futures[fut] = job["job_id"]
            fut_domain[fut] = domain_id

    try:
        _submit_ready()
        save_manifest(manifest_path, manifest)
        _refresh_dashboard(manifest, manifest_path)

        # Cancel from inside the signal handler, not after KeyboardInterrupt has
        # unwound to the loop below. The worker is interrupted at the same instant;
        # it returns, and the executor dispatches the next queued job before the
        # main thread ever reaches its except clause. Cancelling here closes that
        # window — Future.cancel() is a no-op on one already running.
        def _on_sigint(_signum, _frame):
            for f in list(futures) + list(render_futures):
                f.cancel()
            raise KeyboardInterrupt

        # SIGTERM takes the same path: a run stopped from another shell (kill,
        # a supervisor, a timeout) must stop its simulators and leave its jobs
        # pending exactly as Ctrl-C does — the default handler killed run_all
        # and left every psim, reproducer and recorder running in its own
        # session, and a running job marked "running" in the manifest.
        previous_sigint = signal.signal(signal.SIGINT, _on_sigint)
        previous_sigterm = signal.signal(signal.SIGTERM, _on_sigint)
        # A closed terminal (SIGHUP) too: otherwise run_all dies silently and
        # two full Autoware stacks keep running as orphans (2026-09-16).
        previous_sighup = signal.signal(signal.SIGHUP, _on_sigint)

        finished = 0
        try:
            while futures:
                # Wake every 15 s even when nothing finished: the dashboard is
                # rebuilt so a job that has started shows RUNNING, not QUEUED.
                done, _ = wait(list(futures), timeout=15.0, return_when=FIRST_COMPLETED)
                if not done:
                    _refresh_dashboard(manifest, manifest_path)
                    continue
                fut = next(iter(done))
                finished += 1
                job_id = futures.pop(fut)
                free_domains.append(fut_domain.pop(fut))
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    res = {"job_id": job_id, "status": "failed", "error": str(exc)}
                for job in manifest["jobs"]:
                    if job["job_id"] == res["job_id"]:
                        job["status"] = res["status"]
                        job["error"] = res.get("error")
                        job["domain_id"] = res.get("domain_id")
                        job["interventions"] = res.get("interventions") or {}
                        job["clean_run"] = bool(res.get("clean_run", True))
                        break
                _submit_ready()
                save_manifest(manifest_path, manifest)
                _refresh_dashboard(manifest, manifest_path)
                if res.get("render_goal") is not None and render_pool is not None:
                    job_of = next(j for j in manifest["jobs"] if j["job_id"] == res["job_id"])
                    rf = render_pool.submit(
                        _render_job, {"job": job_of, "opts": opts, "goal": res["render_goal"]}
                    )
                    render_futures[rf] = res["job_id"]
                n_done = sum(1 for j in manifest["jobs"] if j["status"] == "done")
                n_fail = sum(1 for j in manifest["jobs"] if j["status"] == "failed")
                print(
                    f"[progress] {finished}/{len(pending)} batch | "
                    f"overall {n_done}/{total} done, {n_fail} failed | last={job_id}→{res['status']}"
                )
            if render_futures:
                print(f"[info] all drives finished; {len(render_futures)} video(s) still rendering")
            for rf in as_completed(render_futures):
                r = rf.result()
                note = "rendered" if r.get("ok") else f"failed: {str(r.get('error'))[:120]}"
                print(f"[video] {r['job_id']}: {note}")
                _refresh_dashboard(manifest, manifest_path)
        except KeyboardInterrupt:
            interrupted = True
            print("\n[warn] Interrupted — cancelling queued jobs and stopping simulators")
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGHUP, previous_sighup)
    finally:
        # Ctrl+C reaches this process and the pool workers, but never the simulator:
        # psim, the reproducer and the recorder are each started with os.setsid, so
        # they sit in their own sessions and keep running. Cancel what has not begun,
        # do not wait on what has, then kill the process groups directly.
        if interrupted:
            # Terminate the pool workers before shutting down. shutdown(wait=False)
            # leaves a running worker alive, and multiprocessing's atexit hook then
            # joins it on the way out — which is why a second Ctrl+C was needed and
            # the exit still ended in "Exception ignored in atexit callback".
            for worker in list(getattr(pool, "_processes", {}).values()):
                try:
                    worker.terminate()
                except Exception:  # noqa: BLE001
                    pass
        pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
        if render_pool is not None:
            render_pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
        if interrupted:
            cleanup_evaluation_processes()
            # A job cut off mid-flight is not "running" any more. Leave it pending so
            # a later run picks it up instead of skipping it as already in progress.
            for job_id in futures.values():
                for job in manifest["jobs"]:
                    if job["job_id"] == job_id and job["status"] == "running":
                        job["status"] = "pending"
                        job["error"] = "interrupted"
            save_manifest(manifest_path, manifest)
            _refresh_dashboard(manifest, manifest_path)
            raise KeyboardInterrupt

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
        for key in PIPELINE_KNOBS:
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
