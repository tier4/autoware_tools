"""Phase 7 — Streamlit GUI for non-engineers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.build_dashboard import (
    build_dashboard,
    collect_stop_scatter_points,
    render_stop_scatter_section,
)
from dp_multi_eval.path_picker import pick_folder
from dp_multi_eval.process_utils import cleanup_evaluation_processes, stop_pid_group
from dp_multi_eval.scenario_sim import (
    build_scenario_runner_command,
    discover_scenarios,
    scenario_key,
)

# Soft dependency: streamlit
try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "streamlit is required for the GUI. Install with: pip install streamlit"
    ) from exc


DEFAULT_CONFIG = Path.home() / "dp_multi_eval_pipeline.yaml"
STATE_FILE_NAME = ".gui_run_state.json"
SESSION_KEYS = ("models_dir", "rosbag_dir", "scenario_dir", "results_root")


def _widget_key(session_key: str) -> str:
    return f"input_{session_key}"


def _pending_key(session_key: str) -> str:
    return f"_pending_folder_{session_key}"


def _set_folder_path(session_key: str, path: str) -> None:
    """Keep session path and Streamlit text_input widget key in sync.

    Must be called only before the matching text_input widget is created.
    """
    path = str(path or "").strip()
    st.session_state[session_key] = path
    st.session_state[_widget_key(session_key)] = path


def _init_session(defaults: dict[str, str], *, config_path: str) -> None:
    """Seed folder paths from pipeline YAML.

    Streamlit text_input widgets own a separate key (`input_*`). Once created empty,
    `value=` is ignored — so we must write both keys, and re-seed when the pipeline
    config file changes or a field is still empty.
    """
    prev_config = st.session_state.get("_pipeline_config_path")
    config_changed = prev_config != config_path
    st.session_state["_pipeline_config_path"] = config_path

    for key in SESSION_KEYS:
        default = str(defaults.get(key, "") or "").strip()
        widget_key = _widget_key(key)
        current = str(st.session_state.get(widget_key) or st.session_state.get(key) or "").strip()
        if config_changed or not current:
            if default:
                _set_folder_path(key, default)
            elif key not in st.session_state:
                _set_folder_path(key, "")


def folder_input(
    label: str,
    session_key: str,
    *,
    picker_title: str,
    help_text: str = "",
) -> Path:
    widget_key = _widget_key(session_key)
    pending_key = _pending_key(session_key)

    # Apply browse result before instantiating the text_input (Streamlit forbids
    # writing widget keys after the widget exists).
    pending = st.session_state.pop(pending_key, None)
    if pending is not None:
        _set_folder_path(session_key, pending)
    elif widget_key not in st.session_state:
        st.session_state[widget_key] = str(st.session_state.get(session_key, "") or "")

    col_path, col_btn = st.sidebar.columns([5, 1])
    with col_path:
        value = st.text_input(
            label,
            key=widget_key,
            help=help_text or None,
        )
    with col_btn:
        st.write("")
        if st.button("…", key=f"browse_{session_key}", help=f"Browse for {label.lower()}"):
            picked = pick_folder(picker_title, st.session_state.get(widget_key, ""))
            if picked:
                st.session_state[pending_key] = picked
                st.rerun()
    path = str(value or "").strip()
    st.session_state[session_key] = path
    return Path(path).expanduser()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def discover_model_configs(models_dir: Path) -> list[tuple[str, Path]]:
    if not models_dir.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for path in sorted(models_dir.rglob("*.param.yaml")):
        name = path.stem.replace(".param", "")
        found.append((name, path))
    for path in sorted(models_dir.rglob("diffusion_planner.param.yaml")):
        name = path.parent.name
        found.append((name, path))
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "diffusion_planner.onnx").is_file() and (child / "args.json").is_file():
            found.append((child.name, child))
    # unique by path
    seen: set[Path] = set()
    out: list[tuple[str, Path]] = []
    for name, path in found:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        out.append((name, path))
    return out


def discover_bag_dirs(rosbag_dir: Path) -> list[Path]:
    if not rosbag_dir.is_dir():
        return []
    bags: list[Path] = []
    for child in sorted(rosbag_dir.iterdir()):
        if child.is_dir():
            if (child / "metadata.yaml").exists() or list(child.glob("*.db3")):
                bags.append(child)
            else:
                for sub in sorted(child.iterdir()):
                    if sub.is_dir() and (
                        (sub / "metadata.yaml").exists() or list(sub.glob("*.db3"))
                    ):
                        bags.append(sub)
    return bags


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    return Path(f"/proc/{pid}").exists()


def load_gui_run_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return state if isinstance(state, dict) else None


def gui_run_is_active(state_path: Path) -> tuple[bool, dict[str, Any] | None]:
    state = load_gui_run_state(state_path)
    if not state:
        return False, None
    if pid_alive(int(state.get("pid") or 0)):
        return True, state
    return False, state


def mark_manifest_jobs_cancelled(manifest_path: Path) -> int:
    if not manifest_path.is_file():
        return 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    cancelled = 0
    for job in manifest.get("jobs", []):
        if job.get("status") != "running":
            continue
        job["status"] = "failed"
        job["error"] = "cancelled_by_user"
        cancelled += 1
    if cancelled:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        try:
            build_dashboard(manifest, manifest_path.parent / "dashboard.html")
        except OSError:
            pass
    return cancelled


def stop_gui_run(state_path: Path, state: dict[str, Any]) -> str:
    pid = int(state.get("pid") or 0)
    run_name = state.get("run_name", "evaluation")
    stopped = stop_pid_group(pid)
    cleanup_evaluation_processes()
    manifest_path = Path(state["manifest"]) if state.get("manifest") else None
    cancelled = 0
    if manifest_path:
        cancelled = mark_manifest_jobs_cancelled(manifest_path)
    try:
        state_path.unlink(missing_ok=True)
    except OSError:
        pass
    parts = [f"Stopped “{run_name}”."]
    if stopped:
        parts.append(f"Orchestrator PID {pid} terminated.")
    else:
        parts.append("Orchestrator was not running (cleaning up simulators anyway).")
    if cancelled:
        parts.append(f"Marked {cancelled} running scenario(s) as cancelled.")
    return " ".join(parts)


def render_stop_run_controls(
    state_path: Path,
    *,
    key_prefix: str,
    show_when_inactive: bool = False,
) -> None:
    running, state = gui_run_is_active(state_path)
    if not running and not show_when_inactive:
        return
    if not running:
        if state_path.is_file():
            try:
                state_path.unlink(missing_ok=True)
            except OSError:
                pass
        return
    assert state is not None
    st.warning(
        f"Run **{state.get('run_name', '?')}** is active "
        f"(orchestrator PID {state.get('pid', '?')})."
    )
    confirm = st.checkbox(
        "Confirm stop — kills orchestrator and any running simulators",
        key=f"{key_prefix}_confirm_stop",
    )
    if st.button("Stop evaluation", type="primary", disabled=not confirm, key=f"{key_prefix}_stop"):
        st.info(stop_gui_run(state_path, state))
        st.rerun()


def write_run_config(
    base: dict[str, Any],
    *,
    run_name: str,
    models: list[dict[str, str]],
    rosbag_dir: Path | None = None,
    scenario_dir: Path | None = None,
    selected_bags: list[Path] | None = None,
    selected_scenarios: list[Path] | None = None,
) -> Path:
    results_root = Path(base.get("results_root", "/tmp/dp_multi_eval_results")) / run_name
    cfg = dict(base)
    cfg["models"] = models
    cfg["results_root"] = str(results_root)
    if rosbag_dir is not None:
        cfg["rosbag_dir"] = str(rosbag_dir)
    if scenario_dir is not None:
        cfg["scenario_dir"] = str(scenario_dir)
    if selected_bags is not None:
        cfg["bag_whitelist"] = [str(b) for b in selected_bags]
    if selected_scenarios is not None:
        cfg["scenario_whitelist"] = [str(s) for s in selected_scenarios]
    out = results_root / "pipeline_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def main() -> None:
    st.set_page_config(page_title="Planner Evaluation", layout="wide")
    st.title("Planner evaluation")
    st.caption("Run model scenarios and review results — no terminal required.")

    config_path = Path(
        st.sidebar.text_input("Pipeline config file", value=str(DEFAULT_CONFIG))
    ).expanduser()
    base = load_yaml(config_path)
    if not config_path.is_file():
        st.sidebar.warning(f"Config not found: `{config_path}`")
    _init_session(
        {
            "models_dir": str(base.get("models_dir", "/opt/autoware/mlmodels")),
            "rosbag_dir": str(base.get("rosbag_dir", "")),
            "scenario_dir": str(base.get("scenario_dir", "")),
            "results_root": str(base.get("results_root", "/tmp/dp_multi_eval_results")),
        },
        config_path=str(config_path.resolve()) if config_path.exists() else str(config_path),
    )
    models_dir = folder_input(
        "Models folder",
        "models_dir",
        picker_title="Select models folder",
        help_text="Folder with ONNX model dirs or diffusion_planner.param.yaml files.",
    )
    results_root = folder_input(
        "Results folder",
        "results_root",
        picker_title="Select results output folder",
        help_text="Manifest, dashboard, and per-job outputs are written here.",
    )

    tab_setup, tab_progress, tab_results = st.tabs(
        ["Setup & Run", "Live Progress", "Results"]
    )

    with tab_setup:
        mode_labels = {
            "reproducer": "Reproducer (ROS bags)",
            "scenario_simulator": "Scenario Simulator (.yaml / .xosc)",
        }
        default_mode = str(base.get("mode", "reproducer")).strip().lower()
        if default_mode not in mode_labels:
            default_mode = "reproducer"
        mode = st.radio(
            "Evaluation mode",
            options=list(mode_labels.keys()),
            format_func=lambda m: mode_labels[m],
            index=list(mode_labels.keys()).index(default_mode),
            horizontal=True,
            help="Reproducer = perception_reproducer on bags. "
            "Scenario Simulator = scenario_test_runner on OpenSCENARIO / T4 YAML.",
        )

        if mode == "reproducer":
            rosbag_dir = folder_input(
                "Rosbag scenarios folder",
                "rosbag_dir",
                picker_title="Select ROS bag scenarios folder",
                help_text="Parent folder of ID1, ID2, … rosbag subfolders.",
            )
            scenario_dir = Path(st.session_state.get("scenario_dir") or "")
        else:
            scenario_dir = folder_input(
                "Scenario files folder",
                "scenario_dir",
                picker_title="Select scenario folder",
                help_text="Folder of .yaml / .yml / .xosc scenario files (one nesting level OK).",
            )
            rosbag_dir = Path(st.session_state.get("rosbag_dir") or "")

        st.subheader("Choose models")
        model_opts = discover_model_configs(models_dir)
        if not model_opts:
            st.warning(
                "No models found. Pick a folder containing ONNX model directories "
                "(diffusion_planner.onnx + args.json) or *.param.yaml files."
            )
        selected_models = []
        for name, path in model_opts:
            if st.checkbox(name, value=False, key=f"m_{path}"):
                selected_models.append({"name": name, "model_config": str(path)})

        selected_bags: list[Path] = []
        selected_scenarios: list[Path] = []
        if mode == "reproducer":
            st.subheader("Choose rosbag scenarios")
            bags = discover_bag_dirs(rosbag_dir)
            if not str(rosbag_dir):
                st.info(
                    "Set **Rosbag scenarios folder** to the parent of ID1, ID2, … "
                    f"(from config: `{base.get('rosbag_dir', '') or 'not set'}`)."
                )
            elif not rosbag_dir.is_dir():
                st.warning(
                    f"Rosbag folder does not exist or is not a directory: `{rosbag_dir}`"
                )
            elif not bags:
                st.warning(
                    f"No bags found under `{rosbag_dir}`. "
                    "Each subfolder needs `metadata.yaml` or a `.db3` file."
                )
            else:
                st.caption(f"Found {len(bags)} bag(s) under `{rosbag_dir}`")
            select_all = st.checkbox("Select all scenarios", value=False, key="select_all_bags")
            for bag in bags:
                try:
                    label = str(bag.resolve().relative_to(rosbag_dir.resolve()))
                except ValueError:
                    label = bag.name
                checked = select_all or st.checkbox(label, value=False, key=f"b_{bag}")
                if checked:
                    selected_bags.append(bag)
        else:
            st.subheader("Choose scenario files")
            try:
                scenarios = discover_scenarios(scenario_dir) if scenario_dir.is_dir() else []
            except FileNotFoundError:
                scenarios = []
            if not str(scenario_dir):
                st.info(
                    "Set **Scenario files folder** "
                    f"(from config: `{base.get('scenario_dir', '') or 'not set'}`)."
                )
            elif not scenario_dir.is_dir():
                st.warning(f"Scenario folder missing: `{scenario_dir}`")
            elif not scenarios:
                st.warning(
                    f"No scenario files under `{scenario_dir}`. "
                    "Expected `.yaml` / `.yml` / `.xosc`."
                )
            else:
                st.caption(f"Found {len(scenarios)} scenario file(s) under `{scenario_dir}`")
            select_all = st.checkbox(
                "Select all scenario files", value=False, key="select_all_scenarios"
            )
            for scn in scenarios:
                try:
                    label = scenario_key(scn, scenario_dir)
                except Exception:  # noqa: BLE001
                    label = scn.name
                checked = select_all or st.checkbox(label, value=False, key=f"s_{scn}")
                if checked:
                    selected_scenarios.append(scn)

        st.subheader("Vehicle / Autoware")
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            vehicle_model = st.text_input(
                "vehicle_model",
                value=str(base.get("vehicle_model", "lv828l")),
                help="Passed to scenario_test_runner / planning_simulator as vehicle_model:=…",
            )
            sensor_model = st.text_input(
                "sensor_model",
                value=str(base.get("sensor_model", "aip_x2_gen2")),
            )
        with col_a2:
            vehicle_id = st.text_input(
                "vehicle_id (autoware.vehicle_id)",
                value=str(base.get("vehicle_id", "6_lv828l")),
                help="Matches manual launch: autoware.vehicle_id:=6_lv828l",
            )
            planning_setting = st.text_input(
                "planning_setting",
                value=str(base.get("planning_setting", "diffusion_planner")),
                help="autoware.planning_setting:=… (use diffusion_planner)",
            )
        with col_a3:
            map_path = st.text_input(
                "map_path",
                value=str(base.get("map_path", "/opt/autoware/maps")),
                help="Lanelet map used for OOB / offline metrics.",
            )
            launch_rviz = st.checkbox(
                "Launch RViz",
                value=bool(base.get("rviz", False)),
                help="On = open RViz (debug). Off = headless (default). "
                "Reproducer: rviz:=…  "
                "Scenario Simulator: launch_rviz:=… and autoware.rviz:=… "
                "(Autoware defaults rviz on — both must be off).",
            )

        initialize_duration_sec = float(base.get("psim_startup_sec", 120.0))
        scenario_timeout_sec = float(base.get("scenario_timeout_sec", 300.0))
        architecture_type = str(base.get("architecture_type", "awf/universe/20250130"))
        scenario_record_warmup_sec = float(base.get("scenario_record_warmup_sec", 15.0))

        if mode == "scenario_simulator":
            st.subheader("Scenario Simulator launch")
            st.caption(
                "Mapped to `ros2 launch scenario_test_runner scenario_test_runner.launch.py` "
                "(same args as a successful local single-scenario run)."
            )
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                architecture_type = st.text_input(
                    "architecture_type",
                    value=architecture_type,
                    help="e.g. awf/universe/20250130",
                )
                initialize_duration_sec = st.number_input(
                    "initialize_duration [s]",
                    min_value=30.0,
                    max_value=600.0,
                    value=float(initialize_duration_sec),
                    step=10.0,
                    help="SS2 initialize_duration — Autoware bring-up allowance.",
                )
            with col_s2:
                scenario_timeout_sec = st.number_input(
                    "global_timeout [s]",
                    min_value=30.0,
                    max_value=3600.0,
                    value=float(scenario_timeout_sec),
                    step=30.0,
                    help="SS2 global_timeout — scenario wall-clock budget.",
                )
                scenario_record_warmup_sec = st.number_input(
                    "Record warmup [s]",
                    min_value=0.0,
                    max_value=120.0,
                    value=float(scenario_record_warmup_sec),
                    step=5.0,
                    help="Wait after SS2 start before ros2 bag record (sim time).",
                )
            if selected_scenarios:
                sample = selected_scenarios[0]
                preview_cmd = build_scenario_runner_command(
                    scenario_path=sample,
                    output_directory=Path("/tmp/dp_multi_eval_ss2_preview"),
                    vehicle_model=vehicle_model,
                    sensor_model=sensor_model,
                    architecture_type=architecture_type,
                    planning_setting=planning_setting,
                    initialize_duration_sec=initialize_duration_sec,
                    global_timeout_sec=scenario_timeout_sec,
                    rviz=launch_rviz,
                    record=False,
                    vehicle_id=vehicle_id or None,
                )
                with st.expander("Preview launch command (first selected scenario)", expanded=False):
                    st.code(" \\\n  ".join(preview_cmd), language="bash")

        run_name = st.text_input("Run name", value=time.strftime("run_%Y%m%d_%H%M%S"))
        max_workers = st.number_input(
            "Parallel workers",
            min_value=1,
            max_value=8,
            value=int(base.get("max_workers", 1)),
        )

        st.subheader("Preview video")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            view_opts = ["base_link", "map"]
            default_view = str(base.get("video_view_frame", "base_link"))
            if default_view not in view_opts:
                default_view = "base_link"
            video_view_frame = st.selectbox(
                "View frame",
                view_opts,
                index=view_opts.index(default_view),
                help="base_link = ego-centered (recommended); map = fit full path (can look zoomed out).",
            )
            show_planning_factors = st.checkbox(
                "Show planning-factor virtual walls",
                value=bool(base.get("show_planning_factors", True)),
                help="Draw walls from modifier_obstacle_stop, diffusion_planner, stop_point_fixer.",
            )
        with col_v2:
            video_view_range_m = st.number_input(
                "View range [m] (base_link half-extent)",
                min_value=10.0,
                max_value=200.0,
                value=float(base.get("video_view_range_m", 40.0)),
                step=5.0,
            )
            video_fps = st.number_input(
                "Video FPS",
                min_value=1.0,
                max_value=30.0,
                value=float(base.get("video_fps", 6.0)),
                step=1.0,
            )

        state_path = results_root / STATE_FILE_NAME
        running, _state = gui_run_is_active(state_path)

        if running:
            render_stop_run_controls(state_path, key_prefix="setup")
            st.info("A run is already in progress. See the Live Progress tab.")
        elif st.button("Start evaluation", type="primary"):
            if not selected_models:
                st.error("Select at least one model.")
            elif mode == "reproducer" and not selected_bags:
                st.error("Select at least one rosbag scenario.")
            elif mode == "scenario_simulator" and not selected_scenarios:
                st.error("Select at least one scenario file.")
            else:
                cfg_base = dict(base)
                cfg_base["mode"] = mode
                cfg_base["results_root"] = str(results_root)
                cfg_base["max_workers"] = int(max_workers)
                cfg_base["map_path"] = str(map_path).strip() or "/opt/autoware/maps"
                cfg_base["vehicle_model"] = str(vehicle_model).strip()
                cfg_base["sensor_model"] = str(sensor_model).strip()
                cfg_base["vehicle_id"] = str(vehicle_id).strip()
                cfg_base["planning_setting"] = str(planning_setting).strip()
                cfg_base["psim_startup_sec"] = float(initialize_duration_sec)
                cfg_base["architecture_type"] = str(architecture_type).strip()
                cfg_base["scenario_timeout_sec"] = float(scenario_timeout_sec)
                cfg_base["scenario_record_warmup_sec"] = float(scenario_record_warmup_sec)
                cfg_base["rviz"] = bool(launch_rviz)
                cfg_base["video_view_frame"] = video_view_frame
                cfg_base["video_view_range_m"] = float(video_view_range_m)
                cfg_base["show_planning_factors"] = bool(show_planning_factors)
                cfg_base["video_fps"] = float(video_fps)
                cfg_base["render_video"] = True
                import shutil

                if mode == "reproducer":
                    run_input_root = results_root / run_name / "_selected_bags"
                    if run_input_root.exists():
                        shutil.rmtree(run_input_root)
                    run_input_root.mkdir(parents=True)
                    for bag in selected_bags:
                        link = run_input_root / bag.name
                        if not link.exists():
                            link.symlink_to(bag, target_is_directory=bag.is_dir())
                    cfg_path = write_run_config(
                        cfg_base,
                        run_name=run_name,
                        models=selected_models,
                        rosbag_dir=run_input_root,
                        selected_bags=selected_bags,
                    )
                else:
                    run_input_root = results_root / run_name / "_selected_scenarios"
                    if run_input_root.exists():
                        shutil.rmtree(run_input_root)
                    run_input_root.mkdir(parents=True)
                    for scn in selected_scenarios:
                        # Preserve relative key via flat unique names
                        link = run_input_root / scn.name
                        n = 1
                        while link.exists():
                            link = run_input_root / f"{scn.stem}_{n}{scn.suffix}"
                            n += 1
                        link.symlink_to(scn)
                    cfg_path = write_run_config(
                        cfg_base,
                        run_name=run_name,
                        models=selected_models,
                        scenario_dir=run_input_root,
                        selected_scenarios=selected_scenarios,
                    )
                log_path = results_root / run_name / "run_all.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_f = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "dp_multi_eval.run_all",
                        "-c",
                        str(cfg_path),
                        "--max_workers",
                        str(int(max_workers)),
                    ],
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(
                    json.dumps(
                        {
                            "pid": proc.pid,
                            "run_name": run_name,
                            "config": str(cfg_path),
                            "manifest": str(results_root / run_name / "manifest.json"),
                            "log": str(log_path),
                            "started": time.time(),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                st.success(f"Started run “{run_name}”. Open Live Progress.")

    with tab_progress:
        st.subheader("Progress")
        render_stop_run_controls(results_root / STATE_FILE_NAME, key_prefix="progress")
        # Find latest manifest under results_root
        manifests = sorted(results_root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
        if not manifests:
            st.info("No evaluation runs yet. Start one from Setup & Run.")
        else:
            manifest_path = manifests[-1]
            st.write(f"Tracking: `{manifest_path}`")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                st.warning("Manifest is being written…")
                manifest = {"jobs": []}
            jobs = manifest.get("jobs", [])
            n = len(jobs) or 1
            n_done = sum(1 for j in jobs if j.get("status") == "done")
            n_fail = sum(1 for j in jobs if j.get("status") == "failed")
            n_run = sum(1 for j in jobs if j.get("status") == "running")
            st.progress(n_done / n, text=f"Scenarios completed: {n_done}/{len(jobs)}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Completed", n_done)
            c2.metric("Running", n_run)
            c3.metric("Failed", n_fail)

            # Per-model bars
            models = sorted({j["model_name"] for j in jobs})
            for model in models:
                mj = [j for j in jobs if j["model_name"] == model]
                md = sum(1 for j in mj if j["status"] == "done")
                st.write(model)
                st.progress((md / len(mj)) if mj else 0.0, text=f"{md}/{len(mj)}")

            failed = [j for j in jobs if j.get("status") == "failed"]
            if failed:
                st.markdown("#### Failed scenarios")
                st.dataframe(
                    [
                        {
                            "scenario": j["bag_key"],
                            "model": j["model_name"],
                            "error": (j.get("error") or "")[:200],
                        }
                        for j in failed
                    ],
                    width="stretch",
                )
            running_jobs = [j for j in jobs if j.get("status") == "running"]
            if running_jobs:
                st.markdown("#### Currently running")
                rows = []
                for j in running_jobs:
                    phase = ""
                    total_elapsed = ""
                    phase_elapsed = ""
                    domain = j.get("domain_id")
                    js_path = Path(j["output_dir"]) / "job_status.json"
                    if js_path.is_file():
                        try:
                            js = json.loads(js_path.read_text(encoding="utf-8"))
                            phase = str(js.get("phase") or "")
                            domain = js.get("domain_id", domain)
                            from datetime import datetime, timezone

                            now = datetime.now(timezone.utc)
                            start = js.get("start_time")
                            if start:
                                t0 = datetime.fromisoformat(str(start))
                                total_elapsed = (
                                    f"{(now - t0.astimezone(timezone.utc)).total_seconds():.0f}s"
                                )
                            phase_start = js.get("phase_start_time")
                            if phase_start:
                                t1 = datetime.fromisoformat(str(phase_start))
                                phase_elapsed = (
                                    f"{(now - t1.astimezone(timezone.utc)).total_seconds():.0f}s"
                                )
                        except (json.JSONDecodeError, ValueError):
                            pass
                    rows.append(
                        {
                            "scenario": j["bag_key"],
                            "model": j["model_name"],
                            "phase": phase,
                            "total_elapsed": total_elapsed,
                            "phase_elapsed": phase_elapsed,
                            "domain": domain,
                        }
                    )
                st.dataframe(rows, width="stretch")

            if st.button("Refresh now"):
                st.rerun()
            st.caption("This page does not auto-refresh in all Streamlit versions — click Refresh.")

    with tab_results:
        st.subheader("Results")
        render_stop_run_controls(results_root / STATE_FILE_NAME, key_prefix="results")
        manifests = sorted(results_root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
        if not manifests:
            st.info("No results yet — start an evaluation from the Setup tab.")
        else:
            manifest_path = manifests[-1]
            run_dir = manifest_path.parent
            st.caption(f"Latest run: `{run_dir}`")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                st.error(f"Could not read {manifest_path}")
                manifest = None

            if manifest:
                jobs = manifest.get("jobs", [])
                stop_points = collect_stop_scatter_points(jobs)

                # KPI strip from completed metrics
                n_done = sum(1 for j in jobs if j.get("status") == "done")
                n_pass = 0
                goal_ok = 0
                for job in jobs:
                    mp = Path(job.get("output_dir", "")) / "metrics.json"
                    if not mp.is_file():
                        continue
                    try:
                        m = json.loads(mp.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if m.get("pass"):
                        n_pass += 1
                    g = m.get("goal_stop_precision") or {}
                    if g.get("status") == "ok":
                        goal_ok += 1

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Jobs done", f"{n_done}/{len(jobs)}")
                k2.metric("Pass", f"{n_pass}/{max(n_done, 1)}")
                k3.metric("Stops on map", f"{len(stop_points)}")
                k4.metric("Goal reached", f"{goal_ok}/{max(n_done, 1)}")

                lat_tol, lon_tol = 2.0, 2.0
                precision_rows: list[dict[str, Any]] = []
                for job in jobs:
                    mp = Path(job.get("output_dir", "")) / "metrics.json"
                    if not mp.is_file():
                        continue
                    try:
                        metrics = json.loads(mp.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    thr = metrics.get("thresholds") or {}
                    if isinstance(thr.get("goal_lateral_tolerance_m"), (int, float)):
                        lat_tol = float(thr["goal_lateral_tolerance_m"])
                    if isinstance(thr.get("goal_longitudinal_tolerance_m"), (int, float)):
                        lon_tol = float(thr["goal_longitudinal_tolerance_m"])
                    goal = metrics.get("goal_stop_precision", {}) or {}
                    bag = str(job.get("bag_key") or "")
                    precision_rows.append(
                        {
                            "id": bag,
                            "scenario": bag,
                            "model": job.get("model_name") or "",
                            "pass": bool(metrics.get("pass")),
                            "goal_status": goal.get("status"),
                            "pos_m": goal.get("position_error_m"),
                            "lat_m": goal.get("lateral_m"),
                            "lon_m": goal.get("longitudinal_m"),
                            "abs_lat_m": goal.get("abs_lateral_m"),
                            "abs_lon_m": goal.get("abs_longitudinal_m"),
                            "heading_deg": goal.get("heading_error_deg"),
                            "samples": goal.get("sample_count"),
                            "flagged": goal.get("flagged"),
                        }
                    )

                # Build compact HTML table for the unified panel
                def _cell(v: Any, nd: int = 2) -> str:
                    if isinstance(v, (int, float)) and v == v:  # not NaN
                        return f"{float(v):.{nd}f}"
                    return "—"

                table_rows = []
                for r in precision_rows:
                    cls = "pass" if r["pass"] else "fail"
                    label = "PASS" if r["pass"] else "FAIL"
                    table_rows.append(
                        "<tr>"
                        f"<td><b>{r['id']}</b></td>"
                        f"<td>{r['scenario']}</td>"
                        f"<td>{r['model']}</td>"
                        f"<td class='{cls}'>{label}</td>"
                        f"<td>{r['goal_status']}</td>"
                        f"<td>{_cell(r['pos_m'])}</td>"
                        f"<td>{_cell(r['lat_m'])}</td>"
                        f"<td>{_cell(r['lon_m'])}</td>"
                        f"<td>{_cell(r['heading_deg'])}</td>"
                        "</tr>"
                    )
                precision_table = (
                    "<table class='summary'>"
                    "<tr><th>ID</th><th>Scenario</th><th>Model</th><th>Result</th>"
                    "<th>Status</th><th>pos</th><th>lat</th><th>lon</th><th>yaw</th></tr>"
                    + "".join(table_rows)
                    + "</table>"
                )
                scatter = render_stop_scatter_section(
                    stop_points,
                    lat_tol_m=lat_tol,
                    lon_tol_m=lon_tol,
                    precision_table_html=precision_table if precision_rows else "",
                )

                _SCATTER_CSS = """
<style>
body{margin:0;background:#0b0f14;color:#e7ecf1;font-family:system-ui,sans-serif}
.goal-stop-panel{background:#141c26;border:1px solid #2a3440;border-radius:10px;padding:14px 16px}
.goal-stop-panel>h2{margin:0 0 6px;font-size:1.15rem}
.scatter-card{background:#1a2330;border:1px solid #2a3440;border-radius:8px;padding:12px;margin:8px 0}
.scatter-row{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
.hint{color:#9eb4c8;font-size:12px}
.precision-table-wrap{margin-top:12px}
.precision-table-wrap h3{margin:8px 0 4px;font-size:14px}
table.summary{border-collapse:collapse;font-size:12px;color:#e7ecf1;width:100%}
table.summary th,table.summary td{border:1px solid #2a3440;padding:4px 8px;text-align:left}
td.pass{color:#7dffa6}td.fail{color:#ffb4b4}
.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.swatch.pt-pass{background:#3dffa6}.swatch.pt-fail{background:#ff6b6b}
.swatch.avg{background:#ffd866}.avg-row td{color:#ffd866}
svg.stop-scatter .plot-bg{fill:#0f1419}
svg.stop-scatter .grid{stroke:#243040}
svg.stop-scatter .axis{stroke:#6a7f96;stroke-width:1.5}
svg.stop-scatter .tol-box{fill:rgba(125,170,255,.08);stroke:#7daaff;stroke-dasharray:6 4}
svg.stop-scatter .goal-arrow{fill:#c8d6e8;stroke:#7daaff;stroke-width:1.2}
svg.stop-scatter .goal-label,.pt-id,.avg-label{fill:#e7ecf1;font-size:12px;font-weight:700}
svg.stop-scatter .pt-pass{fill:#3dffa6;stroke:#0f1419}
svg.stop-scatter .pt-fail{fill:#ff6b6b;stroke:#0f1419}
svg.stop-scatter .avg-ring{fill:none;stroke:#ffd866;stroke-width:2.5}
svg.stop-scatter .avg-arrow{fill:#ffd866;stroke:#0f1419}
svg.stop-scatter .triad-x{stroke:#ff6b6b;fill:#ff6b6b;stroke-width:2.5}
svg.stop-scatter .triad-y{stroke:#3dffa6;fill:#3dffa6;stroke-width:2.5}
svg.stop-scatter .triad-origin{fill:#e7ecf1}
svg.stop-scatter .triad-label{fill:#c9d4e0;font-size:11px}
svg.stop-scatter .tick{fill:#6a7f96;font-size:10px}
svg.stop-scatter .axis-title,svg.stop-scatter .axis-title-y{fill:#9eb4c8;font-size:11px;text-anchor:middle}
</style>
"""
                panel_height = 720 if stop_points else 220
                if precision_rows:
                    panel_height = min(1100, panel_height + 40 + 28 * len(precision_rows))
                st.components.v1.html(_SCATTER_CSS + scatter, height=panel_height, scrolling=True)

                if precision_rows:
                    with st.expander("Stop precision table (sortable)", expanded=False):
                        st.dataframe(precision_rows, use_container_width=True, hide_index=True)

                dash = run_dir / "dashboard.html"
                # Keep HTML dashboard in sync for the expander / external browser
                try:
                    build_dashboard(manifest, dash)
                except OSError:
                    pass
                with st.expander("Full HTML dashboard (PASS/FAIL matrix, rates, videos)", expanded=False):
                    if dash.is_file():
                        st.caption(
                            "Same run as above — scenario × model matrix and live progress. "
                            f"File: `{dash}`"
                        )
                        st.iframe(dash, height=900)
                    else:
                        st.info("dashboard.html not found for this run.")

                if st.button("Refresh results", key="results_refresh"):
                    st.rerun()



if __name__ == "__main__":
    main()
