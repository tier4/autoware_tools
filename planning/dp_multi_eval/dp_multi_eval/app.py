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

from dp_multi_eval.build_dashboard import build_dashboard
from dp_multi_eval.path_picker import pick_folder
from dp_multi_eval.process_utils import cleanup_evaluation_processes, stop_pid_group

# Soft dependency: streamlit
try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "streamlit is required for the GUI. Install with: pip install streamlit"
    ) from exc


DEFAULT_CONFIG = Path.home() / "dp_multi_eval_pipeline.yaml"
STATE_FILE_NAME = ".gui_run_state.json"
SESSION_KEYS = ("models_dir", "rosbag_dir", "results_root")


def _init_session(defaults: dict[str, str]) -> None:
    for key in SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = defaults.get(key, "")


def folder_input(
    label: str,
    session_key: str,
    *,
    picker_title: str,
    help_text: str = "",
) -> Path:
    col_path, col_btn = st.sidebar.columns([5, 1])
    with col_path:
        value = st.text_input(
            label,
            value=st.session_state.get(session_key, ""),
            key=f"input_{session_key}",
            help=help_text or None,
        )
    with col_btn:
        st.write("")
        if st.button("…", key=f"browse_{session_key}", help=f"Browse for {label.lower()}"):
            picked = pick_folder(picker_title, st.session_state.get(session_key, ""))
            if picked:
                st.session_state[session_key] = picked
                st.rerun()
    st.session_state[session_key] = value
    return Path(value).expanduser()


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
    rosbag_dir: Path,
    selected_bags: list[Path] | None,
) -> Path:
    results_root = Path(base.get("results_root", "/tmp/dp_multi_eval_results")) / run_name
    cfg = dict(base)
    cfg["models"] = models
    cfg["rosbag_dir"] = str(rosbag_dir)
    cfg["results_root"] = str(results_root)
    # If subset of bags selected, write a temp bag list file consumed by custom logic:
    # For simplicity regenerate full dir but document; orchestrator uses whole rosbag_dir.
    # Users selecting subset: copy manifest approach — write bag_whitelist
    if selected_bags is not None:
        cfg["bag_whitelist"] = [str(b) for b in selected_bags]
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
    _init_session(
        {
            "models_dir": str(base.get("models_dir", "/opt/autoware/mlmodels")),
            "rosbag_dir": str(base.get("rosbag_dir", "")),
            "results_root": str(base.get("results_root", "/tmp/dp_multi_eval_results")),
        }
    )
    models_dir = folder_input(
        "Models folder",
        "models_dir",
        picker_title="Select models folder",
        help_text="Folder with ONNX model dirs or diffusion_planner.param.yaml files.",
    )
    rosbag_dir = folder_input(
        "Scenarios folder",
        "rosbag_dir",
        picker_title="Select ROS bag scenarios folder",
        help_text="Parent folder of ID1, ID2, … scenario subfolders.",
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

        st.subheader("Choose scenarios")
        bags = discover_bag_dirs(rosbag_dir)
        if not bags and rosbag_dir:
            st.warning(
                f"No scenarios found under `{rosbag_dir}`. "
                "Each subfolder needs metadata.yaml or a .db3 file."
            )
        select_all = st.checkbox("Select all scenarios", value=False)
        selected_bags: list[Path] = []
        for bag in bags:
            try:
                label = str(bag.resolve().relative_to(rosbag_dir.resolve()))
            except ValueError:
                label = bag.name
            checked = select_all or st.checkbox(label, value=False, key=f"b_{bag}")
            if checked:
                selected_bags.append(bag)

        run_name = st.text_input("Run name", value=time.strftime("run_%Y%m%d_%H%M%S"))
        max_workers = st.number_input("Parallel workers", min_value=1, max_value=8, value=int(base.get("max_workers", 1)))

        state_path = results_root / STATE_FILE_NAME
        running, _state = gui_run_is_active(state_path)

        if running:
            render_stop_run_controls(state_path, key_prefix="setup")
            st.info("A run is already in progress. See the Live Progress tab.")
        elif st.button("Start evaluation", type="primary"):
            if not selected_models:
                st.error("Select at least one model.")
            elif not selected_bags:
                st.error("Select at least one scenario.")
            else:
                cfg_base = dict(base)
                cfg_base["results_root"] = str(results_root)
                cfg_base["max_workers"] = int(max_workers)
                cfg_base["map_path"] = cfg_base.get("map_path", "/opt/autoware/maps")
                # Temporary rosbag_dir containing only selected bags via symlink farm
                run_bag_root = results_root / run_name / "_selected_bags"
                if run_bag_root.exists():
                    import shutil

                    shutil.rmtree(run_bag_root)
                run_bag_root.mkdir(parents=True)
                for bag in selected_bags:
                    link = run_bag_root / bag.name
                    if not link.exists():
                        link.symlink_to(bag, target_is_directory=bag.is_dir())
                cfg_path = write_run_config(
                    cfg_base,
                    run_name=run_name,
                    models=selected_models,
                    rosbag_dir=run_bag_root,
                    selected_bags=selected_bags,
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
                    use_container_width=True,
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
                st.dataframe(rows, use_container_width=True)

            if st.button("Refresh now"):
                st.rerun()
            st.caption("This page does not auto-refresh in all Streamlit versions — click Refresh.")

    with tab_results:
        st.subheader("Results dashboard")
        render_stop_run_controls(results_root / STATE_FILE_NAME, key_prefix="results")
        dashboards = sorted(results_root.glob("*/dashboard.html"), key=lambda p: p.stat().st_mtime)
        if not dashboards:
            st.info("No results yet — start an evaluation from the Setup tab.")
        else:
            dash = dashboards[-1]
            st.write(f"Showing `{dash}`")
            st.caption(
                "The dashboard auto-refreshes every 5s while jobs are pending or running. "
                "Click Refresh now on Live Progress to reload Streamlit."
            )
            html = dash.read_text(encoding="utf-8")
            st.components.v1.html(html, height=900, scrolling=True)


if __name__ == "__main__":
    main()
