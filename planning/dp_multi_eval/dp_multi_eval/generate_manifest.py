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

"""Phase 4a — generate job manifest from models + bags/scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.model_config import planning_setting_for, resolve_model_config
from dp_multi_eval.scenario_sim import (
    SCENARIO_EXTENSIONS,
    discover_scenarios,
    scenario_key,
)


def default_param_cache_dir() -> Path:
    return Path.home() / ".cache" / "dp_multi_eval" / "model_params"


def discover_bags(rosbag_dir: Path) -> list[Path]:
    rosbag_dir = rosbag_dir.expanduser().resolve()
    bags: list[Path] = []
    if not rosbag_dir.is_dir():
        raise FileNotFoundError(f"rosbag_dir missing: {rosbag_dir}")

    # rosbag_dir itself is one bag (metadata.yaml + chunked *.db3 files)
    if (rosbag_dir / "metadata.yaml").exists():
        return [rosbag_dir]

    # Prefer directories that contain metadata.yaml or *.db3
    for child in sorted(rosbag_dir.iterdir()):
        if child.is_dir():
            if (child / "metadata.yaml").exists() or list(child.glob("*.db3")):
                bags.append(child)
            else:
                # One level deeper (ID folders)
                for sub in sorted(child.iterdir()):
                    if sub.is_dir() and (
                        (sub / "metadata.yaml").exists() or list(sub.glob("*.db3"))
                    ):
                        bags.append(sub)
        elif child.suffix == ".db3":
            bags.append(child)
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for b in bags:
        b = b.resolve()
        if b not in seen:
            seen.add(b)
            unique.append(b)
    return unique


def _planning_setting(model: dict[str, Any]) -> str | None:
    """The planner a model's jobs launch with when it is not the run's default:
    an explicit ``planning_setting``, or "ml_planner" for an ML planner model."""
    explicit = str(model.get("planning_setting") or "").strip()
    if explicit:
        return explicit
    return "ml_planner" if planning_setting_for(model["model_config"]) == "ml_planner" else None


def bag_key(bag: Path, rosbag_dir: Path) -> str:
    try:
        rel = bag.resolve().relative_to(rosbag_dir.resolve())
        if rel == Path("."):  # rosbag_dir is itself the bag
            return bag.resolve().name
        return str(rel.with_suffix("") if rel.suffix else rel)
    except ValueError:
        return bag.stem if bag.is_file() else bag.name


def generate_manifest(
    *,
    models: list[dict[str, Any]],
    results_root: Path,
    mode: str = "reproducer",
    rosbag_dir: Path | None = None,
    scenario_dir: Path | None = None,
    bags: list[Path] | None = None,
    scenarios: list[Path] | None = None,
    param_template: Path | None = None,
    scenario_repeat_count: int = 1,
) -> dict[str, Any]:
    """Build jobs for reproducer (rosbags) or scenario_simulator (scenario files)."""
    mode = (mode or "reproducer").strip().lower()
    if mode not in ("reproducer", "scenario_simulator"):
        raise ValueError(f"unsupported mode: {mode}")

    results_root = results_root.expanduser().resolve()
    param_cache = default_param_cache_dir()
    jobs: list[dict[str, Any]] = []
    repeats = max(1, int(scenario_repeat_count or 1))

    if mode == "reproducer":
        if rosbag_dir is None:
            raise ValueError("rosbag_dir is required for mode=reproducer")
        rosbag_dir = rosbag_dir.expanduser().resolve()
        bag_list = bags or discover_bags(rosbag_dir)
        if not bag_list:
            print(
                f"[warn] No ROS bags found under {rosbag_dir}. "
                "Each scenario folder needs metadata.yaml or a .db3 file."
            )
        for model in models:
            name = str(model["name"]).rstrip("/")
            model_config = str(
                resolve_model_config(
                    model["model_config"],
                    cache_dir=param_cache,
                    model_name=name,
                    param_template=param_template,
                    model_opts=model,
                )
            )
            setting = _planning_setting(model)
            for bag in bag_list:
                key = bag_key(bag, rosbag_dir)
                # Departures proved near-threshold stochastic — the same model
                # and config swung one bag 44% -> 100% -> 100% stub across
                # identical runs — so a single run per bag is a coin flip, not a
                # verdict. Repeats existed only for scenario mode; bags get the
                # same treatment: N jobs, __runN suffixes, separate outputs.
                for run_i in range(1, repeats + 1):
                    if repeats == 1:
                        job_id = f"{name}__{key.replace('/', '__')}"
                        out = results_root / name / key
                    else:
                        job_id = f"{name}__{key.replace('/', '__')}__run{run_i}"
                        out = results_root / name / f"{key}__run{run_i}"
                    jobs.append(
                        {
                            "job_id": job_id,
                            "mode": mode,
                            "model_name": name,
                            "model_config_path": model_config,
                            **({"planning_setting": setting} if setting else {}),
                            "bag_path": str(bag.resolve()),
                            "scenario_path": None,
                            "bag_key": key,
                            "output_dir": str(out),
                            "status": "pending",
                            "error": None,
                            "attempts": 0,
                            "repeat_index": run_i,
                            "repeat_count": repeats,
                        }
                    )
        return {
            "version": 2,
            "mode": mode,
            "rosbag_dir": str(rosbag_dir),
            "scenario_dir": None,
            "results_root": str(results_root),
            "jobs": jobs,
        }

    # scenario_simulator
    if scenario_dir is None:
        raise ValueError("scenario_dir is required for mode=scenario_simulator")
    scenario_dir = scenario_dir.expanduser().resolve()
    scenario_list = scenarios or discover_scenarios(scenario_dir)
    if not scenario_list:
        print(
            f"[warn] No scenario files found under {scenario_dir}. "
            f"Expected extensions: {', '.join(SCENARIO_EXTENSIONS)}"
        )
    if repeats > 1:
        print(f"[info] scenario_repeat_count={repeats} → {repeats} jobs per scenario file")
    for model in models:
        name = str(model["name"]).rstrip("/")
        model_config = str(
            resolve_model_config(
                model["model_config"],
                cache_dir=param_cache,
                model_name=name,
                param_template=param_template,
                model_opts=model,
            )
        )
        setting = _planning_setting(model)
        for scenario in scenario_list:
            base_key = scenario_key(scenario, scenario_dir)
            for run_i in range(1, repeats + 1):
                if repeats == 1:
                    key = base_key
                else:
                    key = f"{base_key}/run{run_i:02d}"
                job_id = f"{name}__{key.replace('/', '__')}"
                out = results_root / name / key
                jobs.append(
                    {
                        "job_id": job_id,
                        "mode": mode,
                        "model_name": name,
                        "model_config_path": model_config,
                        **({"planning_setting": setting} if setting else {}),
                        "bag_path": None,
                        "scenario_path": str(scenario.resolve()),
                        "bag_key": key,
                        "repeat_index": run_i,
                        "repeat_count": repeats,
                        "output_dir": str(out),
                        "status": "pending",
                        "error": None,
                        "attempts": 0,
                    }
                )
    return {
        "version": 2,
        "mode": mode,
        "rosbag_dir": None,
        "scenario_dir": str(scenario_dir),
        "results_root": str(results_root),
        "scenario_repeat_count": repeats,
        "jobs": jobs,
    }


def load_pipeline_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.expanduser().read_text(encoding="utf-8")) or {}
    return raw


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 4: generate evaluation manifest.json")
    p.add_argument("-c", "--config", type=Path, required=True, help="Pipeline YAML")
    p.add_argument("-o", "--output", type=Path, default=None, help="manifest.json path")
    args = p.parse_args(argv)

    cfg = load_pipeline_config(args.config)
    models = cfg.get("models") or []
    if not models:
        print("[error] config.models is empty")
        return 1
    for m in models:
        if "name" not in m or "model_config" not in m:
            print("[error] each model needs name + model_config")
            return 1

    mode = str(cfg.get("mode", "reproducer")).strip().lower()
    try:
        if mode == "scenario_simulator":
            scenario_dir = cfg.get("scenario_dir")
            if not scenario_dir:
                print("[error] mode=scenario_simulator requires scenario_dir")
                return 1
            manifest = generate_manifest(
                models=models,
                results_root=Path(cfg["results_root"]),
                mode=mode,
                scenario_dir=Path(scenario_dir),
                param_template=(
                    Path(cfg["param_template_path"]).expanduser()
                    if cfg.get("param_template_path")
                    else None
                ),
                scenario_repeat_count=int(cfg.get("scenario_repeat_count", 1)),
            )
        else:
            if not cfg.get("rosbag_dir"):
                print("[error] mode=reproducer requires rosbag_dir")
                return 1
            manifest = generate_manifest(
                models=models,
                results_root=Path(cfg["results_root"]),
                mode="reproducer",
                rosbag_dir=Path(cfg["rosbag_dir"]),
                param_template=(
                    Path(cfg["param_template_path"]).expanduser()
                    if cfg.get("param_template_path")
                    else None
                ),
                # The scenario branch always forwarded this; the bag branch
                # dropped it, so repeats silently stayed at 1 from the CLI.
                scenario_repeat_count=int(cfg.get("scenario_repeat_count", 1)),
            )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[error] {exc}")
        return 1

    out = args.output or Path(cfg["results_root"]).expanduser() / "manifest.json"
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {len(manifest['jobs'])} jobs (mode={mode}) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
