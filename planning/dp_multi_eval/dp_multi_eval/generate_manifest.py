"""Phase 4a — generate job manifest from models + bags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from dp_multi_eval.model_config import resolve_model_config


def default_param_cache_dir() -> Path:
    return Path.home() / ".cache" / "dp_multi_eval" / "model_params"


def discover_bags(rosbag_dir: Path) -> list[Path]:
    rosbag_dir = rosbag_dir.expanduser().resolve()
    bags: list[Path] = []
    if not rosbag_dir.is_dir():
        raise FileNotFoundError(f"rosbag_dir missing: {rosbag_dir}")

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


def bag_key(bag: Path, rosbag_dir: Path) -> str:
    try:
        rel = bag.resolve().relative_to(rosbag_dir.resolve())
        return str(rel.with_suffix("") if rel.suffix else rel)
    except ValueError:
        return bag.stem if bag.is_file() else bag.name


def generate_manifest(
    *,
    models: list[dict[str, Any]],
    rosbag_dir: Path,
    results_root: Path,
    bags: list[Path] | None = None,
    param_template: Path | None = None,
) -> dict[str, Any]:
    rosbag_dir = rosbag_dir.expanduser().resolve()
    results_root = results_root.expanduser().resolve()
    bag_list = bags or discover_bags(rosbag_dir)
    if not bag_list:
        print(
            f"[warn] No ROS bags found under {rosbag_dir}. "
            "Each scenario folder needs metadata.yaml or a .db3 file."
        )
    param_cache = default_param_cache_dir()
    jobs: list[dict[str, Any]] = []
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
        for bag in bag_list:
            key = bag_key(bag, rosbag_dir)
            job_id = f"{name}__{key.replace('/', '__')}"
            out = results_root / name / key
            jobs.append(
                {
                    "job_id": job_id,
                    "model_name": name,
                    "model_config_path": model_config,
                    "bag_path": str(bag.resolve()),
                    "bag_key": key,
                    "output_dir": str(out),
                    "status": "pending",
                    "error": None,
                    "attempts": 0,
                }
            )
    return {
        "version": 1,
        "rosbag_dir": str(rosbag_dir),
        "results_root": str(results_root),
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

    manifest = generate_manifest(
        models=models,
        rosbag_dir=Path(cfg["rosbag_dir"]),
        results_root=Path(cfg["results_root"]),
    )
    out = args.output or Path(cfg["results_root"]).expanduser() / "manifest.json"
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {len(manifest['jobs'])} jobs → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
