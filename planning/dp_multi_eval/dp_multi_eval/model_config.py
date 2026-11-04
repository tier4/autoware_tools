"""Resolve model_config paths (param YAML file or ONNX model directory)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


ML_PLANNER_ONNX = "ml_planner.onnx"


def _launch_param_path(filename: str) -> Path:
    rel = Path("share/autoware_launch/config/planning/neural_net_planner") / filename
    try:
        prefix = subprocess.check_output(
            ["ros2", "pkg", "prefix", "autoware_launch"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        path = Path(prefix) / rel
        if path.is_file():
            return path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    fallback = Path("/opt/autoware") / rel
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"{filename} not found. Source autoware_launch or set param_template_path "
        "in the pipeline config."
    )


def default_param_template_path() -> Path:
    return _launch_param_path("diffusion_planner.param.yaml")


def default_ml_param_template_path() -> Path:
    return _launch_param_path("ml_planner.param.yaml")


def is_ml_planner_source(path: Path) -> bool:
    """True for a lone ml_planner.onnx, or a directory that only has that file."""
    path = Path(path)
    if path.is_file():
        return path.name == ML_PLANNER_ONNX
    if path.is_dir() and (path / ML_PLANNER_ONNX).is_file():
        return not (path / "diffusion_planner.onnx").is_file()
    return False


def patch_param_yaml(
    template_path: Path,
    output_path: Path,
    onnx_path: str,
    args_path: str,
    *,
    ignore_neighbors: bool | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for line in template_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("onnx_model_path:"):
            lines.append(f"    onnx_model_path: {onnx_path}")
        elif stripped.startswith("args_path:"):
            lines.append(f"    args_path: {args_path}")
        elif stripped.startswith("ignore_neighbors:") and ignore_neighbors is not None:
            lines.append(f"    ignore_neighbors: {str(ignore_neighbors).lower()}")
        else:
            lines.append(line)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _patch_ml_param_yaml(template_path: Path, output_path: Path, onnx_path: Path) -> None:
    """Point ml_planner.param.yaml at one ml_planner.onnx file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    replacements = {
        "base_model_directory": str(onnx_path.parent),
        "onnx_model_filename": onnx_path.name,
    }
    lines: list[str] = []
    for line in template_path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip(" ")
        key = stripped.split(":", 1)[0]
        if key in replacements:
            indent = line[: len(line) - len(stripped)]
            lines.append(f"{indent}{key}: {replacements[key]}")
        else:
            lines.append(line)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ml_param(onnx_path: Path, cache_dir: Path, model_name: str | None) -> Path:
    fallback_name = onnx_path.parent.name
    safe_name = (model_name or fallback_name).replace("/", "_")
    target = cache_dir / f"{safe_name}.param.yaml"
    _patch_ml_param_yaml(default_ml_param_template_path(), target, onnx_path)
    return target


def resolve_model_config(
    model_config: str | Path,
    *,
    cache_dir: Path,
    model_name: str | None = None,
    param_template: Path | None = None,
    model_opts: dict[str, Any] | None = None,
) -> Path:
    """
    Accept either:
    - a planner param YAML file, or
    - ml_planner.onnx (single-file model, no args.json), or
    - a model directory containing diffusion_planner.onnx + args.json,
      or only ml_planner.onnx.
    Directories are turned into a cached param YAML from the autoware_launch template.
    """
    path = Path(model_config).expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    opts = model_opts or {}

    if path.is_file() and path.name == ML_PLANNER_ONNX:
        return _write_ml_param(path, cache_dir, model_name)

    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"model_config not found: {path}")

    nested = path / "diffusion_planner.param.yaml"
    if nested.is_file():
        return nested

    ml_onnx = path / ML_PLANNER_ONNX
    onnx = path / "diffusion_planner.onnx"
    if ml_onnx.is_file() and not onnx.is_file():
        return _write_ml_param(ml_onnx, cache_dir, model_name)

    args_json = path / "args.json"
    if not onnx.is_file():
        raise FileNotFoundError(
            f"model_config directory has no diffusion_planner.param.yaml, "
            f"diffusion_planner.onnx, or {ML_PLANNER_ONNX}: {path}"
        )
    if not args_json.is_file():
        raise FileNotFoundError(f"args.json missing next to ONNX model: {path}")

    template = param_template or default_param_template_path()
    safe_name = (model_name or path.name).replace("/", "_")
    target = cache_dir / f"{safe_name}.param.yaml"
    ignore = opts.get("ignore_neighbors")
    patch_param_yaml(
        template,
        target,
        str(onnx),
        str(args_json),
        ignore_neighbors=ignore if isinstance(ignore, bool) else None,
    )
    return target
