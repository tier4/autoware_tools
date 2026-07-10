"""Resolve model_config paths (param YAML file or ONNX model directory)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def default_param_template_path() -> Path:
    try:
        prefix = subprocess.check_output(
            ["ros2", "pkg", "prefix", "autoware_launch"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        path = (
            Path(prefix)
            / "share/autoware_launch/config/planning/neural_net_planner/diffusion_planner.param.yaml"
        )
        if path.is_file():
            return path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    fallback = Path(
        "/opt/autoware/share/autoware_launch/config/planning/neural_net_planner/diffusion_planner.param.yaml"
    )
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        "Diffusion planner param template not found. "
        "Source autoware_launch or set param_template_path in the pipeline config."
    )


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
    - a diffusion_planner.param.yaml file, or
    - a model directory containing diffusion_planner.onnx + args.json
      (generates a cached param YAML from the autoware_launch template).
    """
    path = Path(model_config).expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    opts = model_opts or {}

    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"model_config not found: {path}")

    nested = path / "diffusion_planner.param.yaml"
    if nested.is_file():
        return nested

    onnx = path / "diffusion_planner.onnx"
    args_json = path / "args.json"
    if not onnx.is_file():
        raise FileNotFoundError(
            f"model_config directory has no diffusion_planner.param.yaml or "
            f"diffusion_planner.onnx: {path}"
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
