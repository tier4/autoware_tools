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
        f"{filename} template not found. "
        "Source autoware_launch or set param_template_path in the pipeline config."
    )


def default_param_template_path() -> Path:
    return _launch_param_path("diffusion_planner.param.yaml")


def default_ml_param_template_path() -> Path:
    return _launch_param_path("ml_planner.param.yaml")


def _param_kind(text: str) -> str | None:
    """Which planner a param YAML configures: the diffusion planner sets
    args_path, the ML planner (one ONNX, no args.json) does not."""
    if "args_path:" in text:
        return "diffusion_planner"
    ml_markers = ("ml_planner", "onnx_model_filename:", "remap_unsupported_objects_to_pedestrian:")
    if any(m in text for m in ml_markers):
        return "ml_planner"
    return None


def planning_setting_for(model_config: str | Path) -> str:
    """The launch ``planning_setting`` a model needs: "ml_planner" for an
    ml_planner.onnx (or its folder, or its param YAML), else "diffusion_planner"."""
    path = Path(model_config).expanduser()
    if path.is_dir():
        if (path / "diffusion_planner.onnx").is_file() or (
            path / "diffusion_planner.param.yaml"
        ).is_file():
            return "diffusion_planner"
        return "ml_planner" if (path / ML_PLANNER_ONNX).is_file() else "diffusion_planner"
    if path.name == ML_PLANNER_ONNX:
        return "ml_planner"
    if path.suffix == ".yaml" and path.is_file():
        return _param_kind(path.read_text(encoding="utf-8", errors="ignore")) or "diffusion_planner"
    return "diffusion_planner"


def discover_model_configs(models_dir: Path) -> list[tuple[str, Path]]:
    """Planner models under ``models_dir``, at any depth: a diffusion planner
    directory (diffusion_planner.onnx + args.json), an ml_planner.onnx file, or
    a planner param YAML. Other *.param.yaml files (the perception models next
    to the planners in autoware_data) are skipped."""
    if not models_dir.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for path in sorted(models_dir.rglob("*.param.yaml")):
        try:
            if _param_kind(path.read_text(encoding="utf-8", errors="ignore")) is None:
                continue
        except OSError:
            continue
        if path.name in ("diffusion_planner.param.yaml", "ml_planner.param.yaml"):
            found.append((path.parent.name, path))
        else:
            found.append((path.name.removesuffix(".param.yaml"), path))
    for onnx in sorted(models_dir.rglob("diffusion_planner.onnx")):
        if (onnx.parent / "args.json").is_file():
            found.append((onnx.parent.name, onnx.parent))
    for onnx in sorted(models_dir.rglob(ML_PLANNER_ONNX)):
        found.append((onnx.parent.name, onnx))
    return unique_models(found)


def unique_models(models: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """Keep the first entry per resolved path."""
    seen: set[Path] = set()
    out: list[tuple[str, Path]] = []
    for name, path in models:
        path = path.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        out.append((name, path))
    return out


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
        # Keep the template's indent: since the 2026-10-15 package update the
        # paths sit under model: / single_step_model:, not at the top level.
        indent = line[: len(line) - len(line.lstrip())]
        if stripped.startswith("onnx_model_path:"):
            lines.append(f"{indent}onnx_model_path: {onnx_path}")
        elif stripped.startswith("args_path:"):
            lines.append(f"{indent}args_path: {args_path}")
        elif stripped.startswith("ignore_neighbors:") and ignore_neighbors is not None:
            lines.append(f"{indent}ignore_neighbors: {str(ignore_neighbors).lower()}")
        else:
            lines.append(line)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def patch_ml_param_yaml(template_path: Path, output_path: Path, onnx_path: Path) -> None:
    """Point ml_planner.param.yaml at one ONNX file. The template sets
    model.onnx_model_path; older ones split it into base_model_directory +
    onnx_model_filename. Indent is kept."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "onnx_model_path": str(onnx_path),
        "base_model_directory": str(onnx_path.parent),
        "onnx_model_filename": onnx_path.name,
    }
    lines: list[str] = []
    for line in template_path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        key = stripped.split(":", 1)[0]
        if key in values and ":" in stripped:
            lines.append(f"{line[: len(line) - len(stripped)]}{key}: {values[key]}")
        else:
            lines.append(line)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_model_config(
    model_config: str | Path,
    *,
    cache_dir: Path,
    model_name: str | None = None,
    param_template: Path | None = None,
    ml_param_template: Path | None = None,
    model_opts: dict[str, Any] | None = None,
) -> Path:
    """
    Accept either:
    - a planner param YAML file (diffusion or ML planner), or
    - an ml_planner.onnx file, or a directory holding one (no args.json), or
    - a model directory containing diffusion_planner.onnx + args.json.
    ONNX inputs get a cached param YAML from the autoware_launch template.
    """
    path = Path(model_config).expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    opts = model_opts or {}

    ml_onnx = path if path.name == ML_PLANNER_ONNX else path / ML_PLANNER_ONNX
    if ml_onnx.is_file() and planning_setting_for(path) == "ml_planner":
        target = cache_dir / f"{(model_name or ml_onnx.parent.name).replace('/', '_')}.param.yaml"
        patch_ml_param_yaml(ml_param_template or default_ml_param_template_path(), target, ml_onnx)
        return target

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
            f"model_config directory has no diffusion_planner.param.yaml, "
            f"diffusion_planner.onnx or {ML_PLANNER_ONNX}: {path}"
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
