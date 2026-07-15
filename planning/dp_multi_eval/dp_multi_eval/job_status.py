"""Job status file helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LEGACY_PREVIEW_NAME = "preview.mp4"
_VIDEOS_DIRNAME = "videos"
_UNSAFE_FILENAME = re.compile(r"[^\w.\-]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_job_status(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "job_status.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def preview_video_filename(job: dict[str, Any]) -> str:
    """Unique per-job preview name so flat cloud uploads do not collide."""
    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        model = str(job.get("model_name") or "model")
        key = str(job.get("bag_key") or job.get("scenario_key") or "scenario")
        job_id = f"{model}__{key}"
    safe = _UNSAFE_FILENAME.sub("_", job_id).strip("._") or "job"
    return f"{safe}_preview.mp4"


def resolve_results_root(
    job: dict[str, Any],
    results_root: Path | str | None = None,
) -> Path | None:
    if results_root is not None:
        return Path(results_root)
    if job.get("results_root"):
        return Path(str(job["results_root"]))
    return None


def videos_dir(results_root: Path | str) -> Path:
    return Path(results_root) / _VIDEOS_DIRNAME


def preview_video_path(
    job: dict[str, Any],
    *,
    results_root: Path | str | None = None,
    for_write: bool = False,
) -> Path:
    """Preferred path: ``{results_root}/videos/{job_id}_preview.mp4``.

    When reading, also accepts older per-job paths:
    ``{output_dir}/{job_id}_preview.mp4`` and ``{output_dir}/preview.mp4``.
    """
    name = preview_video_filename(job)
    root = resolve_results_root(job, results_root)
    out_dir = Path(job["output_dir"])

    shared = (videos_dir(root) / name) if root is not None else None
    per_job_unique = out_dir / name
    per_job_legacy = out_dir / _LEGACY_PREVIEW_NAME

    if for_write:
        if shared is not None:
            shared.parent.mkdir(parents=True, exist_ok=True)
            return shared
        out_dir.mkdir(parents=True, exist_ok=True)
        return per_job_unique

    for candidate in (shared, per_job_unique, per_job_legacy):
        if candidate is not None and candidate.is_file():
            return candidate
    return shared if shared is not None else per_job_unique
