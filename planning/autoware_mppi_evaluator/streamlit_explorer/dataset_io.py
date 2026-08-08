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

"""Store synchronized MPPI frames without losing ROS CDR data."""

import base64
import json
import os
from pathlib import Path
import tempfile
from typing import Dict
from typing import Iterable
from typing import List

from mcap_reader import SynchronizedFrame
import yaml

SCHEMA_VERSION = 1


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_frame(frame, synchronizer, dataset_directory: str, tags: Iterable[str]) -> Path:
    root = Path(dataset_directory).expanduser().resolve()
    frame_path = root / "curated" / f"{frame.frame_id}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "tags": sorted({tag for tag in tags if tag}),
        "topics": synchronizer.topic_map,
        "types": synchronizer.topic_types,
        "ages_ms": frame.ages_ms,
        "messages_cdr_base64": {
            key: base64.b64encode(value).decode("ascii") for key, value in frame.messages.items()
        },
    }
    _atomic_text_write(frame_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    manifest_path = root / "manifest.yaml"
    manifest: Dict[str, List[Dict]] = {"schema_version": SCHEMA_VERSION, "frames": []}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream) or manifest
    entries = [entry for entry in manifest.get("frames", []) if entry.get("id") != frame.frame_id]
    entries.append(
        {
            "id": frame.frame_id,
            "path": str(frame_path.relative_to(root)),
            "tags": payload["tags"],
            "timestamp_ns": frame.timestamp_ns,
        }
    )
    entries.sort(key=lambda entry: entry["timestamp_ns"])
    manifest = {"schema_version": SCHEMA_VERSION, "frames": entries}
    _atomic_text_write(manifest_path, yaml.safe_dump(manifest, sort_keys=False))
    return frame_path


def load_dataset(dataset_path: str) -> List[SynchronizedFrame]:
    requested_path = Path(dataset_path).expanduser().resolve()
    manifest_path = requested_path if requested_path.is_file() else requested_path / "manifest.yaml"
    root = manifest_path.parent
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream) or {}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset schema in {manifest_path}")

    frames = []
    for index, entry in enumerate(manifest.get("frames", [])):
        frame_path = root / entry["path"]
        with frame_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported frame schema in {frame_path}")
        messages = {
            key: base64.b64decode(value, validate=True)
            for key, value in payload["messages_cdr_base64"].items()
        }
        frames.append(
            SynchronizedFrame(
                index=index,
                timestamp_ns=int(payload["timestamp_ns"]),
                messages=messages,
                ages_ms=payload.get("ages_ms", {}),
                warnings=[],
                is_usable=True,
            )
        )
    frames.sort(key=lambda frame: frame.timestamp_ns)
    return [
        SynchronizedFrame(
            index=index,
            timestamp_ns=frame.timestamp_ns,
            messages=frame.messages,
            ages_ms=frame.ages_ms,
            warnings=frame.warnings,
            is_usable=frame.is_usable,
        )
        for index, frame in enumerate(frames)
    ]
