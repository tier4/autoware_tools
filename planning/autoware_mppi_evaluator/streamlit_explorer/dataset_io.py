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
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

from autoware_mppi_evaluator.mcap_reader import SynchronizedFrame
import yaml

SCHEMA_VERSION = 2
STATIC_MESSAGE_KEYS = ("lanelet_map", "route")


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


def _atomic_binary_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _environment_id(messages: Dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for key in STATIC_MESSAGE_KEYS:
        data = messages[key]
        digest.update(key.encode("utf-8"))
        digest.update(len(data).to_bytes(8, byteorder="big"))
        digest.update(data)
    return digest.hexdigest()


def _write_environment(root: Path, messages: Dict[str, bytes]) -> Tuple[str, Dict[str, Any]]:
    environment_id = _environment_id(messages)
    environment: Dict[str, Any] = {"messages": {}}
    for key in STATIC_MESSAGE_KEYS:
        data = messages[key]
        message_hash = _sha256(data)
        relative_path = Path("blobs") / f"{message_hash}.cdr.gz"
        blob_path = root / relative_path
        if not blob_path.exists():
            _atomic_binary_write(blob_path, gzip.compress(data, compresslevel=6, mtime=0))
        environment["messages"][key] = {
            "path": str(relative_path),
            "encoding": "cdr+gzip",
            "sha256": message_hash,
            "size": len(data),
        }
    return environment_id, environment


def _empty_manifest(synchronizer) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "topics": dict(synchronizer.topic_map),
        "types": dict(synchronizer.topic_types),
        "environments": {},
        "frames": [],
    }


def save_frame(frame, synchronizer, dataset_directory: str, tags: Iterable[str]) -> Path:
    root = Path(dataset_directory).expanduser().resolve()
    frame_path = root / "curated" / f"{frame.frame_id}.json"
    missing_static = [key for key in STATIC_MESSAGE_KEYS if key not in frame.messages]
    if missing_static:
        raise ValueError(f"The frame is missing static messages: {missing_static}")

    manifest_path = root / "manifest.yaml"
    manifest = _empty_manifest(synchronizer)
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream) or {}
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported dataset schema in {manifest_path}")
        for key, current in (
            ("topics", dict(synchronizer.topic_map)),
            ("types", dict(synchronizer.topic_types)),
        ):
            if manifest.get(key) != current:
                raise ValueError(f"The dataset {key} do not match those in the selected recording")

    environment_id, environment = _write_environment(root, frame.messages)
    environments = dict(manifest.get("environments", {}))
    environments[environment_id] = environment
    selected_tags = sorted({tag for tag in tags if tag})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_ns": frame.timestamp_ns,
        "environment_id": environment_id,
        "ages_ms": frame.ages_ms,
        "messages_cdr_base64": {
            key: base64.b64encode(value).decode("ascii")
            for key, value in frame.messages.items()
            if key not in STATIC_MESSAGE_KEYS
        },
    }
    _atomic_text_write(frame_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    entries = [entry for entry in manifest.get("frames", []) if entry.get("id") != frame.frame_id]
    entries.append(
        {
            "id": frame.frame_id,
            "path": str(frame_path.relative_to(root)),
            "environment_id": environment_id,
            "tags": selected_tags,
            "timestamp_ns": frame.timestamp_ns,
        }
    )
    entries.sort(key=lambda entry: entry["timestamp_ns"])
    manifest["environments"] = environments
    manifest["frames"] = entries
    _atomic_text_write(manifest_path, yaml.safe_dump(manifest, sort_keys=False))
    return frame_path


def _load_environment(
    root: Path, environment_id: str, manifest: Dict[str, Any]
) -> Dict[str, bytes]:
    try:
        description = manifest["environments"][environment_id]
    except KeyError as error:
        raise ValueError(f"Unknown dataset environment: {environment_id}") from error

    messages = {}
    for key in STATIC_MESSAGE_KEYS:
        message = description["messages"][key]
        if message.get("encoding") != "cdr+gzip":
            raise ValueError(f"Unsupported encoding for {key}: {message.get('encoding')}")
        blob_path = root / message["path"]
        with gzip.open(blob_path, "rb") as stream:
            data = stream.read()
        if len(data) != int(message["size"]) or _sha256(data) != message["sha256"]:
            raise ValueError(f"Dataset blob integrity check failed: {blob_path}")
        messages[key] = data
    return messages


def _load_frames(root: Path, manifest: Dict[str, Any]) -> List[SynchronizedFrame]:
    frames = []
    environment_cache: Dict[str, Dict[str, bytes]] = {}
    for index, entry in enumerate(manifest.get("frames", [])):
        frame_path = root / entry["path"]
        with frame_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported frame schema in {frame_path}")
        environment_id = payload.get("environment_id")
        if environment_id != entry.get("environment_id"):
            raise ValueError(f"Environment mismatch in {frame_path}")
        if environment_id not in environment_cache:
            environment_cache[environment_id] = _load_environment(root, environment_id, manifest)
        messages = dict(environment_cache[environment_id])
        messages.update(
            {
                key: base64.b64decode(value, validate=True)
                for key, value in payload["messages_cdr_base64"].items()
            }
        )
        timestamp_ns = int(payload["timestamp_ns"])
        if timestamp_ns != int(entry["timestamp_ns"]):
            raise ValueError(f"Timestamp mismatch in {frame_path}")
        frames.append(
            SynchronizedFrame(
                index=index,
                timestamp_ns=timestamp_ns,
                messages=messages,
                ages_ms=payload.get("ages_ms", {}),
                warnings=[],
                is_usable=True,
            )
        )
    return frames


def load_dataset(dataset_path: str) -> List[SynchronizedFrame]:
    requested_path = Path(dataset_path).expanduser().resolve()
    manifest_path = requested_path if requested_path.is_file() else requested_path / "manifest.yaml"
    root = manifest_path.parent
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream) or {}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset schema in {manifest_path}")
    frames = _load_frames(root, manifest)
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
