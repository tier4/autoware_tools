#!/usr/bin/env python3

# Copyright 2025 TIER IV, Inc.
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

from __future__ import annotations

from pathlib import Path

import rosbag2_py
from rosbag2_py import ConverterOptions
from rosbag2_py import SequentialReader
from rosbag2_py import StorageOptions


def resolve_rosbag_uri(bag_path: Path) -> tuple[str, str]:
    """Return rosbag2 (uri, storage_id) for a bag file or directory."""
    bag_path = bag_path.expanduser()

    if bag_path.is_file():
        if bag_path.suffix != ".db3":
            raise ValueError(f"Unsupported rosbag file type: {bag_path}")
        return str(bag_path), "sqlite3"

    if not bag_path.is_dir():
        raise FileNotFoundError(f"Rosbag path does not exist: {bag_path}")

    if (bag_path / "metadata.yaml").exists():
        storage_id = "mcap" if any(bag_path.glob("*.mcap")) else "sqlite3"
        return str(bag_path), storage_id

    db3_files = sorted(bag_path.glob("*.db3"))
    if len(db3_files) == 1:
        return str(db3_files[0]), "sqlite3"
    if len(db3_files) > 1:
        raise ValueError(
            f"Ambiguous rosbag directory {bag_path}: multiple .db3 files without metadata.yaml"
        )

    raise ValueError(f"No rosbag data found in {bag_path}")


def open_bag_reader(bag_path: Path) -> SequentialReader:
    uri, storage_id = resolve_rosbag_uri(bag_path)
    storage_options = StorageOptions(uri=uri, storage_id=storage_id)
    converter_options = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def get_bag_duration_sec(bag_path: Path) -> float | None:
    try:
        uri, storage_id = resolve_rosbag_uri(bag_path)
        info = rosbag2_py.Info().read_metadata(uri, storage_id)
        return max(0.0, info.duration.nanoseconds / 1e9)
    except Exception as error:  # noqa: BLE001
        print(f"[warn] Failed to read bag duration for {bag_path}: {error}")
        return None


def discover_rosbags(rosbag_dir: Path) -> list[Path]:
    rosbag_dir = rosbag_dir.expanduser()
    if rosbag_dir.is_file():
        return [rosbag_dir]

    if not rosbag_dir.is_dir():
        raise FileNotFoundError(f"rosbag_dir does not exist: {rosbag_dir}")

    candidates: list[Path] = []
    seen: set[Path] = set()

    for db3_path in sorted(rosbag_dir.rglob("*.db3")):
        parent = db3_path.parent
        if (parent / "metadata.yaml").exists():
            if parent not in seen:
                candidates.append(parent)
                seen.add(parent)
        elif db3_path not in seen:
            candidates.append(db3_path)
            seen.add(db3_path)

    if candidates:
        return candidates

    raise FileNotFoundError(f"No rosbag (.db3 or metadata.yaml directory) found in {rosbag_dir}")


def bag_label(bag_path: Path) -> str:
    if bag_path.is_file():
        return bag_path.stem
    if any(bag_path.glob("*.db3")):
        db3_files = sorted(bag_path.glob("*.db3"))
        if len(db3_files) == 1:
            return db3_files[0].stem
    return bag_path.name


def bag_video_path(bag_path: Path, rosbag_dir: Path, model_output_dir: Path) -> Path:
    """One mp4 per bag, mirroring folder layout under rosbag_dir when possible."""
    bag_path = bag_path.expanduser().resolve()
    rosbag_dir = rosbag_dir.expanduser().resolve()
    try:
        rel = bag_path.relative_to(rosbag_dir)
        if rel.is_file():
            return model_output_dir / rel.with_suffix(".mp4")
        return model_output_dir / rel / f"{bag_label(bag_path)}.mp4"
    except ValueError:
        return model_output_dir / f"{bag_label(bag_path)}.mp4"


def bag_trace_dir(bag_path: Path, rosbag_dir: Path, model_output_dir: Path) -> Path:
    """Directory for CSV traces, parallel to the per-bag mp4 path."""
    return bag_video_path(bag_path, rosbag_dir, model_output_dir).with_suffix("")


def bag_relative_key(bag_path: Path, rosbag_dir: Path) -> str:
    bag_path = bag_path.expanduser().resolve()
    rosbag_dir = rosbag_dir.expanduser().resolve()
    try:
        rel = bag_path.relative_to(rosbag_dir)
        if rel.is_file():
            return str(rel.with_suffix(""))
        return str(rel / bag_label(bag_path))
    except ValueError:
        return bag_label(bag_path)


def trace_logs_complete(trace_dir: Path) -> bool:
    ego_csv = trace_dir / "ego_pose.csv"
    traj_csv = trace_dir / "planned_trajectory.csv"
    return (
        ego_csv.exists()
        and ego_csv.stat().st_size > 0
        and traj_csv.exists()
        and traj_csv.stat().st_size > 0
    )
