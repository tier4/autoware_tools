#!/usr/bin/env python3
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

"""Evaluate MPPI configurations against synchronized MCAP frames."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

from autoware_mppi_evaluator import mppi_optimizer_py as mppi_cpp
from autoware_mppi_evaluator.dataset_io import load_dataset
from autoware_mppi_evaluator.evaluator_config import make_configuration
from autoware_mppi_evaluator.mcap_reader import McapZohSynchronizer


def parse_named_path(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for each optimizer configuration")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Both NAME and PATH are required")
    return name, path


def finite_or_none(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def evaluate_frame(session, frame):
    messages = frame.messages
    return session.evaluate(
        frame.frame_id,
        frame.timestamp_ns,
        messages["reference_trajectory"],
        messages["odometry"],
        messages["tracked_objects"],
        messages.get("acceleration"),
        messages.get("steering"),
        frame.ages_ms["odometry"],
        frame.ages_ms.get("acceleration"),
        frame.ages_ms.get("steering"),
        frame.ages_ms.get("tracked_objects"),
    )


def percentile(values: Iterable[float], fraction: float):
    ordered = sorted(values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: List[Dict]) -> Dict:
    summary = {}
    names = sorted({row["config_name"] for row in rows})
    for name in names:
        selected = [row for row in rows if row["config_name"] == name and "error" not in row]
        latencies = [row["execution_time_ms"] for row in selected]
        summary[name] = {
            "evaluated_frames": len(selected),
            "failed_frames": sum(row["config_name"] == name and "error" in row for row in rows),
            "rejected_frames": sum(row["was_rejected"] for row in selected),
            "invalid_frames": sum(not row["is_valid"] for row in selected),
            "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
            "latency_ms_p50": percentile(latencies, 0.50),
            "latency_ms_p95": percentile(latencies, 0.95),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="MCAP path or curated dataset path")
    parser.add_argument("--input-format", choices=("bag", "dataset"), default="bag")
    parser.add_argument("--topics", help="Topic configuration YAML for MCAP input")
    parser.add_argument(
        "--optimizer-config",
        action="append",
        required=True,
        type=parse_named_path,
        metavar="NAME=PATH",
    )
    parser.add_argument("--vehicle-info", required=True)
    parser.add_argument("--simulator-model", required=True)
    parser.add_argument("--mode", choices=("isolated", "chronological"), default="isolated")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--output-stride", type=int, default=1)
    parser.add_argument("--output", required=True, help="Output path without an extension")
    arguments = parser.parse_args()

    if arguments.output_stride < 1:
        parser.error("--output-stride must be positive")

    if arguments.input_format == "bag":
        if not arguments.topics:
            parser.error("--topics is required for MCAP input")
        frame_source = McapZohSynchronizer(arguments.input, arguments.topics)
        frame_count = len(frame_source)

        def selected_frames(start, stop):
            return frame_source.iter_frames(start, stop)

    else:
        all_frames = load_dataset(arguments.input)
        frame_count = len(all_frames)

        def selected_frames(start, stop):
            return iter(all_frames[start:stop])

    stop = frame_count if arguments.stop is None else min(arguments.stop, frame_count)
    if arguments.start < 0 or arguments.start >= stop:
        parser.error("The selected frame range is empty")

    rows: List[Dict] = []
    for config_name, config_path in arguments.optimizer_config:
        configuration = make_configuration(
            mppi_cpp,
            config_path,
            arguments.vehicle_info,
            arguments.simulator_model,
            config_name,
        )
        session = None
        environment_key = None
        for frame in selected_frames(arguments.start, stop):
            if not frame.is_usable:
                rows.append(
                    {
                        "frame_id": frame.frame_id,
                        "timestamp_ns": frame.timestamp_ns,
                        "config_name": config_name,
                        "error": "; ".join(frame.warnings),
                    }
                )
                continue

            next_environment_key = (
                hash(frame.messages["lanelet_map"]),
                hash(frame.messages["route"]),
            )
            if session is None or environment_key != next_environment_key:
                session = mppi_cpp.EvaluationSession(
                    frame.messages["lanelet_map"],
                    frame.messages["route"],
                    configuration,
                    arguments.mode,
                )
                environment_key = next_environment_key
            try:
                result = evaluate_frame(session, frame)
                if (frame.index - arguments.start) % arguments.output_stride == 0:
                    row = {
                        "frame_id": result["frame_id"],
                        "timestamp_ns": result["timestamp_ns"],
                        "config_name": result["config_name"],
                    }
                    row.update(
                        {key: finite_or_none(value) for key, value in result["metrics"].items()}
                    )
                    rows.append(row)
            except Exception as error:
                rows.append(
                    {
                        "frame_id": frame.frame_id,
                        "timestamp_ns": frame.timestamp_ns,
                        "config_name": config_name,
                        "error": str(error),
                    }
                )
                if arguments.mode == "chronological":
                    session.reset()

    output_base = Path(arguments.output).expanduser().resolve()
    json_payload = {"schema_version": 1, "summary": summarize(rows), "frames": rows}
    atomic_write(output_base.with_suffix(".json"), json.dumps(json_payload, indent=2) + "\n")

    fieldnames = sorted({key for row in rows for key in row})
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        atomic_write(output_base.with_suffix(".csv"), stream.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
