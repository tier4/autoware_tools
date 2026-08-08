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

"""Index the MPPI input topics in a ROS 2 MCAP file."""

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import rosbag2_py
import yaml


@dataclass(frozen=True)
class SerializedRecord:
    timestamp_ns: int
    data: bytes


@dataclass(frozen=True)
class SynchronizedFrame:
    index: int
    timestamp_ns: int
    messages: Dict[str, bytes]
    ages_ms: Dict[str, Optional[float]]
    warnings: List[str]
    is_usable: bool

    @property
    def frame_id(self) -> str:
        return f"frame_{self.timestamp_ns}"


class McapZohSynchronizer:
    """Synchronize MPPI inputs against each reference trajectory timestamp."""

    _STATIC_KEYS = ("lanelet_map", "route")
    _REQUIRED_ZOH_KEYS = ("odometry", "tracked_objects")
    _OPTIONAL_ZOH_KEYS = ("acceleration", "steering")

    def __init__(self, bag_path: str, topics_config_path: str):
        self.bag_path = Path(bag_path).expanduser().resolve()
        self.config_path = Path(topics_config_path).expanduser().resolve()
        with self.config_path.open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream) or {}

        self.topic_map: Dict[str, str] = self.config["topics"]
        thresholds = self.config.get("thresholds", {})
        self.required_stale_ms = float(thresholds.get("required_stale_ms", 100.0))
        self.optional_stale_ms = float(thresholds.get("optional_stale_ms", 250.0))
        self.topic_types: Dict[str, str] = {}
        self.records: Dict[str, List[SerializedRecord]] = {key: [] for key in self.topic_map}
        self._timestamps: Dict[str, List[int]] = {key: [] for key in self.topic_map}
        self._index_bag()

        reference_key = "reference_trajectory"
        self.ref_timestamps = self._timestamps[reference_key]
        if not self.ref_timestamps:
            raise ValueError(f"The bag contains no messages for {self.topic_map[reference_key]}")

    def _index_bag(self) -> None:
        if not self.bag_path.exists():
            raise FileNotFoundError(self.bag_path)

        storage_options = rosbag2_py.StorageOptions(uri=str(self.bag_path), storage_id="mcap")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        bag_topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
        required_keys = (
            "reference_trajectory",
            *self._STATIC_KEYS,
            *self._REQUIRED_ZOH_KEYS,
        )
        missing = [
            self.topic_map[key] for key in required_keys if self.topic_map[key] not in bag_topics
        ]
        if missing:
            raise ValueError(f"The bag does not contain configured topics: {missing}")

        reverse_topics = {
            topic: key for key, topic in self.topic_map.items() if topic in bag_topics
        }
        self.topic_types = {key: bag_topics[topic] for topic, key in reverse_topics.items()}
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(reverse_topics)))
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            key = reverse_topics[topic]
            self.records[key].append(SerializedRecord(timestamp_ns, bytes(data)))

        for key, records in self.records.items():
            records.sort(key=lambda record: record.timestamp_ns)
            self._timestamps[key] = [record.timestamp_ns for record in records]

    def __len__(self) -> int:
        return len(self.ref_timestamps)

    def _latest_at_or_before(
        self, key: str, timestamp_ns: int
    ) -> Tuple[Optional[SerializedRecord], Optional[float]]:
        timestamps = self._timestamps[key]
        position = bisect_right(timestamps, timestamp_ns)
        if position == 0:
            return None, None
        record = self.records[key][position - 1]
        return record, (timestamp_ns - record.timestamp_ns) / 1.0e6

    def get_synchronized_frame(self, index: int) -> SynchronizedFrame:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        reference = self.records["reference_trajectory"][index]
        timestamp_ns = reference.timestamp_ns
        messages = {"reference_trajectory": reference.data}
        ages_ms: Dict[str, Optional[float]] = {"reference_trajectory": 0.0}
        warnings: List[str] = []
        is_usable = True

        for key in self._STATIC_KEYS + self._REQUIRED_ZOH_KEYS + self._OPTIONAL_ZOH_KEYS:
            record, age_ms = self._latest_at_or_before(key, timestamp_ns)
            ages_ms[key] = age_ms
            if record is not None:
                messages[key] = record.data
            elif key in self._STATIC_KEYS + self._REQUIRED_ZOH_KEYS:
                warnings.append(f"No {key} message exists at or before this frame")
                is_usable = False
            else:
                warnings.append(f"No optional {key} message exists at or before this frame")

            if age_ms is None:
                continue
            if key in self._REQUIRED_ZOH_KEYS and age_ms > self.required_stale_ms:
                warnings.append(f"{key} is stale by {age_ms:.1f} ms")
                is_usable = False
            if key in self._OPTIONAL_ZOH_KEYS and age_ms > self.optional_stale_ms:
                warnings.append(f"{key} is stale by {age_ms:.1f} ms and will be omitted")
                messages.pop(key, None)

        return SynchronizedFrame(index, timestamp_ns, messages, ages_ms, warnings, is_usable)

    def iter_frames(
        self, start: int = 0, stop: Optional[int] = None
    ) -> Iterator[SynchronizedFrame]:
        final = len(self) if stop is None else min(stop, len(self))
        for index in range(max(0, start), final):
            yield self.get_synchronized_frame(index)
