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
# limitations under the License.# mcap_reader.py
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import yaml


class McapZohSynchronizer:
    def __init__(self, bag_path: str, topics_config_path: str):
        self.bag_path = Path(bag_path)
        with open(topics_config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.topic_map = self.config["topics"]
        self.stale_threshold_ns = self.config["thresholds"]["stale_warning_ms"] * 1e6

        # Caches for indexed messages
        self.ref_timestamps: List[int] = []
        self.ref_messages: Dict[int, any] = {}
        self.history: Dict[str, List[Tuple[int, any]]] = {"odom": [], "objects": [], "borders": []}

        self._index_bag()

    def _index_bag(self):
        """Pass through the bag once to cache timestamps and message histories."""
        with Reader(self.bag_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                msg = deserialize_cdr(rawdata, connection.msgtype)

                if connection.topic == self.topic_map["reference_trajectory"]:
                    self.ref_timestamps.append(timestamp)
                    self.ref_messages[timestamp] = msg
                elif connection.topic == self.topic_map["odometry"]:
                    self.history["odom"].append((timestamp, msg))
                elif connection.topic == self.topic_map["tracked_objects"]:
                    self.history["objects"].append((timestamp, msg))
                elif connection.topic == self.topic_map["road_borders"]:
                    self.history["borders"].append((timestamp, msg))

        self.ref_timestamps.sort()
        for key in self.history:
            self.history[key].sort(key=lambda x: x[0])

    def _get_latest_before(self, history_key: str, target_ts: int) -> Tuple[Optional[any], float]:
        """Find the latest message where t_msg <= target_ts (Zero-Order Hold)."""
        records = self.history[history_key]
        best_msg = None
        best_ts = 0
        for ts, msg in records:
            if ts <= target_ts:
                best_msg, best_ts = msg, ts
            else:
                break

        lag_ms = (target_ts - best_ts) / 1e6 if best_msg else float("inf")
        return best_msg, lag_ms

    def get_synchronized_frame(self, index: int) -> Dict:
        """Return the fully synchronized frame and data-lags for UI warnings."""
        target_ts = self.ref_timestamps[index]
        ref_traj = self.ref_messages[target_ts]

        odom, odom_lag = self._get_latest_before("odom", target_ts)
        objects, obj_lag = self._get_latest_before("objects", target_ts)
        borders, border_lag = self._get_latest_before("borders", target_ts)

        # Check if perception or odom is too stale
        warnings = []
        if odom_lag > (self.stale_threshold_ns / 1e6):
            warnings.append(f"Odometry stale by {odom_lag:.1f} ms")
        if obj_lag > (self.stale_threshold_ns / 1e6):
            warnings.append(f"TrackedObjects stale by {obj_lag:.1f} ms")

        return {
            "timestamp_ns": target_ts,
            "reference_trajectory": ref_traj,
            "odometry": odom,
            "tracked_objects": objects,
            "road_borders": borders,
            "warnings": warnings,
        }
