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

import sys
from threading import Lock
import types
import unittest

try:
    import rosbag2_py  # noqa: F401
except ModuleNotFoundError:
    rosbag2_py = types.ModuleType("rosbag2_py")

    class StorageFilter:
        def __init__(self, topics):
            self.topics = topics

    rosbag2_py.StorageFilter = StorageFilter
    sys.modules["rosbag2_py"] = rosbag2_py

from autoware_mppi_evaluator.mcap_reader import McapZohSynchronizer
from autoware_mppi_evaluator.mcap_reader import MessageReference
from autoware_mppi_evaluator.mcap_reader import SerializedRecord


class FakeReader:
    def __init__(self, messages):
        self.messages = messages
        self.filtered_messages = []
        self.index = 0
        self.read_count = 0
        self.topics = []

    def set_filter(self, storage_filter):
        self.topics = storage_filter.topics

    def seek(self, timestamp_ns):
        self.filtered_messages = [
            message
            for message in self.messages
            if message[0] in self.topics and message[2] >= timestamp_ns
        ]
        self.index = 0

    def has_next(self):
        return self.index < len(self.filtered_messages)

    def read_next(self):
        message = self.filtered_messages[self.index]
        self.index += 1
        self.read_count += 1
        return message


def make_synchronizer():
    synchronizer = McapZohSynchronizer.__new__(McapZohSynchronizer)
    synchronizer.topic_map = {
        "reference_trajectory": "/reference",
        "lanelet_map": "/map",
        "route": "/route",
        "odometry": "/odometry",
        "tracked_objects": "/objects",
        "acceleration": "/acceleration",
        "steering": "/steering",
    }
    synchronizer.required_stale_ms = 100.0
    synchronizer.optional_stale_ms = 250.0
    synchronizer.records = {
        "reference_trajectory": [MessageReference(100, 0)],
        "lanelet_map": [MessageReference(10, 0)],
        "route": [MessageReference(20, 0)],
        "odometry": [MessageReference(90, 0)],
        "tracked_objects": [MessageReference(80, 0)],
        "acceleration": [],
        "steering": [],
    }
    synchronizer.ref_timestamps = [100]
    payloads = {
        (key, reference.timestamp_ns, reference.occurrence): key.encode()
        for key, records in synchronizer.records.items()
        for reference in records
    }
    synchronizer._materialize = lambda key, reference: SerializedRecord(
        reference.timestamp_ns,
        payloads[(key, reference.timestamp_ns, reference.occurrence)],
    )
    return synchronizer


class TestMcapReader(unittest.TestCase):
    def test_get_synchronized_frame_preserves_public_behavior(self):
        synchronizer = make_synchronizer()

        frame = synchronizer.get_synchronized_frame(0)

        self.assertEqual(frame.index, 0)
        self.assertEqual(frame.timestamp_ns, 100)
        self.assertEqual(frame.messages["reference_trajectory"], b"reference_trajectory")
        self.assertEqual(frame.messages["odometry"], b"odometry")
        self.assertEqual(frame.ages_ms["odometry"], 10 / 1.0e6)
        self.assertTrue(frame.is_usable)
        self.assertEqual(len(synchronizer), 1)

    def test_payload_cache_preserves_duplicate_timestamp_payloads(self):
        synchronizer = McapZohSynchronizer.__new__(McapZohSynchronizer)
        synchronizer.topic_map = {"odometry": "/odometry"}
        synchronizer._payload_reader = FakeReader(
            [
                ("/odometry", b"first", 100),
                ("/odometry", b"second", 100),
                ("/odometry", b"later", 200),
            ]
        )
        synchronizer._payload_reader_lock = Lock()
        synchronizer._load_payload.cache_clear()

        first_result = synchronizer._load_payload("odometry", 100, 1)
        read_count = synchronizer._payload_reader.read_count
        second_result = synchronizer._load_payload("odometry", 100, 1)

        self.assertEqual(first_result, b"second")
        self.assertIs(second_result, first_result)
        self.assertEqual(synchronizer._payload_reader.read_count, read_count)
        self.assertEqual(synchronizer._load_payload.cache_info().maxsize, 128)


if __name__ == "__main__":
    unittest.main()
