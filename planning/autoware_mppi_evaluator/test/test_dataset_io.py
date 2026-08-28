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

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

try:
    import rosbag2_py  # noqa: F401
except ModuleNotFoundError:
    rosbag2_py = types.ModuleType("rosbag2_py")
    sys.modules["rosbag2_py"] = rosbag2_py

from autoware_mppi_evaluator.dataset_io import load_dataset
from autoware_mppi_evaluator.dataset_io import save_frame
from autoware_mppi_evaluator.mcap_reader import SynchronizedFrame
import yaml


class FakeSynchronizer:
    topic_map = {
        "reference_trajectory": "/reference",
        "lanelet_map": "/map",
        "route": "/route",
        "odometry": "/odometry",
        "tracked_objects": "/objects",
    }
    topic_types = {key: f"test_msgs/msg/{key}" for key in topic_map}


def make_frame(index: int) -> SynchronizedFrame:
    timestamp_ns = 100 + index
    return SynchronizedFrame(
        index=index,
        timestamp_ns=timestamp_ns,
        messages={
            "lanelet_map": b"shared map",
            "route": b"shared route",
            "reference_trajectory": f"reference {index}".encode(),
            "odometry": f"odometry {index}".encode(),
            "tracked_objects": f"objects {index}".encode(),
        },
        ages_ms={"odometry": 1.0},
        warnings=[],
        is_usable=True,
    )


class TestDatasetIo(unittest.TestCase):
    def test_deduplicates_and_shares_static_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_frame(make_frame(0), FakeSynchronizer(), directory, ["first"])
            save_frame(make_frame(1), FakeSynchronizer(), directory, ["second"])

            with (root / "manifest.yaml").open(encoding="utf-8") as stream:
                manifest = yaml.safe_load(stream)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(len(manifest["environments"]), 1)
            self.assertEqual(manifest["frames"][0]["tags"], ["first"])

            frame_path = root / manifest["frames"][0]["path"]
            with frame_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            self.assertNotIn("lanelet_map", payload["messages_cdr_base64"])
            self.assertNotIn("route", payload["messages_cdr_base64"])
            self.assertNotIn("topics", payload)
            self.assertNotIn("types", payload)
            self.assertNotIn("tags", payload)

            frames = load_dataset(directory)
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].messages["lanelet_map"], b"shared map")
            self.assertEqual(frames[1].messages["reference_trajectory"], b"reference 1")
            self.assertIs(frames[0].messages["lanelet_map"], frames[1].messages["lanelet_map"])


if __name__ == "__main__":
    unittest.main()
