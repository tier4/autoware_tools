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

"""bag_whitelist (the GUI's selection, no symlink staging): run_all mints jobs
only for the listed bags, keyed relative to rosbag_dir; a rosbag_dir that is
itself one bag is keyed by its name, not ".".
Run: python3 tests/test_bag_whitelist.py"""
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval import run_all as run_all_mod  # noqa: E402
from dp_multi_eval.generate_manifest import bag_key  # noqa: E402
from dp_multi_eval.run_orchestrator import load_manifest  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for b in ("H45", "H78", "H95"):
            (root / "bags" / b).mkdir(parents=True)
            (root / "bags" / b / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
        model = root / "m.param.yaml"
        model.write_text("/**:\n  ros__parameters: {}\n")
        (root / "lanelet2_map.osm").write_text("<osm/>\n")   # run_all refuses a map folder without one
        cfg = {
            "rosbag_dir": str(root / "bags"),
            "results_root": str(root / "res"),
            "map_path": str(root),
            "models": [{"name": "modelX", "model_config": str(model)}],
            "bag_whitelist": [str(root / "bags" / "H78"), str(root / "bags" / "H95")],
        }
        cfg_path = root / "pipeline.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        # Stop after the manifest is written: the orchestrator would launch simulators.
        class Stop(Exception):
            pass

        def no_orchestrator(*_a, **_k):
            raise Stop

        run_all_mod.run_orchestrator = no_orchestrator
        try:
            run_all_mod.run_all(cfg_path, dry_run=True)
        except Stop:
            pass
        m = load_manifest(root / "res" / "manifest.json")
        keys = sorted(j["bag_key"] for j in m["jobs"])
        assert keys == ["H78", "H95"], keys

        one = root / "bags" / "H45"
        assert bag_key(one, one) == "H45", bag_key(one, one)
        assert bag_key(one, root / "bags") == "H45"
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
