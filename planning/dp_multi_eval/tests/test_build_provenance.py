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

"""Results carry the planner/checkpoint versions and the code they came from.

    python3 tests/test_build_provenance.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval import build_provenance as bp  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        header = root / "autoware" / bp._CONSTANTS_REL
        header.parent.mkdir(parents=True)
        header.write_text("constexpr int WEIGHT_MINIMUM_VERSION = 4;\n")
        assert bp.planner_weight_rule(root / "autoware") == ("minimum", 4)
        header.write_text("constexpr int WEIGHT_MAJOR_VERSION = 3;\n")
        assert bp.planner_weight_rule(root / "autoware") == ("exact", 3)
        print("PASS: the planner's rule is read in both spellings (exact / minimum)")

        model = root / "model"
        model.mkdir()
        (model / "args.json").write_text(json.dumps({"major_version": 4}))
        assert bp.checkpoint_version(model) == 4, "a model directory (as the pipeline names it)"
        param = root / "dp.param.yaml"
        param.write_text(f"onnx_model_path: {model}/diffusion_planner.onnx\n")
        assert bp.checkpoint_version(param) == 4, "a param YAML pointing at the ONNX"
        print("PASS: the checkpoint version is read from a model directory or a param YAML")

        repo = root / "repo"
        repo.mkdir()
        run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
        run("init", "-q")
        (repo / "f.txt").write_text("a\n")
        run("add", "f.txt")
        run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
        clean = bp.repo_state(repo)
        assert clean and clean["dirty"] is False and len(clean["commit"]) >= 7, clean
        (repo / "f.txt").write_text("b\n")
        assert bp.repo_state(repo)["dirty"] is True
        assert bp.repo_state(root / "not_a_repo") is None
        print("PASS: repo state — commit, branch, dirty; None off git")
    return 0


if __name__ == "__main__":
    sys.exit(main())
