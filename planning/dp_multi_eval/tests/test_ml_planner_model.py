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

"""ML planner models: an ml_planner.onnx (or its folder, or its param YAML) is an
ML model; its param comes from the ml_planner template with the ONNX path in
place (model.onnx_model_path, or the older base_model_directory +
onnx_model_filename); its jobs carry planning_setting=ml_planner while
diffusion planner jobs carry none; the overlay writes ml_planner.param.yaml.
Run: python3 tests/test_ml_planner_model.py"""
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.generate_manifest import generate_manifest  # noqa: E402
from dp_multi_eval.model_config import (  # noqa: E402
    patch_ml_param_yaml,
    planning_setting_for,
    resolve_model_config,
)

TEMPLATE = """/**:
  ros__parameters:
    model:
      backend: "tensorrt"
      onnx_model_path: /opt/autoware/mlmodels/diffusion_planner_for_erga/ml_planner.onnx
    planning_frequency_hz: 10.0
"""
OLD_TEMPLATE = """/**:
  ros__parameters:
    base_model_directory: /opt/m
    onnx_model_filename: ml_planner.onnx
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ml_dir = root / "ml_v1"
        ml_dir.mkdir()
        onnx = ml_dir / "ml_planner.onnx"
        onnx.write_bytes(b"")
        dp_dir = root / "dp_v4"
        dp_dir.mkdir()
        (dp_dir / "diffusion_planner.onnx").write_bytes(b"")
        (dp_dir / "args.json").write_text("{}")
        tpl = root / "ml_planner.param.yaml"
        tpl.write_text(TEMPLATE)

        assert planning_setting_for(onnx) == "ml_planner"
        assert planning_setting_for(ml_dir) == "ml_planner"
        assert planning_setting_for(dp_dir) == "diffusion_planner"
        assert planning_setting_for(tpl) == "ml_planner"

        for src in (onnx, ml_dir):
            out = resolve_model_config(src, cache_dir=root / "cache", model_name="ml_v1",
                                       ml_param_template=tpl)
            p = yaml.safe_load(out.read_text())["/**"]["ros__parameters"]
            assert p["model"]["onnx_model_path"] == str(onnx.resolve()), p
            assert p["model"]["backend"] == "tensorrt" and p["planning_frequency_hz"] == 10.0

        old = root / "old.yaml"
        old.write_text(OLD_TEMPLATE)
        patch_ml_param_yaml(old, root / "old_out.yaml", onnx)
        p = yaml.safe_load((root / "old_out.yaml").read_text())["/**"]["ros__parameters"]
        assert p == {"base_model_directory": str(onnx.parent), "onnx_model_filename": "ml_planner.onnx"}, p

        # Jobs: the ML model carries its planner; the DP one leaves the run default.
        bags = root / "bags" / "H45"
        bags.mkdir(parents=True)
        (bags / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
        dp_param = root / "dp.param.yaml"
        dp_param.write_text("/**:\n  ros__parameters:\n    args_path: /m/args.json\n")
        ml_param = root / "ml.param.yaml"
        ml_param.write_text(TEMPLATE)
        m = generate_manifest(
            models=[{"name": "dp", "model_config": str(dp_param)},
                    {"name": "ml", "model_config": str(ml_param)}],
            results_root=root / "res", mode="reproducer", rosbag_dir=root / "bags",
        )
        settings = {j["model_name"]: j.get("planning_setting") for j in m["jobs"]}
        assert settings == {"dp": None, "ml": "ml_planner"}, settings

        # Overlay: the ML param lands in ml_planner.param.yaml (needs autoware_launch).
        try:
            from dp_multi_eval.ament_overlay import (
                ML_PARAM_REL,
                PARAM_REL,
                make_model_overlay,
                resolve_autoware_launch_share,
            )
            resolve_autoware_launch_share()
        except FileNotFoundError:
            print("PASS (overlay part skipped: no sourced autoware_launch)")
            return 0
        ov = root / "ov"
        make_model_overlay(ml_param, ov, planning_setting="ml_planner")
        share = ov / "share" / "autoware_launch"
        assert (share / ML_PARAM_REL).read_text() == ml_param.read_text()
        assert (share / PARAM_REL).read_text() != ml_param.read_text()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
