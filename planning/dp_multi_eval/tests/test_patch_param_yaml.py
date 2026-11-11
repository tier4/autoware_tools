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

"""patch_param_yaml keeps the template's indent: the flat (pre 2026-10-15) and
the nested (model: / single_step_model:) diffusion_planner.param.yaml both stay
valid YAML with the model's paths in place.
Run: python3 tests/test_patch_param_yaml.py"""
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.model_config import patch_param_yaml  # noqa: E402

FLAT = """/**:
  ros__parameters:
    onnx_model_path: /opt/m/diffusion_planner.onnx
    args_path: /opt/m/args.json
    ignore_neighbors: false
"""

NESTED = """/**:
  ros__parameters:
    model:
      type: "single_step"
      args_path: /opt/m/args.json
      single_step_model:
        onnx_model_path: /opt/m/diffusion_planner.onnx
      multi_step_model:
        encoder_onnx_model_path: /opt/m/diffusion_planner_encoder.onnx
    ignore_neighbors: false
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for name, text in (("flat", FLAT), ("nested", NESTED)):
            tpl = root / f"{name}.yaml"
            tpl.write_text(text)
            out = root / f"{name}_out.yaml"
            patch_param_yaml(tpl, out, "/new/dp.onnx", "/new/args.json", ignore_neighbors=True)
            p = yaml.safe_load(out.read_text())["/**"]["ros__parameters"]
            if name == "flat":
                assert p["onnx_model_path"] == "/new/dp.onnx", p
                assert p["args_path"] == "/new/args.json", p
            else:
                assert p["model"]["single_step_model"]["onnx_model_path"] == "/new/dp.onnx", p
                assert p["model"]["args_path"] == "/new/args.json", p
                assert p["model"]["type"] == "single_step"
                # only the single-step path is the model's; the others stay as written
                enc = p["model"]["multi_step_model"]["encoder_onnx_model_path"]
                assert enc == "/opt/m/diffusion_planner_encoder.onnx", enc
            assert p["ignore_neighbors"] is True, p
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
