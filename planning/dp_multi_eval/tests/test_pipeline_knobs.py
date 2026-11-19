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

"""Every JobConfig field named in PIPELINE_KNOBS is forwarded from the pipeline
YAML into the job: a knob accepted by run_all but dropped by
_job_config_from_payload is a silent default (indicator_merge, 2026-09-16).
Run: python3 tests/test_pipeline_knobs.py"""
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.job_config import PIPELINE_KNOBS, JobConfig  # noqa: E402
from dp_multi_eval.run_orchestrator import _job_config_from_payload  # noqa: E402

job = {"model_config_path": "/tmp/m.param.yaml", "bag_path": "/tmp/bag", "output_dir": "/tmp/out"}
defaults = {f.name: f.default for f in fields(JobConfig)}
opts, expect = {"map_path": "/tmp/map"}, {}
for name in PIPELINE_KNOBS:
    if name not in defaults or name in ("mode", "multi_goal_stop_order", "goal_backoff_steps_m"):
        continue  # post-processing knobs; mode comes from the job; lists are normalised
    d = defaults[name]
    if isinstance(d, bool):
        v = not d
    elif isinstance(d, (int, float)):
        v = type(d)(d + 1)
    elif isinstance(d, str):
        v = "changed_" + d
    else:
        continue
    opts[name] = expect[name] = v
cfg = _job_config_from_payload(job, opts, domain_id=42)
wrong = {k: (getattr(cfg, k), v) for k, v in expect.items() if getattr(cfg, k) != v}
assert not wrong, f"knobs not forwarded (got, expected): {wrong}"
assert set(PIPELINE_KNOBS) >= {"indicator_merge", "render_workers", "window_eval"}
print(f"PASS: pipeline knobs — {len(expect)} JobConfig fields reach the job from the yaml")
