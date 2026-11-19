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

"""merge_manifests keeps every existing job the fresh manifest does not
mention. Run: python3 tests/test_merge_manifests.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.run_all import merge_manifests  # noqa: E402

existing = {"jobs": [
    {"job_id": "v4__A", "status": "done", "output_dir": "/nonexistent/v4/A"},
    {"job_id": "v4__B", "status": "failed", "error": "x", "attempts": 2, "model_config_path": "p"},
    {"job_id": "other__A", "status": "done", "output_dir": "/nonexistent/other/A"},
]}
fresh = {"jobs": [
    {"job_id": "v4__B", "status": "pending", "attempts": 0, "model_config_path": "p"},
    {"job_id": "v4__C", "status": "pending", "attempts": 0},
]}
out = merge_manifests(existing, fresh)
ids = [j["job_id"] for j in out["jobs"]]
assert ids == ["v4__B", "v4__C", "v4__A", "other__A"], ids
assert next(j for j in out["jobs"] if j["job_id"] == "v4__B")["status"] == "failed"
assert next(j for j in out["jobs"] if j["job_id"] == "other__A")["status"] == "done"
print("PASS: merge_manifests — fresh jobs first, every unmentioned existing job kept")
