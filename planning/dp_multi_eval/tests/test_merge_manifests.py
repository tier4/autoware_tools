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
