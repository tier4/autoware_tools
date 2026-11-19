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

"""rescore: walks the manifest's done jobs, recomputes metrics.json in place
keeping the orchestrator's run stamps, reports changed verdicts, rebuilds the
dashboard; --dry-run writes metrics_rescored.json and touches nothing else.
Run: python3 tests/test_rescore.py"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dp_multi_eval.rescore as rs  # noqa: E402


def setup(root: Path):
    job_dir = root / "m" / "bagA"
    (job_dir / "output").mkdir(parents=True)
    (job_dir / "goal_pose.json").write_text(json.dumps({"x": 1.0, "y": 2.0, "yaw": 0.5}))
    (job_dir / "metrics.json").write_text(json.dumps({
        "pass": False, "windows": {"legs_passed": 0, "legs_total": 1},
        "run_status": "done", "interventions": {"reinits": 0}, "clean_run": True,
    }))
    no_out = root / "m" / "bagB"
    no_out.mkdir(parents=True)
    (no_out / "metrics.json").write_text("{}")
    manifest = {"version": 2, "results_root": str(root), "jobs": [
        {"job_id": "m__bagA", "status": "done", "output_dir": str(job_dir), "bag_path": "/tmp/rec/bagA"},
        {"job_id": "m__bagB", "status": "done", "output_dir": str(no_out), "bag_path": "/tmp/rec/bagB"},
        {"job_id": "m__bagC", "status": "pending", "output_dir": str(root / "m" / "bagC")},
    ]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    cfg = root / "pipeline.yaml"
    cfg.write_text(f"results_root: {root}\nmap_path: /tmp/map\n")
    return cfg, job_dir


def main() -> int:
    calls = {"compute": [], "dashboard": 0}

    def fake_compute(bag, **kw):
        calls["compute"].append((bag, kw))
        return {"pass": True, "windows": {"legs_passed": 1, "legs_total": 1}}

    rs.compute_metrics = fake_compute
    rs.build_dashboard = lambda manifest, out: calls.__setitem__("dashboard", calls["dashboard"] + 1) or out

    with tempfile.TemporaryDirectory() as d:
        cfg, job_dir = setup(Path(d))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rs.rescore(cfg, dry_run=True)
        out = buf.getvalue()
        assert rc == 0 and "1 done job(s)" not in out and "2 done job(s)" in out, out
        assert "m__bagA: FAIL 0/1 -> PASS 1/1  CHANGED" in out and "m__bagB: no output to score" in out, out
        new = json.loads((job_dir / "metrics_rescored.json").read_text())
        assert new["pass"] is True and new["run_status"] == "done" and new["clean_run"] is True, new
        assert json.loads((job_dir / "metrics.json").read_text())["pass"] is False, "dry run must not touch metrics.json"
        assert calls["dashboard"] == 0
        bag, kw = calls["compute"][0]
        assert bag == job_dir / "output" and kw["goal_x"] == 1.0 and kw["goal_yaw"] == 0.5
        assert kw["reference_bag"] == Path("/tmp/rec/bagA") and kw["map_path"] == Path("/tmp/map")
        print("PASS: dry run scores the job, keeps the run stamps, leaves metrics.json alone")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rs.rescore(cfg, only="bagA")
        assert rc == 0 and "1 done job(s) to rescore matching --only bagA" in buf.getvalue(), buf.getvalue()
        cur = json.loads((job_dir / "metrics.json").read_text())
        assert cur["pass"] is True and cur["interventions"] == {"reinits": 0} and cur["run_status"] == "done", cur
        assert calls["dashboard"] == 1 and "1 verdict(s) changed" in buf.getvalue()
        print("PASS: --only rescored one job in place and rebuilt the dashboard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
