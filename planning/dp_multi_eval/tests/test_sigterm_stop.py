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

"""SIGTERM (and SIGINT) stop the orchestrator cleanly: the running job goes
back to pending with error "interrupted", the manifest is saved and the
process exits through KeyboardInterrupt. Runs the orchestrator in a child
process with a slow fake job. Run: python3 tests/test_sigterm_stop.py"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

DRIVER = textwrap.dedent(
    """
    import sys, time, pathlib
    sys.path.insert(0, sys.argv[1])
    import dp_multi_eval.run_orchestrator as ro
    marker = pathlib.Path(sys.argv[3])

    def slow_job(payload):
        marker.write_text("started")
        time.sleep(120)
        return {"job_id": payload["job"]["job_id"], "status": "done"}

    ro._process_job = slow_job                                  # the pool submits this name
    ro.cleanup_evaluation_processes = lambda *a, **k: None     # never touch real processes here
    ro._refresh_dashboard = lambda *a, **k: None
    try:
        ro.run_orchestrator(pathlib.Path(sys.argv[2]), max_workers=1, domain_ids=[199], render_video_flag=False)
    except KeyboardInterrupt:
        print("INTERRUPTED", flush=True)
        sys.exit(130)
    print("FINISHED", flush=True)
    """
)


def stop_with(sig) -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"version": 2, "results_root": str(root), "jobs": [{
            "job_id": "m__bag", "status": "pending", "attempts": 0, "error": None,
            "output_dir": str(root / "m" / "bag"), "model_config_path": "/tmp/m.param.yaml",
            "bag_path": "/tmp/bag", "mode": "reproducer",
        }]}))
        marker = root / "started"
        driver = root / "driver.py"
        driver.write_text(DRIVER)
        proc = subprocess.Popen([sys.executable, str(driver), str(PKG), str(manifest), str(marker)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            t0 = time.monotonic()
            while not marker.exists():
                assert proc.poll() is None, f"driver exited early:\n{proc.stdout.read()}"
                assert time.monotonic() - t0 < 60, "job never started"
                time.sleep(0.2)
            os.kill(proc.pid, sig)
            out, _ = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == 130 and "INTERRUPTED" in out, f"rc={proc.returncode}\n{out}"
        job = json.loads(manifest.read_text())["jobs"][0]
        assert job["status"] == "pending" and job["error"] == "interrupted" and job["attempts"] == 1, job
        assert "Interrupted — cancelling queued jobs" in out, out


def main() -> int:
    stop_with(signal.SIGTERM)
    print("PASS: SIGTERM — running job back to pending, error=interrupted, exit 130")
    stop_with(signal.SIGINT)
    print("PASS: SIGINT — same path")
    stop_with(signal.SIGHUP)
    print("PASS: SIGHUP (terminal closed) — same path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
