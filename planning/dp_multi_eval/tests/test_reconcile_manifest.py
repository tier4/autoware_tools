"""reconcile_manifest at run_all start: a "running" job is stale (its run_all
died) and is re-queued; done/dry_run jobs without an output bag are re-queued;
done jobs with a bag and pending/failed jobs are left alone.
Run: python3 tests/test_reconcile_manifest.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.run_all import reconcile_manifest  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    with_bag = Path(d) / "done_with_bag"
    (with_bag / "output").mkdir(parents=True)
    jobs = [
        {"job_id": "a", "status": "running", "output_dir": str(Path(d) / "a")},
        {"job_id": "b", "status": "done", "output_dir": str(with_bag)},
        {"job_id": "c", "status": "done", "output_dir": str(Path(d) / "c")},
        {"job_id": "d", "status": "dry_run", "output_dir": str(Path(d) / "d")},
        {"job_id": "e", "status": "pending", "output_dir": str(Path(d) / "e")},
        {"job_id": "f", "status": "failed", "error": "x", "output_dir": str(Path(d) / "f")},
    ]
    n = reconcile_manifest({"jobs": jobs})
    assert n == 3, n
    st = {j["job_id"]: j["status"] for j in jobs}
    assert st == {"a": "pending", "b": "done", "c": "pending", "d": "pending", "e": "pending", "f": "failed"}, st
    assert "died" in jobs[0]["error"] and jobs[2]["error"] is None
print("PASS: reconcile — stale running and output-less done/dry_run re-queued; real results and failures untouched")
