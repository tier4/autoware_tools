"""generate_manifest: bags are discovered one level deep; job ids and output
dirs follow the naming; repeats are numbered runs.
Run: python3 tests/test_manifest_defaults.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.generate_manifest import generate_manifest  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for b in ("bagA", "sub/bagB"):
            (root / "bags" / b).mkdir(parents=True)
            (root / "bags" / b / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
        (root / "bags" / "notabag").mkdir()  # no metadata, no db3: ignored
        model = root / "m.param.yaml"
        model.write_text("/**:\n  ros__parameters: {}\n")
        models = [{"name": "modelX", "model_config": str(model)}]

        m = generate_manifest(models=models, results_root=root / "res", mode="reproducer", rosbag_dir=root / "bags")
        ids = sorted(j["job_id"] for j in m["jobs"])
        assert ids == ["modelX__bagA", "modelX__sub__bagB"], ids
        job = next(j for j in m["jobs"] if j["bag_key"] == "bagA")
        assert job["status"] == "pending" and job["attempts"] == 0 and "scene_mode" not in job
        assert job["output_dir"].endswith("res/modelX/bagA"), job["output_dir"]
        print("PASS: job ids and output dirs, nested bag keys")

        m3 = generate_manifest(models=models, results_root=root / "res", mode="reproducer",
                               rosbag_dir=root / "bags", scenario_repeat_count=2)
        assert sorted(j["job_id"] for j in m3["jobs"] if j["bag_key"] == "bagA") == [
            "modelX__bagA__run1", "modelX__bagA__run2"]
        print("PASS: repeats are numbered runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
