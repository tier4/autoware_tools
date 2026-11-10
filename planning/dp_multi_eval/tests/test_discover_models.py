"""discover_model_configs (GUI "Choose models"): model dirs are found at any
depth, perception *.param.yaml files are skipped, a planner param yaml (with
args_path) is kept, ml_planner.onnx files are found at any depth, and
unique_models keeps the first entry per path.
Run: python3 tests/test_discover_models.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.model_config import discover_model_configs, unique_models  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # autoware_data layout: planner models two levels down, perception params beside them
        for model in ("diffusion_planner_for_erga/v4", "top_level_model"):
            (root / model).mkdir(parents=True)
            (root / model / "diffusion_planner.onnx").write_bytes(b"")
            (root / model / "args.json").write_text("{}")
        (root / "no_args").mkdir()
        (root / "no_args" / "diffusion_planner.onnx").write_bytes(b"")
        (root / "centerpoint").mkdir()
        (root / "centerpoint" / "centerpoint_ml_package.param.yaml").write_text(
            "/**:\n  ros__parameters:\n    model_path: x.onnx\n"
        )
        (root / "custom").mkdir()
        (root / "custom" / "diffusion_planner.param.yaml").write_text(
            "/**:\n  ros__parameters:\n    model:\n      args_path: /m/args.json\n"
        )
        (root / "my_run.param.yaml").write_text("    args_path: /m/args.json\n")
        # ML planner: a lone ONNX at depth, and one beside a diffusion planner
        (root / "ml" / "ml_v1").mkdir(parents=True)
        (root / "ml" / "ml_v1" / "ml_planner.onnx").write_bytes(b"")
        (root / "diffusion_planner_for_erga" / "v4" / "ml_planner.onnx").write_bytes(b"")
        (root / "ml" / "ml_planner.param.yaml").write_text(
            "/**:\n  ros__parameters:\n    model:\n      onnx_model_path: /m/ml_planner.onnx\n"
        )

        found = discover_model_configs(root)
        names = sorted(n for n, _ in found)
        assert names == ["custom", "ml", "ml_v1", "my_run", "top_level_model", "v4", "v4"], names
        ml = sorted(str(p.relative_to(root.resolve())) for _, p in found if p.name == "ml_planner.onnx")
        assert ml == ["diffusion_planner_for_erga/v4/ml_planner.onnx", "ml/ml_v1/ml_planner.onnx"], ml
        assert all(p.is_absolute() for _, p in found)

        assert discover_model_configs(root / "missing") == []

        v4 = root / "diffusion_planner_for_erga" / "v4"
        merged = unique_models([("from_yaml", v4)] + found)
        assert len(merged) == len(found), merged
        assert dict((p, n) for n, p in merged)[v4.resolve()] == "from_yaml"
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
