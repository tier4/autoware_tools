"""Window runs start the scene driver (the original perception_reproducer on
dp_multi_eval's clock) with --window-clock; a dashboard column is the model.
Run: python3 tests/test_scene_mode.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.build_dashboard import _model_key  # noqa: E402
from dp_multi_eval.run_single_job import build_reproducer_command  # noqa: E402

bag = Path("/tmp/bag")
window = build_reproducer_command(bag, window_clock=True)
assert window[:4] == ["ros2", "run", "dp_multi_eval", "scene_driver.py"], window
assert window[-1] == "--window-clock", window
assert "--window-clock" not in build_reproducer_command(bag, window_clock=False)
assert build_reproducer_command(bag, window_clock=True, objects=False)[-1] == "--no-objects"
assert _model_key({"model_name": "v4"}) == "v4"
print("PASS: scene command — the scene driver, --window-clock for window runs")
