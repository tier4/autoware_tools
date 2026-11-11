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

"""run_all refuses to start in a Python that lacks what a run imports later
(2026-11-11: a virtualenv without scipy drove for ten minutes, then failed at
the metrics). Run: python3 tests/test_preflight.py"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        from dp_multi_eval import run_all
    except ImportError as exc:   # no ROS / yaml here: nothing to check
        print(f"SKIP: {exc}")
        return 0
    assert run_all.missing_modules(("json", "no_such_module_xyz", "os")) == ["no_such_module_xyz"]
    assert "scipy" in run_all.REQUIRED_MODULES and "lanelet2" in run_all.REQUIRED_MODULES
    # every third-party module declared here is in package.xml (by its rosdep key or its ROS package)
    xml = (ROOT / "package.xml").read_text(encoding="utf-8")
    key = {"yaml": "python3-yaml", "numpy": "python3-numpy", "scipy": "python3-scipy", "matplotlib": "python3-matplotlib",
           "lanelet2": "lanelet2_python"}
    for mod in run_all.REQUIRED_MODULES:
        assert f">{key.get(mod, mod)}<" in xml, f"{mod} is not declared in package.xml"
    print("PASS: missing modules are named; each required module is declared in package.xml")
    # a key no code reads is named; the example config has none
    import yaml

    assert run_all.unknown_keys({"models": [], "window_eval": True, "scene_mode": "realtime", "max_workerz": 2}) == [
        "max_workerz", "scene_mode"]
    example = yaml.safe_load((ROOT / "config" / "example_pipeline.yaml").read_text(encoding="utf-8"))
    assert run_all.unknown_keys(example) == [], run_all.unknown_keys(example)
    print("PASS: config keys this version does not read are named; the example config has none")
    # a config that cannot run says why in one line, and nothing is started
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "map").mkdir()
        (tmp / "map" / "lanelet2_map.osm").write_text("<osm/>")
        base = {"results_root": str(tmp / "res"), "rosbag_dir": str(tmp / "bags"), "map_path": str(tmp / "map"),
                "models": [{"name": "m", "model_config": str(tmp / "no_such_model")}]}
        cases = (
            ({"results_root": None}, "results_root is not set"),
            ({"models": []}, "models is not set"),
            ({"map_path": str(tmp / "nowhere")}, "map_path has no lanelet2_map.osm"),
            ({}, "rosbag_dir missing"),
        )
        for change, said in cases:
            cfg_file = tmp / "cfg.yaml"
            cfg_file.write_text(yaml.safe_dump(base | change), encoding="utf-8")
            r = subprocess.run([sys.executable, "-c", "import sys; from dp_multi_eval import run_all; "
                                f"sys.exit(run_all.run_all(__import__('pathlib').Path(r'{cfg_file}'), dry_run=True))"],
                               capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")))
            assert r.returncode == 2 and said in r.stderr and "Traceback" not in r.stderr, (said, r.returncode, r.stderr[-300:])
            assert "cfg.yaml" in r.stderr and "Nothing was started" in r.stderr, r.stderr
        (tmp / "bags").mkdir()
        cfg_file.write_text(yaml.safe_dump(base), encoding="utf-8")
        r = subprocess.run([sys.executable, "-c", "import sys; from dp_multi_eval import run_all; "
                            f"sys.exit(run_all.run_all(__import__('pathlib').Path(r'{cfg_file}'), dry_run=True))"],
                           capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")))
        assert r.returncode == 2 and "Traceback" not in r.stderr, r.stderr[-300:]     # no recording / no model: one line too
    print("PASS: a config with a missing result folder, model list, map or recordings folder says so in one line")
    # the entry point stops before anything is started, and says which Python it is
    code = ("import sys; sys.argv=['run_all.py','-c','/nonexistent.yaml']\n"
            "from dp_multi_eval import run_all\n"
            "run_all.REQUIRED_MODULES = ('json', 'no_such_module_xyz')\n"
            "run_all.missing_modules.__defaults__ = (run_all.REQUIRED_MODULES,)\n"
            "sys.exit(run_all.main())")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                                VIRTUAL_ENV="/some/venv"))
    assert r.returncode == 2 and "no_such_module_xyz" in r.stderr and sys.executable in r.stderr, (r.returncode, r.stderr)
    assert "/some/venv" in r.stderr and "deactivate" in r.stderr, r.stderr
    print("PASS: run_all stops at once, naming the modules, the interpreter and an active virtualenv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
