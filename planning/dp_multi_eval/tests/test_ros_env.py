"""A job sees packages built after the GUI started; a launch that died says why.

    python3 tests/test_ros_env.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.ros_env import launch_failure_reason, workspace_env  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        install = Path(tmp) / "install"
        install.mkdir()
        # The install's setup as it stands now: a package added since the GUI started.
        (install / "setup.bash").write_text(
            f'export AMENT_PREFIX_PATH="{install}/new_pkg${{AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}}"\n'
            "export DP_TEST_SOURCED=1\n"
        )
        base = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "COLCON_PREFIX_PATH": str(install),
            "AMENT_PREFIX_PATH": f"{install}/old_pkg",
            "KEEP_ME": "yes",
        }
        env = workspace_env(base)
        paths = env["AMENT_PREFIX_PATH"].split(":")
        assert f"{install}/new_pkg" in paths and f"{install}/old_pkg" in paths, paths
        assert env.get("DP_TEST_SOURCED") == "1" and env.get("KEEP_ME") == "yes", env
        assert base["AMENT_PREFIX_PATH"] == f"{install}/old_pkg", "the caller's env was changed"
        print("PASS: a package built after the GUI started reaches the job's environment")

        assert workspace_env({"PATH": base["PATH"], "X": "1"}) == {"PATH": base["PATH"], "X": "1"}
        print("PASS: no workspace on COLCON_PREFIX_PATH: the inherited environment as is")

        log = Path(tmp) / "psim_launch.log"
        log.write_text(
            "[INFO] [launch]: All log files can be found below /tmp\n"
            "[ERROR] [launch]: Caught exception in launch (see debug for traceback): \"package "
            "'autoware_trajectory_selector' not found, searching: ['/tmp/x', '/ws/install/a']\"\n"
        )
        reason = launch_failure_reason(log)
        assert reason and "autoware_trajectory_selector" in reason and "colcon build" in reason, reason
        assert "searching" not in reason, reason
        log.write_text("[ERROR] [launch]: something else broke\n")
        assert launch_failure_reason(log) == "something else broke"
        log.write_text("[INFO] all fine\n")
        assert launch_failure_reason(log) is None
        assert launch_failure_reason(Path(tmp) / "missing.log") is None
        print("PASS: a launch that died early names its cause (a missing package spelled out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
