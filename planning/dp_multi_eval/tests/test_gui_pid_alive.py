"""The GUI's run check: a run_all that has finished but was never waited for
(a zombie child) is not a run in progress; a live one is, but only if it
is still run_all (not a reused pid).
Run: python3 tests/test_gui_pid_alive.py"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.app import _is_run_all, pid_alive  # noqa: E402

done = subprocess.Popen(["true"])
time.sleep(0.3)                                   # exited, not waited for: a zombie
assert not pid_alive(done.pid), "a finished run counted as in progress"
live = subprocess.Popen(["sleep", "5"])
try:
    assert pid_alive(live.pid), "a live run not seen"
finally:
    live.kill()
    live.wait()
assert not pid_alive(0) and not pid_alive(None)
other = subprocess.Popen(["sleep", "5"])          # a live pid that is not a run (a reused number)
try:
    assert pid_alive(other.pid) and not _is_run_all(other.pid)
finally:
    other.kill()
    other.wait()
print("PASS: GUI run check — a finished (zombie) run is not in progress, a live one is")
