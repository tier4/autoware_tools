"""Intermediate legs on a map bus stop require a stop. Run: python3 tests/test_bus_stop_leg.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.run_single_job import leg_is_bus_stop  # noqa: E402

stops = [(50280.2, 9965.6), (50235.3, 9818.5)]
assert leg_is_bus_stop((50280.0, 9965.0, 0.0), stops)
assert leg_is_bus_stop((50236.0, 9820.0, 0.0), stops)
assert not leg_is_bus_stop((50294.0, 9863.0, 0.0), stops), "a junction wait is not a bus stop"
assert not leg_is_bus_stop(None, stops)
print("PASS: bus-stop legs identified within 10 m; junction legs stay drive-through")
