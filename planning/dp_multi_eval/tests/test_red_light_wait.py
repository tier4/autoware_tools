"""_at_red_light: a run stop spent under a red or amber light ahead is a wait, not
a stop the recording did not make; under green, without a light, or with no
signal timeline it is not.
Run: python3 tests/test_red_light_wait.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.compute_metrics import _at_red_light  # noqa: E402

RED, AMBER, GREEN = 1, 2, 3
# H95: group 1061 red from 7.7 s, green from 38.8 s; another light (1049) green throughout
states = [(1000.0, {1061: GREEN, 1049: GREEN}), (1007.7, {1061: RED, 1049: GREEN}), (1038.8, {1061: GREEN, 1049: GREEN})]
group_of = lambda x, y, yaw: {(0.0, 0.0): 1061, (50.0, 0.0): 1049}.get((x, y))
stops = [
    (1036.5, 1040.2, 0.0, 0.0, 3.1),   # stood 36.5-40.2 s at the 1061 stop line: red until 38.8 -> a red-light wait
    (1040.0, 1045.0, 0.0, 0.0, 3.1),   # after the green: not
    (1036.5, 1040.2, 50.0, 0.0, 3.1),  # under a light that was green: not
    (1036.5, 1040.2, 99.0, 0.0, 3.1),  # no light ahead: not
    (1037.0, 1043.0, 0.0, 0.0, 3.1),   # red for 1.8 of 6 s: less than half, not
]
assert _at_red_light(stops, states, group_of) == [True, False, False, False, False]
assert _at_red_light(stops, [], group_of) == [False] * 5
# amber counts as the light being against the ego
amber = [(1000.0, {1061: AMBER})]
assert _at_red_light([stops[0]], amber, group_of) == [True]
print("PASS: a stop under a red/amber light ahead is a red-light wait; green, no light or no signals is not")
