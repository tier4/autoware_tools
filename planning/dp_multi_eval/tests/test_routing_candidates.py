"""route_candidates: what the planner is asked, in order, for a refused goal.
Run: python3 tests/test_routing_candidates.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.routing import MapStop, RouteResult, route_candidates  # noqa: E402

ORDER = ["平塚駅南口", "商工会議所前（往路）", "教会前（往路）", "松風公園入口", "花水小学校前", "すみれ平", "松風町"]
STOPS = [MapStop(n, 100.0 * i, 0.0, math.pi, lane_id=str(1000 + i)) for i, n in enumerate(ORDER)]


def main() -> int:
    # a goal 0.4 m off 松風公園入口 is that stop; heading west, pull-backs move east
    cands = route_candidates((300.4, 0.0, math.pi), STOPS, ORDER, allow_walk=True)
    labels = [c.label for c in cands]
    assert labels == ["exact", "pull_back=1m", "pull_back=3m", "walk", "walk", "walk"], labels
    assert cands[0].goal == (300.0, 0.0, math.pi) and cands[0].name == "松風公園入口"
    assert abs(cands[1].goal[0] - 301.0) < 1e-9 and abs(cands[2].goal[0] - 303.0) < 1e-9
    assert cands[3].name == "花水小学校前" and cands[3].skipped == ("松風公園入口",)
    assert [w[0] for w in cands[3].waypoints] == [300.0]
    assert cands[4].name == "すみれ平" and cands[4].skipped == ("松風公園入口", "花水小学校前")
    assert len(cands[4].waypoints) == 2
    # the walk is the caller's decision: an intermediate leg never skips its stop
    assert [c.label for c in route_candidates((300.4, 0.0, math.pi), STOPS, ORDER, allow_walk=False)] == labels[:3]
    # a goal that is not a map stop gets no walk even when allowed
    free = route_candidates((250.0, 30.0, 0.0), STOPS, ORDER, allow_walk=True)
    assert [c.label for c in free] == labels[:3] and free[0].name is None
    # the last stop in the order has nothing to walk to
    assert [c.label for c in route_candidates((600.0, 0.0, math.pi), STOPS, ORDER, allow_walk=True)] == labels[:3]
    # a stop drawn 1 m off its lane centre: the centred pose comes second, pull-backs start from it
    lines = {"1003": [(320.0, 1.0), (280.0, 1.0)]}  # lanelet of 松風公園入口, running west at y=1
    cc = route_candidates((300.4, 0.0, math.pi), STOPS, ORDER, allow_walk=False, centerlines=lines)
    assert [c.label for c in cc] == ["exact", "lane_centre", "pull_back=1m", "pull_back=3m"], [c.label for c in cc]
    assert abs(cc[1].goal[0] - 300.0) < 1e-9 and abs(cc[1].goal[1] - 1.0) < 1e-9 and abs(cc[1].goal[2] - math.pi) < 1e-6
    assert abs(cc[2].goal[0] - 301.0) < 1e-9 and abs(cc[2].goal[1] - 1.0) < 1e-9
    # a stop already on its centreline gets no duplicate candidate
    on_line = route_candidates((300.4, 0.0, math.pi), STOPS, ORDER, allow_walk=False, centerlines={"1003": [(320.0, 0.0), (280.0, 0.0)]})
    assert [c.label for c in on_line] == ["exact", "pull_back=1m", "pull_back=3m"]
    # the message contract the drive loop parses
    r = RouteResult(True, "route_set", (400.0, 0.0, math.pi), "花水小学校前", ("松風公園入口",), 1, "walk", 4)
    text = r.describe()
    assert text.startswith("effective_goal_xy=400.0000,0.0000,3.141593; effective_goal=花水小学校前; skipped_goals=松風公園入口; ")
    plain = RouteResult(True, "route_set", (300.0, 0.0, math.pi), "松風公園入口", (), 0, "exact", 1).describe()
    assert "effective_goal=" not in plain and plain.startswith("effective_goal_xy=300.0000")
    print("test_routing_candidates: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
