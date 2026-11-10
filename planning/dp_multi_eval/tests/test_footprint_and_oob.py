"""The ego is measured with the bus's real body, and a road-border crossing
the recording made at the same place is the map's, not the model's.

    python3 tests/test_footprint_and_oob.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.compute_metrics import split_recorded_too  # noqa: E402
from dp_multi_eval.geometry import load_vehicle_footprint  # noqa: E402


class F:
    def __init__(self, t, x, y):
        self.stamp_sec, self.x, self.y = t, x, y


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "vehicle_info.param.yaml"
        yml.write_text(
            "/**:\n  ros__parameters:\n    wheel_base: 4.9901\n    wheel_tread: 2.096\n"
            "    front_overhang: 2.5667\n    rear_overhang: 3.3244\n"
            "    left_overhang: 0.3998\n    right_overhang: 0.3778\n"
        )
        fp = load_vehicle_footprint(yml)
        xs, ys = [p[0] for p in fp], [p[1] for p in fp]
        assert abs(max(xs) - 7.5568) < 1e-3 and abs(min(xs) + 3.3244) < 1e-3, xs
        assert abs(max(ys) - 1.4478) < 1e-3 and abs(min(ys) + 1.4258) < 1e-3, ys
        assert abs((max(ys) - min(ys)) - 2.8736) < 1e-3, "lv828l is 2.87 m wide (tread + both overhangs)"
        print("PASS: footprint from vehicle_info — 10.88 x 2.87 m, side overhangs included (was tread + 0.5 = 2.60 m)")
    assert load_vehicle_footprint(None) == [(7.8, 1.25), (7.8, -1.25), (-2.5, -1.25), (-2.5, 1.25)]
    print("PASS: no vehicle_info: the old default box, unchanged")

    frames = [F(t * 0.2, t * 1.0, 0.0) for t in range(100)]       # ego drives +x, 5 m/s
    ep_shared = [2.0, 4.0, "road_border"]                           # x 10..20
    ep_own = [12.0, 14.0, "road_border"]                            # x 60..70
    recorded_hits = [(x * 0.5, 0.3) for x in range(18, 42)]         # recording crossed x 9..20.5
    own, rec = split_recorded_too([ep_shared, ep_own], frames, recorded_hits)
    assert rec == [ep_shared] and own == [ep_own], (own, rec)
    own, rec = split_recorded_too([ep_shared, ep_own], frames, [])
    assert own == [ep_shared, ep_own] and rec == [], "no recording: every episode is the ego's"
    print("PASS: a crossing where the recording crossed too is the map's; one elsewhere stays the ego's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
