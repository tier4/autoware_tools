"""waits.explain_stop: why a standing ego waited. A pedestrian crossing ahead
or about to step in, a walker on or at the kerb of a crosswalk the way
crosses, a vehicle on the recorded path ahead, cross traffic entering it and a
red light are justified; a walker standing on the pavement away from it,
a truck the recorded vehicle drove round and an oncoming car in its own lane
are not.
Run: python3 tests/test_waits.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval import waits as W  # noqa: E402
from dp_multi_eval.bag_reader import ObjectSample  # noqa: E402

BODY = W.EgoBody(front_m=8.4, half_width_m=1.44)
EGO = (50.0, 0.0, 0.0)   # standing at x=50 facing +x


def straight_ref():
    return [(x * 0.5, 0.0, 0.0) for x in range(0, 400)]


def swerve_ref():
    """Recorded path: y=0 to x=60, over to y=3.5 by x=70, then straight."""
    out = []
    for k in range(0, 400):
        x = k * 0.5
        y = 0.0 if x <= 60 else (3.5 if x >= 70 else 3.5 * (x - 60) / 10.0)
        out.append((x, y, 0.0))
    return out


def track(oid, label, fn, t0=0.0, t1=10.0, length=0.5, width=0.5, yaw=0.0):
    """fn(t) -> (x, y); samples every 0.2 s."""
    out, k = [], 0
    while t0 + 0.2 * k <= t1 + 1e-9:
        t = t0 + 0.2 * k
        x, y = fn(t)
        out.append(ObjectSample(stamp_sec=t, object_id=oid, label=label, x=x, y=y, yaw_rad=yaw,
                                length_m=length, width_m=width))
        k += 1
    return out


def explain(objects, ref, light_red=None, crosswalks=None):
    return W.explain_stop(0.0, 10.0, *EGO, frames=W.ObjectFrames(objects), ref=ref, body=BODY,
                          light_red=light_red, crosswalks=crosswalks)


def main() -> int:
    # a pedestrian crossing 6 m ahead of the front at 1.2 m/s, -6 m to +6 m
    cross = track("p", "PEDESTRIAN", lambda t: (64.4, -6.0 + 1.2 * t), yaw=math.pi / 2)
    w = explain(cross, straight_ref())
    assert w["reasons"].get("vru", 0) >= 5.0 and w["justified_sec"] >= 5.0, w
    # ... standing on the pavement 5 m to the side
    still = track("p", "PEDESTRIAN", lambda t: (64.4, 5.0))
    w = explain(still, straight_ref())
    assert w["justified_sec"] == 0.0, w
    print("PASS: a crossing pedestrian justifies the wait; one standing on the pavement does not")

    # a crosswalk 64-68 m across the road (the way crosses it 6-10 m ahead of the front)
    cw = [[(64.0, -12.0), (68.0, -12.0), (68.0, 12.0), (64.0, 12.0)]]
    on_cw = track("p", "PEDESTRIAN", lambda t: (66.0, 6.0))       # standing on it, 4.6 m off the body
    kerb = track("p", "PEDESTRIAN", lambda t: (66.0, 13.5))       # waiting at its kerb
    away = track("p", "PEDESTRIAN", lambda t: (80.0, 6.0))        # on the pavement 12 m further on
    assert explain(on_cw, straight_ref())["justified_sec"] == 0.0   # no map: not on the way
    assert explain(on_cw, straight_ref(), crosswalks=cw)["reasons"] == {"crosswalk": 10.0}
    assert explain(kerb, straight_ref(), crosswalks=cw)["reasons"] == {"crosswalk": 10.0}
    assert explain(away, straight_ref(), crosswalks=cw)["justified_sec"] == 0.0
    behind = [[(30.0, -12.0), (34.0, -12.0), (34.0, 12.0), (30.0, 12.0)]]   # a crosswalk behind the ego
    assert explain(track("p", "PEDESTRIAN", lambda t: (32.0, 6.0)), straight_ref(), crosswalks=behind)["justified_sec"] == 0.0
    print("PASS: a walker on or at the kerb of a crosswalk ahead justifies the wait; one off it or behind does not")

    # a stopped truck 20-30 m ahead of the base_link on the ego's lane
    truck = track("t", "TRUCK", lambda t: (75.0, 0.0), length=10.0, width=2.5)
    w = explain(truck, straight_ref())
    assert w["reasons"].get("vehicle", 0) >= 9.5, w
    # ... but the recorded vehicle drove round it (the path swerves to y=3.5)
    w = explain(truck, swerve_ref())
    assert w["justified_sec"] == 0.0, w
    # ... or brushed past with nothing to spare: its box 0.1 m into the swept body (H63)
    brushed = track("t", "CAR", lambda t: (75.0, 1.44 + 0.9 - 0.1), length=4.5, width=1.8)   # 0.1 m in
    w = explain(brushed, straight_ref())
    assert w["justified_sec"] == 0.0, w
    # ... unless the shield held it for the ego: it waited for the bus (a mutual wait)
    held = W.held_index([{"t": 0.5 * k, "id": "t", "reason": "ego"} for k in range(21)])
    w = W.explain_stop(0.0, 10.0, *EGO, frames=W.ObjectFrames(truck), ref=straight_ref(), body=BODY, held_for_ego=held)
    assert w["justified_sec"] == 0.0, w
    deadlocked = W.held_index([{"t": 0.5 * k, "id": "t", "reason": "deadlock"} for k in range(21)])
    w = W.explain_stop(0.0, 10.0, *EGO, frames=W.ObjectFrames(truck), ref=straight_ref(), body=BODY, held_for_ego=deadlocked)
    assert w["reasons"].get("vehicle", 0) >= 9.5, w
    print("PASS: a vehicle on the recorded path ahead is a reason; one the recording drove or brushed round, "
          "or one the shield held for the ego, is not")

    # cross traffic: a car at 8 m/s heading -y, crossing the way 20 m ahead
    car = track("c", "CAR", lambda t: (70.0, 40.0 - 8.0 * t), yaw=-math.pi / 2, length=4.5, width=1.8)
    w = explain(car, straight_ref())
    assert w["reasons"].get("crossing", 0) > 0.0 or w["reasons"].get("vehicle", 0) > 0.0, w
    first = next(t * 0.5 for t in range(20) if W.reasons_at(W.ObjectFrames(car), W.build_way(straight_ref(), *EGO, 60.0), BODY, t * 0.5))
    assert first < 4.5, first   # justified before it is on the way (it gets there at ~4.8 s)
    # an oncoming car in its own lane (y=-3.5) passes by
    onc = track("o", "CAR", lambda t: (120.0 - 10.0 * t, -3.5), yaw=math.pi, length=4.5, width=1.8)
    w = explain(onc, straight_ref())
    assert w["justified_sec"] == 0.0, w
    print("PASS: cross traffic entering the way is a reason; an oncoming car in its own lane is not")

    w = explain([], straight_ref(), light_red=lambda t, x, y, yaw: t < 4.0)
    assert w["reasons"] == {"red_light": 4.0} and w["justified_sec"] == 4.0, w
    print("PASS: red light seconds are counted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
