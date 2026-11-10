"""_index_objects_per_frame: one box per object per frame. Run: python3 tests/test_render_objects_per_frame.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.bag_reader import EgoSample, ObjectSample  # noqa: E402
from dp_multi_eval.render_video import _index_objects_per_frame  # noqa: E402


def ego(t):
    return EgoSample(stamp_sec=t, x=0.0, y=0.0, z=0.0, yaw_rad=0.0, vx=0.0, vy=0.0, speed_mps=0.0)


def obj(t, oid, x):
    return ObjectSample(stamp_sec=t, object_id=oid, label="CAR", x=x, y=0.0, yaw_rad=0.0, length_m=4.0, width_m=2.0)


def main() -> int:
    frames = [ego(10.0), ego(10.5)]
    # object A sampled at 10 Hz, object B once
    objs = [obj(9.6 + 0.1 * k, "A", float(k)) for k in range(10)] + [obj(10.4, "B", 50.0)]
    per = _index_objects_per_frame(objs, frames, 0.5)
    assert len(per[0]) == 2 and len(per[1]) == 2, per
    a0 = next(o for o in per[0] if o.object_id == "A"); a1 = next(o for o in per[1] if o.object_id == "A")
    assert abs(a0.stamp_sec - 10.0) < 1e-9 and abs(a1.stamp_sec - 10.5) < 1e-9, (a0.stamp_sec, a1.stamp_sec)
    print("PASS: one box per object per frame, the sample nearest the frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
