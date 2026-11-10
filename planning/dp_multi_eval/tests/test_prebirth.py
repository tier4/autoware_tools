"""Pre-birth: an object's approach before the recording first saw it.

* its past is what physics gives: one first seen standing stood there (as long
  before as it is seen to stand after), one first seen moving came from behind
  along that motion; UNKNOWN on the move, creepers and blips get none;
* never a moving object frozen at its first pose (H78, 2026-11-10: walking
  pedestrians stood 30 s mid-crosswalk in the ego's lane);
* visible once the ego is as near as the bus was when the bus first saw it,
  and from then on; not if its way to its birth runs through the ego;
* the past is part of the object's track: the shield moves it like any NPC;
* a past that would have touched the recorded vehicle is not one it had: it is
  cut to after the recorded vehicle passed.

    python3 tests/test_prebirth.py
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.prebirth import PreBirth  # noqa: E402
from dp_multi_eval.reactive_shield import ReactiveShield, ShieldParams  # noqa: E402
from dp_multi_eval.scene_driver import ObjectGate  # noqa: E402

S = 1_000_000_000
BASE_NS = 2 * 10**18      # real stamps are ns since the epoch
T0 = BASE_NS * 1e-9


def sample(n, x, y, v=0.0, label=1, yaw=0.0, length=4.5, width=1.8):
    return NS(
        object_id=NS(uuid=bytes([n] * 16)),
        classification=[NS(label=label)],
        kinematics=NS(
            pose_with_covariance=NS(pose=NS(
                position=NS(x=x, y=y, z=0.0),
                orientation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))),
            twist_with_covariance=NS(twist=NS(linear=NS(x=v, y=0.0, z=0.0))),
        ),
        shape=NS(dimensions=NS(x=length, y=width, z=1.5)),
    )


def frames(tracks_fn, n):
    """[(stamp ns, msg)] at 10 Hz from {id: fn(k) -> sample or None}."""
    return [(BASE_NS + k * S // 10, NS(objects=[s for s in (fn(k) for fn in tracks_fn) if s is not None]))
            for k in range(n)]


def ego(x, y=0.0, yaw=0.0):
    return NS(position=NS(x=x, y=y, z=0.0), orientation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))


def ids(objs):
    return sorted(o.object_id.uuid[0] for o in objs)


def build(tracks_fn, n=1000):
    """A shield over the recording and its PreBirth; the bus drives +x at 2 m/s."""
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, vru_real_body=True))
    sh.build_tracks(frames(tracks_fn, n))
    et = [T0 + 0.1 * i for i in range(n)]
    pb = PreBirth(sh.tracks, sh.parent_of, et, [0.2 * i for i in range(n)], [0.0] * n, clearance=sh._rect_clearance)
    return sh, pb


def between(k0, k1, fn):
    return lambda k: fn(k) if k0 <= k < k1 else None


RECORDING = [
    lambda k: sample(1, 500.0, 0.0),                                                     # there from the start
    between(200, 400, lambda k: sample(2, 80.0, 6.0)),                                   # parked, first seen at 20 s
    between(200, 300, lambda k: sample(3, 80.0 + 0.5 * (k - 200), -6.0, 5.0)),           # car driving +x at 5 m/s
    between(300, 302, lambda k: sample(4, 90.0, 6.0)),                                   # 0.2 s blip
    between(500, 505, lambda k: sample(5, 150.0 + 0.14 * (k - 500), 6.0, 1.4, label=7, length=0.6, width=0.6)),
    between(506, 540, lambda k: sample(6, 150.8 + 0.14 * (k - 506), 6.0, 1.4, label=7, length=0.6, width=0.6)),
    between(600, 660, lambda k: sample(7, 200.0, 0.14 * (k - 600), 1.4, label=7, yaw=math.pi / 2, length=0.6, width=0.6)),
    between(700, 760, lambda k: sample(8, 250.0 + 0.2 * (k - 700), 6.0, 0.0, label=0)),  # UNKNOWN on the move
    between(800, 860, lambda k: sample(9, 300.0 + 0.04 * (k - 800), 6.0, 0.4)),          # creeping at 0.4 m/s
    between(900, 960, lambda k: sample(10, 350.0 + (0.0 if k < 925 else 0.3 * (k - 925)), 6.0, label=7,
                                       length=0.6, width=0.6)),                          # paused 2.5 s, then walked
]


def main() -> int:
    sh, pb = build(RECORDING)
    kinds = {p.key[0]: p.kind for p in pb._pasts}
    assert kinds == {2: "stood", 3: "moving", 7: "moving", 10: "stood"}, kinds
    print("PASS: a past for what stood or moved at birth — not for blips, re-born ids too short to judge, "
          "UNKNOWN on the move, creepers")

    past = {p.key[0]: p for p in pb._pasts}
    assert abs((past[2].t_birth - past[2].times[0]) - 20.0) < 0.6, "parked 20 s in view: 20 s before"
    assert abs((past[10].t_birth - past[10].times[0]) - 2.5) < 0.6, (
        "a pedestrian that paused 2.5 s is not shown standing 30 s", past[10].t_birth - past[10].times[0])
    p7 = past[7]
    assert abs((p7.t_birth - p7.times[0]) - 8.0) < 0.2 and abs(p7.v - 1.4) < 0.05
    y_first = p7.samples[0].kinematics.pose_with_covariance.pose.position.y
    assert abs(y_first + 1.4 * 8.0) < 0.3, f"walked in from 11 m back, not {y_first:.1f}"
    steps = [b.kinematics.pose_with_covariance.pose.position.y - a.kinematics.pose_with_covariance.pose.position.y
             for a, b in zip(p7.samples, p7.samples[1:])]
    assert min(steps) > 0.12, "a walking pedestrian is never frozen on its past"
    tw = p7.samples[0].kinematics.twist_with_covariance.twist.linear
    assert abs(math.hypot(tw.x, tw.y) - 1.4) < 0.05, "it reports the speed it is shown at"
    p3 = past[3]
    assert abs((p3.t_birth - p3.times[0]) - 6.0) < 0.2, "a car is traced back 30 m at most"
    print("PASS: stood as long before as after (≤ 30 s); a mover traced back at its speed (≤ 8 s, ≤ 30 m), never frozen")

    # -- visibility. At scene t=14 the bus is at x=28; the parked car (bus first saw it from 40 m).
    def sees(t, e, n=2, present=frozenset()):
        return n in ids(pb.objects_at(T0 + t, e, set(present)))
    assert not sees(12.0, ego(30.0)), "farther than the bus first saw it"
    assert sees(12.0, ego(50.0))
    assert sees(12.5, ego(20.0)), "seen once, it stays in view"
    pb.reset()
    assert not sees(12.5, ego(20.0)), "after the scene was moved it is judged anew"
    assert not sees(12.0, ego(50.0), present={bytes([2] * 16)}), "never a second copy"
    assert not sees(25.0, ego(60.0)), "born: the recording shows it"
    print("PASS: visible once the ego is as near as the bus first was, then for good; anew after a scene jump")

    # The car's way in (y=-6, x 50..80) with the ego standing across it: no past through the ego.
    pb.reset()
    assert 3 not in ids(pb.objects_at(T0 + 18.0, ego(70.0, -2.0, -math.pi / 2), set()))
    assert 3 in ids(pb.objects_at(T0 + 18.0, ego(60.0, 10.0), set()))
    out = [o for o in pb.objects_at(T0 + 18.0, ego(60.0, 10.0), set()) if o.object_id.uuid[0] == 3][0]
    assert abs(out.kinematics.pose_with_covariance.pose.position.x - 70.0) < 0.3
    assert sh.tracks[bytes([3] * 16)].samples[0].kinematics.pose_with_covariance.pose.position.x == 80.0, (
        "the recorded samples are not touched")
    print("PASS: no past through the ego; injected objects are copies")

    # -- one path, one motion logic: the past is part of the track in the shield.
    sh, pb = build(RECORDING)
    gate = ObjectGate(shield=sh, prebirth=pb)
    standing_end = set(gate._standing_at_end)
    pb.extend(sh)
    tr3 = sh.tracks[bytes([3] * 16)]
    assert tr3.ext == 60 and abs(tr3.times[0] - T0 - 14.0) < 0.11 and abs(tr3.s[-1] - (30.0 + 49.5)) < 0.6
    assert ObjectGate(shield=sh)._standing_at_end == standing_end, "a given past is no evidence of standing at the end"
    assert ReactiveShield.end_speed(sh.tracks[bytes([7] * 16)]) > 1.0

    def play(k0, k1, ego_fn, want):
        shown = []
        for k in range(k0, k1):
            msg = NS(objects=[s for s in (fn(k) for fn in RECORDING) if s is not None])
            out = gate.apply(msg, ego_pose=ego_fn(k), ego_speed=0.0, ts_ns=BASE_NS + k * S // 10, at_end=False,
                             objects_on=True, now_ns=BASE_NS + k * S // 10).objects
            got = [o for o in out if o.object_id.uuid[0] == want]
            shown.append(got[0].kinematics.pose_with_covariance.pose.position if got else None)
        return shown

    # The pedestrian (id 7) walks in along x=200 from y=-11 and is born at y=0.
    walk = play(515, 640, lambda k: ego(180.0), 7)
    ys = [p.y for p in walk if p is not None]
    assert walk[0] is None and len(ys) > 100 and ys[0] < -10.0 and ys[-1] > 5.0, (len(ys), ys[:1], ys[-1:])
    assert all(b >= a - 1e-6 for a, b in zip(ys, ys[1:])) and max(b - a for a, b in zip(ys, ys[1:])) < 0.5, (
        "one continuous walk across its birth")
    # The same pedestrian with the bus standing on its way (body over y -3..0 at x 195..206):
    # it stops short of the bus, as any NPC does, and is not pushed through it.
    gate.reset()
    # (the bus pulls onto its way once it is in view; with the bus there from
    # the start it has no past and is born as recorded)
    blocked = play(515, 640, lambda k: ego(180.0) if k < 540 else ego(198.0, -1.5), 7)
    ys = [p.y for p in blocked if p is not None]
    assert ys and ys[0] < -10.0 and max(ys) < -3.5, f"a pre-born pedestrian walked into the bus: y up to {max(ys):.1f}"
    gate.reset()
    born = [p.y for p in play(515, 640, lambda k: ego(198.0, -1.5), 7) if p is not None]
    assert born and min(born) >= 0.0, "a past through a standing bus was shown"
    print("PASS: the past is part of the track — a continuous walk across the birth, and the shield holds it like any NPC")

    assert gate.apply(NS(objects=[]), ego_pose=ego(180.0), ego_speed=0.0, ts_ns=BASE_NS + 560 * S // 10, at_end=False,
                      objects_on=False, now_ns=0).objects == []
    assert gate.apply(NS(objects=[]), ego_pose=ego(180.0), ego_speed=0.0, ts_ns=BASE_NS + 560 * S // 10, at_end=True,
                      objects_on=True, now_ns=0).objects == []
    print("PASS: nothing is added with objects off or past the recording's end")

    # The recording is the reference (H39, 2026-11-10): the bus drives +x at
    # 2 m/s along y=0 and is at x=100 at 50 s. A pedestrian first seen at 52 s
    # at (97, -2.2), walking -y at 1 m/s, traced straight back 8 s would have
    # crossed y=0 at x=97 at 49.8 s — through the bus's body (x 97.9-108.8
    # then; its rear clears x=97 only by 0.9 m... within REC_CLEAR of its side).
    def walker(k):
        return sample(9, 97.0 + 0.0 * k, -2.2 - 0.1 * (k - 520), 1.0, label=7, yaw=-math.pi / 2, length=0.6, width=0.6)

    def with_poses(tracks_fn, n=1000, poses=True):
        sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, vru_real_body=True))
        sh.build_tracks(frames(tracks_fn, n))
        et = [T0 + 0.1 * i for i in range(n)]
        xs = [0.2 * i for i in range(n)]
        return PreBirth(sh.tracks, sh.parent_of, et, xs, [0.0] * n, clearance=sh._rect_clearance,
                        ego_poses=[ego(x) for x in xs] if poses else None)

    rec = [lambda k: sample(1, 500.0, 0.0), between(520, 600, walker)]   # 1: there from the start
    free = with_poses(rec, poses=False)
    cut = with_poses(rec)
    p_free = next(p for p in free._pasts if p.key == bytes([9] * 16))
    assert len(p_free.times) == 80 and free.cut_by_recording == 0, len(p_free.times)
    ys_free = [q.kinematics.pose_with_covariance.pose.position.y for q in p_free.samples]
    assert max(ys_free) > 5.0, "scenario changed: the straight past does not cross the bus's lane"
    assert cut.cut_by_recording == 1
    p_cut = next((p for p in cut._pasts if p.key == bytes([9] * 16)), None)
    assert p_cut is not None and 0 < len(p_cut.times) < 80, None if p_cut is None else len(p_cut.times)
    sh2 = ReactiveShield(ShieldParams(vehicle_real_body=True, vru_real_body=True))
    for t, q in zip(p_cut.times, p_cut.samples):
        k = round((t - T0) / 0.1)
        assert sh2._rect_clearance(q, ego(0.2 * k)) >= 0.3, (t - T0, sh2._rect_clearance(q, ego(0.2 * k)))
    print(f"PASS: a past through the recorded vehicle is cut to after it passed ({len(p_cut.times) / 10:.1f} s of 8 s kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
