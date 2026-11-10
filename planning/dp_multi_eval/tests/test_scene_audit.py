"""The scene audit: defects of the scene player, found on what it publishes.

* an NPC standing on a crosswalk longer than its recording stood there;
* an NPC appearing within 15 m of the ego (a recorded blip is counted apart);
* an NPC that vanishes while it waits beside the ego; a jump;
* nothing "appears" or "jumps" when the scene is moved or switched on, nobody "stands
  too long" while the scene clock is held.

    python3 tests/test_scene_audit.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.scene_audit import SceneAudit, summarize  # noqa: E402
from dp_multi_eval.shield_map import ShieldMap  # noqa: E402

T0 = 2e9
EGO = NS(position=NS(x=0.0, y=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))
CROSSWALK = ShieldMap(crosswalks=[[(18.0, -6.0), (22.0, -6.0), (22.0, 6.0), (18.0, 6.0)]])


def obj(n, x, y, v=0.0, label=7):
    return NS(object_id=NS(uuid=bytes([n] * 16)), classification=[NS(label=label)],
              kinematics=NS(pose_with_covariance=NS(pose=NS(position=NS(x=x, y=y))),
                            twist_with_covariance=NS(twist=NS(linear=NS(x=v, y=0.0)))))


def track(n, k0, k1, fn):
    ks = range(k0, k1)
    return NS(times=[T0 + 0.1 * k for k in ks], samples=[obj(n, *fn(k)) for k in ks], ext=0)


def run(audit, k0, k1, scene, **kw):
    out = []
    for k in range(k0, k1):
        out += audit.check(T0 + 0.1 * k, T0 + 0.1 * k, scene(k), EGO, **kw)
    return out


def kinds(found):
    return sorted({(f["audit"], f["id"][:2]) for f in found})


def main() -> int:
    # 1: walks across the crosswalk in the recording (x=20, y -8 -> 8 in 10 s) but is shown standing on it.
    # 2: stood 20 s on it in the recording too. 3: parked off the road.
    tracks = {
        bytes([1] * 16): track(1, 0, 100, lambda k: (20.0, -8.0 + 0.16 * k)),
        bytes([2] * 16): track(2, 0, 300, lambda k: (21.0, 3.0)),
        bytes([3] * 16): track(3, 0, 300, lambda k: (30.0, 9.0)),
        bytes([4] * 16): track(4, 150, 155, lambda k: (8.0, 4.0)),          # 0.5 s blip
        bytes([5] * 16): track(5, 150, 250, lambda k: (9.0, -4.0)),
        bytes([6] * 16): track(6, 0, 400, lambda k: (12.0 + 0.0 * k, 5.0)),
        bytes([7] * 16): track(7, 0, 220, lambda k: (10.0, -5.0)),
    }
    audit = SceneAudit(tracks, CROSSWALK)
    scene = lambda k: [obj(1, 20.0, 0.0), obj(2, 21.0, 3.0), obj(3, 30.0, 9.0)]  # noqa: E731
    found = run(audit, 0, 120, scene, held={bytes([1] * 16): "ego"})
    assert kinds(found) == [("stand_in_zone", "01")], kinds(found)
    assert found[0]["why"] == "ego" and found[0]["rec_sec"] < 1.0 and 4.0 <= found[0]["sec"] <= 4.5, found[0]
    assert len(found) == 2 and found[1]["sec"] >= 9.0, "reported again while it lasts"
    print("PASS: standing on a crosswalk where its recording walked is found (with the rule that held it); "
          "standing where the recording stood, or off the road, is not")

    # The scene clock held (a leg's start): nobody stands too long.
    audit = SceneAudit(tracks, CROSSWALK)
    found = []
    for k in range(120):
        found += audit.check(T0 + 0.1 * k, T0 + 5.0, scene(k), EGO)
    assert found == [], found
    print("PASS: nothing is found while the scene clock is held")

    # Appearing near the ego: a recorded track of 10 s is a pop-in, of 0.5 s a blip; far away neither.
    audit = SceneAudit(tracks, CROSSWALK)
    scene = lambda k: (([obj(4, 8.0, 4.0)] if 150 <= k < 155 else [])  # noqa: E731
                       + ([obj(5, 9.0, -4.0)] if k >= 150 else []) + ([obj(3, 30.0, 9.0)] if k >= 160 else []))
    found = run(audit, 100, 200, scene)
    assert kinds(found) == [("blip", "04"), ("pop_in", "05")], kinds(found)
    assert [f["src"] for f in found if f["audit"] == "pop_in"] == ["birth"]
    print("PASS: appearing within 15 m is a pop-in — a recorded blip is counted apart, farther away is fine")

    # Switched on, or moved: the scene that is there is not an appearance.
    audit = SceneAudit(tracks, CROSSWALK)
    near = lambda k: [obj(6, 12.0, 5.0), obj(7, 10.0, -5.0)]  # noqa: E731
    assert run(audit, 0, 20, near) == []
    audit.reset()
    assert run(audit, 20, 40, near) == []
    assert audit.check(T0 + 4.0, T0 + 4.0, None, EGO) == [] and run(audit, 41, 60, near) == []
    print("PASS: nothing appears when the scene is switched on or moved")

    # 7 waits for the ego (held) until its recorded time runs out at 22 s and is gone: found.
    # 6 is taken out while its recording goes on: found. A track that just ends near the ego: not.
    audit = SceneAudit(tracks, CROSSWALK)
    scene = lambda k: ([obj(7, 10.0, -5.0)] if k < 220 else []) + ([obj(6, 12.0, 5.0)] if k < 250 else [])  # noqa: E731
    found = run(audit, 100, 300, scene, held={bytes([7] * 16): "ego"})
    assert sorted((f["audit"], f["id"][:2], f["why"]) for f in found) == [
        ("vanish", "06", "dropped"), ("vanish", "07", "held_to_its_end")], found
    audit = SceneAudit(tracks, CROSSWALK)
    assert run(audit, 100, 300, lambda k: [obj(7, 10.0, -5.0)] if k < 220 else []) == []
    print("PASS: vanishing while held, or while its recording goes on, is found; a recorded track's end is not")

    audit = SceneAudit(tracks, CROSSWALK)
    scene = lambda k: [obj(6, 12.0 if k < 150 else 6.0, 5.0)]  # noqa: E731
    assert kinds(run(audit, 100, 200, scene)) == [("jump", "06")]
    # ... but not right after the scene was moved: a frame of the old instant
    # lands after the reset, then the new one (the harness moved the scene)
    audit = SceneAudit(tracks, CROSSWALK)
    run(audit, 100, 150, lambda k: [obj(6, 12.0, 5.0)])
    audit.reset()
    assert run(audit, 150, 200, lambda k: [obj(6, 12.0 if k < 151 else 6.0, 5.0)]) == []
    print("PASS: a 6 m step in one frame is a jump; not within the grace after the scene was moved")

    s = summarize([
        {"t": T0 + 12.0, "audit": "stand_in_zone", "id": "01" * 16, "label": 7, "x": 20.0, "y": 0.0, "d": 12.0, "sec": 3.1, "why": "ego"},
        {"t": T0 + 17.0, "audit": "stand_in_zone", "id": "01" * 16, "label": 7, "x": 20.0, "y": 0.0, "d": 12.0, "sec": 8.1, "why": "ego"},
        {"t": T0 + 15.0, "audit": "blip", "id": "04" * 16, "label": 0, "x": 8.0, "y": 4.0, "d": 1.0, "src": "recorded"},
    ], T0)
    assert s["counts"] == {"stand_in_zone": 1, "blip": 1} and s["defects"] == 1
    assert s["events"] == [{"t_rel": 12.0, "kind": "stand_in_zone", "object_id": "01010101", "label": 7,
                            "at": [20.0, 0.0], "ego_dist_m": 12.0, "sec": 8.1, "why": "ego"}], s["events"]
    print("PASS: summary — one finding per object and kind (the longest standing), blips counted, not listed")

    # -- what is the recording's own, not the player's (2026-11-10)
    # The bus (recorded ego) stood at x=-20 when 8 was first tracked 12 m in front of the
    # ego at x=12, and at x=0 when 9 was: 8 appears 32 m nearer than the bus first had it
    # (the player's pop-in), 9 as near as the bus had it (the recording's own).
    ego_path = ([T0 + 0.1 * k for k in range(400)], [(-20.0 if k < 150 else 0.0) for k in range(400)], [0.0] * 400)
    tracks2 = {
        bytes([8] * 16): track(8, 100, 300, lambda k: (12.0, 4.0)),
        bytes([9] * 16): track(9, 160, 300, lambda k: (12.0, -4.0)),
    }
    audit = SceneAudit(tracks2, CROSSWALK, ego_path=ego_path)
    scene = lambda k: ([obj(8, 12.0, 4.0)] if k >= 100 else []) + ([obj(9, 12.0, -4.0)] if k >= 160 else [])  # noqa: E731
    found = run(audit, 90, 200, scene)
    assert kinds(found) == [("pop_in", "08")], kinds(found)
    print("PASS: appearing no nearer than the bus first saw it is the recording's own, not a pop-in")

    # A new id where one of its class just left (a tracker handover) is not a pop-in.
    tracks3 = {bytes([10] * 16): track(10, 0, 120, lambda k: (9.0, 4.0)),
               bytes([11] * 16): track(11, 125, 300, lambda k: (9.5, 4.0))}
    audit = SceneAudit(tracks3, CROSSWALK, ego_path=([T0], [-100.0], [0.0]))
    scene = lambda k: [obj(10, 9.0, 4.0)] if k < 120 else ([obj(11, 9.5, 4.0)] if k >= 125 else [])  # noqa: E731
    found = run(audit, 0, 200, scene)
    assert kinds(found) == [], kinds(found)
    print("PASS: a new tracker id where the same object just was is not a pop-in")

    # The recording loses 12 for 3 s and finds it 10 m on: neither a vanish nor a jump.
    def gappy(k):
        return (5.0 + 0.1 * k, 4.0)
    tr12 = track(12, 0, 300, gappy)
    keep = [i for i, t in enumerate(tr12.times) if not (T0 + 10.0 <= t < T0 + 13.0)]
    tr12.times = [tr12.times[i] for i in keep]
    tr12.samples = [tr12.samples[i] for i in keep]
    still_bus = ([T0], [0.0], [0.0])     # the bus stood where the ego stands
    audit = SceneAudit({bytes([12] * 16): tr12}, CROSSWALK, ego_path=still_bus)
    scene = lambda k: [] if 100 <= k < 130 else [obj(12, *gappy(k), v=1.0)]  # noqa: E731
    found = run(audit, 0, 200, scene)
    assert kinds(found) == [], kinds(found)
    # ... and the same object shown frozen through the gap, then catching up, is a jump
    audit = SceneAudit({bytes([12] * 16): tr12}, CROSSWALK, ego_path=still_bus)
    scene = lambda k: [obj(12, *gappy(min(k, 99) if k < 130 else k), v=1.0)]  # noqa: E731
    assert kinds(run(audit, 0, 200, scene)) == [("jump", "0c")]
    print("PASS: the recording's own gaps are not vanishes, its own steps not jumps; a player's step is")
    return 0


if __name__ == "__main__":
    sys.exit(main())
