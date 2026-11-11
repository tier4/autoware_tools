# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavioural suite for reactive_shield.

Every scenario here reproduces a failure that once cost a full evaluation
pass; keep them green. Run directly:

    python3 tests/test_reactive_shield.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp_multi_eval.reactive_shield import ReactiveShield, ShieldParams  # noqa: E402


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def mkobj(uuid, x, y, yaw=0.0, vx=6.7):
    q = NS(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))
    return NS(
        object_id=NS(uuid=uuid),
        shape=NS(dimensions=NS(x=4.5, y=1.8, z=1.5)),
        kinematics=NS(
            pose_with_covariance=NS(pose=NS(position=NS(x=x, y=y, z=0.0), orientation=q)),
            twist_with_covariance=NS(twist=NS(linear=NS(x=vx, y=0.0, z=0.0))),
        ),
    )


def ego_at(x, y, yaw=0.0):
    return NS(position=NS(x=x, y=y, z=0.0),
              orientation=NS(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2)))


BASE = 2e18 * 1e-9
U = bytes(range(16))
U2 = bytes(range(16, 32))
U3 = bytes(range(32, 48))
UP = bytes(range(48, 64))
UX = bytes(range(64, 80))


def drive_bag(n=600, speed=6.7, y=0.0, uuid=U, yaw=0.0):
    return [(int(k * 0.1 * 1e9 + 2e18),
             NS(objects=[mkobj(uuid, k * 0.1 * speed, y, yaw, speed)]))
            for k in range(n)]


def test_hold_no_pass_through_catchup():
    """A queue NPC waits behind a parked ego, resumes bounded, expires at end."""
    sh = ReactiveShield(ShieldParams())
    sh.build_tracks(drive_bag())
    ego = ego_at(120.0, 0.0)
    worst = 1e9
    last_x = None
    for k in range(400):
        ft = BASE + k * 0.1
        msg = NS(objects=[mkobj(U, (ft - BASE) * 6.7, 0.0)])
        sh.apply(msg, ego, ft, 0.1, ego_speed_mps=0.0)
        if msg.objects:
            p = msg.objects[0].kinematics.pose_with_covariance.pose.position
            worst = min(worst, sh._rect_clearance(msg.objects[0], ego))
            last_x = p.x
    assert worst >= 0.2, f"hold margin violated: {worst:.2f}"
    assert last_x is not None and last_x < 120.0, "passed through a parked ego"
    ego2 = ego_at(120.0, 50.0)
    prev = last_x
    maxstep = 0.0
    expired = False
    for k in range(400, 1200):
        ft = BASE + k * 0.1
        msg = NS(objects=[mkobj(U, min((ft - BASE) * 6.7, 59.9 * 6.7), 0.0)])
        sh.apply(msg, ego2, ft, 0.1, ego_speed_mps=6.0)
        if not msg.objects:
            expired = True  # bag-end mover leaves once its clock passes the end
            break
        p = msg.objects[0].kinematics.pose_with_covariance.pose.position
        maxstep = max(maxstep, p.x - prev)
        prev = p.x
    assert maxstep <= 6.7 * 0.1 * 2 + 0.05, f"resume step too large: {maxstep:.2f}"
    assert prev > 300.0 and expired, f"resume/expiry broken: x={prev:.1f} expired={expired}"


def test_frame_flap_no_teleport():
    """Frame-selector flapping cannot teleport a governed NPC near the ego."""
    sh = ReactiveShield(ShieldParams())
    sh.build_tracks(drive_bag())
    ego = ego_at(100.0, 3.5)
    seq = [10.0 + k * 0.1 for k in range(50)]
    seq += [21.0 + k * 0.1 for k in range(30)]
    seq += [15.0 + k * 0.1 for k in range(30)]
    seq += [24.0 + (k % 2) * (-6.0) + k * 0.1 for k in range(60)]
    prevp = None
    for ft in seq:
        msg = NS(objects=[mkobj(U, ft * 6.7, 0.0)])
        sh.apply(msg, ego, BASE + ft, 0.1, ego_speed_mps=6.0)
        if not msg.objects:
            prevp = None
            continue
        p = msg.objects[0].kinematics.pose_with_covariance.pose.position
        if prevp is not None:
            near = min(math.hypot(p.x - 100.0, p.y - 3.5),
                       math.hypot(prevp[0] - 100.0, prevp[1] - 3.5)) <= 120.0
            step = math.hypot(p.x - prevp[0], p.y - prevp[1])
            assert not (near and step > 6.7 * 0.1 * 2 + 0.3), f"teleport {step:.1f} m"
        prevp = (p.x, p.y)


def test_rewound_frames_do_not_freeze():
    """A rewound repeat window must not pin an NPC in a moving ego's lane."""
    sh = ReactiveShield(ShieldParams())
    sh.build_tracks(drive_bag(1200))
    ego_far = ego_at(70.0, 40.0)
    msg = None
    for k in range(200):
        ft = BASE + k * 0.1
        msg = NS(objects=[mkobj(U, (ft - BASE) * 6.7, 0.0)])
        sh.apply(msg, ego_far, ft, 0.1, ego_speed_mps=0.0)
    x0 = msg.objects[0].kinematics.pose_with_covariance.pose.position.x
    ex = x0 - 60.0
    frozen = 0
    prevx = x0
    last = prevx
    for k in range(200, 400):
        ft = BASE + 20.0 + (k - 200) * 0.001  # frame clock nearly frozen
        ex += 0.85
        msg = NS(objects=[mkobj(U, 20.0 * 6.7, 0.0)])
        sh.apply(msg, ego_at(ex, 0.0), ft, 0.1, ego_speed_mps=8.5)
        if not msg.objects:
            continue
        p = msg.objects[0].kinematics.pose_with_covariance.pose.position
        if p.x - prevx < 0.001:
            frozen += 1
        prevx = p.x
        last = p.x
    assert last - x0 > 100.0, f"froze under rewound frames: +{last - x0:.1f} m"
    assert frozen < 20, f"{frozen} frozen ticks"


def test_transient_ghost_expiry_and_parker():
    """A mid-bag transient track leaves after its life; a bag-end parker stays."""
    bag = []
    for k in range(1200):
        t = k * 0.1
        objs = []
        if t <= 8.0:
            objs.append(mkobj(U, t * 5.0, 0.0, vx=5.0))
        objs.append(mkobj(U2, 30.0, 5.0, vx=0.0))
        bag.append((int(t * 1e9 + 2e18), NS(objects=objs)))
    sh = ReactiveShield(ShieldParams())
    sh.build_tracks(bag)
    ego = ego_at(20.0, -4.0)
    ghost_tail = 0
    parker_all = True
    for k in range(300):
        ft = BASE + (k % 80) * 0.1
        objs = []
        if (k % 80) * 0.1 <= 8.0:
            objs.append(mkobj(U, (k % 80) * 0.1 * 5.0, 0.0, vx=5.0))
        objs.append(mkobj(U2, 30.0, 5.0, vx=0.0))
        msg = NS(objects=objs)
        sh.apply(msg, ego, ft, 0.1, ego_speed_mps=0.0)
        ids = [bytes(o.object_id.uuid) for o in msg.objects]
        if k >= 120 and U in ids:
            ghost_tail += 1
        if U2 not in ids:
            parker_all = False
    assert ghost_tail == 0, f"ghost survived {ghost_tail} late ticks"
    assert parker_all, "bag-end parker was dropped"


def test_subsample_tick_flows():
    """dt below the track's sample spacing must still accumulate progress."""
    sh = ReactiveShield(ShieldParams())
    sh.build_tracks(drive_bag(400, speed=12.0))
    ego = ego_at(20.0, -3.5)
    prevx = None
    maxstep = 0.0
    for k in range(1000):
        msg = NS(objects=[mkobj(U, 0.0, 0.0, vx=12.0)])
        sh.apply(msg, ego, BASE + 0.0, 0.03, ego_speed_mps=8.5)
        if not msg.objects:
            break
        p = msg.objects[0].kinematics.pose_with_covariance.pose.position
        if prevx is not None:
            maxstep = max(maxstep, p.x - prevx)
        prevx = p.x
    assert prevx is not None and prevx > 250.0, f"sub-sample freeze at x={prevx}"
    assert maxstep <= 12.0 * 0.03 * 1.5 + 1.3, f"step bound exceeded: {maxstep:.2f}"


def test_headon_held_with_body_clearance():
    """A head-on oncoming car keeps true body-to-body clearance."""
    bag = [(int(k * 0.1 * 1e9 + 2e18),
            NS(objects=[mkobj(U, 250.0 - k * 0.1 * 8.0, 0.0, math.pi, 8.0)]))
           for k in range(370)]
    sh = ReactiveShield(ShieldParams())
    sh.build_tracks(bag)
    ex = 0.0
    minclr = 1e9
    for k in range(370):
        ft = BASE + k * 0.1
        ex += 0.03
        ego = ego_at(ex, 0.0)
        msg = NS(objects=[mkobj(U, 250.0 - k * 0.1 * 8.0, 0.0, math.pi, 8.0)])
        sh.apply(msg, ego, ft, 0.1, ego_speed_mps=0.3)
        if msg.objects:
            minclr = min(minclr, sh._rect_clearance(msg.objects[0], ego))
    assert minclr >= 0.15, f"head-on overlap: {minclr:.2f}"


def test_fast_headon_no_overlap_and_overtake_passes():
    """18+ m/s closing never overlaps; a parallel overtake still gets past."""
    bag = [(int(k * 0.1 * 1e9 + 2e18),
            NS(objects=[mkobj(U, 300.0 - k * 0.1 * 12.0, 0.0, math.pi, 12.0)]))
           for k in range(300)]
    sh = ReactiveShield(ShieldParams())
    sh.build_tracks(bag)
    ex = 0.0
    minclr = 1e9
    advanced_inside_margin = 0
    prev_x = None
    for k in range(300):
        ft = BASE + k * 0.1
        ex += 0.1 * 6.5
        ego = ego_at(ex, 0.0)
        msg = NS(objects=[mkobj(U, 300.0 - k * 0.1 * 12.0, 0.0, math.pi, 12.0)])
        sh.apply(msg, ego, ft, 0.1, ego_speed_mps=6.5)
        if msg.objects:
            clr = sh._rect_clearance(msg.objects[0], ego)
            minclr = min(minclr, clr)
            x = msg.objects[0].kinematics.pose_with_covariance.pose.position.x
            # it drives -x; advancing means x decreasing. Once the ego has
            # driven through it, advancing takes it away from the ego: only
            # a step toward the ego counts.
            if prev_x is not None and clr < ShieldParams().margin_m and x < prev_x - 1e-6:
                moved = msg.objects[0]
                back = mkobj(U, prev_x, 0.0, math.pi, 0.0)
                if clr < sh._rect_clearance(back, ego) - 1e-6:
                    advanced_inside_margin += 1
            prev_x = x
    # The shield no longer hides a car the ego closes in on (operator,
    # 2026-09-15): what it still guarantees is that the car never DRIVES into
    # the ego — inside the margin its pose stops advancing, so any remaining
    # closing is the ego's own doing and is reported as a contact.
    assert minclr < 1e8, "object never shown"
    assert advanced_inside_margin == 0, f"{advanced_inside_margin} ticks advanced inside the margin"

    bag2 = [(int(k * 0.1 * 1e9 + 2e18),
             NS(objects=[mkobj(U3, k * 0.1 * 8.0, 3.4, 0.0, 8.0)]))
            for k in range(300)]
    sh2 = ReactiveShield(ShieldParams())
    sh2.build_tracks(bag2)
    ego2 = ego_at(100.0, 0.0)
    maxx = -1e9
    for k in range(300):
        ft = BASE + k * 0.1
        msg = NS(objects=[mkobj(U3, k * 0.1 * 8.0, 3.4, 0.0, 8.0)])
        sh2.apply(msg, ego2, ft, 0.1, ego_speed_mps=0.0)
        if msg.objects:
            maxx = max(maxx, msg.objects[0].kinematics.pose_with_covariance.pose.position.x)
    assert maxx > 200.0, f"parallel overtake blocked at x={maxx:.1f}"


def test_only_the_ego_can_close_the_gap():
    """A parked car the ego drives into stays visible all the way to contact.

    Until 2026-09-15 it was hidden the moment the ego came inside the shield
    margin: on the video a car brushed past at 0.4 m and then vanished, and
    the metric recorded nothing. Now it keeps its pose, so a real touch is a
    contact in the report.
    """
    sh = ReactiveShield(ShieldParams())
    parked = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=[mkobj(U, 100.0, 0.0, 0.0, 0.0)]))
              for k in range(200)]
    sh.build_tracks(parked)
    shown = 0
    moved = 0
    prev_x = None
    last_clr = None
    ex = 80.0
    for k in range(60):
        ex += 0.3
        ego = ego_at(ex, 0.0)
        msg = NS(objects=[mkobj(U, 100.0, 0.0, 0.0, 0.0)])
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=3.0)
        if msg.objects:
            shown += 1
            x = msg.objects[0].kinematics.pose_with_covariance.pose.position.x
            if prev_x is not None and abs(x - prev_x) > 1e-6:
                moved += 1
            prev_x = x
            last_clr = sh._rect_clearance(msg.objects[0], ego)
    assert shown == 60, f"parked car hidden on {60 - shown} of 60 ticks"
    assert moved == 0, f"parked car moved on {moved} ticks"
    assert last_clr is not None and last_clr < 0.35, (
        f"ego never reached the car ({last_clr}); scenario no longer tests the rule"
    )


def test_a_car_the_ego_comes_up_behind_is_not_held():
    """H82, 2026-11-09: a car born 16 m ahead of an ego 150 m ahead of the
    recording drove on at 5.6 m/s; the ego came up behind it at 8.6 m/s and
    the shield, seeing the gap close, held the car — the ego ran into a car
    standing in its lane. Moving away from the ego is never held."""
    car = lambda k: mkobj(U, 20.0 + k * 0.1 * 5.6, 0.0, 0.0, 5.6)  # noqa: E731
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=[car(k)])) for k in range(100)]
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True))
    sh.build_tracks(bag)
    xs = []
    for k in range(40):
        ego = ego_at(k * 0.1 * 8.6, 0.0)
        msg = NS(objects=[car(k)])
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=8.6)
        xs.append(msg.objects[0].kinematics.pose_with_covariance.pose.position.x)
    stalls = sum(1 for a, b in zip(xs, xs[1:]) if b - a < 0.3)
    assert stalls == 0, f"held {stalls} ticks in front of the ego closing in from behind"


def test_materialising_inside_the_ego_is_omitted():
    """An object whose first pose is already inside the ego is not published."""
    sh = ReactiveShield(ShieldParams())
    # a car parked exactly where the ego stands, seen for the first time here
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=[mkobj(U2, 100.0, 0.0, 0.0, 0.0)]))
           for k in range(100)]
    sh.build_tracks(bag)
    ego = ego_at(100.0, 0.0)
    shown = 0
    for k in range(60):
        msg = NS(objects=[mkobj(U2, 100.0, 0.0, 0.0, 0.0)])
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        shown += len(msg.objects)
    assert shown == 0, f"car materialised inside the ego on {shown} ticks"


def test_born_driving_at_the_nose_comes_out_of_its_shadow():
    """H78, 2026-11-10: a car crossing in front of the bus was first tracked
    0.1 m from the nose of the sim ego standing mid-turn; the shield held it
    there 30 s (143 contact frames). A track born driving within the standoff
    is born at its first pose out of the ego's way; a parked one beside the
    ego is shown where it is."""
    P = ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, vru_real_body=True)
    ego = ego_at(0.0, 0.0)
    # crossing the nose (ego front ~7.6 m) at 5 m/s, -y, 1 m off it
    car = lambda k: mkobj(U, 10.0, 2.0 - 0.5 * k, -math.pi / 2, 5.0)  # noqa: E731
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=[car(k)])) for k in range(60)]
    sh = ReactiveShield(P)
    sh.build_tracks(bag)
    gaps = []
    for k in range(60):
        msg = NS(objects=[car(k)])
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        gaps += [sh._rect_clearance(o, ego) for o in msg.objects]
    assert gaps, "never born: a standing ego must not keep a passing car away for good"
    assert gaps[0] >= P.standoff_m, f"born {gaps[0]:.2f} m from the ego"
    # parked 1.5 m beside the ego, first seen now: shown
    sh2 = ReactiveShield(P)
    park = lambda: mkobj(U2, 2.0, 3.0, 0.0, 0.0)  # noqa: E731
    sh2.build_tracks([(int(k * 0.1 * 1e9 + 2e18), NS(objects=[park()])) for k in range(60)])
    shown = 0
    for k in range(30):
        msg = NS(objects=[park()])
        sh2.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        shown += len(msg.objects)
    assert shown == 30, shown
    # H82, 2026-11-11: a car that drove up at 5 m/s and has stood 1 m beside
    # the ego since — first shown there after a scene jump (reset): no birth,
    # shown where it stands (it was withheld 10 s and came out behind the ego)
    queue = lambda k: mkobj(U3, min(-40.0 + 0.5 * k, 5.0), 3.4, 0.0, 5.0 if k < 90 else 0.0)  # noqa: E731
    sh3 = ReactiveShield(P)
    sh3.build_tracks([(int(k * 0.1 * 1e9 + 2e18), NS(objects=[queue(k)])) for k in range(200)])
    sh3.apply(NS(objects=[queue(0)]), ego, BASE, 0.1, ego_speed_mps=0.0)
    sh3.reset()
    shown = 0
    for k in range(120, 150):
        msg = NS(objects=[queue(k)])
        sh3.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        shown += len(msg.objects)
    assert shown == 30, f"a car standing beside the ego after a scene jump was shown on {shown}/30 ticks"


def test_scene_jump_keeps_vehicles_driving_away():
    """H82, 2026-11-11: the ego drove on past a recorded stand and the scene
    jumped 29 s to the recorded departure — three cars ahead of it, going its
    way, were gone in the next frame while the ego kept its place and speed.
    A vehicle going the ego's way ahead of it keeps its own clock over
    the jump: it drives on along its recorded track, at its recorded speed,
    until it is out of reach. One coming toward the ego, and one standing, are the
    new instant's. When the scene is moved with the ego (reset) nothing stays."""
    P = ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, vru_real_body=True, npc_following=True)
    ego = ego_at(0.0, 0.0)

    def car(uuid, x, y, yaw, v):
        o = mkobj(uuid, x, y, yaw, v)
        o.classification = [NS(label=1)]
        return o
    def frame(t):   # the recording at t seconds
        out = [car(U3, 30.0, -4.0, 0.0, 0.0)]                                    # parked beside the road
        if t <= 15.0:
            out.append(car(U, 20.0 + 8.0 * t, 0.0, 0.0, 8.0))                    # ahead, driving away; lost at 15 s
        out.append(car(U2, 80.0 - 8.0 * t, 3.5, math.pi, 8.0))                   # oncoming
        out.append(car(UP, 25.0 - 8.0 * t, 3.5, math.pi, 8.0))                   # oncoming, past the ego by the jump
        out.append(car(UX, 30.0, -20.0 - 3.0 * t, -math.pi / 2, 3.0))            # driving off down a side street
        return out
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=frame(k * 0.1))) for k in range(600)]

    def play(jump):
        sh = ReactiveShield(P)
        sh.build_tracks(bag)
        shown = {U: [], U2: [], U3: [], UP: [], UX: []}
        def tick(t_scene, t_wall):
            msg = NS(objects=frame(t_scene))
            msg.objects += sh.carried({bytes(o.object_id.uuid) for o in msg.objects}, BASE + t_scene)
            sh.apply(msg, ego, BASE + t_scene, 0.1, ego_speed_mps=0.0)
            for o in msg.objects:
                shown[bytes(o.object_id.uuid)].append((t_wall, o.kinematics.pose_with_covariance.pose.position.x))
        for k in range(50):
            tick(k * 0.1, k * 0.1)
        kept = jump(sh)
        for k in range(50, 200):
            tick(30.0 + k * 0.1, k * 0.1)
        return kept, shown

    kept, shown = play(lambda sh: sh.carry_over(ego, 30.0))
    assert kept == {U}, kept   # not the oncoming car that has passed, nor the one turning off: both "drive away" too
    lead = shown[U]
    steps = [b[1] - a[1] for a, b in zip(lead, lead[1:])]
    assert all(b[0] - a[0] < 0.15 for a, b in zip(lead, lead[1:])), "the car driving away vanished at the jump"
    assert min(steps) >= -1e-6 and max(steps) <= 8.0 * 0.1 * 1.3, (min(steps), max(steps))
    one_sec = next(x for t, x in lead if t >= 6.0) - next(x for t, x in lead if t >= 5.0)
    assert 7.0 <= one_sec <= 9.0, f"it drove {one_sec:.1f} m in the second after the jump (recorded: 8)"
    # ... until it is scenery (twice the active radius away) or its track ends
    assert lead[-1][1] > 2.0 * P.active_radius_m, lead[-1]
    after = [x for t, x in shown[U2] if t >= 5.5]
    assert after and max(after) < -150.0, "the oncoming car is the new instant's"
    assert len([1 for t, _ in shown[U3] if t >= 5.5]) > 100, "the parked car is still there"

    kept, shown = play(lambda sh: sh.reset())
    assert not [1 for t, _ in shown[U] if t >= 5.5], "moved with the ego, the scene is the recorded instant: nothing stays"


def test_held_npc_never_reverses():
    """A blocked NPC holds its pose; it never walks back down its own path.

    H81 realtime, 2026-09-15: a car queueing behind an ego that had stopped at
    a bus stop stepped 0.6 m back down its track again and again — on the
    video, a car reversing for no reason.
    """
    sh = ReactiveShield(ShieldParams())
    sh.build_tracks(drive_bag(400, speed=6.7))
    ex = 90.0
    shown = []
    for k in range(260):
        ft = BASE + k * 0.1
        # the ego stands in the car's path, then creeps onto it
        if k > 120:
            ex -= 0.15
        ego = ego_at(ex, 0.0, yaw=math.pi)
        msg = NS(objects=[mkobj(U, min(k * 0.1, 39.9) * 6.7, 0.0, vx=6.7)])
        sh.apply(msg, ego, ft, 0.1, ego_speed_mps=0.0 if k <= 120 else 1.5)
        if msg.objects:
            shown.append((k, msg.objects[0].kinematics.pose_with_covariance.pose.position.x,
                          msg.objects[0].kinematics.twist_with_covariance.twist.linear.x))
    assert len(shown) > 50, f"object hardly ever shown ({len(shown)})"
    worst = min((b[1] - a[1] for a, b in zip(shown, shown[1:])), default=0.0)
    assert worst >= -1e-6, f"NPC reversed {worst:.2f} m along its path"


def test_frozen_npc_reports_standing_still():
    """A pose that does not move is published with zero speed.

    H81 realtime, 2026-09-15: a car held at a standing ego kept its recorded
    5.6 m/s in its twist, flickering to 0 and back every few frames, because
    on a 0.02 s tick the object's clock often did not reach the next 0.1 s
    sample and the hold bookkeeping called it "not held".
    """
    sh = ReactiveShield(ShieldParams())
    sh.build_tracks(drive_bag(400, speed=6.7))
    ego = ego_at(90.0, 0.0, yaw=math.pi)
    bad = 0
    standing = 0
    history = []
    for k in range(1500):
        ft = BASE + k * 0.02
        msg = NS(objects=[mkobj(U, min(k * 0.02, 39.9) * 6.7, 0.0, vx=6.7)])
        sh.apply(msg, ego, ft, 0.02, ego_speed_mps=0.0)
        if not msg.objects:
            history = []
            continue
        x = msg.objects[0].kinematics.pose_with_covariance.pose.position.x
        v = msg.objects[0].kinematics.twist_with_covariance.twist.linear.x
        history.append(x)
        # standing = not moved for 0.3 s (a car driving at 6.7 m/s changes
        # sample every 0.1 s, so it never stays put this long)
        if len(history) >= 15 and max(history[-15:]) - min(history[-15:]) < 1e-6:
            standing += 1
            if abs(v) > 0.1:
                bad += 1
    assert standing > 500, f"scenario no longer holds the car ({standing} standing ticks)"
    assert bad == 0, f"{bad} of {standing} standing ticks reported speed"


def test_moving_npc_reports_its_speed_on_every_tick():
    """A car driving freely near the ego keeps its recorded speed on every tick.

    2026-08-24 realtime: on a 0.02 s tick the car sat on one 0.1 s sample for
    several ticks and was published at 0 m/s on most of them while moving
    about 2 m every quarter second.
    """
    sh = ReactiveShield(ShieldParams())
    sh.build_tracks(drive_bag(400, speed=6.7, y=30.0))
    ego = ego_at(60.0, 0.0)  # 30 m to the side: governed, never blocking
    zero = 0
    shown = 0
    for k in range(300):
        msg = NS(objects=[mkobj(U, min(k * 0.02, 39.9) * 6.7, 30.0, vx=6.7)])
        sh.apply(msg, ego, BASE + k * 0.02, 0.02, ego_speed_mps=0.0)
        if msg.objects:
            shown += 1
            if abs(msg.objects[0].kinematics.twist_with_covariance.twist.linear.x) < 0.1:
                zero += 1
    assert shown > 250, f"object hardly shown ({shown})"
    assert zero == 0, f"moving car published at 0 m/s on {zero} of {shown} ticks"


def test_scene_rewind_needs_reset():
    """After the harness rewinds the scene clock, objects must follow the new
    scene time. Without reset a governed car stays ahead of the scene (the
    2026-08-24 leg-3 artefact); with reset it is back on the raw frame."""
    def run(reset):
        sh = ReactiveShield(ShieldParams())
        sh.build_tracks(drive_bag(600, speed=6.7, y=8.0))   # passes 8 m beside the ego: governed, never blocked
        ego = ego_at(150.0, 0.0)
        # drive the scene to 40 s: the car is at x = 268
        for k in range(400):
            ft = BASE + k * 0.1
            sh.apply(NS(objects=[mkobj(U, k * 0.1 * 6.7, 8.0, vx=6.7)]), ego, ft, 0.1, 0.0)
        if reset:
            sh.reset()
        # the harness rewinds the scene to 25 s (car at x = 167.5) and runs on
        shown = []
        for k in range(250, 300):
            ft = BASE + k * 0.1
            msg = NS(objects=[mkobj(U, k * 0.1 * 6.7, 8.0, vx=6.7)])
            sh.apply(msg, ego, ft, 0.1, 0.0)
            if msg.objects:
                shown.append(msg.objects[0].kinematics.pose_with_covariance.pose.position.x - k * 0.1 * 6.7)
        return shown
    ahead = run(reset=False)
    assert ahead and max(ahead) > 50.0, f"scenario no longer reproduces the drift ({max(ahead) if ahead else None})"
    synced = run(reset=True)
    assert synced and max(abs(x) for x in synced) < 1.0, f"after reset the car is still {max(synced):.1f} m off the scene"


def test_frame_clock_queues_behind_a_standing_ego():
    """Position scene (H81, 2026-10-28): the ego stands where the recording drove
    on, the scene plays on, and a car that followed the recorded bus comes up
    behind it. It waits behind the ego, never inside it, and never runs ahead
    of the frame."""
    sh = ReactiveShield(ShieldParams(frame_clock=True))
    sh.build_tracks(drive_bag())
    ego = ego_at(120.0, 0.0)
    worst, last_x, ahead_of_frame = 1e9, None, 0.0
    for k in range(400):
        ft = BASE + k * 0.1
        raw_x = (ft - BASE) * 6.7
        msg = NS(objects=[mkobj(U, raw_x, 0.0)])
        sh.apply(msg, ego, ft, 0.1, ego_speed_mps=0.0)
        if msg.objects:
            p = msg.objects[0].kinematics.pose_with_covariance.pose.position
            worst = min(worst, sh._rect_clearance(msg.objects[0], ego))
            ahead_of_frame = max(ahead_of_frame, p.x - raw_x)
            last_x = p.x
    assert worst >= 0.2, f"hold margin violated: {worst:.2f}"
    assert last_x is not None and last_x < 120.0, "passed through a standing ego"
    assert ahead_of_frame <= 0.7, f"ran {ahead_of_frame:.1f} m ahead of the frame"


def test_frame_clock_follows_a_slow_frame():
    """A governed car keeps to the frame's time: the position scene runs at 0.4x
    behind a slower ego, and the car with it. The 1x clock runs it at real time."""
    def shown_x(frame_clock):
        sh = ReactiveShield(ShieldParams(frame_clock=frame_clock))
        sh.build_tracks(drive_bag())
        ego = ego_at(20.0, 8.0)      # near (governed), never in its path
        x = None
        for k in range(50):
            ft = BASE + k * 0.04     # 0.4 s of frame per 1 s of ticks
            msg = NS(objects=[mkobj(U, (ft - BASE) * 6.7, 0.0)])
            sh.apply(msg, ego, ft, 0.1, ego_speed_mps=0.0)
            x = msg.objects[0].kinematics.pose_with_covariance.pose.position.x
        return x
    frame_x, realtime_x = shown_x(True), shown_x(False)
    assert abs(frame_x - 49 * 0.04 * 6.7) < 1.0, f"frame clock shows x={frame_x:.1f}"
    assert realtime_x > frame_x + 15.0, f"1x clock no longer runs ahead ({realtime_x:.1f} vs {frame_x:.1f})"


def test_frame_clock_never_reverses():
    """A frame that jumps back (the position clock's floor released) does not walk
    a governed car back down its path: it waits until the frame catches up."""
    sh = ReactiveShield(ShieldParams(frame_clock=True))
    sh.build_tracks(drive_bag())
    ego = ego_at(20.0, 8.0)
    xs = []
    for k in list(range(100)) + list(range(50, 120)):
        ft = BASE + k * 0.1
        msg = NS(objects=[mkobj(U, (ft - BASE) * 6.7, 0.0)])
        sh.apply(msg, ego, ft, 0.1, ego_speed_mps=0.0)
        xs.append(msg.objects[0].kinematics.pose_with_covariance.pose.position.x)
    assert all(b >= a - 1e-6 for a, b in zip(xs, xs[1:])), "a governed car stepped back"
    assert abs(xs[-1] - 119 * 0.1 * 6.7) < 1.0, f"it did not resume with the frame ({xs[-1]:.1f})"


def test_cyclist_crossing_behind_a_standing_bus_is_not_held():
    """A cyclist whose recorded path crosses 1.2 m behind a standing bus rides on
    when judged against the bus's own body (H81, 2026-10-29: the car-sized box
    held one still for 15 s); the car-sized box still holds it."""
    def bike_bag():
        out = []
        for k in range(200):        # crosses the road at x=-4.6: 1.3 m behind the bus's rear (-3.32)
            o = mkobj(U, -4.6, 8.0 - k * 0.1 * 4.0, yaw=-math.pi / 2, vx=4.0)
            o.shape.dimensions.x, o.shape.dimensions.y = 1.8, 0.6
            o.classification = [NS(label=6)]
            out.append((int(k * 0.1 * 1e9 + 2e18), NS(objects=[o])))
        return out
    def min_y(vru_real_body):
        sh = ReactiveShield(ShieldParams(frame_clock=True, vru_real_body=vru_real_body))
        sh.build_tracks(bike_bag())
        ego = ego_at(0.0, 0.0)      # the real body's rear: x = 2.12 - 5.44 = -3.32
        ys = []
        for k in range(200):
            ft = BASE + k * 0.1
            o = mkobj(U, -4.6, 8.0 - k * 0.1 * 4.0, yaw=-math.pi / 2, vx=4.0)
            o.shape.dimensions.x, o.shape.dimensions.y = 1.8, 0.6
            o.classification = [NS(label=6)]
            msg = NS(objects=[o])
            sh.apply(msg, ego, ft, 0.1, ego_speed_mps=0.0)
            if msg.objects:
                ys.append(msg.objects[0].kinematics.pose_with_covariance.pose.position.y)
        return min(ys)
    assert min_y(True) < -5.0, "the cyclist was held behind the bus"
    assert min_y(False) > 0.0, "the car-sized box no longer holds it (scenario changed)"


def _headon_run(soft_stop, ego_until=None, n=400):
    """A car at 10 m/s head-on toward a bus standing at x=0 (frame = 1x).
    Returns (frame time offset, x, published speed, clearance) per tick."""
    speed = 10.0
    bag = [(int(k * 0.1 * 1e9 + 2e18),
            NS(objects=[mkobj(U, 150.0 - k * 0.1 * speed, 0.0, math.pi, speed)]))
           for k in range(n)]
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=soft_stop))
    sh.build_tracks(bag)
    out = []
    for k in range(n):
        ft = BASE + k * 0.1
        # the bus stands in the car's way, then (optionally) leaves its lane
        ego = ego_at(0.0, 0.0) if ego_until is None or k < ego_until else ego_at(0.0, 30.0)
        msg = NS(objects=[mkobj(U, 150.0 - k * 0.1 * speed, 0.0, math.pi, speed)])
        sh.apply(msg, ego, ft, 0.1, ego_speed_mps=0.0)
        if msg.objects:
            o = msg.objects[0]
            out.append((k * 0.1, o.kinematics.pose_with_covariance.pose.position.x,
                        abs(o.kinematics.twist_with_covariance.twist.linear.x),
                        sh._rect_clearance(o, ego)))
    return out, sh


def test_soft_stop_brakes_to_a_standoff():
    """H81, 2026-10-29: a blocked oncoming car halted on one tick at its full
    recorded speed. With soft_stop it brakes along its path at about
    soft_decel and stands off ~3 m; the published speed falls with it and
    matches the distance it covers."""
    hard, _ = _headon_run(False)
    soft, sh_soft = _headon_run(True)
    drops = [a[2] - b[2] for a, b in zip(hard, hard[1:])]
    assert max(drops) >= 9.0, "scenario changed: the hard stop is no longer abrupt"
    speeds = [r[2] for r in soft]
    # 0.25 m/s a tick at 2.5 m/s^2; the last tick of the profile ends a little steeper
    assert max(a - b for a, b in zip(speeds, speeds[1:])) <= 0.4, "braked harder than soft_decel"
    assert speeds[-1] < 0.1
    final_clear = soft[-1][3]
    assert 3.0 <= final_clear <= 4.5, f"standoff {final_clear:.2f} m"
    assert min(r[3] for r in soft) >= 3.0, "came closer than the standoff"
    # the motion shown agrees with the speed published (1 m sample steps)
    i0 = next(i for i, v in enumerate(speeds) if v < 9.9)
    covered = soft[i0][1] - soft[-1][1]
    integrated = sum(speeds[i0:]) * 0.1
    assert abs(covered - integrated) <= 2.0, (covered, integrated)
    assert sh_soft.emergency_stops == 0


def test_soft_stop_pulls_away_at_bounded_acceleration():
    """The bus leaves the car's way: the car pulls away from its standstill at
    about soft_accel, not jumping to its recorded 10 m/s on one tick."""
    run, _ = _headon_run(True, ego_until=200)
    after = [r for r in run if r[0] >= 20.0]
    standing = [r for r in run if r[0] < 20.0][-1][2]
    assert standing < 0.1, f"published {standing:.1f} m/s while standing"
    speeds = [r[2] for r in after]
    assert max(b - a for a, b in zip(speeds, speeds[1:])) <= 1.5 * 0.1 + 0.05, "pulled away harder than soft_accel"
    assert max(speeds) >= 9.0, "never got back to its recorded speed"
    first_10m = next(r[0] for r in after if after[0][1] - r[1] >= 10.0) - 20.0
    assert 3.0 <= first_10m <= 6.0, f"first 10 m in {first_10m:.1f} s (1.5 m/s^2 takes ~3.7 s)"

def test_soft_stop_cyclist_waits_at_the_kerb():
    """H81, 2026-10-29: in the recording a cyclist waited at the kerb, then
    crossed behind the stopped bus. The sim ego stood further back, its rear
    across that path: the cyclist rode out at 1.5x (catch-up) and stopped
    beside the bus. Now it waits at the kerb while its way is blocked, and
    near the ego it crosses no faster than recorded."""
    def pose_at(k):   # stands at the kerb 5 s, then crosses +y at 2 m/s along x=-2
        t = k * 0.1
        return (-2.0, -8.0, 0.0) if t < 5.0 else (-2.0, -8.0 + 2.0 * (t - 5.0), 2.0)

    def bike(k):
        x, y, v = pose_at(k)
        o = mkobj(U, x, y, math.pi / 2, v)
        o.shape.dimensions.x, o.shape.dimensions.y = 1.8, 0.6
        o.classification = [NS(label=6)]
        return o

    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=[bike(k)])) for k in range(500)]

    def run(soft):
        sh = ReactiveShield(ShieldParams(frame_clock=True, vru_real_body=True, soft_stop=soft))
        sh.build_tracks(bag)
        out = []
        for k in range(500):
            ego = ego_at(0.0, 0.0) if k < 200 else ego_at(15.0, 0.0)   # stands 20 s, then moves on
            msg = NS(objects=[bike(k)])
            sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
            if msg.objects:
                out.append((k * 0.1, msg.objects[0].kinematics.pose_with_covariance.pose.position.y))
        return out

    hard = run(False)
    assert max(y for t, y in hard if t < 20.0) > -3.5, "scenario changed: the cyclist no longer rides up to the bus"
    soft = run(True)
    waiting = [y for t, y in soft if t < 20.0]
    assert max(waiting) < -7.5, f"left the kerb while blocked (reached y={max(waiting):.1f})"
    after = [(t, y) for t, y in soft if t >= 20.0]
    near = [(t, y) for t, y in after if y < 5.0]   # within no_catchup_radius of the ego
    pace = max((b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(near, near[10:]))
    assert pace <= 2.0 * 1.1, f"crossed at {pace:.1f} m/s near the ego (recorded 2.0)"
    assert after[-1][1] > 5.0, "never crossed once its way was clear"
    set_off = next(t for t, y in after if y > -7.9)
    assert set_off < 21.5, f"set off {set_off - 20.0:.1f} s after its way cleared (replayed its kerb wait)"


def test_soft_stop_never_publishes_above_the_recording():
    """H81, 2026-10-29: a scene-clock step (frame jumping 1 s) with a conflict far
    down the path made the brake cap (sqrt(2 a d), ~17 m/s) the published speed
    of a 3.5 m/s cyclist. A published speed never exceeds the recorded one."""
    bag = [(int(k * 0.1 * 1e9 + 2e18),
            NS(objects=[mkobj(U, -80.0 + k * 0.1 * 3.5, 0.0, 0.0, 3.5)]))
           for k in range(400)]
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True))
    sh.build_tracks(bag)
    ego = ego_at(0.0, 0.0)   # standing on the cyclist's path, 60-80 m ahead
    worst, k = 0.0, 0
    while k < 150:
        k += 10 if k % 30 == 0 else 1   # the frame jumps 1 s now and then
        msg = NS(objects=[mkobj(U, -80.0 + k * 0.1 * 3.5, 0.0, 0.0, 3.5)])
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        if msg.objects:
            worst = max(worst, abs(msg.objects[0].kinematics.twist_with_covariance.twist.linear.x))
    assert worst <= 3.5 + 1e-6, f"published {worst:.1f} m/s (recorded 3.5)"


def test_soft_stop_oncoming_leaves_room_for_the_ego_to_stop():
    """H81, 2026-10-29: an oncoming car braked to 3 m from where a 5 m/s ego
    was, not from where it would stop; the ego drove into it (37 ego-caused
    contact frames). Now it stands off from the ego's stopping point, while a
    car ahead going the same way does not brake for an ego closing in."""
    def oncoming(k):
        return mkobj(U, 120.0 - k * 0.1 * 10.0, 0.0, math.pi, 10.0)
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=[oncoming(k)])) for k in range(300)]

    def run(reach_decel):
        sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, ego_stop_decel_mps2=reach_decel))
        sh.build_tracks(bag)
        x, v, worst = 0.0, 5.0, 1e9
        for k in range(300):
            ego = ego_at(x, 0.0)
            msg = NS(objects=[oncoming(k)])
            sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=v)
            if msg.objects:
                clr = sh._rect_clearance(msg.objects[0], ego)
                worst = min(worst, clr)
                if clr < 8.0:
                    v = max(0.0, v - 2.0 * 0.1)   # the ego brakes at 2 m/s^2 once the car is close
            x += v * 0.1
        return worst

    assert run(1e9) < 1.0, "scenario changed: without the reach the ego no longer closes in"
    assert run(2.0) >= 1.5, f"oncoming car left {run(2.0):.2f} m"

    # a slower car ahead in the ego's lane, same direction, ego closing in from 15 m
    lead = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=[mkobj(U2, 25.0 + k * 0.1 * 3.0, 0.0, 0.0, 3.0)]))
            for k in range(200)]
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True))
    sh.build_tracks(lead)
    xs = []
    for k in range(35):   # 3.5 s: the ego closes to ~4 m, short of ramming it
        msg = NS(objects=[mkobj(U2, 25.0 + k * 0.1 * 3.0, 0.0, 0.0, 3.0)])
        sh.apply(msg, ego_at(k * 0.1 * 6.0, 0.0), BASE + k * 0.1, 0.1, ego_speed_mps=6.0)
        xs.append(msg.objects[0].kinematics.pose_with_covariance.pose.position.x)
    assert xs[-1] - xs[0] >= 0.9 * 3.0 * 3.4, f"lead car braked for the ego behind it ({xs[-1] - xs[0]:.1f} m)"


def test_soft_stop_forgets_a_hold_out_of_reach():
    """A car held behind a standing ego, then driven on out of reach, does not
    brake to a crawl when it comes near the ego again with its way clear."""
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True))
    sh.build_tracks(drive_bag(n=1200))
    k = 0

    def step(ego):
        nonlocal k
        msg = NS(objects=[mkobj(U, min(k, 1199) * 0.1 * 6.7, 0.0)])
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        k += 1
        return msg.objects[0]

    for _ in range(200):
        o = step(ego_at(60.0, 0.0))        # held behind the standing ego
    for _ in range(100):
        o = step(ego_at(60.0, 80.0))       # the ego is out of reach: it drives on
    x = o.kinematics.pose_with_covariance.pose.position.x
    side = ego_at(x + 30.0, 8.0)           # near again, the ego off its path
    for _ in range(5):
        o = step(side)
        tw = o.kinematics.twist_with_covariance.twist.linear
        assert math.hypot(tw.x, tw.y) > 6.0, f"braked with its way clear: {math.hypot(tw.x, tw.y):.2f} m/s"


def test_soft_stop_parked_cars_are_cheap():
    """Parked cars repeat one pose for minutes: the conflict scan must not
    walk every sample on every tick."""
    import time
    uu = [bytes([i]) * 16 for i in range(1, 21)]
    bag = [(int(k * 0.1 * 1e9 + 2e18),
            NS(objects=[mkobj(u, 20.0 + i * 6.0, 4.0, 0.0, 0.0) for i, u in enumerate(uu)]))
           for k in range(6000)]
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True))
    sh.build_tracks(bag)
    t0 = time.perf_counter()
    for k in range(20):
        msg = NS(objects=[mkobj(u, 20.0 + i * 6.0, 4.0, 0.0, 0.0) for i, u in enumerate(uu)])
        sh.apply(msg, ego_at(60.0, 0.0), BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
    ms = (time.perf_counter() - t0) / 20 * 1e3
    assert ms < 50.0, f"{ms:.0f} ms a tick for 20 parked cars"


def test_soft_stop_pull_away_never_publishes_above_the_recording():
    """2026-10-29 (H81 0e35: 14.4 m/s published, 12.7 recorded max; H78 0914):
    a car held behind a standing ego pulls away once the ego leaves, while far
    enough to close its delay at 1.5x. Its pull-away ramp ran on past the
    recorded speed and was published as the car's speed."""
    bag = drive_bag(800, speed=6.7)
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True))
    sh.build_tracks(bag)
    worst, held = 0.0, False
    for k in range(700):
        # stands in the car's lane 20 s, then 30 m to the side: clear of its
        # path, past no_catchup_radius, still inside the active radius
        ego = ego_at(60.0, 0.0) if k < 200 else ego_at(60.0, 30.0)
        msg = NS(objects=[mkobj(U, min(k, 799) * 0.1 * 6.7, 0.0)])
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        if msg.objects:
            v = abs(msg.objects[0].kinematics.twist_with_covariance.twist.linear.x)
            if k < 200 and v < 0.1:
                held = True
            if k >= 200:
                worst = max(worst, v)
    assert held, "scenario changed: the car is no longer held behind the ego"
    assert worst <= 6.7 + 1e-6, f"published {worst:.2f} m/s pulling away (recorded 6.7)"


def _queue_run(following, ego_leaves_at=None, n=500):
    """Two cars on one lane, B 12 m behind A, both 6.7 m/s, toward a bus
    standing at x=60. Returns per-tick (A x, B x, gap between their boxes)."""
    def frame(k):
        return [mkobj(U, k * 0.1 * 6.7, 0.0), mkobj(U2, k * 0.1 * 6.7 - 12.0, 0.0)]
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=frame(k))) for k in range(n)]
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True,
                                     npc_following=following))
    sh.build_tracks(bag)
    out = []
    for k in range(n):
        ego = ego_at(60.0, 0.0) if ego_leaves_at is None or k < ego_leaves_at else ego_at(60.0, 40.0)
        msg = NS(objects=frame(k))
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        by = {bytes(o.object_id.uuid): o for o in msg.objects}
        if U in by and U2 in by:
            a, b = by[U], by[U2]
            gap = ReactiveShield._box_gap(*ReactiveShield._sample_box(a), *ReactiveShield._sample_box(b))
            out.append((a.kinematics.pose_with_covariance.pose.position.x,
                        b.kinematics.pose_with_covariance.pose.position.x, gap))
    return out


def test_npc_following_keeps_a_queue_apart():
    """H81/H35/H95, 2026-11-09: cars queued behind a standing bus all braked to
    the same spot behind it and sat inside each other (three for ~28 s in H81).
    With npc_following the second car brakes for the first and stands ~2 m
    behind it; once the bus has gone both drive off, still apart."""
    plain = _queue_run(False, n=350)
    assert min(r[2] for r in plain[-50:]) == 0.0, "scenario changed: the queue no longer collapses"
    queue = _queue_run(True, ego_leaves_at=350)
    standing = queue[300:350]
    assert min(r[2] for r in standing) >= 1.5, f"queued cars {min(r[2] for r in standing):.2f} m apart"
    assert min(r[2] for r in queue) >= 0.1, "the cars touched at some point"
    assert queue[-1][1] > standing[-1][1] + 20.0, "the second car never drove off once the bus had gone"


def test_npc_following_leaves_recorded_traffic_alone():
    """Two cars keeping to their recording 0.5 m apart, away from the bus: NPC
    following changes nothing (it only brakes for an NPC the shield moved)."""
    def frame(k):
        return [mkobj(U, k * 0.1 * 8.0, 20.0), mkobj(U2, k * 0.1 * 8.0 - 5.0, 20.0)]   # gap 0.5 m
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=frame(k))) for k in range(200)]
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, npc_following=True))
    sh.build_tracks(bag)
    worst = 0.0
    for k in range(200):
        msg = NS(objects=frame(k))
        sh.apply(msg, ego_at(0.0, 0.0), BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        for o, raw in zip(sorted(msg.objects, key=lambda o: bytes(o.object_id.uuid)), sorted(frame(k), key=lambda o: o.object_id.uuid)):
            pa, pb = o.kinematics.pose_with_covariance.pose.position, raw.kinematics.pose_with_covariance.pose.position
            worst = max(worst, math.hypot(pa.x - pb.x, pa.y - pb.y))
    assert worst < 1e-6, f"recorded traffic moved by {worst:.2f} m"


def test_npc_following_tracks_a_moving_leader_without_stop_go():
    """Behind a car that only runs late, the follower keeps the leader's speed
    rather than braking to a standstill and pulling away again (2026-11-09:
    the leader was first treated as a standing obstacle)."""
    def frame(k):
        return [mkobj(U, k * 0.1 * 6.7, 0.0), mkobj(U2, k * 0.1 * 6.7 - 12.0, 0.0)]
    n = 600
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=frame(k))) for k in range(n)]
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=True, npc_following=True))
    sh.build_tracks(bag)
    speeds_b, gaps = [], []
    for k in range(n):
        ego = ego_at(60.0, 0.0) if k < 200 else ego_at(60.0, 40.0)
        msg = NS(objects=frame(k))
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        by = {bytes(o.object_id.uuid): o for o in msg.objects}
        if U in by and U2 in by and k >= 200:
            b = by[U2]
            speeds_b.append(abs(b.kinematics.twist_with_covariance.twist.linear.x))
            gaps.append(ReactiveShield._box_gap(*ReactiveShield._sample_box(by[U]), *ReactiveShield._sample_box(b)))
    moving = next(i for i, v in enumerate(speeds_b) if v > 0.5)
    assert min(speeds_b[moving:]) > 0.3, f"the follower stopped again behind a moving leader ({min(speeds_b[moving:]):.2f} m/s)"
    assert min(gaps) >= 1.0, f"follower {min(gaps):.2f} m from the leader"


def test_every_invariant_holds_with_soft_stop():
    """The position scene runs the shield with soft_stop: every scenario above
    must hold there too."""
    import functools
    g = globals()
    plain = g["ShieldParams"]
    g["ShieldParams"] = functools.partial(plain, soft_stop=True)
    try:
        for name, fn in sorted(g.items()):
            if name.startswith("test_") and "soft_stop" not in name:
                try:
                    fn()
                except AssertionError as e:
                    raise AssertionError(f"{name} with soft_stop: {e}") from None
    finally:
        g["ShieldParams"] = plain


def test_vehicle_real_body_lets_an_adjacent_car_pass():
    """2026-11-09 (offline replay of 8 bags): the widened 12.5 x 3.0 m box held
    cars that were really clear of the bus; judged against the real body
    (+0.25 m) they keep closer to their recording — near NPCs off it by more
    than 1 m: H45 7.3 -> 3.3 %, H78 4.8 -> 2.3 %, H81 p99 44 -> 15 m — with no
    NPC driving into the ego. A car 2.7 m off the bus's centreline (0.35 m
    clear of its real side) passes; the box held it."""
    def car(k):
        return mkobj(U, -30.0 + k * 0.1 * 6.0, -2.7, 0.0, 6.0)   # adjacent lane, passing on the right
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=[car(k)])) for k in range(300)]

    def furthest(real):
        sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, vehicle_real_body=real))
        sh.build_tracks(bag)
        x, worst = -1e9, 1e9
        for k in range(300):
            msg = NS(objects=[car(k)])
            sh.apply(msg, ego_at(0.0, 0.0), BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
            if msg.objects:
                x = max(x, msg.objects[0].kinematics.pose_with_covariance.pose.position.x)
                worst = min(worst, sh._rect_clearance(msg.objects[0], ego_at(0.0, 0.0)))
        return x, worst

    x_box, _ = furthest(False)
    x_real, clr = furthest(True)
    assert x_box < 0.0, f"scenario changed: the box no longer holds the car (reached x={x_box:.1f})"
    assert x_real > 100.0, f"held by the real body (reached x={x_real:.1f})"
    assert clr >= 0.25, f"passed {clr:.2f} m from the real body"


def _slanting_run(old_rule=False):
    """A car waits 3 s, then drives at 3 m/s on a line 20 deg off the bus's
    axis into the middle of its left side (H78, 2026-11-09: cross traffic
    released on its green toward a bus standing in the turn). Returns the
    closest clearance."""
    yaw, speed = -math.radians(20.0), 3.0

    def car(k):
        d = max(0.0, (k - 30) * 0.1 * speed)
        return mkobj(U, -25.0 + d * math.cos(yaw), 12.74 + d * math.sin(yaw), yaw, speed if k >= 30 else 0.0)
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=[car(k)])) for k in range(250)]
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True))
    if old_rule:   # the path-distance standoff this replaced
        plain = sh._conflict_s
        sh._conflict_s = lambda *a, **kw: (lambda h: h and (h[0], h[0]))(plain(*a, **kw))
    sh.build_tracks(bag)
    worst = 1e9
    for k in range(250):
        msg = NS(objects=[car(k)])
        sh.apply(msg, ego_at(0.0, 0.0), BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        if msg.objects:
            worst = min(worst, sh._rect_clearance(msg.objects[0], ego_at(0.0, 0.0)))
    return worst


def test_soft_stop_standoff_is_a_clearance_on_a_slanting_approach():
    """Standing off 3 m along a path that meets the bus at 20 deg left the car
    about a metre from its side (H78: 1.1 m); it now stops where it first comes
    within the standoff."""
    old = _slanting_run(old_rule=True)
    assert old < 2.0, f"scenario changed: the path-distance rule kept {old:.2f} m"
    new = _slanting_run()
    assert 2.7 <= new <= 4.5, f"closest {new:.2f} m"


def _plan_run(plan_until, moving=False, refresh=True, n=300):
    """A car on y=0 eastward; the bus stands at (10, 15) facing south, clear of
    the car's way, while its plan (until tick plan_until) drives it south
    across that way in the next 5 s. moving: the car passes at 6 m/s instead
    of waiting 3 s at x=2. Returns (x, published speed) per tick."""
    def car(k):
        if moving:
            return mkobj(U, -30.0 + k * 0.1 * 6.0, 0.0, 0.0, 6.0)
        return mkobj(U, 2.0 + max(0, k - 30) * 0.1 * 3.0, 0.0, 0.0, 3.0 if k >= 30 else 0.0)
    bag = [(int(k * 0.1 * 1e9 + 2e18), NS(objects=[car(k)])) for k in range(n)]
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, ego_plan_sec=5.0))
    sh.build_tracks(bag)
    plan = [(i * 0.25, 10.0, 15.0 - i * 1.25, -math.pi / 2) for i in range(21)]
    ego = ego_at(10.0, 15.0, -math.pi / 2)
    out = []
    for k in range(n):
        if k < plan_until and (refresh or k == 0):
            sh.set_ego_plan(plan)
        elif k == plan_until:
            sh.set_ego_plan(plan[:1])   # the ego has turned away: its plan stays where it is
        msg = NS(objects=[car(k)])
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        o = msg.objects[0]
        out.append((o.kinematics.pose_with_covariance.pose.position.x,
                    abs(o.kinematics.twist_with_covariance.twist.linear.x)))
    return out


def test_ego_plan_holds_a_standing_car_at_its_line():
    """H78, 2026-11-09: a car waiting at its light set off on its green while
    the bus, still clear of its way, was turning into it; it stood 1 m from
    the bus. A standing NPC waits while the ego's plan sweeps its way and sets
    off once it no longer does; without the plan it went."""
    free = _plan_run(plan_until=0)
    assert free[60][0] > 5.0, "scenario changed: the car no longer sets off on its own"
    held = _plan_run(plan_until=150)
    assert max(x for x, _ in held[:150]) <= 2.6, f"set off into the bus's plan (x={max(x for x, _ in held[:150]):.1f})"
    assert held[-1][0] > 20.0, "never set off after the plan left its way"
    stale = _plan_run(plan_until=150, refresh=False)
    assert stale[80][0] > 5.0, "held by a plan that stopped coming"


def test_ego_plan_never_brakes_a_moving_car():
    """Passing in front of an approaching bus is the NPC's to time: a car
    already driving keeps its recorded speed through the plan's sweep."""
    run = _plan_run(plan_until=300, moving=True)
    assert min(v for _, v in run) >= 5.9, f"slowed to {min(v for _, v in run):.1f} m/s"
    assert run[-1][0] > 100.0


U4 = bytes(range(48, 64))


def _bike(uuid, x, y, v):
    o = mkobj(uuid, x, y, 0.0, v)
    o.shape.dimensions.x, o.shape.dimensions.y = 2.0, 0.7
    o.classification = [NS(label=6)]
    return o


def _split_bag(dup_dy=0.0, n=400):
    """A cyclist rides +x at 3 m/s on y=0 into a bus standing across its lane
    at x=10. From t=20 s to 26 s the recording also holds a second track
    0.8 m behind it (dup_dy=0: the tracker's new id for it; dup_dy=1.2: a
    second rider beside it), born when the first is already past the bus."""
    def frame(k):
        t = k * 0.1
        objs = [_bike(U, -40.0 + 3.0 * t, 0.0, 3.0)]
        if 200 <= k < 260:
            objs.append(_bike(U4, -40.0 + 3.0 * t - 0.8, dup_dy, 3.0))
        return objs
    return frame, [(int(k * 0.1 * 1e9 + 2e18), NS(objects=frame(k))) for k in range(n)]


def test_tracker_duplicate_follows_its_parent():
    """H78, 2026-11-09: the shield held a cyclist behind the standing bus; the
    tracker then gave the cyclist a new id, an object of its own to the
    shield, born at its recorded place — in front of the bus. A new id runs
    no earlier than its parent: it is not shown while the parent is held
    short of its birth."""
    frame, bag = _split_bag()
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, vru_real_body=True))
    sh.build_tracks(bag)
    assert sh.parent_of == {U4: U}, sh.parent_of
    ego = ego_at(10.0, -2.0, math.pi / 2)   # across the lane
    for k in range(260):
        msg = NS(objects=frame(k))
        sh.apply(msg, ego, BASE + k * 0.1, 0.1, ego_speed_mps=0.0)
        shown = {bytes(o.object_id.uuid): o.kinematics.pose_with_covariance.pose.position.x for o in msg.objects}
        assert shown.get(U, 0.0) < 10.0, "scenario changed: the cyclist is not held behind the bus"
        assert U4 not in shown, f"the copy was shown at x={shown[U4]:.1f} while its parent waits at {shown[U]:.1f}"


def test_a_second_rider_beside_is_not_a_duplicate():
    _, bag = _split_bag(dup_dy=1.2)
    sh = ReactiveShield(ShieldParams(frame_clock=True, soft_stop=True, vru_real_body=True))
    sh.build_tracks(bag)
    assert sh.parent_of == {}, sh.parent_of


def test_every_invariant_holds_with_production_shield():
    """The scene driver's shield: soft stop, every class judged against the real
    body, NPC following, the ego plan. Every scenario above must hold with it too."""
    import functools
    g = globals()
    plain = g["ShieldParams"]
    g["ShieldParams"] = functools.partial(plain, soft_stop=True, vru_real_body=True, vehicle_real_body=True,
                                           npc_following=True, ego_plan_sec=5.0)
    try:
        for name, fn in sorted(g.items()):
            # all but the meta tests and the two that compare body choices themselves
            if (name.startswith("test_") and not name.startswith("test_every_")
                    and "real_body" not in name and "cyclist_crossing_behind" not in name
                    and "npc_following" not in name):
                try:
                    fn()
                except AssertionError as e:
                    raise AssertionError(f"{name} with the production shield: {e}") from None
    finally:
        g["ShieldParams"] = plain


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                fails += 1
    sys.exit(1 if fails else 0)
