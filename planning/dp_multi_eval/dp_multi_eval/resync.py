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

"""Re-sync: set the ego back onto the recording.

The recording is the timeline the scene plays. When the ego has fallen
behind it (job_waiters.recorded_lag_sec), the traffic the model is looking at
belongs to a vehicle that is no longer here, and every second of lag adds
cars that catch up and drive through the ego. Rather than bending the scene
to the ego (holds, slow motion, fast-forward — each of them a scene the
vehicle never saw), the ego is bent to the scene: it is placed on the
recorded vehicle's current pose and driven along the recorded track for the
model's history window (INPUT_T = 30 frames at 10 Hz, 3 s), so that at
hand-back its own history and the agents' histories are the recording's, and
the recorded speed is what it is moving at.

Mechanics: simple_planning_simulator takes its state from /initialpose3d
(pose, velocity reset) followed by /simulation/input/initialtwist (velocity
on the stored pose). Both are published for every recorded-ego frame the
reproducer emits (/perception_reproducer/rosbag_ego_odom, the scene's own
clock), so the ego tracks the recorded vehicle whatever the clock does. The
stack stays engaged throughout: the transition manager only judges a
transition in progress, and the longitudinal controller's only emergency
latch (overshoot) is off in this vehicle's configuration, so a pose jump
under AUTONOMOUS costs the model one stale cycle and nothing else. No route
is touched.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

# The model's ego history is INPUT_T + 1 = 31 frames at 10 Hz. A little
# longer, so the whole window is recorded motion at hand-back.
FOLLOW_SEC = 3.5


def _state(odom: Any) -> dict[str, float]:
    q = odom.pose.pose.orientation
    tw = odom.twist.twist.linear
    return {
        "t": odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9,
        "x": round(float(odom.pose.pose.position.x), 2),
        "y": round(float(odom.pose.pose.position.y), 2),
        "yaw": round(math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)), 3),
        "v": round(math.hypot(float(tw.x), float(tw.y)), 2),
    }


def resync_ego(
    env: dict[str, str],
    *,
    follow_sec: float = FOLLOW_SEC,
    wait_sec: float = 10.0,
) -> tuple[bool, str, dict[str, Any]]:
    """Drive the ego along the recorded track for ``follow_sec``, then hand it back.

    Returns (ok, detail, event); ``event`` carries the ego's state before, and
    the recorded state at the start and at hand-back.
    """
    try:
        from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
    except ImportError:
        return False, "resync_unavailable: rclpy_or_msgs_missing", {}

    from dp_multi_eval.job_waiters import close_job_context, job_executor, open_job_context

    context = open_job_context(env)
    old_env = os.environ.copy()
    os.environ.update(env)

    class Sync(Node):
        def __init__(self) -> None:
            super().__init__("dp_multi_eval_resync", context=context)
            self.rec: Any = None
            self.ego: Any = None
            self.create_subscription(
                Odometry, "/perception_reproducer/rosbag_ego_odom", self._cb_rec, 1
            )
            self.create_subscription(Odometry, "/localization/kinematic_state", self._cb_ego, 1)
            self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose3d", 1)
            self._twist_pub = self.create_publisher(
                TwistStamped, "/simulation/input/initialtwist", 1
            )

        def _cb_rec(self, msg: Any) -> None:
            self.rec = msg

        def _cb_ego(self, msg: Any) -> None:
            self.ego = msg

        def place(self, odom: Any) -> None:
            stamp = self.get_clock().now().to_msg()
            pose = PoseWithCovarianceStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = "map"
            pose.pose.pose = odom.pose.pose
            pose.pose.covariance = [0.0] * 36
            for i in (0, 7, 14, 21, 28, 35):
                pose.pose.covariance[i] = 0.01
            self._pose_pub.publish(pose)
            # on_initialtwist re-applies the stored pose with this velocity.
            twist = TwistStamped()
            twist.header.stamp = stamp
            twist.header.frame_id = "base_link"
            twist.twist = odom.twist.twist
            self._twist_pub.publish(twist)

    node = Sync()
    executor = job_executor(context, node)
    try:
        deadline = time.time() + wait_sec
        while (node.rec is None or node.ego is None) and time.time() < deadline and context.ok():
            executor.spin_once(timeout_sec=0.1)
        if node.rec is None or node.ego is None:
            missing = "recorded ego" if node.rec is None else "ego odometry"
            return False, f"resync_failed: no {missing} within {wait_sec:.0f}s", {}

        before = _state(node.ego)
        start = _state(node.rec)
        last = None
        frames = 0
        end_at = time.time() + follow_sec
        while time.time() < end_at and context.ok():
            executor.spin_once(timeout_sec=0.05)
            if node.rec is not last:
                last = node.rec
                node.place(last)
                frames += 1
        end = _state(last)
        jump = math.hypot(start["x"] - before["x"], start["y"] - before["y"])
        event = {
            "ego_before": before,
            "recorded_start": start,
            "recorded_end": end,
            "jump_m": round(jump, 1),
            "frames": frames,
            "follow_sec": follow_sec,
        }
        detail = (
            f"resynced: ego moved {jump:.1f} m onto the recording at "
            f"({start['x']:.1f},{start['y']:.1f}) and followed it for {follow_sec:.1f}s "
            f"({frames} frames), handed back at {end['v']:.2f} m/s"
        )
        return True, detail, event
    finally:
        executor.shutdown()
        node.destroy_node()
        close_job_context(context)
        os.environ.clear()
        os.environ.update(old_env)
