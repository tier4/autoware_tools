"""Low-CPU live drive monitor for long bag runs.

Samples /localization/kinematic_state at a low rate and writes a tiny JSON
file that the HTML dashboard / Streamlit GUI polls. No rosbridge, RViz, or
video encode.
"""

from __future__ import annotations

import html
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any


def live_drive_map_html(data: dict[str, Any] | None, *, width: int = 640, height: int = 420) -> str:
    """Self-contained SVG map for Streamlit (no fetch / http.server needed)."""
    w, h, pad = width, height, 36
    if not data:
        return (
            f'<div style="font:14px system-ui;padding:12px;color:#666">'
            f"Waiting for live_drive.json…</div>"
        )

    trail = [p for p in (data.get("trail") or []) if p.get("x") is not None]
    goals = [g for g in (data.get("goals") or []) if g.get("x") is not None]
    ego_x = data.get("x")
    ego_y = data.get("y")
    pts: list[tuple[float, float]] = [(float(p["x"]), float(p["y"])) for p in trail]
    pts.extend((float(g["x"]), float(g["y"])) for g in goals)
    if ego_x is not None and ego_y is not None:
        pts.append((float(ego_x), float(ego_y)))

    active = bool(data.get("active"))
    badge = "LIVE" if active else "idle"
    badge_bg = "#1b7f4e" if active else "#888"
    speed = data.get("speed_mps")
    speed_s = f"{float(speed):.2f}" if speed is not None else "?"
    xy = (
        f"x={float(ego_x):.1f}, y={float(ego_y):.1f}"
        if ego_x is not None and ego_y is not None
        else "x=?, y=?"
    )
    leg = ""
    if data.get("leg_idx") is not None:
        leg = f" · leg {data.get('leg_idx')}/{data.get('leg_total') or '?'}"
    meta = html.escape(
        f"{data.get('bag_key') or data.get('job_id') or '?'} · "
        f"{data.get('phase') or ''}{leg} · {xy} · v={speed_s} m/s · "
        f"{data.get('updated_iso') or ''}"
        + (f" · {data.get('note')}" if data.get("note") else "")
    )

    if len(pts) < 1:
        body = (
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="#f4f6f8"/>'
            f'<text x="20" y="40" fill="#666" font-size="14">'
            f"Ego pose not received yet…</text>"
        )
    else:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = max(20.0, max_x - min_x)
        dy = max(20.0, max_y - min_y)
        scale = min((w - 2 * pad) / dx, (h - 2 * pad) / dy)

        def to_xy(x: float, y: float) -> tuple[float, float]:
            return (pad + (x - min_x) * scale, h - pad - (y - min_y) * scale)

        trail_pts = " ".join(
            f"{px:.1f},{py:.1f}" for px, py in (to_xy(float(p["x"]), float(p["y"])) for p in trail)
        )
        goal_svg = []
        for i, g in enumerate(goals):
            gx, gy = to_xy(float(g["x"]), float(g["y"]))
            label = html.escape(str(g.get("label") or f"G{i + 1}"))
            goal_svg.append(
                f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="4" fill="#c0392b"/>'
                f'<text x="{gx + 6:.1f}" y="{gy - 6:.1f}" font-size="11" fill="#333">{label}</text>'
            )
        ego_svg = ""
        if ego_x is not None and ego_y is not None:
            ex, ey = to_xy(float(ego_x), float(ego_y))
            yaw = float(data["yaw"]) if data.get("yaw") is not None else 0.0
            deg = -yaw * 180.0 / math.pi
            ego_svg = (
                f'<g transform="translate({ex:.1f} {ey:.1f}) rotate({deg:.1f})">'
                f'<polygon points="14,0 -8,-7 -8,7" fill="#1a5fb4"/></g>'
            )
        grid = []
        for i in range(1, 4):
            x = w * i / 4
            y = h * i / 4
            grid.append(
                f'<line x1="{x}" y1="0" x2="{x}" y2="{h}" stroke="#dde3ea" stroke-width="1"/>'
                f'<line x1="0" y1="{y}" x2="{w}" y2="{y}" stroke="#dde3ea" stroke-width="1"/>'
            )
        body = (
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="#f4f6f8"/>'
            + "".join(grid)
            + f'<polyline points="{trail_pts}" fill="none" stroke="#2d7dd2" stroke-width="2.5"/>'
            + "".join(goal_svg)
            + ego_svg
        )

    return f"""
<div style="font-family:system-ui,sans-serif;max-width:{w}px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
    <span style="background:{badge_bg};color:#fff;padding:2px 10px;border-radius:999px;
      font-size:12px;font-weight:600">{badge}</span>
    <span style="font-size:13px;color:#333">{meta}</span>
  </div>
  <svg viewBox="0 0 {w} {h}" width="100%" height="{h}"
    style="border:1px solid #cfd6dd;border-radius:6px;background:#f4f6f8">
    {body}
  </svg>
</div>
"""


def _yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class LiveDriveMonitor:
    """Daemon sampler: ~0.5 Hz pose → live_drive.json (negligible CPU)."""

    def __init__(
        self,
        *,
        output_path: Path,
        sample_sec: float = 2.0,
        trail_max: int = 240,
        job_id: str = "",
        bag_key: str = "",
        goals: list[dict[str, Any]] | None = None,
        extra_paths: list[Path] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.extra_paths = [Path(p) for p in (extra_paths or [])]
        self.sample_sec = max(0.5, float(sample_sec))
        self.trail_max = max(20, int(trail_max))
        self.job_id = job_id
        self.bag_key = bag_key
        self.goals = list(goals or [])
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "driving"
        self._leg_idx: int | None = None
        self._leg_total: int | None = None
        self._note = ""
        self._trail: list[dict[str, float]] = []

    def set_phase(self, phase: str, *, note: str = "") -> None:
        self._phase = phase
        if note:
            self._note = note

    def set_leg(self, leg_idx: int | None, leg_total: int | None = None) -> None:
        self._leg_idx = leg_idx
        if leg_total is not None:
            self._leg_total = leg_total

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="dp_multi_eval_live_monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, final_note: str = "stopped") -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._write_snapshot(
            active=False,
            x=None,
            y=None,
            yaw=None,
            speed=None,
            note=final_note or self._note,
        )

    def _write_snapshot(
        self,
        *,
        active: bool,
        x: float | None,
        y: float | None,
        yaw: float | None,
        speed: float | None,
        note: str = "",
    ) -> None:
        payload: dict[str, Any] = {
            "active": active,
            "updated_at": time.time(),
            "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "job_id": self.job_id,
            "bag_key": self.bag_key,
            "phase": self._phase,
            "leg_idx": self._leg_idx,
            "leg_total": self._leg_total,
            "note": note or self._note,
            "x": x,
            "y": y,
            "yaw": yaw,
            "speed_mps": speed,
            "trail": self._trail[-self.trail_max :],
            "goals": self.goals,
            "sample_sec": self.sample_sec,
        }
        for path in [self.output_path, *self.extra_paths]:
            try:
                _atomic_write_json(path, payload)
            except OSError:
                pass

    def _run(self) -> None:
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
        except ImportError:
            self._write_snapshot(
                active=False,
                x=None,
                y=None,
                yaw=None,
                speed=None,
                note="rclpy_unavailable",
            )
            return

        # Own ROS context so this daemon thread does not collide with the job's
        # main-thread rclpy.spin_once (route setup / perception wait).
        context = rclpy.Context()
        rclpy.init(context=context)

        class Sampler(Node):
            def __init__(self_inner) -> None:
                super().__init__("dp_multi_eval_live_monitor", context=context)
                self_inner.x = float("nan")
                self_inner.y = float("nan")
                self_inner.yaw = float("nan")
                self_inner.speed = float("nan")
                self_inner.create_subscription(
                    Odometry,
                    "/localization/kinematic_state",
                    self_inner._cb,
                    1,
                )

            def _cb(self_inner, msg: Odometry) -> None:
                pose = msg.pose.pose
                self_inner.x = float(pose.position.x)
                self_inner.y = float(pose.position.y)
                self_inner.yaw = _yaw_from_quat(pose.orientation)
                vx = float(msg.twist.twist.linear.x)
                vy = float(msg.twist.twist.linear.y)
                self_inner.speed = (vx * vx + vy * vy) ** 0.5

        node = Sampler()
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        last_write = 0.0
        try:
            while not self._stop.is_set() and context.ok():
                executor.spin_once(timeout_sec=0.2)
                now = time.time()
                if now - last_write < self.sample_sec:
                    continue
                last_write = now
                x, y, yaw, speed = node.x, node.y, node.yaw, node.speed
                if x == x and y == y:
                    self._trail.append({"x": x, "y": y})
                    if len(self._trail) > self.trail_max * 2:
                        # Keep memory bounded even if sample_sec is tiny.
                        self._trail = self._trail[-self.trail_max :]
                    self._write_snapshot(
                        active=True,
                        x=x,
                        y=y,
                        yaw=yaw if yaw == yaw else None,
                        speed=speed if speed == speed else None,
                    )
                else:
                    self._write_snapshot(
                        active=True,
                        x=None,
                        y=None,
                        yaw=None,
                        speed=None,
                        note="waiting_for_ego_pose",
                    )
        finally:
            try:
                executor.remove_node(node)
            except Exception:  # noqa: BLE001
                pass
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
            try:
                executor.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                if context.ok():
                    rclpy.shutdown(context=context)
            except Exception:  # noqa: BLE001
                pass
