"""Phase 3 — headless top-down video from output bag + metrics."""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Patch

from dp_multi_eval.bag_reader import (
    BagSeries,
    TrajectorySample,
    TURN_LEFT,
    TURN_RIGHT,
    downsample_ego,
    load_bag_series,
)
from dp_multi_eval.geometry import box_local, load_vehicle_footprint, transform_local_polygon
from dp_multi_eval.topics import TopicSet, load_topics

# Virtual walls — high-contrast (avoid ego red #d62728 / NPC green)
_WALL_STYLE = {
    # face, edge, short label
    "modifier_obstacle_stop": ("#ff00aa", "#ffffff", "obst_stop"),
    "diffusion_planner": ("#ffcc00", "#000000", "diff_plan"),
    "stop_point_fixer": ("#00e5ff", "#000000", "stop_fix"),
}
_WALL_DEFAULT_STYLE = ("#7c4dff", "#ffffff", "wall")
# Thicker than RViz default so walls stay visible in top-down preview
_WALL_LOCAL = box_local(0.55, 6.5)

# Blinker chevrons in base_link (left / right of ego)
_BLINKER_LEFT_LOCAL = [(-0.5, 1.6), (0.8, 2.4), (-0.5, 3.2)]
_BLINKER_RIGHT_LOCAL = [(-0.5, -1.6), (0.8, -2.4), (-0.5, -3.2)]


@dataclass
class MapOverlay:
    centerlines: list[list[tuple[float, float]]] = field(default_factory=list)
    left_bounds: list[list[tuple[float, float]]] = field(default_factory=list)
    right_bounds: list[list[tuple[float, float]]] = field(default_factory=list)
    crosswalks: list[list[tuple[float, float]]] = field(default_factory=list)
    stop_lines: list[list[tuple[float, float]]] = field(default_factory=list)
    traffic_signs: list[tuple[float, float, str]] = field(default_factory=list)  # x,y,label
    road_borders: list[list[tuple[float, float]]] = field(default_factory=list)


@lru_cache(maxsize=4)
def _load_lanelet_map(osm_path: str):
    from autoware_lanelet2_extension_python.projection import MGRSProjector
    from lanelet2.io import Origin, load

    return load(osm_path, MGRSProjector(Origin(0.0, 0.0)))


def _lanelet_attr(obj: Any, key: str, default: str = "") -> str:
    try:
        attrs = obj.attributes
        if key in attrs:
            return str(attrs[key])
    except Exception:  # noqa: BLE001
        pass
    return default


def _pts_in_bbox(
    pts: list[tuple[float, float]], bx0: float, bx1: float, by0: float, by1: float
) -> bool:
    return any(bx0 <= x <= bx1 and by0 <= y <= by1 for x, y in pts)


def _linestring_xy(ls: Any) -> list[tuple[float, float]]:
    return [(float(p.x), float(p.y)) for p in ls]


def _load_map_overlay(
    map_path: Path | None,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    margin: float = 40.0,
) -> MapOverlay:
    overlay = MapOverlay()
    if map_path is None:
        return overlay
    map_path = map_path.expanduser().resolve()
    osm = map_path if map_path.suffix == ".osm" else map_path / "lanelet2_map.osm"
    if not osm.is_file():
        return overlay
    try:
        lanelet_map = _load_lanelet_map(str(osm))
    except Exception:  # noqa: BLE001
        return overlay

    bx0, bx1 = xmin - margin, xmax + margin
    by0, by1 = ymin - margin, ymax + margin

    for ll in lanelet_map.laneletLayer:
        subtype = _lanelet_attr(ll, "subtype")
        center = _linestring_xy(ll.centerline)
        left = _linestring_xy(ll.leftBound)
        right = _linestring_xy(ll.rightBound)
        if subtype == "crosswalk":
            try:
                poly = [(float(p.x), float(p.y)) for p in ll.polygon2d()]
            except Exception:  # noqa: BLE001
                poly = left + list(reversed(right)) if left and right else center
            if len(poly) >= 3 and _pts_in_bbox(poly, bx0, bx1, by0, by1):
                overlay.crosswalks.append(poly)
            continue
        if subtype not in ("", "road", "road_shoulder", "bicycle_lane", "highway", "play_street"):
            # skip walkway / pedestrian_lane clutter unless bounds needed
            if subtype in ("walkway", "pedestrian_lane"):
                continue
        if center and _pts_in_bbox(center, bx0, bx1, by0, by1):
            if len(center) >= 2:
                overlay.centerlines.append(center)
            if len(left) >= 2:
                overlay.left_bounds.append(left)
            if len(right) >= 2:
                overlay.right_bounds.append(right)

    for poly in lanelet_map.polygonLayer:
        ptype = _lanelet_attr(poly, "type")
        if ptype not in ("crosswalk_polygon", "pedestrian_marking", "zebra_marking"):
            continue
        try:
            pts = [(float(p.x), float(p.y)) for p in poly]
        except Exception:  # noqa: BLE001
            continue
        if len(pts) >= 3 and _pts_in_bbox(pts, bx0, bx1, by0, by1):
            overlay.crosswalks.append(pts)

    for ls in lanelet_map.lineStringLayer:
        ltype = _lanelet_attr(ls, "type")
        subtype = _lanelet_attr(ls, "subtype")
        pts = _linestring_xy(ls)
        if len(pts) < 2 or not _pts_in_bbox(pts, bx0, bx1, by0, by1):
            continue
        if ltype == "stop_line":
            overlay.stop_lines.append(pts)
        elif ltype == "road_border":
            overlay.road_borders.append(pts)
        elif ltype == "traffic_sign":
            mid = pts[len(pts) // 2]
            label = subtype or "sign"
            if label in ("?", "unknown"):
                label = "sign"
            overlay.traffic_signs.append((mid[0], mid[1], label))
        elif ltype == "zebra_marking":
            overlay.crosswalks.append(pts)

    return overlay


def _draw_map_overlay(ax: Any, overlay: MapOverlay) -> list[Any]:
    artists: list[Any] = []
    for line in overlay.road_borders:
        (ln,) = ax.plot(
            [p[0] for p in line], [p[1] for p in line],
            color="#6e7681", lw=1.0, alpha=0.7, zorder=0,
        )
        artists.append(ln)
    for line in overlay.left_bounds + overlay.right_bounds:
        (ln,) = ax.plot(
            [p[0] for p in line], [p[1] for p in line],
            color="#9aa0a6", lw=0.7, alpha=0.85, zorder=0,
        )
        artists.append(ln)
    for line in overlay.centerlines:
        (ln,) = ax.plot(
            [p[0] for p in line], [p[1] for p in line],
            color="#c5c9ce", lw=0.35, alpha=0.6, zorder=0, linestyle="--",
        )
        artists.append(ln)
    for poly in overlay.crosswalks:
        if len(poly) >= 3 and abs(poly[0][0] - poly[-1][0]) + abs(poly[0][1] - poly[-1][1]) > 1e-3:
            patch = MplPolygon(
                poly, closed=True, facecolor="#f0e68c", edgecolor="#c4a000",
                alpha=0.35, lw=0.6, zorder=0.5,
            )
            ax.add_patch(patch)
            artists.append(patch)
        else:
            (ln,) = ax.plot(
                [p[0] for p in poly], [p[1] for p in poly],
                color="#c4a000", lw=1.2, alpha=0.7, zorder=0.5,
            )
            artists.append(ln)
    for line in overlay.stop_lines:
        (ln,) = ax.plot(
            [p[0] for p in line], [p[1] for p in line],
            color="#d62728", lw=2.0, alpha=0.9, zorder=1,
        )
        artists.append(ln)
    if overlay.traffic_signs:
        xs = [s[0] for s in overlay.traffic_signs]
        ys = [s[1] for s in overlay.traffic_signs]
        sc = ax.scatter(xs, ys, marker="^", s=28, c="#9467bd", zorder=1, edgecolors="k", linewidths=0.3)
        artists.append(sc)
        # Only annotate a few nearby stop signs to avoid clutter
        for x, y, label in overlay.traffic_signs:
            if "stop" in label.lower():
                artists.append(
                    ax.text(x, y, "STOP", fontsize=5, color="#9467bd", ha="left", va="bottom", zorder=1)
                )
    return artists


def _turn_state_at(
    samples: list,
    stamp_sec: float,
    *,
    prefer_source: str = "cmd",
    max_age_sec: float = 1.0,
) -> tuple[int, str]:
    """Return (state, source) preferring planner cmd over vehicle status."""
    if not samples:
        return 1, ""
    stamps = [s.stamp_sec for s in samples]
    end = bisect.bisect_right(stamps, stamp_sec)
    best_cmd = None
    best_status = None
    for s in samples[:end]:
        if stamp_sec - s.stamp_sec > max_age_sec:
            continue
        if s.source == "cmd":
            best_cmd = s
        else:
            best_status = s
    if prefer_source == "cmd" and best_cmd is not None:
        return best_cmd.state, best_cmd.source
    if best_status is not None:
        return best_status.state, best_status.source
    if best_cmd is not None:
        return best_cmd.state, best_cmd.source
    return 1, ""


def _trajectory_at(
    trajectories: list[TrajectorySample], stamp_sec: float, max_age_sec: float = 0.5
) -> TrajectorySample | None:
    if not trajectories:
        return None
    stamps = [t.stamp_sec for t in trajectories]
    idx = bisect.bisect_right(stamps, stamp_sec) - 1
    if idx < 0:
        return None
    sample = trajectories[idx]
    if stamp_sec - sample.stamp_sec > max_age_sec:
        return None
    return sample


def _index_objects_per_frame(
    objs: list,
    ego_frames: list,
    sample_dt: float,
) -> list[list]:
    if not objs or not ego_frames:
        return [[] for _ in ego_frames]
    out: list[list] = [[] for _ in ego_frames]
    oi = 0
    for fi, ego in enumerate(ego_frames):
        t0, t1 = ego.stamp_sec - sample_dt, ego.stamp_sec + sample_dt
        while oi < len(objs) and objs[oi].stamp_sec < t0:
            oi += 1
        j = oi
        while j < len(objs) and objs[j].stamp_sec <= t1:
            out[fi].append(objs[j])
            j += 1
    return out


def _index_walls_per_frame(
    walls: list,
    ego_frames: list,
    sample_dt: float,
    max_age_sec: float = 0.5,
) -> list[list]:
    """Latest planning-factor walls whose stamp is within max_age of the frame."""
    if not walls or not ego_frames:
        return [[] for _ in ego_frames]
    stamps = [w.stamp_sec for w in walls]
    out: list[list] = []
    for ego in ego_frames:
        idx = bisect.bisect_right(stamps, ego.stamp_sec) - 1
        if idx < 0:
            out.append([])
            continue
        # Collect all walls sharing the latest message stamp (per source batch)
        latest_t = walls[idx].stamp_sec
        if ego.stamp_sec - latest_t > max_age_sec:
            out.append([])
            continue
        # Walk back to include sibling factors published at the same stamp window
        t0 = latest_t - max(sample_dt, 0.05)
        j = idx
        while j >= 0 and walls[j].stamp_sec >= t0:
            j -= 1
        frame_walls = walls[j + 1 : idx + 1]
        out.append(frame_walls)
    return out


def _wall_style(source: str) -> tuple[str, str, str]:
    leaf = source.rstrip("/").split("/")[-1]
    return _WALL_STYLE.get(leaf, _WALL_DEFAULT_STYLE)


def render_video(
    bag_path: Path,
    output_path: Path,
    *,
    metrics_json: Path | None = None,
    map_path: Path | None = None,
    goal: tuple[float, float, float] | None = None,
    fps: float = 8.0,
    sample_dt: float = 0.2,
    topics: TopicSet | None = None,
    vehicle_info_yaml: Path | None = None,
    series: BagSeries | None = None,
    dpi: int = 72,
    view_frame: str = "map",
    view_range_m: float = 40.0,
    show_planning_factors: bool = True,
) -> Path:
    topics = topics or TopicSet()
    view_frame = (view_frame or "map").strip().lower()
    if view_frame not in ("map", "base_link"):
        view_frame = "map"
    if series is None:
        series = load_bag_series(
            bag_path,
            ego_topic=topics.ego_pose,
            objects_topic=topics.predicted_objects,
            velocity_topic=topics.vehicle_status_velocity,
            trajectory_topic=topics.trajectory,
            planning_factor_topics=topics.planning_factors if show_planning_factors else [],
            turn_indicators_status_topic=topics.turn_indicators_status,
            turn_indicators_cmd_topic=topics.turn_indicators_cmd,
            object_sample_dt=sample_dt,
            trajectory_sample_dt=sample_dt,
            factor_sample_dt=sample_dt,
            skip_zero_size_objects=True,
        )
    ego_frames = downsample_ego(series.ego, sample_dt)
    if not ego_frames:
        raise RuntimeError("no ego samples in bag — cannot render")

    metrics: dict[str, Any] = {}
    if metrics_json and metrics_json.expanduser().is_file():
        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))

    if goal is None and "meta" in metrics and "goal" in metrics["meta"]:
        g = metrics["meta"]["goal"]
        goal = (float(g["x"]), float(g["y"]), float(g.get("yaw", 0.0)))

    footprint = load_vehicle_footprint(vehicle_info_yaml)

    xs = [e.x for e in series.ego]
    ys = [e.y for e in series.ego]
    pad = 25.0
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    map_overlay = _load_map_overlay(map_path, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    objs = series.objects
    trajectories = series.trajectories
    walls = series.virtual_walls if show_planning_factors else []
    turn_samples = series.turn_indicators
    ego_stamps = [e.stamp_sec for e in series.ego]
    trail_ends = [bisect.bisect_right(ego_stamps, e.stamp_sec) for e in ego_frames]
    objects_per_frame = _index_objects_per_frame(objs, ego_frames, sample_dt)
    walls_per_frame = _index_walls_per_frame(walls, ego_frames, sample_dt)

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    fail = not metrics.get("pass", True) if metrics else False
    fail_reason = []
    if metrics:
        if metrics.get("stuck_rate", {}).get("flagged"):
            fail_reason.append("STUCK")
        if metrics.get("collision_rate", {}).get("flagged"):
            fail_reason.append("COLLISION")
        if metrics.get("out_of_boundary", {}).get("flagged"):
            fail_reason.append("OOB")
        if metrics.get("goal_stop_precision", {}).get("flagged"):
            fail_reason.append("GOAL")

    _draw_map_overlay(ax, map_overlay)

    trail_line, = ax.plot([], [], color="#1f77b4", lw=1.2, alpha=0.85, zorder=1, label="ego path")
    plan_line, = ax.plot([], [], color="#ff7f0e", lw=2.2, alpha=0.95, zorder=2, label="planned trajectory")
    ego_patch = MplPolygon([[0, 0]], closed=True, facecolor="#d62728", edgecolor="k", alpha=0.8, zorder=3)
    ax.add_patch(ego_patch)
    blinker_left = MplPolygon(
        [[0, 0]], closed=True, facecolor="#ffdd57", edgecolor="#b8860b", alpha=0.0, zorder=6, lw=0.8
    )
    blinker_right = MplPolygon(
        [[0, 0]], closed=True, facecolor="#ffdd57", edgecolor="#b8860b", alpha=0.0, zorder=6, lw=0.8
    )
    ax.add_patch(blinker_left)
    ax.add_patch(blinker_right)
    npc_patches: list[MplPolygon] = []
    wall_patches: list[MplPolygon] = []
    wall_labels: list[Any] = []
    goal_artist = None
    if goal is not None:
        goal_artist = ax.scatter([], [], marker="*", s=180, c="gold", zorder=4, edgecolors="k")
        goal_artist.set_offsets([[goal[0], goal[1]]])

    ax.set_aspect("equal")
    if view_frame == "map":
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if fail:
        for spine in ax.spines.values():
            spine.set_color("red")
            spine.set_linewidth(3)

    legend_handles = [
        Patch(facecolor="#1f77b4", edgecolor="none", label="ego path"),
        Patch(facecolor="#ff7f0e", edgecolor="none", label="planned trajectory"),
        Patch(facecolor="#9aa0a6", edgecolor="none", label="lane bounds"),
        Patch(facecolor="#f0e68c", edgecolor="#c4a000", label="crosswalk"),
        Patch(facecolor="#8b0000", edgecolor="none", label="stop line"),
        Patch(facecolor="#9467bd", edgecolor="k", label="traffic sign"),
        Patch(facecolor="#ffdd57", edgecolor="#b8860b", label="blinker ON"),
    ]
    if show_planning_factors:
        for name, (face, edge, short) in _WALL_STYLE.items():
            legend_handles.append(
                Patch(facecolor=face, edgecolor=edge, linewidth=1.5, alpha=0.9, label=f"wall:{short}")
            )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=6, framealpha=0.7)

    title = ax.set_title("")
    half = max(5.0, float(view_range_m))

    def draw_frame(idx: int):
        e = ego_frames[idx]
        end = trail_ends[idx]
        trail_line.set_data([s.x for s in series.ego[:end]], [s.y for s in series.ego[:end]])

        planned = _trajectory_at(trajectories, e.stamp_sec)
        if planned is not None:
            plan_line.set_data([p[0] for p in planned.points], [p[1] for p in planned.points])
            plan_line.set_visible(True)
        else:
            plan_line.set_visible(False)

        ego_poly = transform_local_polygon(footprint, e.x, e.y, e.yaw_rad)
        ego_patch.set_xy(ego_poly)

        turn_state, turn_src = _turn_state_at(turn_samples, e.stamp_sec)
        # Blink ~2 Hz when on
        blink_on = (int(e.stamp_sec * 4) % 2) == 0
        if turn_state == TURN_LEFT and blink_on:
            blinker_left.set_xy(transform_local_polygon(_BLINKER_LEFT_LOCAL, e.x, e.y, e.yaw_rad))
            blinker_left.set_alpha(0.95)
        else:
            blinker_left.set_alpha(0.0)
        if turn_state == TURN_RIGHT and blink_on:
            blinker_right.set_xy(transform_local_polygon(_BLINKER_RIGHT_LOCAL, e.x, e.y, e.yaw_rad))
            blinker_right.set_alpha(0.95)
        else:
            blinker_right.set_alpha(0.0)

        frame_objs = objects_per_frame[idx]
        while len(npc_patches) < len(frame_objs):
            patch = MplPolygon([[0, 0]], closed=True, facecolor="#2ca02c", edgecolor="k", alpha=0.5, zorder=2)
            ax.add_patch(patch)
            npc_patches.append(patch)
        for pi, patch in enumerate(npc_patches):
            if pi < len(frame_objs):
                obj = frame_objs[pi]
                npc = transform_local_polygon(
                    box_local(obj.length_m, obj.width_m), obj.x, obj.y, obj.yaw_rad
                )
                patch.set_xy(npc)
                patch.set_visible(True)
            else:
                patch.set_visible(False)

        frame_walls = walls_per_frame[idx]
        while len(wall_patches) < len(frame_walls):
            face, edge, _ = _WALL_DEFAULT_STYLE
            patch = MplPolygon(
                [[0, 0]],
                closed=True,
                facecolor=face,
                edgecolor=edge,
                alpha=0.92,
                zorder=8,
                linewidth=2.0,
                hatch="///",
            )
            ax.add_patch(patch)
            wall_patches.append(patch)
            wall_labels.append(
                ax.text(
                    0.0,
                    0.0,
                    "",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                    ha="center",
                    va="bottom",
                    zorder=9,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": face,
                        "edgecolor": edge,
                        "linewidth": 1.0,
                        "alpha": 0.95,
                    },
                )
            )
        for wi, patch in enumerate(wall_patches):
            label = wall_labels[wi]
            if wi < len(frame_walls):
                wall = frame_walls[wi]
                face, edge, short = _wall_style(wall.source)
                poly = transform_local_polygon(_WALL_LOCAL, wall.x, wall.y, wall.yaw_rad)
                patch.set_xy(poly)
                patch.set_facecolor(face)
                patch.set_edgecolor(edge)
                patch.set_visible(True)
                label.set_position((wall.x, wall.y + 1.2))
                label.set_text(short)
                label.set_bbox(
                    {
                        "boxstyle": "round,pad=0.15",
                        "facecolor": face,
                        "edgecolor": edge,
                        "linewidth": 1.0,
                        "alpha": 0.95,
                    }
                )
                label.set_visible(True)
            else:
                patch.set_visible(False)
                label.set_visible(False)

        if view_frame == "base_link":
            ax.set_xlim(e.x - half, e.x + half)
            ax.set_ylim(e.y - half, e.y + half)

        t_title = (
            f"t={e.stamp_sec - ego_frames[0].stamp_sec:.1f}s  "
            f"speed={e.speed_mps:.2f} m/s  view={view_frame}"
        )
        if turn_state == TURN_LEFT:
            t_title += f"  blinker=LEFT({turn_src or '?'})"
        elif turn_state == TURN_RIGHT:
            t_title += f"  blinker=RIGHT({turn_src or '?'})"
        if fail_reason:
            t_title += "  FAIL: " + ",".join(fail_reason)
        if frame_walls:
            names = sorted({w.source for w in frame_walls})
            t_title += "  walls=" + ",".join(names)
        title.set_text(t_title)
        return [
            trail_line,
            plan_line,
            ego_patch,
            blinker_left,
            blinker_right,
            title,
            *npc_patches,
            *wall_patches,
            *wall_labels,
        ]

    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    anim = FuncAnimation(
        fig,
        draw_frame,
        frames=len(ego_frames),
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    writer: Any
    try:
        writer = FFMpegWriter(fps=fps, metadata={"artist": "dp_multi_eval"})
        anim.save(str(output_path), writer=writer)
    except Exception as exc:  # noqa: BLE001
        alt = output_path.with_suffix(".gif")
        print(f"[warn] ffmpeg writer failed ({exc}); writing {alt}")
        writer = PillowWriter(fps=max(1, int(fps)))
        anim.save(str(alt), writer=writer)
        output_path = alt
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 3: render top-down mp4 from output bag")
    p.add_argument("--bag_path", type=Path, required=True)
    p.add_argument("--metrics_json", type=Path, default=None)
    p.add_argument("--map_path", type=Path, default=None)
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument("--goal_pose", nargs=3, type=float, metavar=("X", "Y", "YAW"), default=None)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--sample_dt", type=float, default=0.2)
    p.add_argument("--topics-yaml", type=Path, default=None)
    p.add_argument("--vehicle_info_yaml", type=Path, default=None)
    p.add_argument(
        "--view_frame",
        choices=("map", "base_link"),
        default="map",
        help="map = fit full path; base_link = ego-centered window",
    )
    p.add_argument(
        "--view_range_m",
        type=float,
        default=40.0,
        help="Half-extent [m] for base_link view (ignored for map)",
    )
    p.add_argument(
        "--no_planning_factors",
        action="store_true",
        help="Skip virtual walls from /planning/planning_factors/*",
    )
    args = p.parse_args(argv)

    goal = tuple(args.goal_pose) if args.goal_pose else None
    path = render_video(
        args.bag_path,
        args.output_path,
        metrics_json=args.metrics_json,
        map_path=args.map_path,
        goal=goal,  # type: ignore[arg-type]
        fps=args.fps,
        sample_dt=args.sample_dt,
        topics=load_topics(args.topics_yaml),
        vehicle_info_yaml=args.vehicle_info_yaml,
        view_frame=args.view_frame,
        view_range_m=args.view_range_m,
        show_planning_factors=not args.no_planning_factors,
    )
    print(f"[done] video → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
