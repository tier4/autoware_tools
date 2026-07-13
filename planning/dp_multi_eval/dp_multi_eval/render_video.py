"""Phase 3 — headless top-down video from output bag + metrics."""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Patch

from dp_multi_eval.bag_reader import BagSeries, TrajectorySample, downsample_ego, load_bag_series
from dp_multi_eval.geometry import box_local, load_vehicle_footprint, transform_local_polygon
from dp_multi_eval.topics import TopicSet, load_topics

# Virtual wall colors by planning-factor source (topic leaf / module name)
_WALL_COLORS = {
    "modifier_obstacle_stop": "#e41a1c",
    "diffusion_planner": "#ff7f00",
    "stop_point_fixer": "#984ea3",
}
_WALL_DEFAULT_COLOR = "#377eb8"
_WALL_LOCAL = box_local(0.15, 5.0)  # thin along heading, wide laterally (RViz-like)


@lru_cache(maxsize=4)
def _load_lanelet_map(osm_path: str):
    from autoware_lanelet2_extension_python.projection import MGRSProjector
    from lanelet2.io import Origin, load

    return load(osm_path, MGRSProjector(Origin(0.0, 0.0)))


def _load_map_lines(
    map_path: Path | None,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    margin: float = 40.0,
) -> list[list[tuple[float, float]]]:
    if map_path is None:
        return []
    map_path = map_path.expanduser().resolve()
    osm = map_path if map_path.suffix == ".osm" else map_path / "lanelet2_map.osm"
    if not osm.is_file():
        return []
    try:
        lanelet_map = _load_lanelet_map(str(osm))
    except Exception:  # noqa: BLE001
        return []

    bx0, bx1 = xmin - margin, xmax + margin
    by0, by1 = ymin - margin, ymax + margin
    lines: list[list[tuple[float, float]]] = []
    for ll in lanelet_map.laneletLayer:
        center = [(p.x, p.y) for p in ll.centerline]
        if len(center) < 2:
            continue
        if not any(bx0 <= x <= bx1 and by0 <= y <= by1 for x, y in center):
            continue
        lines.append(center)
    return lines


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


def _wall_color(source: str) -> str:
    leaf = source.rstrip("/").split("/")[-1]
    return _WALL_COLORS.get(leaf, _WALL_DEFAULT_COLOR)


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

    map_lines = _load_map_lines(map_path, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    objs = series.objects
    trajectories = series.trajectories
    walls = series.virtual_walls if show_planning_factors else []
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

    # Static map layer — draw once, reuse via artist list
    map_artists = []
    for line in map_lines:
        (ln,) = ax.plot(
            [p[0] for p in line],
            [p[1] for p in line],
            color="#bbbbbb",
            lw=0.4,
            zorder=0,
        )
        map_artists.append(ln)

    trail_line, = ax.plot([], [], color="#1f77b4", lw=1.2, alpha=0.85, zorder=1, label="ego path")
    plan_line, = ax.plot([], [], color="#ff7f0e", lw=2.2, alpha=0.95, zorder=2, label="planned trajectory")
    ego_patch = MplPolygon([[0, 0]], closed=True, facecolor="#d62728", edgecolor="k", alpha=0.8, zorder=3)
    ax.add_patch(ego_patch)
    npc_patches: list[MplPolygon] = []
    wall_patches: list[MplPolygon] = []
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
    ]
    if show_planning_factors:
        for name, color in _WALL_COLORS.items():
            legend_handles.append(Patch(facecolor=color, edgecolor="k", alpha=0.65, label=name))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

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
            patch = MplPolygon(
                [[0, 0]],
                closed=True,
                facecolor=_WALL_DEFAULT_COLOR,
                edgecolor="k",
                alpha=0.7,
                zorder=5,
                linewidth=0.6,
            )
            ax.add_patch(patch)
            wall_patches.append(patch)
        for wi, patch in enumerate(wall_patches):
            if wi < len(frame_walls):
                wall = frame_walls[wi]
                poly = transform_local_polygon(_WALL_LOCAL, wall.x, wall.y, wall.yaw_rad)
                patch.set_xy(poly)
                patch.set_facecolor(_wall_color(wall.source))
                patch.set_visible(True)
            else:
                patch.set_visible(False)

        if view_frame == "base_link":
            ax.set_xlim(e.x - half, e.x + half)
            ax.set_ylim(e.y - half, e.y + half)

        t_title = (
            f"t={e.stamp_sec - ego_frames[0].stamp_sec:.1f}s  "
            f"speed={e.speed_mps:.2f} m/s  view={view_frame}"
        )
        if fail_reason:
            t_title += "  FAIL: " + ",".join(fail_reason)
        if frame_walls:
            names = sorted({w.source for w in frame_walls})
            t_title += "  walls=" + ",".join(names)
        title.set_text(t_title)
        return [trail_line, plan_line, ego_patch, title, *npc_patches, *wall_patches]

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
