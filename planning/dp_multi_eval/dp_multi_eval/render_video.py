"""Phase 3 — headless top-down video from output bag + metrics."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Patch

from dp_multi_eval.bag_reader import (
    BagSeries,
    TrajectorySample,
    TURN_LEFT,
    TURN_RIGHT,
    downsample_ego,
    load_bag_series,
    load_reference_ego,
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
    traffic_lights: list[tuple[float, float, int]] = field(default_factory=list)  # x,y,group_id
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

    # Traffic lights, drawn live-coloured per frame: a 48 s red wait reads as
    # "the ego is broken" when the video shows no reason for it.
    try:
        for elem in lanelet_map.regulatoryElementLayer:
            try:
                if elem.attributes.get("subtype", "") != "traffic_light":
                    continue
                refers = elem.parameters.get("refers", [])
                if not refers:
                    continue
                pts = _linestring_xy(refers[0])
                if not pts or not _pts_in_bbox(pts, bx0, bx1, by0, by1):
                    continue
                mid = pts[len(pts) // 2]
                overlay.traffic_lights.append((mid[0], mid[1], int(elem.id)))
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass

    return overlay


def _draw_map_overlay(ax: Any, overlay: MapOverlay) -> list[Any]:
    """One collection per line class. The map is redrawn every frame (the view
    follows the ego), and as ~1900 separate Line2D artists it was 170 s of a
    290 s render; a handful of collections draw in a few seconds."""
    artists: list[Any] = []

    def lines(segs: list[list[tuple[float, float]]], **style: Any) -> None:
        segs = [seg for seg in segs if len(seg) >= 2]
        if segs:
            coll = LineCollection(segs, **style)
            ax.add_collection(coll)
            artists.append(coll)

    lines(overlay.road_borders, colors="#6e7681", linewidths=1.0, alpha=0.7, zorder=0)
    lines(overlay.left_bounds + overlay.right_bounds, colors="#9aa0a6", linewidths=0.7, alpha=0.85, zorder=0)
    lines(overlay.centerlines, colors="#c5c9ce", linewidths=0.35, alpha=0.6, zorder=0, linestyles="--")
    closed = [
        poly for poly in overlay.crosswalks
        if len(poly) >= 3 and abs(poly[0][0] - poly[-1][0]) + abs(poly[0][1] - poly[-1][1]) > 1e-3
    ]
    open_ = [poly for poly in overlay.crosswalks if poly not in closed]
    if closed:
        pc = PolyCollection(closed, facecolors="#f0e68c", edgecolors="#c4a000", alpha=0.35, linewidths=0.6, zorder=0.5)
        ax.add_collection(pc)
        artists.append(pc)
    lines(open_, colors="#c4a000", linewidths=1.2, alpha=0.7, zorder=0.5)
    lines(overlay.stop_lines, colors="#d62728", linewidths=2.0, alpha=0.9, zorder=1)
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
    # One box per object per frame: the sample nearest the frame's instant.
    # Appending every sample inside the window drew a moving object up to ten
    # times per frame, a smear trailing behind it (2026-09-10).
    out: list[list] = [[] for _ in ego_frames]
    oi = 0
    for fi, ego in enumerate(ego_frames):
        t0, t1 = ego.stamp_sec - sample_dt, ego.stamp_sec + sample_dt
        while oi < len(objs) and objs[oi].stamp_sec < t0:
            oi += 1
        best: dict = {}
        j = oi
        while j < len(objs) and objs[j].stamp_sec <= t1:
            o = objs[j]
            key = o.object_id or id(o)
            if key not in best or abs(o.stamp_sec - ego.stamp_sec) < abs(best[key].stamp_sec - ego.stamp_sec):
                best[key] = o
            j += 1
        out[fi] = list(best.values())
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


from dp_multi_eval.lag import scene_time_map as _scene_time_map  # noqa: E402


def window_labels(windows_json: Path) -> list[tuple[float, float, str]]:
    """(wall from, wall to, title text) per segment of a leg-window run: the
    window number, and while the harness places the ego at a planned cut (its
    lead along the recording, +2 s) that it was placed — a jump the viewer
    should read as the cut, not as a defect (2026-11-10)."""
    try:
        payload = json.loads(Path(windows_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    lead = float((payload.get("params") or {}).get("lead_sec", 3.5))
    legs = payload.get("legs") or []
    out: list[tuple[float, float, str]] = []
    for leg in legs:
        tag = f"window {leg.get('index')}/{len(legs)}" + (" (cut)" if leg.get("opens") == "cut" else "")
        for seg in leg.get("segments") or []:
            a, b = seg.get("wall_handover"), seg.get("wall_end")
            if a is None:
                continue
            if seg.get("placed"):
                out.append((float(a) - lead - 0.5, float(a) + 2.0, tag + "  PLACED on the recording (planned cut)"))
            if seg.get("scene_jumped"):
                out.append((float(a) - 0.5, float(a) + 2.0,
                            tag + f"  SCENE JUMPED {float(seg.get('skipped_sec') or 0):+.0f}s (the ego did not wait here)"))
            out.append((float(a), float(b) if b is not None else float("inf"), tag))
    return out


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
    reference_bag: Path | None = None,
) -> Path:
    topics = topics or TopicSet()
    # The recording this replay is driving against. Showing where that vehicle was
    # at the same place — not the same instant — turns divergence from a number in
    # metrics.json into something visible: a ghost that falls behind is a replay
    # running fast, and that is the root of most conflicts in these runs.
    reference_ego: list[Any] = []
    if reference_bag is not None:
        try:
            reference_ego = load_reference_ego(
                reference_bag,
                ego_topic=topics.ego_pose,
                velocity_topic=topics.vehicle_status_velocity,
            )
        except Exception:  # noqa: BLE001
            reference_ego = []
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

    # Live traffic-light colours from the run's own recorded signal stream.
    signal_series: list[tuple[float, dict[int, int]]] = []
    if map_overlay.traffic_lights:
        try:
            from rclpy.serialization import deserialize_message
            from rosbag2_py import StorageFilter
            from rosidl_runtime_py.utilities import get_message

            from dp_multi_eval.bag_reader import _type_map, open_bag_reader

            _reader = open_bag_reader(bag_path)
            _types = _type_map(_reader)
            if topics.traffic_signals in _types:
                _reader.set_filter(StorageFilter(topics=[topics.traffic_signals]))
                while _reader.has_next():
                    _t, _d, _ns = _reader.read_next()
                    _m = deserialize_message(_d, get_message(_types[topics.traffic_signals]))
                    st = {
                        g.traffic_light_group_id: (g.elements[0].color if g.elements else 0)
                        for g in _m.traffic_light_groups
                    }
                    signal_series.append((_ns * 1e-9, st))
        except Exception:  # noqa: BLE001
            signal_series = []
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

    # Two different pairings, because they answer two different questions.
    #
    # The speed strip pairs by *position*: what was the recorded speed at this place.
    # Pairing it on the clock would compare unrelated stretches of road, since the
    # replay reaches a place at its own moment.
    #
    # The ghost pairs by *elapsed time*: where had the recorded vehicle got to by now.
    # That is the one that has to be on the clock — a ghost matched by position sits
    # under the ego by construction and shows nothing. Matched by time it falls behind
    # when the replay runs fast, which is exactly the divergence worth seeing.
    #
    # A leg-window run (windows.json beside the bag) is different: the scene is
    # on recorded time, jumped at each window and held while the ego is set up,
    # so the recorded vehicle at the SAME SCENE TIME is the one the ego is
    # driving beside, for the strip and the ghost alike; paired by place the
    # strip hid every departure delay behind a curve that rose with the ego.
    speed_pairs: list[Any] = []
    ghost_pairs: list[Any] = []
    scene_time = _scene_time_map(Path(bag_path).parent / "windows.json")
    win_labels = window_labels(Path(bag_path).parent / "windows.json")
    if reference_ego:
        ref_t0 = reference_ego[0].stamp_sec
        ego_t0 = ego_frames[0].stamp_sec
        ref_stamps = [r.stamp_sec for r in reference_ego]
        for e in ego_frames:
            if scene_time is not None:
                import bisect as _b

                k = min(_b.bisect_left(ref_stamps, scene_time(e.stamp_sec)), len(reference_ego) - 1)
                speed_pairs.append(reference_ego[k])
                ghost_pairs.append(reference_ego[k])
                continue
            best = None
            best_d2 = 25.0
            for r in reference_ego:
                d2 = (r.x - e.x) ** 2 + (r.y - e.y) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = r
            speed_pairs.append(best)

            elapsed = e.stamp_sec - ego_t0
            ghost_pairs.append(
                min(reference_ego, key=lambda r: abs((r.stamp_sec - ref_t0) - elapsed))
            )
    ref_label = "recorded, same time" if scene_time is not None else "recorded, same place"
    ghost_label = (
        "recorded ego (same scene time)" if scene_time is not None else "recorded ego (same elapsed time)"
    )

    have_ref = any(g is not None for g in speed_pairs)
    if have_ref:
        fig, (ax, ax_v) = plt.subplots(
            2, 1, figsize=(8, 9.4), dpi=dpi,
            gridspec_kw={"height_ratios": [8, 1.4], "hspace": 0.22},
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
        ax_v = None
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
    ghost_patch = MplPolygon(
        [[0, 0]], closed=True, facecolor="none", edgecolor="#7b5ea7",
        linestyle="--", linewidth=1.6, alpha=0.9, zorder=2,
    )
    ax.add_patch(ghost_patch)
    ghost_patch.set_visible(False)
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
    if have_ref:
        legend_handles.append(
            Patch(facecolor="none", edgecolor="#7b5ea7", linewidth=1.6, label=ghost_label)
        )
    if show_planning_factors:
        # Only the wall sources this run actually published: a legend line for a
        # module that never spoke (the fixer, switched off) read as its flag.
        present = {w.source.rstrip("/").split("/")[-1] for w in walls}
        for name, (face, edge, short) in _WALL_STYLE.items():
            if name in present:
                legend_handles.append(
                    Patch(facecolor=face, edgecolor=edge, linewidth=1.5, alpha=0.9, label=f"wall:{short}")
                )
        if present - set(_WALL_STYLE):
            face, edge, short = _WALL_DEFAULT_STYLE
            legend_handles.append(
                Patch(facecolor=face, edgecolor=edge, linewidth=1.5, alpha=0.9, label=f"wall:{short}")
            )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=6, framealpha=0.7)

    # Speed strip: ego against the recorded speed at the same place, with a playhead.
    # A single glance answers the question the map alone cannot — is this run driving
    # the road the way the road was driven.
    playhead = None
    if ax_v is not None:
        t_rel = [e.stamp_sec - ego_frames[0].stamp_sec for e in ego_frames]
        ego_v = [e.speed_mps for e in ego_frames]
        ref_v = [(g.speed_mps if g is not None else float("nan")) for g in speed_pairs]
        ax_v.plot(t_rel, ref_v, color="#7b5ea7", lw=1.4, ls="--", label=ref_label)
        ax_v.plot(t_rel, ego_v, color="#d62728", lw=1.6, label="ego")
        ax_v.set_xlim(t_rel[0], max(t_rel[-1], t_rel[0] + 1.0))
        top = max([v for v in ego_v + ref_v if v == v] + [1.0])
        ax_v.set_ylim(0, top * 1.15)
        ax_v.set_ylabel("m/s", fontsize=7)
        ax_v.set_xlabel("t [s]", fontsize=7)
        ax_v.tick_params(labelsize=6)
        ax_v.grid(True, alpha=0.25, lw=0.5)
        ax_v.legend(loc="upper right", fontsize=6, framealpha=0.7, ncol=2)
        playhead = ax_v.axvline(t_rel[0], color="#333333", lw=1.0)
        for side in ("top", "right"):
            ax_v.spines[side].set_visible(False)

    title = ax.set_title("")
    half = max(5.0, float(view_range_m))

    _SIG_COLORS = {1: "#d62728", 2: "#ff9900", 3: "#2ca02c"}
    light_dots = None
    if map_overlay.traffic_lights and signal_series:
        light_dots = ax.scatter(
            [tl[0] for tl in map_overlay.traffic_lights],
            [tl[1] for tl in map_overlay.traffic_lights],
            s=42, c="#9aa0a6", edgecolors="k", linewidths=0.6, zorder=9, marker="o",
        )
        _sig_times = [t for t, _ in signal_series]

    def _light_colors_at(t: float) -> list[str]:
        import bisect as _b

        k = min(_b.bisect_left(_sig_times, t), len(signal_series) - 1)
        st = signal_series[k][1]
        return [
            _SIG_COLORS.get(st.get(gid, 0), "#9aa0a6")
            for _x, _y, gid in map_overlay.traffic_lights
        ]

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

        if light_dots is not None:
            light_dots.set_facecolors(_light_colors_at(e.stamp_sec))

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

        ghost = ghost_pairs[idx] if idx < len(ghost_pairs) else None
        same_place = speed_pairs[idx] if idx < len(speed_pairs) else None
        if ghost is not None:
            ghost_patch.set_xy(
                transform_local_polygon(footprint, ghost.x, ghost.y, ghost.yaw_rad)
            )
            ghost_patch.set_visible(True)
        else:
            ghost_patch.set_visible(False)
        if playhead is not None:
            playhead.set_xdata([e.stamp_sec - ego_frames[0].stamp_sec] * 2)

        if view_frame == "base_link":
            ax.set_xlim(e.x - half, e.x + half)
            ax.set_ylim(e.y - half, e.y + half)

        t_title = (
            f"t={e.stamp_sec - ego_frames[0].stamp_sec:.1f}s  "
            f"speed={e.speed_mps:.2f} m/s  view={view_frame}"
        )
        if same_place is not None:
            t_title += (
                f"  ref_here={same_place.speed_mps:.2f} "
                f"({e.speed_mps - same_place.speed_mps:+.2f})"
            )
        if ghost is not None:
            lead = math.hypot(e.x - ghost.x, e.y - ghost.y)
            t_title += f"  ahead_of_ref={lead:.0f}m"
        if turn_state == TURN_LEFT:
            t_title += f"  blinker=LEFT({turn_src or '?'})"
        elif turn_state == TURN_RIGHT:
            t_title += f"  blinker=RIGHT({turn_src or '?'})"
        label = next((txt for a_, b_, txt in win_labels if a_ <= e.stamp_sec <= b_), None)
        if label:
            t_title += "  " + label
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
            ghost_patch,
            *([playhead] if playhead is not None else []),
            blinker_left,
            blinker_right,
            title,
            *npc_patches,
            *wall_patches,
            *wall_labels,
        ]

    from matplotlib.animation import FFMpegWriter, PillowWriter

    # Drive the writer directly: FuncAnimation.save drew every frame twice
    # (once for the animation step, once more inside grab_frame's savefig).
    def _write(writer: Any, path: Path) -> None:
        with writer.saving(fig, str(path), dpi):
            for idx in range(len(ego_frames)):
                draw_frame(idx)
                writer.grab_frame()

    try:
        _write(FFMpegWriter(fps=fps, metadata={"artist": "dp_multi_eval"}), output_path)
    except Exception as exc:  # noqa: BLE001
        alt = output_path.with_suffix(".gif")
        print(f"[warn] ffmpeg writer failed ({exc}); writing {alt}")
        _write(PillowWriter(fps=max(1, int(fps))), alt)
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
