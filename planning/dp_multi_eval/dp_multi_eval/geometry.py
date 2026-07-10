"""Geometry helpers (shapely if available, else pure Python)."""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]
Polygon = list[Point]


def transform_local_polygon(
    local: Sequence[Point], x: float, y: float, yaw: float
) -> Polygon:
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * lx - s * ly, y + s * lx + c * ly) for lx, ly in local]


def ego_local_footprint(
    wheel_base: float = 5.3,
    front_overhang: float = 2.5,
    rear_overhang: float = 2.5,
    width: float = 2.5,
    margin: float = 0.0,
) -> Polygon:
    """Approximate bus footprint in base_link (axis-aligned box)."""
    x_f = front_overhang + wheel_base + margin
    x_r = -(rear_overhang + margin)
    half_w = width * 0.5 + margin
    return [(x_f, half_w), (x_f, -half_w), (x_r, -half_w), (x_r, half_w)]


def box_local(length: float, width: float) -> Polygon:
    hl, hw = length * 0.5, width * 0.5
    return [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]


def _poly_edges(poly: Polygon) -> list[tuple[Point, Point]]:
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_seg(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) < 1e-9 and _on_seg(a, c, b):
        return True
    if abs(o2) < 1e-9 and _on_seg(a, d, b):
        return True
    if abs(o3) < 1e-9 and _on_seg(c, a, d):
        return True
    if abs(o4) < 1e-9 and _on_seg(c, b, d):
        return True
    return False


def point_in_poly(p: Point, poly: Polygon) -> bool:
    # Ray casting
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
    return inside


def polygons_overlap(a: Polygon, b: Polygon) -> bool:
    for e1 in _poly_edges(a):
        for e2 in _poly_edges(b):
            if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True
    if a and point_in_poly(a[0], b):
        return True
    if b and point_in_poly(b[0], a):
        return True
    return False


def _point_seg_dist(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _polygon_area(poly: Polygon) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _point_to_polygon_distance(p: Point, poly: Polygon) -> float:
    if not poly:
        return float("inf")
    if point_in_poly(p, poly):
        return 0.0
    best = float("inf")
    for e in _poly_edges(poly):
        best = min(best, _point_seg_dist(p, e[0], e[1]))
    return best


def polygon_distance(a: Polygon, b: Polygon) -> float:
    """Min distance between polygons; 0 if overlap."""
    if _polygon_area(a) < 1e-6:
        return _point_to_polygon_distance(a[0], b)
    if _polygon_area(b) < 1e-6:
        return _point_to_polygon_distance(b[0], a)
    try:
        from shapely.geometry import Polygon as ShapelyPoly

        pa = ShapelyPoly(a)
        pb = ShapelyPoly(b)
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        if pa.area < 1e-6:
            return _point_to_polygon_distance(a[0], b)
        if pb.area < 1e-6:
            return _point_to_polygon_distance(b[0], a)
        if pa.intersects(pb):
            return 0.0
        return float(pa.distance(pb))
    except Exception:  # noqa: BLE001
        if polygons_overlap(a, b):
            return 0.0
        best = float("inf")
        for pa in a:
            for e in _poly_edges(b):
                best = min(best, _point_seg_dist(pa, e[0], e[1]))
        for pb in b:
            for e in _poly_edges(a):
                best = min(best, _point_seg_dist(pb, e[0], e[1]))
        return best


def load_vehicle_footprint(vehicle_info_yaml: Path | None, margin: float = 0.0) -> Polygon:
    if vehicle_info_yaml is None or not vehicle_info_yaml.expanduser().is_file():
        return ego_local_footprint(margin=margin)
    try:
        import yaml

        raw = yaml.safe_load(vehicle_info_yaml.read_text(encoding="utf-8")) or {}
        params = raw.get("/**", {}).get("ros__parameters", raw.get("ros__parameters", raw))
        return ego_local_footprint(
            wheel_base=float(params.get("wheel_base", 5.3)),
            front_overhang=float(params.get("front_overhang", 2.5)),
            rear_overhang=float(params.get("rear_overhang", 2.5)),
            width=float(params.get("vehicle_width", params.get("wheel_tread", 2.0) + 0.5)),
            margin=margin,
        )
    except Exception:  # noqa: BLE001
        return ego_local_footprint(margin=margin)
