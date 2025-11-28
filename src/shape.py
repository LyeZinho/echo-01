"""
Utility functions to create and validate shapes for the echo-01 simulator.

The module provides a JSON schema (very small) that describes a polygonal
shape as either a list of vertices or as a radial-parameterized shape
(center + radii + spokes). It includes basic validation filters (area,
bounding box aspect ratio, min edge length) to avoid unrealistic shapes such
as a single line or extremely thin geometries.

This is educational; it simplifies many geometric checks and shouldn't be
considered a production-grade mesh module.
"""

from __future__ import annotations
import json
import math
from typing import List, Tuple
import numpy as np

# Type alias for integer vertices (ix, iy)
Vertex = Tuple[int, int]


def radial_polygon(center: Tuple[int, int], radii: List[float], start_angle: float = 0.0) -> List[Vertex]:
    """Create polygon vertices from center and radial distances.

    - center: (ix, iy)
    - radii: list of radii (float) for each spoke (in grid units)
    - start_angle: initial angle in degrees

    Returns a list of integer grid (ix,iy) vertices.
    """
    cx, cy = center
    n = len(radii)
    verts: List[Vertex] = []
    for i, r in enumerate(radii):
        theta = math.radians(start_angle + 360.0 * i / n)
        # note: we use (row, col) = (ix, iy) mapping: ix = cx + r*sin, iy = cy + r*cos
        ix = int(round(cx + r * math.sin(theta)))
        iy = int(round(cy + r * math.cos(theta)))
        verts.append((ix, iy))
    return verts


def polygon_area(vertices: List[Vertex]) -> float:
    """Compute polygon area (positive) using shoelace formula.
    vertices must be in order (clockwise or counterclockwise)
    """
    xs = [v[1] for v in vertices]  # x corresponds to column (iy)
    ys = [v[0] for v in vertices]  # y is row (ix)
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def bounding_box_aspect(vertices: List[Vertex]) -> float:
    xs = [v[1] for v in vertices]
    ys = [v[0] for v in vertices]
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    if height == 0:
        return float('inf')
    return float(width) / float(height) if width >= height else float(height) / float(width)


def min_edge_length(vertices: List[Vertex]) -> float:
    """Compute min edge length in grid units (Euclidean)"""
    n = len(vertices)
    min_len = float('inf')
    for i in range(n):
        j = (i + 1) % n
        dx = vertices[i][0] - vertices[j][0]
        dy = vertices[i][1] - vertices[j][1]
        l = math.hypot(dx, dy)
        if l < min_len:
            min_len = l
    return 0.0 if math.isinf(min_len) else min_len


def is_simple_polygon(vertices: List[Vertex]) -> bool:
    """Rudimentary test for polygon self-intersection. O(n^2) checks for segment intersections.

    Returns True if polygon has no intersections (simple polygon), False otherwise.
    """
    def seg_intersect(a1, a2, b1, b2) -> bool:
        # Based on orientation test
        def orient(p, q, r):
            return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        def on_segment(p, q, r):
            # q is on segment pr
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        p1 = (a1[1], a1[0])
        q1 = (a2[1], a2[0])
        p2 = (b1[1], b1[0])
        q2 = (b2[1], b2[0])
        o1 = orient(p1, q1, p2)
        o2 = orient(p1, q1, q2)
        o3 = orient(p2, q2, p1)
        o4 = orient(p2, q2, q1)
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        # colinear cases
        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True
        return False

    n = len(vertices)
    for i in range(n):
        a1 = vertices[i]
        a2 = vertices[(i + 1) % n]
        for j in range(i + 1, n):
            b1 = vertices[j]
            b2 = vertices[(j + 1) % n]
            # adjacent segments share a vertex - skip
            if a1 == b1 or a1 == b2 or a2 == b1 or a2 == b2:
                continue
            if seg_intersect(a1, a2, b1, b2):
                return False
    return True


def validate_polygon(vertices: List[Vertex], min_area: float = 10.0, max_aspect: float = 10.0, min_edge: float = 1.0) -> Tuple[bool, str]:
    area = polygon_area(vertices)
    if area < min_area:
        return False, f'area {area:.2f} < min_area {min_area}'
    aspect = bounding_box_aspect(vertices)
    if aspect > max_aspect:
        return False, f'aspect {aspect:.2f} > max_aspect {max_aspect}'
    me = min_edge_length(vertices)
    if me < min_edge:
        return False, f'min_edge {me:.2f} < min_edge {min_edge}'
    if not is_simple_polygon(vertices):
        return False, 'polygon not simple (self-intersecting)'
    return True, 'ok'


def save_shape_json(path: str, center: Tuple[int, int], radii: List[float], metadata: dict = None, vertices: List[Vertex] = None) -> None:
    data = {
        'center': list(center),
        'radii': list(radii),
    }
    # If no explicit vertices were provided, create them from center/radii
    if vertices is None and radii is not None:
        vertices = radial_polygon(center, radii)
    if vertices is not None:
        # store as [[ix,iy], ...]
        data['vertices'] = [[int(v[0]), int(v[1])] for v in vertices]
    if metadata is not None:
        data['metadata'] = metadata
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_shape_json(path: str) -> Tuple[Tuple[int, int] | None, List[float] | None, dict, List[Vertex] | None]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    center = tuple(data['center']) if 'center' in data else None
    radii = list(data['radii']) if 'radii' in data else None
    metadata = data.get('metadata', {})
    # vertices are optional, return if present
    vertices = None
    if 'vertices' in data:
        vertices = [tuple(v) for v in data['vertices']]
    return center, radii, metadata, vertices
