"""Shared 2D helpers for the procedural subjects (shapely-based)."""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point, Polygon

#: Overlays snap to this grid, in mm.  GEOS 3.12 - what Pyodide ships, for
#: the browser version at hannesstaehlin.ch/frog - throws on some
#: near-degenerate intersections ("found non-noded intersection"), and under
#: WebAssembly that C++ exception kills the Python runtime outright.  Overlay
#: with a fixed grid uses snap rounding, which cannot fail that way; a
#: ten-thousandth of a millimetre is far below any pen.
GRID = 1e-4


def _gridded(*geoms) -> bool:
    """Can these go through fixed-grid overlay?

    Only shapes of one dimension can: GEOS refuses a line clipped by a
    polygon on a grid, and the refusal is itself a C++ exception, so it has
    to be decided here rather than caught.  Line clipping is the robust case
    anyway; it is polygon against polygon that fails.
    """
    dims = set()
    for g in geoms:
        if g.is_empty:
            continue
        if g.geom_type == "GeometryCollection":
            return False
        dims.add(int(shapely.get_dimensions(g)))
    return len(dims) <= 1


def cut(a, b):
    """a ∩ b, robustly (see GRID)."""
    if _lineal(a) and _polygonal(b):
        return clip(a, b, inside=True)
    if _polygonal(a) and _lineal(b):
        return clip(b, a, inside=True)
    return shapely.intersection(a, b, grid_size=GRID if _gridded(a, b) else None)


def minus(a, b):
    """a − b, robustly (see GRID)."""
    if _lineal(a) and _polygonal(b):
        return clip(a, b, inside=False)
    return shapely.difference(a, b, grid_size=GRID if _gridded(a, b) else None)


def _lineal(g):
    return g.geom_type in ("LineString", "LinearRing", "MultiLineString")


def _polygonal(g):
    return g.geom_type in ("Polygon", "MultiPolygon")


def clip(lines, region, inside=True):
    """The parts of ``lines`` inside (or outside) ``region``, without GEOS overlay.

    GEOS 3.12 cannot be trusted to clip lines either, so this does it
    directly: split every segment where it crosses an edge of the region,
    and keep each piece by whether its midpoint lies inside.  Point-in-polygon
    is a test, not a construction, and cannot fail.
    """
    if lines.is_empty:
        return MultiLineString()
    if region.is_empty:
        return lines if not inside else MultiLineString()
    rings = []
    for poly in getattr(region, "geoms", [region]):
        rings.append(np.asarray(poly.exterior.coords))
        rings += [np.asarray(r.coords) for r in poly.interiors]
    ea = np.vstack([r[:-1] for r in rings])          # edge starts
    ed = np.vstack([r[1:] - r[:-1] for r in rings])  # edge directions
    pieces = []
    for line in getattr(lines, "geoms", [lines]):
        pts = np.asarray(line.coords, float)
        run = []
        for p, q in zip(pts[:-1], pts[1:]):
            d = q - p
            den = d[0] * ed[:, 1] - d[1] * ed[:, 0]
            ok = np.abs(den) > 1e-12
            w = ea[ok] - p
            t = (w[:, 0] * ed[ok, 1] - w[:, 1] * ed[ok, 0]) / den[ok]
            u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / den[ok]
            ts = np.unique(np.concatenate([[0.0, 1.0], t[(t > 0) & (t < 1) & (u >= 0) & (u <= 1)]]))
            mids = p + np.outer((ts[:-1] + ts[1:]) / 2, d)
            keep = shapely.contains_xy(region, mids[:, 0], mids[:, 1]) == inside
            for (t0, t1), k in zip(zip(ts[:-1], ts[1:]), keep):
                if not k:
                    if len(run) > 1:
                        pieces.append(run)
                    run = []
                    continue
                a, b = p + t0 * d, p + t1 * d
                if not run:
                    run = [a]
                run.append(b)
        if len(run) > 1:
            pieces.append(run)
    pieces = [r for r in pieces if np.hypot(*(np.asarray(r[-1]) - r[0])) > 0 or len(r) > 2]
    return MultiLineString([np.asarray(r) for r in pieces]) if pieces else MultiLineString()


def join(a, b):
    """a ∪ b, robustly (see GRID)."""
    return shapely.union(a, b, grid_size=GRID if _gridded(a, b) else None)


def merge(geoms):
    """Union of many shapes, robustly (see GRID)."""
    geoms = list(geoms)
    return shapely.union_all(geoms, grid_size=GRID if _gridded(*geoms) else None)


# ── small geometry helpers ───────────────────────────────────────────────

def catmull(points, closed=True, n=14):
    """Smooth curve through control points (centripetal-ish Catmull-Rom)."""
    p = np.asarray(points, float)
    if closed:
        p = np.vstack([p[-1], p, p[0], p[1]])
    else:
        p = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out = np.vstack(out)
    if not closed:
        out = np.vstack([out, p[-2]])
    return out


def blob(cx, cy, rx, ry, rng, wobble=0.15, k=9, rot=0.0):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(0, 1)
    r = 1 + rng.uniform(-wobble, wobble, k)
    pts = np.stack([np.cos(a) * rx * r, np.sin(a) * ry * r], 1)
    c, s = math.cos(rot), math.sin(rot)
    pts = pts @ np.array([[c, s], [-s, c]])
    return Polygon(catmull(pts + [cx, cy])).buffer(0)


def limb(points, widths, n=10):
    """Tapered, round-ended outline along a smooth spine.

    Built as a chain of convex hulls of consecutive circles, so it has round
    joints and caps and tapers smoothly - a flat-ended strip reads as a
    plank, not a leg.
    """
    spine = catmull(points, closed=False, n=n)
    w = np.interp(np.linspace(0, 1, len(spine)),
                  np.linspace(0, 1, len(widths)), widths)
    pieces = []
    for i in range(len(spine) - 1):
        a = Point(*spine[i]).buffer(w[i] / 2, 16)
        b = Point(*spine[i + 1]).buffer(w[i + 1] / 2, 16)
        pieces.append(merge([a, b]).convex_hull)
    return merge(pieces).buffer(0)


def lines_of(geom):
    """All boundary/line pieces of a geometry, as numpy polylines."""
    out = []
    if geom.is_empty:
        return out
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            out += lines_of(g)
        return out
    if geom.geom_type == "Polygon":
        out.append(np.asarray(geom.exterior.coords))
        out += [np.asarray(r.coords) for r in geom.interiors]
    elif geom.geom_type in ("LineString", "LinearRing"):
        out.append(np.asarray(geom.coords))
    return out


def hatch(region, spacing, angle_deg, rng, fade=0.5, light=(-1, -1)):
    """Parallel hatch lines clipped to ``region``, starts staggered from the lit end."""
    if region.is_empty:
        return []
    minx, miny, maxx, maxy = region.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    r = math.hypot(maxx - minx, maxy - miny) / 2 + spacing
    a = math.radians(angle_deg)
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-u[1], u[0]])
    segs = []
    for o in np.arange(-r, r, spacing):
        c = np.array([cx, cy]) + v * o
        segs.append(LineString([c - u * r, c + u * r]))
    clipped = cut(region, MultiLineString(segs))
    out = []
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    for p in lines_of(clipped):
        if len(p) < 2:
            continue
        if (p[-1] - p[0]) @ L > 0:
            p = p[::-1]                      # p[0] = lit end
        t = rng.uniform(0, fade)
        a0 = p[0] + (p[-1] - p[0]) * t
        if np.hypot(*(p[-1] - a0)) > spacing * 0.6:
            out.append(np.vstack([a0, p[-1]]))
    return out


def crescent(poly, depth, light=(-1, -1)):
    """fishdraw's shadow: the shape minus a copy shifted toward the light."""
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    moved = affinity.translate(poly, xoff=L[0] * depth, yoff=L[1] * depth)
    return minus(poly, moved)


def poisson(region, r, rng, tries=30, limit=400):
    """Points inside ``region`` at least ``r`` apart (dart throwing on a grid).

    A background grid of cell size r/sqrt(2) holds at most one point per
    cell, so each candidate checks only its neighbourhood instead of every
    accepted point - the difference between seconds and milliseconds on a
    densely warted toad.
    """
    from shapely import contains_xy
    if region.is_empty:
        return []
    minx, miny, maxx, maxy = region.bounds
    cell = r / math.sqrt(2)
    gw, gh = int((maxx - minx) / cell) + 1, int((maxy - miny) / cell) + 1
    grid = -np.ones((gh, gw), int)
    pts = []
    n = limit * tries
    cand = np.column_stack([rng.uniform(minx, maxx, n), rng.uniform(miny, maxy, n)])
    cand = cand[contains_xy(region, cand[:, 0], cand[:, 1])]
    for p in cand:
        if len(pts) >= limit:
            break
        gx, gy = int((p[0] - minx) / cell), int((p[1] - miny) / cell)
        ok = True
        for yy in range(max(0, gy - 2), min(gh, gy + 3)):
            for xx in range(max(0, gx - 2), min(gw, gx + 3)):
                k = grid[yy, xx]
                if k >= 0 and (pts[k][0] - p[0]) ** 2 + (pts[k][1] - p[1]) ** 2 < r * r:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[gy, gx] = len(pts)
            pts.append(p)
    return pts
