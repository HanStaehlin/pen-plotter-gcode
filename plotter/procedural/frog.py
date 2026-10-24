"""Procedural frog, in the manner of LingDong's fishdraw.

Every part is a closed polygon built from a handful of random parameters.
Parts are listed front to back; each one is clipped by everything already
drawn in front of it, which is what lets a leg sit in front of the body
without any 3D model. Shading is fishdraw's: hatch the part of each shape
not covered by a copy of itself shifted toward the light, so every form gets
a crescent of shadow on its far side.

Coordinates are millimetres, x right, y down, frog facing left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union


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
        pieces.append(unary_union([a, b]).convex_hull)
    return unary_union(pieces).buffer(0)


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
    clipped = region.intersection(MultiLineString(segs))
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
    return poly.difference(moved)


def poisson(region, r, rng, tries=30, limit=400):
    """Points inside ``region`` at least ``r`` apart (dart throwing)."""
    minx, miny, maxx, maxy = region.bounds
    pts = []
    for _ in range(limit * tries):
        if len(pts) >= limit:
            break
        p = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)])
        if not region.contains(Point(p)):
            continue
        if all(np.hypot(*(p - q)) >= r for q in pts):
            pts.append(p)
    return pts


# ── the frog ─────────────────────────────────────────────────────────────

@dataclass
class Part:
    name: str
    shape: object                    # shapely polygon
    shade: float = 1.0               # crescent depth multiplier
    outline: bool = True


def params(rng):
    return dict(
        length=rng.uniform(95, 115),
        hump=rng.uniform(0.85, 1.25),
        snout=rng.uniform(0.85, 1.15),
        plump=rng.uniform(0.9, 1.15),
        eye=rng.uniform(0.85, 1.2),
        warts=rng.uniform(0.0, 1.0),
        spots=int(rng.integers(0, 9)),
        toe=rng.uniform(0.9, 1.2),
    )


def build(seed: int = 0):
    """Return a list of polylines (mm, y down) for one frog."""
    rng = np.random.default_rng(seed)
    P = params(rng)
    s = P["length"] / 100.0
    hp, sn, pl = P["hump"], P["snout"], P["plump"]

    def S(pts):
        return [(x * s, y * s) for x, y in pts]

    body = Polygon(catmull(S([
        (-52 * sn, -2), (-46 * sn, -12), (-30, -21), (-10, -24 * hp),
        (12, -30 * hp), (32, -26 * hp), (46, -12), (44, 8 * pl),
        (25, 20 * pl), (0, 23 * pl), (-25, 19 * pl), (-42, 10), (-50, 4)]), n=16)).buffer(0)

    er = 9.5 * P["eye"] * s
    ex, ey = -30 * sn * s, -21 * s
    eye = Point(ex, ey).buffer(er, 48)
    iris = Point(ex - 0.8 * s, ey + 0.3 * s).buffer(er * 0.72, 48)
    pupil = affinity.scale(Point(ex - 1.0 * s, ey + 0.5 * s).buffer(er * 0.36, 32), 1.45, 0.62)
    glint = Point(ex - 3.2 * s, ey - 2.4 * s).buffer(er * 0.13, 16)
    lid = LineString(catmull([(ex - er * 1.1, ey - er * 0.15), (ex - er * 0.2, ey - er * 1.25),
                              (ex + er * 0.95, ey - er * 0.55)], closed=False))
    tymp = Point(-14 * s, -11 * s).buffer(6.5 * s * P["eye"], 40)

    # Near hind leg, folded the way a sitting frog's is: thigh runs forward
    # and down from the hip to the knee, shank runs back from the knee to the
    # heel, and the long foot lies flat on the ground pointing forward.
    toe = P["toe"]
    thigh = blob(24 * s, 6 * s, 24 * s, 14 * s * pl, rng, 0.05, 9, rot=-0.42)
    knee = np.array([2, 18]) * s
    heel = np.array([33, 24]) * s
    shank = limb([knee, np.array([18, 23]) * s, heel], [12 * s, 10 * s, 7 * s])
    ankle = np.array([10, 30]) * s
    tarsus = limb([heel, np.array([22, 29.5]) * s, ankle], [6.5 * s, 5.5 * s, 5 * s])
    toes, tips, knuckles = [], [], []
    for i, (dx, dy) in enumerate([(-19, -3.5), (-25, -1.2), (-30, 0.6), (-25, 2.0), (-17, 3.0)]):
        tip = ankle + np.array([dx * toe, dy]) * s
        mid = ankle + (tip - ankle) * 0.55 + np.array([0, 0.6 * s])
        toes.append(limb([ankle, mid, tip], [3.6 * s, 2.6 * s, 2.0 * s]))
        toes.append(Point(*tip).buffer(1.5 * s, 16))        # toe pad
        tips.append(tip)
        knuckles.append(ankle + (tip - ankle) * 0.86)
    # webbing: scalloped membrane between neighbouring toes, sagging back
    # toward the ankle between each pair
    rim = [knuckles[0]]
    for a, b in zip(knuckles[:-1], knuckles[1:]):
        sag = (a + b) / 2 + (ankle - (a + b) / 2) * 0.22
        rim += [sag, b]
    web = Polygon(np.vstack([ankle, catmull(rim, closed=False, n=5), ankle])).buffer(0)
    foot = unary_union(toes + [web, tarsus])

    # near front leg: upper arm back and down, forearm forward to the hand
    shoulder = np.array([-30, 8]) * s
    elbow = np.array([-25, 23]) * s
    wrist = np.array([-37, 31]) * s
    arm = limb([shoulder, elbow, wrist], [9 * s, 6.5 * s, 4.5 * s])
    fingers = []
    for dx, dy in [(-9, -1.2), (-10, 1.0), (-6, 2.6), (2, 2.0)]:
        tip = wrist + np.array([dx, dy]) * s
        fingers.append(limb([wrist, (wrist + tip) / 2, tip], [3.2 * s, 2.3 * s, 1.8 * s]))
        fingers.append(Point(*tip).buffer(1.3 * s, 14))
    hand = unary_union(fingers)

    # far legs, mostly hidden: shifted back and up
    far_hand = affinity.translate(unary_union([arm, hand]), 9 * s, -3 * s)
    far_foot = affinity.translate(foot, -6 * s, -2.5 * s)

    parts = [
        Part("hand", hand, 0.6), Part("arm", arm, 0.8),
        Part("foot", foot, 0.6), Part("shank", shank, 1.0), Part("thigh", thigh, 1.2),
        Part("eye", eye, 0.0), Part("tymp", tymp, 0.0), Part("body", body, 1.6),
        Part("far_hand", far_hand, 0.0), Part("far_foot", far_foot, 0.0),
    ]

    out = []
    front = Polygon()
    light = (-1, -1.2)

    def emit(geoms, mask=None):
        for g in geoms if isinstance(geoms, list) else [geoms]:
            g = g if mask is None else g.difference(mask)
            out.extend(lines_of(g))

    for part in parts:
        visible = part.shape.difference(front)
        if part.outline:
            emit(LineString(part.shape.exterior.coords) if part.shape.geom_type == "Polygon"
                 else part.shape.boundary, front)
        if part.shade > 0:
            band = crescent(part.shape, 4.5 * s * part.shade, light).intersection(visible)
            for h in hatch(band, 1.15 * s, 35, rng, 0.5, light):
                out.append(h)
        front = front.union(part.shape)

    # eye detail sits on top of everything behind it
    out += lines_of(iris.boundary)
    out += hatch(pupil.difference(glint), 0.45 * s, 10, rng, 0.0, light)
    out += lines_of(pupil.boundary.difference(glint))
    out += lines_of(lid)
    out += hatch(crescent(iris, 2.5 * s, light).difference(pupil), 0.9 * s, 35, rng, 0.4, light)
    out += lines_of(Point(*tymp.centroid.coords[0]).buffer(4.2 * s * P["eye"], 32).boundary)

    # body surface: mouth, dorsolateral fold, spots, warts
    others = unary_union([hand, arm, foot, shank, thigh, eye, tymp])
    skin = body.difference(others)
    mouth = LineString(catmull(S([(-51 * sn, 1), (-40, 4), (-26, 4.5), (-17, 1)]), closed=False))
    out += lines_of(mouth.intersection(body.buffer(-0.6 * s)).difference(others))
    fold = LineString(catmull(S([(-18, -18), (0, -23 * hp), (20, -26 * hp), (36, -18)]), closed=False))
    out += lines_of(fold.difference(others))

    back = skin.union(thigh.difference(unary_union([shank, foot])))
    back = back.difference(unary_union([eye, tymp]).buffer(2.0 * s))
    for _ in range(P["spots"]):
        c = np.array(back.representative_point().coords[0])
        for _ in range(20):
            minx, miny, maxx, maxy = back.bounds
            q = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy * 0.6 + miny * 0.4)])
            if back.contains(Point(q)):
                c = q
                break
        spot = blob(*c, rng.uniform(3, 6.5) * s, rng.uniform(2.2, 4.5) * s, rng, 0.25,
                    rot=rng.uniform(0, 3)).intersection(back)
        out += lines_of(spot.boundary)
        out += hatch(spot, 0.8 * s, -40, rng, 0.0, light)

    n_warts = 0
    if P["warts"] > 0.15:
        upper = skin.intersection(affinity.translate(body, 0, -6 * s))
        for p in poisson(upper, (5.5 - 2.5 * P["warts"]) * s, rng, limit=60):
            r = rng.uniform(0.6, 1.4) * s
            w = Point(*p).buffer(r, 12)
            # a wart is a tiny dome: an arc on its shadow side reads better
            # than a full circle, which looks like a hole
            arc = w.boundary.intersection(crescent(w.buffer(0.01), r * 0.9, light).buffer(0.05))
            out += lines_of(arc)
            n_warts += 1

    # a little ground under the feet: a few short strokes along the line
    # the feet rest on, densest right under them
    feet = unary_union([foot, hand])
    minx, miny, maxx, maxy = feet.bounds
    for i in range(6):
        gy = maxy + (0.8 + i * 1.4) * s
        x0 = rng.uniform(minx - 6 * s, maxx - 12 * s)
        L = rng.uniform(10, 30) * s * (1 - i / 8)
        out.append(np.array([[x0, gy], [x0 + L, gy + rng.uniform(-0.4, 0.4) * s]]))

    return [np.asarray(p, float) for p in out if len(p) >= 2]
