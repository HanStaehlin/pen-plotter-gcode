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


SPECIES = ("frog", "toad", "treefrog")


def params(rng, species="frog"):
    """Random proportions within the range that still reads as the species."""
    u = rng.uniform
    if species == "toad":
        # squat, dry, densely warty; big parotoid gland behind the eye;
        # short thick legs and little webbing
        return dict(length=u(100, 110), hump=u(1.0, 1.2), snout=u(0.78, 0.9),
                    plump=u(1.1, 1.25), eye=u(0.95, 1.08), warts=1.0,
                    wart_size=(0.8, 2.0), wart_all=True, spots=int(rng.integers(2, 7)),
                    toe=u(0.8, 0.92), web=0.45, pad=0.6, limb_w=1.15,
                    pupil="h", parotoid=True, bumpy=True, stripe=False, branch=False)
    if species == "treefrog":
        # slender and smooth, huge protruding eyes, long thin limbs ending in
        # big round adhesive discs; sits on a branch
        return dict(length=u(90, 102), hump=u(0.72, 0.88), snout=u(0.85, 1.0),
                    plump=u(0.74, 0.88), eye=u(1.35, 1.6), warts=0.0,
                    wart_size=(0.6, 1.4), wart_all=False, spots=0,
                    toe=u(1.05, 1.25), web=0.42, pad=3.4, limb_w=0.72,
                    pupil=str(rng.choice(["h", "v"])), parotoid=False, bumpy=False,
                    stripe=True, branch=True)
    return dict(length=u(95, 115), hump=u(0.85, 1.25), snout=u(0.85, 1.15),
                plump=u(0.9, 1.15), eye=u(0.85, 1.2), warts=u(0.0, 1.0),
                wart_size=(0.6, 1.4), wart_all=False, spots=int(rng.integers(0, 9)),
                toe=u(0.9, 1.2), web=0.86, pad=1.0, limb_w=1.0,
                pupil="h", parotoid=False, bumpy=False, stripe=False, branch=False)


def build(seed: int = 0, species: str = "frog"):
    """Return a list of polylines (mm, y down) for one frog-like animal."""
    if species not in SPECIES:
        raise ValueError(f"unknown species {species!r}; expected one of {SPECIES}")
    rng = np.random.default_rng(seed)
    P = params(rng, species)
    lw = P["limb_w"]
    s = P["length"] / 100.0
    hp, sn, pl = P["hump"], P["snout"], P["plump"]

    def S(pts):
        return [(x * s, y * s) for x, y in pts]

    body = Polygon(catmull(S([
        (-52 * sn, -2), (-46 * sn, -12), (-30, -21), (-10, -24 * hp),
        (12, -30 * hp), (32, -26 * hp), (46, -12), (44, 8 * pl),
        (25, 20 * pl), (0, 23 * pl), (-25, 19 * pl), (-42, 10), (-50, 4)]), n=16)).buffer(0)
    if P["bumpy"]:
        # lumps along the upper silhouette: a toad's outline is never smooth
        ring = body.exterior
        cy = body.centroid.y
        lumps = []
        for d in np.arange(0, ring.length, 3.2 * s):
            q = ring.interpolate(d + rng.uniform(-1, 1) * s)
            if q.y < cy + 4 * s:
                lumps.append(q.buffer(rng.uniform(0.9, 2.0) * s, 12))
        body = unary_union([body] + lumps).buffer(0.3 * s).buffer(-0.3 * s)

    er = 9.5 * P["eye"] * s
    ex, ey = -30 * sn * s, (-21 - 2.5 * (P['eye'] - 1)) * s
    eye = Point(ex, ey).buffer(er, 48)
    iris = Point(ex - 0.8 * s, ey + 0.3 * s).buffer(er * 0.72, 48)
    pupil = affinity.scale(Point(ex - 1.0 * s, ey + 0.5 * s).buffer(er * 0.36, 32),
                           *((1.45, 0.62) if P['pupil'] == 'h' else (0.5, 1.5)))
    glint = Point(ex - 3.2 * s, ey - 2.4 * s).buffer(er * 0.13, 16)
    lid = LineString(catmull([(ex - er * 1.1, ey - er * 0.15), (ex - er * 0.2, ey - er * 1.25),
                              (ex + er * 0.95, ey - er * 0.55)], closed=False))
    tymp = Point(-14 * s, -11 * s).buffer(6.5 * s * min(P['eye'], 1.15), 40)
    parotoid = (blob(-6 * s, -17 * s, 11 * s, 5 * s, rng, 0.08, 9, rot=-0.22)
                if P['parotoid'] else Polygon())

    # Near hind leg, folded the way a sitting frog's is: thigh runs forward
    # and down from the hip to the knee, shank runs back from the knee to the
    # heel, and the long foot lies flat on the ground pointing forward.
    toe = P["toe"]
    thigh = blob(24 * s, 6 * s, 24 * s, 14 * s * pl * (0.6 + 0.4 * lw), rng, 0.05, 9, rot=-0.42)
    knee = np.array([2, 18]) * s
    heel = np.array([33, 24]) * s
    shank = limb([knee, np.array([18, 23]) * s, heel], [12 * s * lw, 10 * s * lw, 7 * s * lw])
    ankle = np.array([10, 30]) * s
    tarsus = limb([heel, np.array([22, 29.5]) * s, ankle], [6.5 * s * lw, 5.5 * s * lw, 5 * s * lw])
    toes, tips, knuckles = [], [], []
    for i, (dx, dy) in enumerate([(-19, -3.5), (-25, -1.2), (-30, 0.6), (-25, 2.0), (-17, 3.0)]):
        tip = ankle + np.array([dx * toe, dy]) * s
        mid = ankle + (tip - ankle) * 0.55 + np.array([0, 0.6 * s])
        toes.append(limb([ankle, mid, tip], [3.6 * s * lw, 2.6 * s * lw, 2.0 * s * lw]))
        toes.append(Point(*tip).buffer(1.5 * s * P['pad'] ** 0.8 * max(lw, 0.85), 16))  # pad
        tips.append(tip)
        knuckles.append(ankle + (tip - ankle) * P['web'])
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
    arm = limb([shoulder, elbow, wrist], [9 * s * lw, 6.5 * s * lw, 4.5 * s * lw])
    fingers = []
    for dx, dy in [(-9, -1.2), (-10, 1.0), (-6, 2.6), (2, 2.0)]:
        tip = wrist + np.array([dx, dy]) * s
        fingers.append(limb([wrist, (wrist + tip) / 2, tip], [3.2 * s * lw, 2.3 * s * lw, 1.8 * s * lw]))
        fingers.append(Point(*tip).buffer(1.3 * s * P['pad'] ** 0.8 * max(lw, 0.85), 14))
    hand = unary_union(fingers)

    # far legs, mostly hidden: shifted back and up
    far_hand = affinity.translate(unary_union([arm, hand]), 9 * s, -3 * s)
    far_foot = affinity.translate(foot, -6 * s, -2.5 * s)

    parts = [
        Part("hand", hand, 0.6), Part("arm", arm, 0.8),
        Part("foot", foot, 0.6), Part("shank", shank, 1.0), Part("thigh", thigh, 1.2),
        Part("eye", eye, 0.0), Part("parotoid", parotoid, 0.8), Part("tymp", tymp, 0.0),
        Part("body", body, 1.6),
        Part("far_hand", far_hand, 0.0), Part("far_foot", far_foot, 0.0),
    ]

    out = []
    front = Polygon()
    light = (-1, -1.2)

    def emit(geoms, mask=None):
        for g in geoms if isinstance(geoms, list) else [geoms]:
            g = g if mask is None else g.difference(mask)
            out.extend(lines_of(g))

    parts = [q for q in parts if not q.shape.is_empty]
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
    out += lines_of(Point(*tymp.centroid.coords[0]).buffer(4.2 * s * min(P['eye'], 1.15), 32)
                    .boundary.difference(parotoid))

    # body surface: mouth, dorsolateral fold, spots, warts
    others = unary_union([hand, arm, foot, shank, thigh, eye, tymp, parotoid])
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

    if P["stripe"]:
        # dark band from the snout through the eye and along the flank,
        # edged below by a pale line, as on many tree frogs
        path = LineString(catmull(S([(-50 * sn, -6), (-30 * sn, -19), (-12, -12),
                                     (12, -10 * hp), (38, -6)]), closed=False))
        band = path.buffer(2.0 * s).intersection(skin).difference(eye.buffer(0.5 * s))
        out += lines_of(band.boundary)
        out += hatch(band, 0.6 * s, -35, rng, 0.0, light)

    if P["warts"] > 0.15:
        region = skin if P["wart_all"] else skin.intersection(affinity.translate(body, 0, -6 * s))
        if P["wart_all"]:
            region = region.union(thigh.difference(unary_union([shank, foot])))
            region = region.union(parotoid)
        lo, hi = P["wart_size"]
        limit = 140 if P["wart_all"] else 60
        for p in poisson(region, (5.5 - 2.5 * P["warts"]) * s * (hi / 1.4) ** 0.5, rng, limit=limit):
            r = rng.uniform(lo, hi) * s
            w = Point(*p).buffer(r, 12)
            # a wart is a tiny dome: an arc on its shadow side reads better
            # than a full circle, which looks like a hole
            arc = w.boundary.intersection(crescent(w.buffer(0.01), r * 0.9, light).buffer(0.05))
            out += lines_of(arc)

    # a little ground under the feet: a few short strokes along the line
    # the feet rest on, densest right under them
    feet = unary_union([foot, hand])
    minx, miny, maxx, maxy = feet.bounds
    if P["branch"]:
        # a branch under the feet: two long edges, shadow hatching along the
        # underside and a few bark ticks, clipped behind the feet
        y0 = maxy - 1.2 * s
        th = rng.uniform(7, 10) * s
        x0, x1 = minx - 30 * s, maxx + 45 * s
        top = [(x0, y0 + 1.5 * s), ((x0 + x1) / 2, y0), (x1, y0 + 2.0 * s)]
        bot = [(x1, y0 + 2.0 * s + th), ((x0 + x1) / 2, y0 + th + 0.5 * s), (x0, y0 + 1.5 * s + th)]
        branch = Polygon(catmull(top, closed=False).tolist() + catmull(bot, closed=False).tolist()).buffer(0)
        mask = unary_union([q.shape for q in parts])
        out += lines_of(LineString(catmull(top, closed=False)).difference(mask))
        out += lines_of(LineString(catmull(bot, closed=False)).difference(mask))
        shade = crescent(branch, th * 0.45, light).difference(mask)
        out += hatch(shade, 1.0 * s, 35, rng, 0.5, light)
        for _ in range(int((x1 - x0) / (6 * s))):
            bx = rng.uniform(x0 + 3 * s, x1 - 3 * s)
            by = rng.uniform(y0 + 0.3 * th, y0 + 0.8 * th)
            tick = LineString([(bx, by), (bx + rng.uniform(3, 7) * s, by + rng.uniform(-0.6, 0.6) * s)])
            out += lines_of(tick.intersection(branch).difference(mask))
        return [np.asarray(q, float) for q in out if len(q) >= 2]
    for i in range(6):
        gy = maxy + (0.8 + i * 1.4) * s
        x0 = rng.uniform(minx - 6 * s, maxx - 12 * s)
        L = rng.uniform(10, 30) * s * (1 - i / 8)
        out.append(np.array([[x0, gy], [x0 + L, gy + rng.uniform(-0.4, 0.4) * s]]))

    return [np.asarray(p, float) for p in out if len(p) >= 2]


# ── seen from above ──────────────────────────────────────────────────────

def build_topdown(seed: int = 0):
    """A frog seen from directly above, legs splayed, facing up the page.

    Every part is a different shape from this angle, so it is built
    separately rather than projected from the side view: the body is a
    symmetric egg, the eyes bulge out of the head's outline, and the legs
    fan out sideways instead of folding under.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform
    s = u(0.95, 1.1)
    wd = u(1.15, 1.35)         # body width
    hd = u(0.82, 0.95)         # head length
    lw = u(0.9, 1.1)           # limb thickness
    light = (-1, -1.2)

    def mirror(pts):
        pts = np.asarray(pts, float)
        return np.vstack([pts, pts[::-1][1:-1] * [-1, 1]])

    half = [(0, -46 * hd), (-9, -45 * hd), (-17 * wd, -36 * hd), (-21 * wd, -20),
            (-25 * wd, 0), (-24 * wd, 18), (-17 * wd, 32), (-7, 37), (0, 38)]
    body = Polygon(catmull(mirror(half) * s, n=12)).buffer(0)

    parts = []
    eyes, details = [], []
    for side in (-1, 1):
        j = lambda: u(-1.2, 1.2) * s           # small asymmetry, so it isn't a stamp
        # eyes bulge out past the head outline
        e = Point(side * 15 * wd * s + j(), -30 * hd * s + j()).buffer(7.5 * s, 40)
        eyes.append(e)
        c = np.array(e.centroid.coords[0])
        details.append(affinity.scale(Point(*c).buffer(3.4 * s, 24), 0.62, 1.35))   # pupil
        # front leg: shoulder out to the elbow, forearm forward to a splayed hand
        sh = np.array([side * 18 * wd, -10]) * s
        el = np.array([side * 31 * wd, -2]) * s + [j(), j()]
        wr = np.array([side * 34 * wd, -19]) * s + [j(), j()]
        arm = limb([sh, el, wr], [8 * s * lw, 6 * s * lw, 4.5 * s * lw])
        fingers = []
        for a in np.radians([-150, -115, -80, -50]):
            tip = wr + np.array([side * -math.cos(a) * 9, math.sin(a) * 9]) * s
            fingers.append(limb([wr, tip], [2.8 * s * lw, 1.8 * s * lw]))
            fingers.append(Point(*tip).buffer(1.4 * s, 12))
        hand = unary_union(fingers)
        # hind leg: thigh out and back, shank folded forward, long foot
        # fanning back and out with webbed toes
        hip = np.array([side * 15 * wd, 30]) * s
        knee = np.array([side * 40 * wd, 26]) * s + [j(), j()]
        heel = np.array([side * 30 * wd, 46]) * s + [j(), j()]
        thigh = limb([hip, (hip + knee) / 2 + [0, 3 * s], knee], [17 * s * lw, 14 * s * lw, 9 * s * lw])
        shank = limb([knee, heel], [9 * s * lw, 6 * s * lw])
        ankle = heel + np.array([side * 9, 8]) * s
        tarsus = limb([heel, ankle], [6 * s * lw, 5 * s * lw])
        toes, knuckles = [], []
        for a in np.radians([10, 35, 58, 80, 100]):
            ln = (16 + 7 * math.sin(math.radians(58) - abs(a - math.radians(58)))) * s
            tip = ankle + np.array([side * math.cos(a), math.sin(a)]) * ln
            toes.append(limb([ankle, (ankle + tip) / 2, tip], [3.4 * s * lw, 2.4 * s * lw, 1.8 * s * lw]))
            toes.append(Point(*tip).buffer(1.5 * s, 12))
            knuckles.append(ankle + (tip - ankle) * 0.8)
        rim = [knuckles[0]]
        for a, b in zip(knuckles[:-1], knuckles[1:]):
            rim += [(a + b) / 2 + (ankle - (a + b) / 2) * 0.25, b]
        web = Polygon(np.vstack([ankle, catmull(rim, closed=False, n=5), ankle])).buffer(0)
        foot = unary_union(toes + [web, tarsus])
        parts += [("hand", hand, 0.6), ("arm", arm, 0.8), ("thigh", thigh, 1.1),
                  ("shank", shank, 0.9), ("foot", foot, 0.6)]

    order = ([("eye", e, 0.6) for e in eyes] + [("body", body, 1.4)]
             + [q for q in parts if q[0] in ("thigh",)]
             + [q for q in parts if q[0] in ("shank", "foot", "arm", "hand")])

    out, front = [], Polygon()
    for name, shape, shade in order:
        visible = shape.difference(front)
        out += lines_of(shape.boundary.difference(front))
        if shade > 0:
            band = crescent(shape, 4.0 * s * shade, light).intersection(visible)
            out += hatch(band, 1.1 * s, 35, rng, 0.5, light)
        front = front.union(shape)

    for e, pupil in zip(eyes, details):
        out += hatch(pupil, 0.45 * s, 80, rng, 0.0, light)
        out += lines_of(pupil.boundary)
        out += lines_of(e.buffer(-1.8 * s).boundary)

    back = body.difference(unary_union(eyes).buffer(2 * s))
    # two dorsolateral ridges and a nostril pair
    for side in (-1, 1):
        ridge = LineString(catmull(np.array([(side * 11, -24), (side * 13 * wd, 0),
                                             (side * 12 * wd, 24), (side * 8, 36)]) * s, closed=False))
        out += lines_of(ridge.intersection(back))
        out += lines_of(Point(side * 4 * s, -41 * hd * s).buffer(0.8 * s, 8).boundary)

    for _ in range(int(rng.integers(3, 10))):
        for _ in range(20):
            minx, miny, maxx, maxy = back.bounds
            q = np.array([u(minx, maxx), u(miny * 0.6, maxy)])
            if back.contains(Point(q)):
                break
        spot = blob(*q, u(3, 6.5) * s, u(2.5, 5) * s, rng, 0.25, rot=u(0, 3)).intersection(back)
        out += lines_of(spot.boundary)
        out += hatch(spot, 0.8 * s, -40, rng, 0.0, light)

    for q in poisson(back.buffer(-2 * s), u(3.5, 5.5) * s, rng, limit=70):
        r = u(0.6, 1.3) * s
        w = Point(*q).buffer(r, 12)
        out += lines_of(w.boundary.intersection(crescent(w.buffer(0.01), r * 0.9, light).buffer(0.05)))

    return [np.asarray(q, float) for q in out if len(q) >= 2]
