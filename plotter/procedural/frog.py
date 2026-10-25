"""Procedural frogs seen from above, in the manner of LingDong's fishdraw.

fishdraw's richness comes from structure, not from any one clever shape:
every part of the animal has several anatomical variants, each chosen
independently, so the combinations multiply.  The same is done here, with
one addition for faithfulness: a frog first draws a real *family*, and the
family sets the distribution every other choice is drawn from.  Mixing
freely produces animals that never existed - a toad's body with a tree
frog's toe discs and a dart frog's colouring - while sampling within a
family keeps each result plausible and still gives endless variety.

Coordinates are millimetres, x right, y down, snout pointing up the page.
Only the left half is designed; the right is its mirror with independent
jitter, so the animal is symmetric the way a real one is and not the way a
stamp is.
"""

from __future__ import annotations

import math

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

from .geom import (blob, catmull, crescent, cut, hatch, join, limb, lines_of, merge,
                   minus, poisson)

LIGHT = (-1.0, -1.25)

FAMILIES = ("truefrog", "toad", "treefrog", "dartfrog", "hornedfrog", "clawedfrog")
WEIGHTS = (0.22, 0.2, 0.2, 0.15, 0.12, 0.11)


# ── parameters ───────────────────────────────────────────────────────────

def params(rng, family=None):
    """Every choice for one frog, sampled within ``family``'s ranges."""
    if family is None:
        family = FAMILIES[int(rng.choice(len(FAMILIES), p=WEIGHTS))]
    u = lambda a, b: float(rng.uniform(a, b))
    pick = lambda opts, p=None: opts[int(rng.choice(len(opts), p=p))]
    coin = lambda p: bool(rng.random() < p)
    fills = ["hatch", "cross", "dots"]

    if family == "truefrog":           # Ranidae
        P = dict(
            shape=dict(snout=u(.05, .08), head=u(.2, .24), neck=u(.18, .21),
                       shoulder=u(.21, .25), waist=u(.16, .2), hip=u(.19, .23), point=u(.55, .9)),
            eye=dict(r=u(.075, .09), y=u(.2, .25), out=u(.9, 1.05), pupil="h", horn=False),
            limbs=dict(hind=u(1.0, 1.2), front=u(.9, 1.05), thick=u(.9, 1.05),
                       pose=pick(["splay", "rest"], [.6, .4])),
            digits=dict(length=u(1.0, 1.2), tip="point", disc=0.0,
                        web_hind=u(.65, .9), web_front=0.0, claws=False),
            skin=dict(warts=u(0, .35), wart_size=(.5, 1.0), granules=0.0, ridges=2,
                      crests=False, parotoid=False, stitches=False),
            pattern=dict(spots=int(rng.integers(0, 12)), spot_size=(.025, .05), blotches=0,
                         mid_stripe=coin(.25), lat_bands=False, stripes=0,
                         leg_bands=coin(.75), reticulate=False, chevron=coin(.25),
                         bar=coin(.4), pale_border=False, fill=pick(fills, [.6, .2, .2])))
    elif family == "toad":             # Bufonidae
        P = dict(
            shape=dict(snout=u(.06, .09), head=u(.22, .25), neck=u(.23, .26),
                       shoulder=u(.27, .3), waist=u(.26, .29), hip=u(.21, .24), point=u(.1, .3)),
            eye=dict(r=u(.06, .07), y=u(.22, .27), out=u(.62, .75), pupil="h", horn=False),
            limbs=dict(hind=u(.95, 1.05), front=u(.95, 1.05), thick=u(1.0, 1.12),
                       pose=pick(["rest", "splay"], [.6, .4])),
            digits=dict(length=u(.75, .9), tip="round", disc=0.0,
                        web_hind=u(.25, .45), web_front=0.0, claws=False),
            skin=dict(warts=u(.7, 1.0), wart_size=(.7, 1.9), granules=u(0, .35), ridges=0,
                      crests=coin(.75), parotoid=True, stitches=False),
            pattern=dict(spots=0, spot_size=(.03, .06), blotches=int(rng.integers(3, 9)),
                         mid_stripe=coin(.35), lat_bands=False, stripes=0,
                         leg_bands=coin(.45), reticulate=False, chevron=False,
                         bar=False, pale_border=coin(.3), fill=pick(fills, [.4, .2, .4])))
    elif family == "treefrog":         # Hylidae
        P = dict(
            shape=dict(snout=u(.09, .12), head=u(.25, .28), neck=u(.23, .25),
                       shoulder=u(.25, .28), waist=u(.2, .23), hip=u(.19, .22), point=u(0, .3)),
            eye=dict(r=u(.095, .115), y=u(.21, .25), out=u(.95, 1.08),
                     pupil=pick(["h", "v"], [.6, .4]), horn=False),
            limbs=dict(hind=u(1.05, 1.2), front=u(1.0, 1.12), thick=u(.84, .95),
                       pose=pick(["splay", "rest"], [.5, .5])),
            digits=dict(length=u(1.0, 1.2), tip="disc", disc=u(1.4, 2.0),
                        web_hind=u(.3, .55), web_front=u(0, .3), claws=False),
            skin=dict(warts=0.0, wart_size=(.4, .8), granules=0.0, ridges=0,
                      crests=False, parotoid=False, stitches=False),
            pattern=dict(spots=pick([0, int(rng.integers(3, 12))], [.5, .5]), spot_size=(.015, .035),
                         blotches=0, mid_stripe=False, lat_bands=coin(.55), stripes=0,
                         leg_bands=coin(.3), reticulate=False, chevron=False,
                         bar=False, pale_border=False, fill=pick(fills, [.5, .2, .3])))
    elif family == "dartfrog":         # Dendrobatidae
        style = pick(["stripes", "reticulate", "blotches", "spots"])
        P = dict(
            shape=dict(snout=u(.08, .11), head=u(.2, .23), neck=u(.18, .2),
                       shoulder=u(.2, .23), waist=u(.16, .19), hip=u(.17, .2), point=u(.2, .5)),
            eye=dict(r=u(.08, .095), y=u(.2, .24), out=u(.9, 1.0), pupil="dark", horn=False),
            limbs=dict(hind=u(.95, 1.1), front=u(.95, 1.1), thick=u(.75, .9),
                       pose=pick(["rest", "splay"], [.6, .4])),
            digits=dict(length=u(.9, 1.05), tip="disc", disc=u(1.1, 1.4),
                        web_hind=0.0, web_front=0.0, claws=False),
            skin=dict(warts=0.0, wart_size=(.4, .8), granules=0.0, ridges=0,
                      crests=False, parotoid=False, stitches=False),
            pattern=dict(spots=int(rng.integers(6, 18)) if style == "spots" else 0,
                         spot_size=(.02, .045), blotches=int(rng.integers(3, 7)) if style == "blotches" else 0,
                         mid_stripe=False, lat_bands=style == "stripes",
                         stripes=int(rng.integers(1, 3)) if style == "stripes" else 0,
                         leg_bands=coin(.5), reticulate=style == "reticulate", chevron=False,
                         bar=False, pale_border=False, fill=pick(fills, [.4, .4, .2])))
    elif family == "hornedfrog":       # Ceratophryidae
        P = dict(
            shape=dict(snout=u(.1, .13), head=u(.3, .33), neck=u(.31, .33),
                       shoulder=u(.3, .33), waist=u(.27, .3), hip=u(.21, .24), point=u(0, .15)),
            eye=dict(r=u(.065, .08), y=u(.25, .29), out=u(.62, .72), pupil="h", horn=True),
            limbs=dict(hind=u(.9, 1.0), front=u(.95, 1.05), thick=u(1.0, 1.1), pose="rest"),
            digits=dict(length=u(.75, .85), tip="round", disc=0.0,
                        web_hind=u(.2, .4), web_front=0.0, claws=False),
            skin=dict(warts=u(.5, .9), wart_size=(.6, 1.4), granules=u(0, .3), ridges=0,
                      crests=False, parotoid=False, stitches=False),
            pattern=dict(spots=0, spot_size=(.03, .06), blotches=int(rng.integers(5, 11)),
                         mid_stripe=coin(.3), lat_bands=False, stripes=0,
                         leg_bands=coin(.7), reticulate=False, chevron=False,
                         bar=coin(.5), pale_border=True, fill=pick(fills, [.5, .3, .2])))
    elif family == "clawedfrog":       # Pipidae
        P = dict(
            shape=dict(snout=u(.07, .1), head=u(.17, .2), neck=u(.19, .22),
                       shoulder=u(.24, .27), waist=u(.27, .3), hip=u(.26, .29), point=u(.3, .6)),
            eye=dict(r=u(.04, .05), y=u(.17, .21), out=u(.42, .55), pupil="round", horn=False),
            limbs=dict(hind=u(1.15, 1.3), front=u(.75, .9), thick=u(1.0, 1.15), pose="splay"),
            digits=dict(length=u(1.35, 1.6), tip="point", disc=0.0,
                        web_hind=u(.92, 1.0), web_front=0.0, claws=True),
            skin=dict(warts=0.0, wart_size=(.4, .8), granules=0.0, ridges=0,
                      crests=False, parotoid=False, stitches=True),
            pattern=dict(spots=pick([0, int(rng.integers(8, 20))], [.4, .6]), spot_size=(.015, .03),
                         blotches=0, mid_stripe=False, lat_bands=False, stripes=0,
                         leg_bands=False, reticulate=False, chevron=False,
                         bar=False, pale_border=False, fill=pick(fills, [.4, .2, .4])))
    else:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")

    P["family"] = family
    P["L"] = u(92, 108)
    P["fill_angle"] = u(-55, -25)
    return P


# ── shapes ───────────────────────────────────────────────────────────────

def _y(L, t):
    """Position along the body: t=0 snout tip, t=1 vent."""
    return (t - 0.5) * L


def body_outline(P):
    L, sh = P["L"], P["shape"]
    pt = sh["point"]
    left = [(-sh["snout"] * L * (1 - .7 * pt), _y(L, .015)),
            (-sh["snout"] * L, _y(L, .05)),
            (-sh["head"] * L * .82, _y(L, .13)),
            (-sh["head"] * L, _y(L, .25)),
            (-sh["neck"] * L, _y(L, .36)),
            (-sh["shoulder"] * L, _y(L, .48)),
            (-sh["waist"] * L, _y(L, .67)),
            (-sh["hip"] * L, _y(L, .85)),
            (-sh["hip"] * L * .55, _y(L, .96))]
    tip = [(0.0, _y(L, -.02 * pt))]
    vent = [(0.0, _y(L, 1.0))]
    right = [(-x, y) for x, y in reversed(left)]
    return Polygon(catmull(tip + left + vent + right, n=12)).buffer(0)


def digit_set(base, ang, lengths, spread, w, D, rng, web, scale):
    """Fingers or toes fanned from ``base`` around direction ``ang``.

    ``lengths`` runs from the first angle in the fan to the last; webbing,
    when present, is a membrane with a scalloped edge sagging toward the
    base between each pair of digits.
    """
    n = len(lengths)
    angs = ang + np.linspace(-spread / 2, spread / 2, n)
    shapes, knuckles, tips = [], [], []
    for a, ln in zip(angs, lengths):
        d = np.array([math.cos(a), math.sin(a)])
        perp = np.array([-d[1], d[0]])
        tip = base + d * ln
        mid = base + d * ln * .55 + perp * rng.uniform(-.04, .04) * ln
        shapes.append(limb([base, mid, tip], [w, w * .78, w * .6]))
        if D["tip"] == "disc":
            shapes.append(Point(*tip).buffer(w * .5 * D["disc"], 18))
        elif D["tip"] == "round":
            shapes.append(Point(*tip).buffer(w * .42, 14))
        tips.append((tip, d))
        knuckles.append(base + (tip - base) * web)
    web_poly = Polygon()
    if web > .08:
        rim = [knuckles[0]]
        for a, b in zip(knuckles[:-1], knuckles[1:]):
            rim += [(a + b) / 2 + (base - (a + b) / 2) * .28, b]
        web_poly = Polygon(np.vstack([base, catmull(rim, closed=False, n=5), base])).buffer(0)
    claws = []
    if D["claws"]:
        # Xenopus carries small black claws on its three inner toes
        for tip, d in tips[-3:]:
            claws.append(LineString([tip - d * w * .2, tip + d * w * .9]))
    return merge(shapes + [web_poly]), claws


def side_limbs(P, rng, sign):
    """Front and hind leg for one side; sign=-1 builds the left side."""
    L, sh, lm, D = P["L"], P["shape"], P["limbs"], P["digits"]
    j = lambda k=.012: rng.uniform(-k, k) * L          # per-side jitter
    th = lm["thick"]

    if lm["pose"] == "splay":
        S = np.array([-sh["shoulder"] * L * .9, _y(L, .43)])
        E = S + np.array([-.16 * L * lm["front"] + j(), .03 * L + j()])
        W = E + np.array([-.03 * L + j(), -.15 * L * lm["front"]])
        hand_ang = math.radians(-118)
        H = np.array([-sh["hip"] * L * .75, _y(L, .84)])
        K = H + np.array([-.33 * L * lm["hind"] + j(), -.04 * L + j()])
        Hl = K + np.array([.07 * L + j(), .2 * L * lm["hind"]])
        A = Hl + np.array([-.07, .08]) * L * lm["hind"]
        foot_ang = math.radians(118)
        toe_len = np.array([.45, .6, .8, 1.0, .72])        # inner -> outer
    else:
        S = np.array([-sh["shoulder"] * L * .92, _y(L, .44)])
        E = S + np.array([-.07 * L + j(), .07 * L + j()])
        W = E + np.array([-.025 * L + j(), -.14 * L * lm["front"]])
        hand_ang = math.radians(-102)
        H = np.array([-sh["hip"] * L * .8, _y(L, .82)])
        K = H + np.array([-.19 * L * lm["hind"] + j(), -.17 * L * lm["hind"] + j()])
        Hl = K + np.array([-.035 * L + j(), .31 * L * lm["hind"]])
        A = Hl + np.array([-.11 * L, -.07 * L]) * lm["hind"]
        foot_ang = math.radians(-155)
        toe_len = np.array([.72, 1.0, .8, .6, .45])        # outer -> inner

    arm = limb([S, E, W], [.085 * L * th, .062 * L * th, .046 * L * th])
    fingers = np.array([.75, 1.0, .7, .55]) * .09 * L * D["length"]   # outer -> inner
    hand, hclaws = digit_set(W, hand_ang, fingers, math.radians(rng.uniform(60, 80)),
                             .03 * L * th, D, rng, D["web_front"], L)
    thigh = limb([H, (H + K) / 2 + np.array([0, .02 * L]), K],
                 [.17 * L * th, .14 * L * th, .09 * L * th])
    shank = limb([K, Hl], [.09 * L * th, .062 * L * th])
    tarsus = limb([Hl, A], [.06 * L * th, .05 * L * th])
    toes, tclaws = digit_set(A, foot_ang, toe_len * .2 * L * D["length"],
                             math.radians(rng.uniform(75, 95)), .034 * L * th, D, rng,
                             D["web_hind"], L)
    foot = merge([tarsus, toes])

    parts = dict(arm=arm, hand=hand, thigh=thigh, shank=shank, foot=foot)
    spines = dict(arm=[S, E, W], thigh=[H, K], shank=[K, Hl])
    if sign > 0:   # mirror to the right side
        flip = lambda g: affinity.scale(g, -1, 1, origin=(0, 0))
        parts = {k: flip(v) for k, v in parts.items()}
        spines = {k: [np.array([-p[0], p[1]]) for p in v] for k, v in spines.items()}
        hclaws = [flip(c) for c in hclaws]
        tclaws = [flip(c) for c in tclaws]
    return parts, spines, hclaws + tclaws


def eye_shapes(P, rng, sign):
    L, sh, E = P["L"], P["shape"], P["eye"]
    r = E["r"] * L
    cx = -sh["head"] * L * E["out"] * .92 + rng.uniform(-.006, .006) * L
    cy = _y(L, E["y"]) + rng.uniform(-.006, .006) * L
    eye = affinity.scale(Point(cx, cy).buffer(r, 40), 1.0, 1.12)
    iris = affinity.scale(Point(cx - r * .05, cy).buffer(r * .74, 40), 1.0, 1.12)
    kind = E["pupil"]
    if kind == "h":            # horizontal slit: wide, not tall
        pupil = affinity.scale(Point(cx - r * .08, cy).buffer(r * .34, 24), 1.3, .55)
    elif kind == "v":          # vertical slit
        pupil = affinity.scale(Point(cx - r * .08, cy).buffer(r * .34, 24), .5, 1.25)
    elif kind == "dark":
        pupil = iris.buffer(-r * .05)
    else:
        pupil = Point(cx - r * .05, cy).buffer(r * .32, 20)
    glint = Point(cx - r * .32, cy - r * .3).buffer(r * .14, 12)
    lid = LineString(catmull([(cx - r * .9, cy - r * .7), (cx - r * .1, cy - r * 1.18),
                              (cx + r * .75, cy - r * .9)], closed=False))
    horn = Polygon()
    if E["horn"]:
        # Ceratophrys: a soft triangular projection of the upper eyelid
        base_a = np.array([cx - r * .55, cy - r * .55])
        base_b = np.array([cx + r * .5, cy - r * .85])
        tip = np.array([cx - r * .5, cy - r * 2.0]) + rng.uniform(-.2, .2, 2) * r
        horn = Polygon(catmull([base_a, (base_a + tip) / 2 + [-r * .15, 0], tip,
                                (base_b + tip) / 2 + [r * .2, 0], base_b], closed=True, n=6)).buffer(0)
    out = dict(eye=eye, iris=iris, pupil=pupil, glint=glint, lid=lid, horn=horn)
    if sign > 0:
        out = {k: affinity.scale(v, -1, 1, origin=(0, 0)) for k, v in out.items()}
    return out


# ── surface pattern and texture ──────────────────────────────────────────

def fill(region, P, rng, s):
    """Dark pattern fill in the frog's chosen style."""
    if region.is_empty:
        return []
    a = P["fill_angle"]
    style = P["pattern"]["fill"]
    if style == "dots":
        out = []
        for q in poisson(region, .95 * s, rng, limit=1200):
            out += lines_of(Point(*q).buffer(.28 * s, 6).boundary)
        return out
    out = hatch(region, .62 * s, a, rng, 0.0, LIGHT)
    if style == "cross":
        out += hatch(region, .9 * s, a + 75, rng, 0.0, LIGHT)
    return out


def bands_across(spine, shape, n, rng, L):
    """Transverse bands across a limb, perpendicular to its spine."""
    a, b = np.asarray(spine[0], float), np.asarray(spine[-1], float)
    d = b - a
    ln = np.hypot(*d)
    if ln < 1e-6:
        return []
    d /= ln
    perp = np.array([-d[1], d[0]])
    out = []
    for t in np.linspace(.18, .85, n):
        c = a + d * ln * t
        w = rng.uniform(.035, .06) * L
        quad = Polygon([c - d * w / 2 - perp * L * .2, c + d * w / 2 - perp * L * .2,
                        c + d * w / 2 + perp * L * .2, c - d * w / 2 + perp * L * .2])
        band = cut(quad, shape)
        if not band.is_empty:
            out.append(band)
    return out


def reticulate(region, rng, L):
    """Dendrobates-style network: dark cells separated by pale lines."""
    from shapely.ops import voronoi_diagram
    from shapely.geometry import MultiPoint
    pts = poisson(region, .07 * L, rng, limit=40)
    if len(pts) < 3:
        return []
    cells = voronoi_diagram(MultiPoint([tuple(p) for p in pts]), envelope=region.envelope.buffer(L))
    out = []
    for c in cells.geoms:
        g = cut(c.buffer(-.014 * L), region)
        if not g.is_empty and g.area > (.02 * L) ** 2:
            out.append(g.buffer(.006 * L).buffer(-.006 * L))
    return out


def warts_on(region, P, rng, s, density):
    out = []
    lo, hi = P["skin"]["wart_size"]
    gap = (6.0 - 3.0 * density) * s * (hi / 1.2) ** .5
    for q in poisson(region, gap, rng, limit=180):
        r = rng.uniform(lo, hi) * s
        w = Point(*q).buffer(r, 12)
        # a wart is a dome: the arc on its shadow side reads as a bump,
        # where a full circle reads as a hole punched in the skin
        arc = cut(w.boundary, crescent(w.buffer(.01), r * .9, LIGHT).buffer(.05))
        out += lines_of(arc)
    return out


# ── the frog ─────────────────────────────────────────────────────────────

def build(seed: int = 0, family: str | None = None):
    """Polylines (mm, y down) for one frog seen from above."""
    rng = np.random.default_rng(seed)
    P = params(rng, family)
    L = P["L"]
    s = L / 100.0
    pat, skin = P["pattern"], P["skin"]

    body = body_outline(P)
    eyes = [eye_shapes(P, rng, sg) for sg in (-1, 1)]
    sides = [side_limbs(P, rng, sg) for sg in (-1, 1)]

    extras = []
    if skin["parotoid"]:
        for sg in (-1, 1):
            g = blob(sg * -P["shape"]["head"] * L * .62, _y(L, P["eye"]["y"] + .14),
                     .055 * L, .1 * L, rng, .06, 9, rot=sg * -.25)
            extras.append(g)

    # draw order, front to back: eyes on top, then body, then limbs
    order = []
    for e in eyes:
        order += [("eye", e["eye"], .5), ("horn", e["horn"], .6)]
    order += [("parotoid", g, .7) for g in extras]
    order.append(("body", body, 1.3))
    for parts, _, _ in sides:
        order += [("hand", parts["hand"], .5), ("arm", parts["arm"], .8)]
    for parts, _, _ in sides:
        order += [("thigh", parts["thigh"], 1.0), ("shank", parts["shank"], .8),
                  ("foot", parts["foot"], .5)]
    order = [o for o in order if not o[1].is_empty]

    out = []
    front = Polygon()
    visible = {}
    for name, shape, shade in order:
        vis = minus(shape, front)
        visible.setdefault(name, []).append(vis)
        out += lines_of(minus(shape.boundary, front))
        if shade > 0:
            band = cut(crescent(shape, .055 * L * shade, LIGHT), vis)
            out += hatch(band, 1.1 * s, 35, rng, .5, LIGHT)
        front = join(front, shape)
    silhouette = front
    vis = {k: merge(v) for k, v in visible.items()}

    # eye detail
    for e in eyes:
        out += lines_of(e["iris"].boundary)
        dark = minus(e["pupil"], e["glint"])
        out += hatch(dark, .42 * s, 80, rng, 0.0, LIGHT)
        out += lines_of(dark.boundary)
        out += lines_of(cut(minus(e["lid"], e["eye"].buffer(-.4 * s)), e["eye"].buffer(1.2 * s)))

    eyes_all = merge([e["eye"] for e in eyes] + [e["horn"] for e in eyes])
    back = minus(vis.get("body", Polygon()), eyes_all.buffer(1.5 * s))
    for g in extras:
        back = minus(back, g.buffer(.8 * s))

    # nostrils and, for toads, cranial crests
    for sg in (-1, 1):
        out += lines_of(Point(sg * .045 * L, _y(L, .07)).buffer(.7 * s, 8).boundary)
        if skin["crests"]:
            ex = -sg * P["shape"]["head"] * L * P["eye"]["out"] * .92
            cx = ex + sg * P["eye"]["r"] * L * 1.25
            crest = LineString(catmull([(cx * .9, _y(L, P["eye"]["y"] - .09)),
                                        (cx, _y(L, P["eye"]["y"])),
                                        (cx * 1.15, _y(L, P["eye"]["y"] + .1))], closed=False))
            out += lines_of(cut(crest, back))

    # dorsolateral folds
    if skin["ridges"]:
        for sg in (-1, 1):
            for k in range(skin["ridges"] // 2):
                x0 = sg * (.09 + .05 * k) * L
                ridge = LineString(catmull([(x0 * .9, _y(L, .3)), (x0 * 1.15, _y(L, .55)),
                                            (x0 * 1.05, _y(L, .8)), (x0 * .7, _y(L, .93))],
                                           closed=False))
                out += lines_of(cut(ridge, back))

    # pattern: the dark marks
    marks = []
    if pat["mid_stripe"]:
        stripe = LineString([(0, _y(L, .06)), (0, _y(L, .97))]).buffer(.012 * L)
        out += lines_of(cut(stripe, back).boundary)
    if pat["bar"]:
        e0 = np.array(eyes[0]["eye"].centroid.coords[0])
        e1 = np.array(eyes[1]["eye"].centroid.coords[0])
        marks.append(cut(LineString([e0, e1]).buffer(.018 * L), back))
    if pat["chevron"]:
        w = LineString([(-.11 * L, _y(L, .37)), (-.04 * L, _y(L, .45)), (0, _y(L, .4)),
                        (.04 * L, _y(L, .45)), (.11 * L, _y(L, .37))])
        marks.append(cut(w.buffer(.014 * L), back))
    if pat["lat_bands"]:
        for sg in (-1, 1):
            band = LineString(catmull([(sg * .05 * L, _y(L, .02)),
                                       (sg * P["shape"]["head"] * L * .62, _y(L, .2)),
                                       (sg * P["shape"]["shoulder"] * L * .72, _y(L, .5)),
                                       (sg * P["shape"]["hip"] * L * .7, _y(L, .9))], closed=False))
            marks.append(cut(band.buffer(.028 * L), back))
    for k in range(pat["stripes"]):
        x0 = 0 if pat["stripes"] == 1 else (k - .5) * .1 * L
        marks.append(cut(LineString([(x0, _y(L, .12)), (x0, _y(L, .95))]).buffer(.022 * L), back))
    for _ in range(pat["spots"]):
        q = _point_in(back, rng)
        if q is not None:
            lo, hi = pat["spot_size"]
            marks.append(cut(blob(*q, rng.uniform(lo, hi) * L, rng.uniform(lo, hi) * L, rng, .25,
                              rot=rng.uniform(0, 3)), back))
    for _ in range(pat["blotches"]):
        q = _point_in(back, rng)
        if q is not None:
            marks.append(cut(blob(*q, rng.uniform(.05, .09) * L, rng.uniform(.035, .07) * L, rng, .3,
                              k=11, rot=rng.uniform(0, 3)), back))
    if pat["reticulate"]:
        marks += reticulate(back, rng, L)
    if pat["leg_bands"]:
        for parts, spines, _ in sides:
            for limb_name in ("thigh", "shank", "arm"):
                region = cut(vis.get(limb_name, Polygon()), parts[limb_name])
                n = 3 if limb_name == "thigh" else 2
                marks += bands_across(spines[limb_name], region, n, rng, L)

    marks = [m for m in marks if not m.is_empty and m.area > (.4 * s) ** 2]
    # Overlapping marks become one mark: two blotches drawn separately leave
    # a stray outline running through the middle of the dark patch.
    if marks:
        merged = merge(marks)
        marks = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    for m in marks:
        out += lines_of(m.boundary)
        out += fill(m, P, rng, s)
        if pat["pale_border"]:
            out += lines_of(cut(m.buffer(.9 * s).boundary, back))
    marked = merge(marks) if marks else Polygon()

    # skin texture
    plain = minus(back, marked.buffer(.5 * s))
    if skin["warts"] > 0:
        region = plain
        if P["family"] in ("toad", "hornedfrog"):
            for g in extras:
                region = join(region, g.buffer(-1 * s))
            region = join(region, minus(vis.get("thigh", Polygon()), marked))
        out += warts_on(region, P, rng, s, skin["warts"])
    if skin["granules"] > 0:
        for q in poisson(plain, (2.6 - 1.2 * skin["granules"]) * s, rng, limit=500):
            out += lines_of(Point(*q).buffer(.25 * s, 5).boundary)
    if skin["stitches"]:
        # Xenopus lateral-line organs: rows of short dashes along the flanks
        for sg in (-1, 1):
            for k, f in enumerate((.6, .8)):
                path = LineString(catmull([(sg * P["shape"]["head"] * L * .5 * f, _y(L, .15)),
                                           (sg * P["shape"]["shoulder"] * L * f, _y(L, .45)),
                                           (sg * P["shape"]["waist"] * L * f, _y(L, .7)),
                                           (sg * P["shape"]["hip"] * L * f * .9, _y(L, .9))],
                                          closed=False))
                n = int(path.length / (3.0 * s))
                for i in range(n):
                    a = path.interpolate(i * 3.0 * s)
                    b = path.interpolate(i * 3.0 * s + 1.4 * s)
                    seg = cut(LineString([a, b]), back)
                    out += lines_of(seg)

    for _, _, claws in sides:
        for c in claws:
            out += lines_of(c.buffer(.25 * s).boundary)

    # soft drop shadow down-right of the animal, a few faint strokes
    off = affinity.translate(silhouette, .025 * L, .035 * L)
    shadow = minus(off, silhouette.buffer(.6 * s))
    out += hatch(shadow, 1.5 * s, 35, rng, .65, LIGHT)

    return [np.asarray(q, float) for q in out if len(q) >= 2]


def _point_in(region, rng, tries=40):
    if region.is_empty:
        return None
    minx, miny, maxx, maxy = region.bounds
    for _ in range(tries):
        q = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)])
        if region.contains(Point(q)):
            return q
    return None
