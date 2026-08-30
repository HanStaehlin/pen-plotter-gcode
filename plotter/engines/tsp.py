"""TSP art: one unbroken line through the whole picture.

Points are distributed so their *density* matches image darkness (weighted
Lloyd relaxation, i.e. Secord's stippling), then joined into a single tour.
The result is literally one stroke - one pen down at the start, one pen up at
the end - and tone survives because a denser cloud of points forces the tour
to double back more often per unit area.

The tour matters as much as the points.  Nearest-neighbour alone leaves long
panic-stricken jumps across the page that read as obvious errors, so it is
built by greedy edge matching and then relaxed with Or-opt and 2-opt over
near neighbours.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:                                   # pragma: no cover
    cKDTree = None

from ..geometry import InkBuffer


def generate(value, coverage, canvas, pen, cfg, features=None, form=None, image_rgb=None):
    h, w = coverage.shape
    px_per_mm = canvas.px_per_mm
    pen_px = max(0.4, pen.width_mm * px_per_mm)

    n_points = int(np.clip(cfg.tsp_points, 100, 60000))
    pts = _stipple(coverage, n_points, int(cfg.tsp_relax))
    if len(pts) < 4:
        return [], InkBuffer(coverage, pen_px, features=features)

    tour = _refine(pts, _greedy_tour(pts))

    path = pts[tour].astype(np.float32)
    paths = _cut_long_jumps(path)

    ink = InkBuffer(coverage, pen_px, features=features)
    for p in paths:
        ink.deposit(p)
    return paths, ink


def _cut_long_jumps(path: np.ndarray, factor: float = 6.0) -> list[np.ndarray]:
    """Break the tour wherever it had to make a jump that would show.

    Refinement gets the tour down to a handful of over-long edges, but on a
    single-stroke drawing each one is a line ruled straight across the
    picture and is the first thing the eye finds.  Cutting them trades about
    ten pen lifts - nothing against the tens of thousands this engine already
    avoids - for a drawing with no stray marks in it.
    """
    seg = np.hypot(*np.diff(path, axis=0).T)
    if len(seg) < 8:
        return [path]
    limit = factor * float(np.median(seg))
    cuts = np.nonzero(seg > limit)[0]
    if cuts.size == 0:
        return [path]
    pieces = np.split(path, cuts + 1)
    return [q for q in pieces if len(q) >= 2]


# ── point distribution ───────────────────────────────────────────────────

def _stipple(coverage: np.ndarray, n: int, iterations: int) -> np.ndarray:
    """Weighted Lloyd relaxation: even spacing, density set by darkness."""
    h, w = coverage.shape
    rng = np.random.default_rng(1234)

    density = np.clip(coverage.astype(np.float64), 0.0, None)
    if density.sum() <= 0:
        return np.empty((0, 2))

    # Rejection-sample a starting cloud, then let relaxation even it out.
    flat = density.ravel() / density.sum()
    pick = rng.choice(flat.size, size=n, p=flat)
    pts = np.stack([(pick % w) + rng.random(n), (pick // w) + rng.random(n)],
                   axis=1)

    if cKDTree is None or iterations <= 0:
        return pts

    # Relax against a subsampled pixel grid; full resolution buys nothing
    # once there are far more pixels than points.
    stride = max(1, int(np.sqrt(density.size / (n * 24.0))))
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    gx = xs.ravel().astype(np.float64) + 0.5
    gy = ys.ravel().astype(np.float64) + 0.5
    gw = density[ys.ravel(), xs.ravel()]
    keep = gw > 1e-6
    gx, gy, gw = gx[keep], gy[keep], gw[keep]
    if gx.size == 0:
        return pts

    for _ in range(iterations):
        _, idx = cKDTree(pts).query(np.stack([gx, gy], axis=1), workers=-1)
        mass = np.bincount(idx, weights=gw, minlength=len(pts))
        sx = np.bincount(idx, weights=gw * gx, minlength=len(pts))
        sy = np.bincount(idx, weights=gw * gy, minlength=len(pts))
        live = mass > 1e-9
        pts[live, 0] = sx[live] / mass[live]
        pts[live, 1] = sy[live] / mass[live]
        # Cells that captured nothing are dead weight; re-seed them where the
        # image still wants ink instead of leaving them stranded.
        dead = int((~live).sum())
        if dead:
            resample = rng.choice(flat.size, size=dead, p=flat)
            pts[~live, 0] = (resample % w) + rng.random(dead)
            pts[~live, 1] = (resample // w) + rng.random(dead)

    return pts


# ── tour construction ────────────────────────────────────────────────────

def _greedy_tour(pts: np.ndarray, k: int = 14) -> np.ndarray:
    """Greedy edge matching over k-nearest candidates, then chain the pieces.

    Greedy edge beats nearest-neighbour construction by roughly 10% on tour
    length and, more visibly, it does not produce the long recovery jumps
    that make a nearest-neighbour tour look broken.
    """
    n = len(pts)
    if cKDTree is None:
        return _nn_tour(pts)

    tree = cKDTree(pts)
    k = min(k + 1, n)
    dist, nbr = tree.query(pts, k=k, workers=-1)

    a = np.repeat(np.arange(n), k - 1)
    b = nbr[:, 1:].ravel()
    d = dist[:, 1:].ravel()
    keep = a < b                      # each undirected edge once
    a, b, d = a[keep], b[keep], d[keep]
    order = np.argsort(d)

    degree = np.zeros(n, dtype=np.int8)
    parent = np.arange(n)
    ends: list[list[int]] = [[] for _ in range(n)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in order:
        i, j = int(a[e]), int(b[e])
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:                  # would close a premature cycle
            continue
        parent[ri] = rj
        degree[i] += 1
        degree[j] += 1
        ends[i].append(j)
        ends[j].append(i)

    return _chain_fragments(_fragments(ends, n), pts)


def _fragments(ends, n) -> list[list[int]]:
    """Split the matching into its open paths (and any leftover cycles)."""
    seen = np.zeros(n, dtype=bool)
    frags: list[list[int]] = []

    def walk(start):
        seen[start] = True
        frag = [start]
        cur, prev = start, -1
        while True:
            nxt = -1
            for c in ends[cur]:
                if c != prev and not seen[c]:
                    nxt = c
                    break
            if nxt < 0:
                return frag
            seen[nxt] = True
            frag.append(nxt)
            prev, cur = cur, nxt

    for start in range(n):            # open ends first, so paths come out whole
        if not seen[start] and len(ends[start]) <= 1:
            frags.append(walk(start))
    for start in range(n):            # anything left is a closed cycle
        if not seen[start]:
            frags.append(walk(start))
    return frags


def _chain_fragments(frags: list[list[int]], pts: np.ndarray) -> np.ndarray:
    """Join fragments end to end, always taking the nearest free endpoint.

    Greedy matching always leaves the tour in pieces.  How those pieces are
    joined is what decides whether the drawing has a stray line ruled across
    it: picking the nearest endpoint keeps every join local, where taking
    them in index order throws chords clean across the page.
    """
    if len(frags) == 1:
        return np.asarray(frags[0], dtype=np.int64)

    m = len(frags)
    ends = np.empty((2 * m, 2), dtype=np.float64)
    for i, f in enumerate(frags):
        ends[2 * i] = pts[f[0]]
        ends[2 * i + 1] = pts[f[-1]]

    tree = cKDTree(ends) if cKDTree is not None else None
    used = np.zeros(m, dtype=bool)
    order: list[int] = []
    used[0] = True
    order.extend(frags[0])
    tail = ends[1]

    for _ in range(m - 1):
        pick = _nearest_free_end(tree, ends, tail, used)
        if pick is None:
            break
        idx, at_end = pick
        used[idx] = True
        piece = frags[idx][::-1] if at_end else frags[idx]
        order.extend(piece)
        tail = pts[piece[-1]]

    for i in range(m):                # safety net, should not normally fire
        if not used[i]:
            order.extend(frags[i])
    return np.asarray(order, dtype=np.int64)


def _nearest_free_end(tree, ends, point, used):
    m = len(used)
    if tree is None:
        d = np.hypot(*(ends - point).T)
        d[np.repeat(used, 2)] = np.inf
        e = int(np.argmin(d))
        return (e // 2, bool(e % 2)) if np.isfinite(d[e]) else None

    k = 8
    while k <= 2 * m:
        k = min(k, 2 * m)
        _, idx = tree.query(point, k=k)
        for e in np.atleast_1d(idx):
            e = int(e)
            if e < 2 * m and not used[e // 2]:
                return e // 2, bool(e % 2)
        if k == 2 * m:
            break
        k *= 4
    free = np.nonzero(~used)[0]
    if free.size == 0:
        return None
    d = np.hypot(*(ends[2 * free] - point).T)
    return int(free[int(np.argmin(d))]), False


def _nn_tour(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    seen = np.zeros(n, dtype=bool)
    order = [0]
    seen[0] = True
    for _ in range(n - 1):
        d = np.hypot(*(pts - pts[order[-1]]).T)
        d[seen] = np.inf
        nxt = int(np.argmin(d))
        seen[nxt] = True
        order.append(nxt)
    return np.asarray(order, dtype=np.int64)


# ── tour refinement ──────────────────────────────────────────────────────

def _refine(pts, order, passes: int = 4) -> np.ndarray:
    """Alternate 2-opt and Or-opt until the tour stops getting shorter.

    Each pass also runs on the reversed tour.  The 2-opt scan only ever looks
    forward from a city, so a long edge whose useful partner sits *behind* it
    is invisible until the direction flips.
    """
    best = _tour_length(pts, order)
    for _ in range(passes):
        order = _two_opt(pts, order)
        order = _two_opt(pts, order[::-1])[::-1]
        order = _or_opt(pts, order)
        length = _tour_length(pts, order)
        if length > best - 1e-6:
            break
        best = length
    return np.ascontiguousarray(order)


def _tour_length(pts, order) -> float:
    return float(np.hypot(*np.diff(pts[order], axis=0).T).sum())


def _two_opt(pts, order, k: int = 10, rounds: int = 8) -> np.ndarray:
    """2-opt driven by neighbour lists rather than a sliding window.

    The moves worth making are the ones that remove a *long* edge, and those
    two endpoints are usually far apart in tour order - exactly what a
    windowed scan cannot reach.  Searching each city's k nearest neighbours
    finds them directly, and the search can stop as soon as the candidate is
    further than the edge being replaced, since no improving move survives
    past that point.
    """
    order = np.asarray(order, dtype=np.int64).copy()
    n = len(order)
    if n < 5 or cKDTree is None:
        return order

    dist, nbr = cKDTree(pts).query(pts, k=min(k + 1, n), workers=-1)
    pos = np.empty(n, dtype=np.int64)

    for _ in range(rounds):
        pos[order] = np.arange(n)
        improved = False
        for i in range(n - 2):
            a = order[i]
            b = order[i + 1]
            d_ab = float(np.hypot(*(pts[b] - pts[a])))
            for col in range(1, nbr.shape[1]):
                d_ac = float(dist[a, col])
                if d_ac >= d_ab:
                    break                     # neighbours are sorted
                j = int(pos[nbr[a, col]])
                if j <= i + 1 or j >= n - 1:
                    continue
                c = order[j]
                d_dd = order[j + 1]
                gain = (d_ab + float(np.hypot(*(pts[d_dd] - pts[c]))) -
                        d_ac - float(np.hypot(*(pts[d_dd] - pts[b]))))
                if gain > 1e-9:
                    order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
                    pos[order[i + 1:j + 1]] = np.arange(i + 1, j + 1)
                    improved = True
                    break
        if not improved:
            break
    return order


def _or_opt(pts, order, max_seg: int = 3, rounds: int = 3) -> np.ndarray:
    """Relocate short runs of points to a cheaper place in the tour.

    2-opt cannot move a group of stipples that landed in the wrong part of
    the sequence; Or-opt can, and those are exactly the stray excursions that
    show up as a line shooting across the drawing and back.
    """
    if cKDTree is None:
        return order
    order = list(np.asarray(order))
    n = len(order)
    if n < 8:
        return np.asarray(order)

    tree = cKDTree(pts)
    for _ in range(rounds):
        improved = False
        pos = np.empty(n, dtype=np.int64)
        pos[np.asarray(order)] = np.arange(n)
        i = 1
        while i < n - max_seg - 1:
            for L in range(1, max_seg + 1):
                if i + L >= n - 1:
                    break
                a, b = order[i - 1], order[i]
                c, d = order[i + L - 1], order[i + L]
                removed = (np.hypot(*(pts[b] - pts[a])) +
                           np.hypot(*(pts[d] - pts[c])) -
                           np.hypot(*(pts[d] - pts[a])))
                if removed <= 1e-9:
                    continue
                for cand in tree.query_ball_point(pts[b], removed):
                    j = int(pos[cand])
                    if i - 1 <= j <= i + L:
                        continue
                    if j >= n - 1:
                        continue
                    u, v = order[j], order[j + 1]
                    added = (np.hypot(*(pts[b] - pts[u])) +
                             np.hypot(*(pts[c] - pts[v])) -
                             np.hypot(*(pts[v] - pts[u])))
                    if added + 1e-9 < removed:
                        seg = order[i:i + L]
                        del order[i:i + L]
                        j2 = order.index(u)
                        order[j2 + 1:j2 + 1] = seg
                        pos[np.asarray(order)] = np.arange(n)
                        improved = True
                        break
                break
            i += 1
        if not improved:
            break
    return np.asarray(order)
