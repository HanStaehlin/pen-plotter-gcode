"""Adaptive space-filling curve shading.

The whole image is drawn as a single continuous curve - one pen down, one pen
up.  A Hilbert curve visits every cell of a quadtree without ever jumping,
and crucially every subtree occupies a *contiguous* run of the curve.  That
property is what makes adaptive subdivision possible: collapsing a run of
fine cells down to one coarse cell keeps the curve connected, so darkness can
control local depth - deeper where the image is dark, laying more line into
the same area - without ever breaking the path.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry import InkBuffer


def generate(value, coverage, canvas, pen, cfg, features=None, form=None):
    h, w = coverage.shape
    px_per_mm = canvas.px_per_mm
    pen_px = max(0.4, pen.width_mm * px_per_mm)

    max_depth = int(np.clip(cfg.hilbert_max_depth, 2, 10))
    side = float(max(w, h))

    # Depth sets cell size and ink density together - that is the engine's
    # one real limitation.  A shallow cell is not just pale, it is *large*:
    # at depth 3 on A4 it spans 24 mm and carries a single stroke, which the
    # eye reads as a bare patch rather than as light tone.  So the floor is
    # set by how big a cell may get on paper, not by an abstract depth.
    coarsest_px = max(1.0, cfg.hilbert_coarsest_mm * px_per_mm)
    min_depth = int(np.ceil(np.log2(max(1.0, side / coarsest_px))))
    min_depth = int(np.clip(min_depth, 1, max_depth))
    n = 1 << max_depth

    # Blank areas narrower than a few cells are not worth lifting for.
    white_px = 3.0 * coarsest_px
    depth = _leaf_depths(coverage, side, n, min_depth, max_depth, pen_px,
                         white_px, features, cfg.hilbert_detail)

    d = np.arange(n * n, dtype=np.int64)
    hx, hy = _hilbert_d2xy(n, d)

    # Which leaf does each finest-level cell belong to, and which coarse cell
    # does it collapse into?  Runs of equal keys are contiguous along the
    # curve precisely because Hilbert order is subtree-contiguous, so
    # collapsing them cannot break the path.
    dep = depth[hy, hx].astype(np.int64)
    shift = max_depth - dep
    cell_x = hx >> shift
    cell_y = hy >> shift
    key = (dep << 42) | (cell_x << 21) | cell_y

    keep = np.empty(len(key), dtype=bool)
    keep[0] = True
    np.not_equal(key[1:], key[:-1], out=keep[1:])
    starts = np.nonzero(keep)[0]
    counts = np.diff(np.append(starts, len(key)))

    # Place each collapsed point at the centroid of the fine cells its run
    # actually visited, not at the geometric centre of the coarse cell.  Where
    # a run is not a perfect subtree the two differ, and the centre version
    # throws a spike out to the side at every depth transition.
    scale = side / n
    cx = (np.add.reduceat(hx.astype(np.float64), starts) / counts + 0.5) * scale
    cy = (np.add.reduceat(hy.astype(np.float64), starts) / counts + 0.5) * scale
    cx -= (side - w) / 2.0
    cy -= (side - h) / 2.0

    # Depth 0 marks cells with no ink at all.  They must be dropped by that
    # flag, not by testing the collapsed point: every empty run collapses to
    # a similar place, and keeping them draws a starburst through it.
    drawn = (dep[starts] > 0) & (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)

    paths: list[np.ndarray] = []
    run: list[tuple[float, float]] = []
    for x, y, ok in zip(cx, cy, drawn):
        if ok:
            run.append((x, y))
        elif run:
            if len(run) >= 2:
                paths.append(np.asarray(run, dtype=np.float32))
            run = []
    if len(run) >= 2:
        paths.append(np.asarray(run, dtype=np.float32))

    if cfg.hilbert_smooth > 0:
        cell_px = side / (1 << max_depth)
        paths = [_round_corners(p, cfg.hilbert_smooth, cell_px) for p in paths]

    ink = InkBuffer(coverage, pen_px, features=features)
    for p in paths:
        ink.deposit(p)
    return paths, ink


def _round_corners(path: np.ndarray, factor: float, cell_px: float,
                   segments: int = 3) -> np.ndarray:
    """Replace the curve's right angles with short arcs.

    A Hilbert curve is nothing but 90-degree corners, and a corner is where a
    plotter has to stop: junction speed falls off with the turn, so a square
    corner costs roughly four times the dwell of a gentle one.  Cutting each
    corner into a few shallow turns keeps the machine moving and softens the
    hard maze texture into something closer to drawn line.
    """
    n = len(path)
    if n < 3 or factor <= 0:
        return path
    f = float(np.clip(factor, 0.0, 0.5))
    p = np.asarray(path, dtype=np.float32)

    seg = np.diff(p, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    # Corner radius is capped by the shorter of the two edges meeting there,
    # so adjacent corners on a short edge cannot overrun each other.
    cut = np.minimum(length[:-1], length[1:]) * f

    out = [p[0]]
    t = np.linspace(0.0, 1.0, segments + 2)[1:-1][:, None]
    for i in range(1, n - 1):
        r = cut[i - 1]
        if r < 1e-4:
            out.append(p[i])
            continue
        a = p[i] + (p[i - 1] - p[i]) * (r / max(length[i - 1], 1e-6))
        b = p[i] + (p[i + 1] - p[i]) * (r / max(length[i], 1e-6))
        out.append(a)
        # Quadratic Bezier through the corner: a -> p[i] -> b.
        out.extend((1 - t) ** 2 * a + 2 * (1 - t) * t * p[i] + t ** 2 * b)
        out.append(b)
    out.append(p[-1])
    return np.asarray(out, dtype=np.float32)


def _leaf_depths(coverage, side, n, min_depth, max_depth, pen_px,
                 white_px: float = 0.0, features=None,
                 detail: float = 0.0) -> np.ndarray:
    """Subdivision depth per finest-level cell, sized to match coverage.

    At depth ``d`` the curve lays about ``side * 2^d`` of line across a
    ``side^2`` square, so its pitch is ``side / 2^d`` and the coverage it
    produces is ``pen * 2^d / side``.  Inverting that gives the depth a tone
    wants: ``log2(c * side / pen)``.
    """
    h, w = coverage.shape
    isize = int(side)
    sq = np.zeros((isize, isize), dtype=np.float32)
    oy, ox = (isize - h) // 2, (isize - w) // 2
    sq[oy:oy + h, ox:ox + w] = coverage

    want = np.log2(np.maximum(sq, 1e-6) * side / pen_px)

    # Depth normally follows darkness alone, so a pale but finely detailed
    # region - an eye, a lip seam - is drawn at the same coarse pitch as the
    # flat skin beside it and loses its structure.  Each extra level halves
    # the cell, so adding the detail response here buys resolution exactly
    # where the picture has something to resolve.
    if features is not None and detail > 0:
        fsq = np.zeros_like(sq)
        fsq[oy:oy + h, ox:ox + w] = features
        want = want + float(detail) * fsq

    # Depth is decided on cell averages, not per pixel: one dark pixel should
    # not drag an entire subtree deep and speckle the tone.
    # Build a genuine quadtree by deciding coarse-to-fine: start every cell
    # at the finest level, then walk levels outward and stop any cell whose
    # mean demand it already satisfies.  Deciding each level independently
    # instead leaves siblings at different depths, and the collapsed runs stop
    # lining up with subtrees.
    depth = np.full((n, n), max_depth, dtype=np.int32)
    for d in range(max_depth - 1, min_depth - 1, -1):
        m = 1 << d
        avg = cv2.resize(want, (m, m), interpolation=cv2.INTER_AREA)
        # Each depth doubles line density, so plain rounding offers only one
        # tone per octave and every gradient collapses into steps.  An
        # ordered dither on the threshold makes a region wanting depth 8.3
        # subdivide 30% of its cells, and the mix reads as the tone between.
        stop = cv2.resize(((avg + _dither(m)) < d + 0.5).astype(np.uint8),
                          (n, n), interpolation=cv2.INTER_NEAREST)
        depth[stop > 0] = d

    # Paper white breaks the curve, and every break is a pen lift.  Tone just
    # under the ink cutoff speckles the light parts of an image with isolated
    # blank cells, which would shred one continuous curve into hundreds of
    # dashes for no visible gain.  Opening the mask keeps genuine white areas
    # white at full size while closing over the speckle.
    empty = (cv2.resize(sq, (n, n), interpolation=cv2.INTER_AREA) <= 0)
    empty = empty.astype(np.uint8)
    span = max(1, int(round(white_px * n / side)))
    if span > 1:
        k = np.ones((span, span), np.uint8)
        empty = cv2.morphologyEx(empty, cv2.MORPH_OPEN, k)
    depth[empty > 0] = 0
    return depth


_BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32)


def _dither(m: int) -> np.ndarray:
    """Bayer threshold offsets in [-0.5, 0.5), tiled to an m x m grid."""
    pattern = (_BAYER8 + 0.5) / 64.0 - 0.5
    reps = -(-m // 8)
    return np.tile(pattern, (reps, reps))[:m, :m]


def _hilbert_d2xy(n: int, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Hilbert index -> (x, y) on an n x n grid (n a power of 2)."""
    x = np.zeros_like(d)
    y = np.zeros_like(d)
    t = d.copy()
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        # Rotate the quadrant so successive sub-curves join end to end.
        flip = ry == 0
        swap_x = np.where(flip & (rx == 1), s - 1 - x, x)
        swap_y = np.where(flip & (rx == 1), s - 1 - y, y)
        nx = np.where(flip, swap_y, swap_x)
        ny = np.where(flip, swap_x, swap_y)
        x, y = nx, ny
        x = x + s * rx
        y = y + s * ry
        t = t // 4
        s *= 2
    return x.astype(np.int32), y.astype(np.int32)
