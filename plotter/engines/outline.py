"""Outline tracing: thin the strokes, then walk them.

Canny plus ``findContours`` - what the old converter did - traces the *rim*
of every edge, so each line in the source comes back as a long thin loop that
the plotter draws twice, once up each side.  Thinning to a one-pixel skeleton
first and then walking that skeleton as a graph draws each line once, and
walking *through* junctions in the straightest available direction keeps the
result as a few long strokes instead of a confetti of little ones.
"""

from __future__ import annotations

import cv2
import numpy as np

# 8-neighbourhood, ordered so opposite directions are 4 apart.
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1))


def skeleton(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, vectorised over the whole image per pass."""
    if hasattr(cv2, "ximgproc"):
        try:
            return (cv2.ximgproc.thinning(
                (mask > 0).astype(np.uint8) * 255) > 0).astype(np.uint8)
        except Exception:
            pass

    img = (mask > 0).astype(np.uint8)
    for _ in range(80):
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(img, dy, 0), dx, 1)
                 for dy, dx in ((-1, 0), (-1, 1), (0, 1), (1, 1),
                                (1, 0), (1, -1), (0, -1), (-1, -1))]
            # p[0]=N, p[1]=NE, p[2]=E, p[3]=SE, p[4]=S, p[5]=SW, p[6]=W, p[7]=NW
            b = sum(p)
            seq = p + [p[0]]
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
                    for i in range(8))
            if step == 0:
                c1 = p[0] * p[2] * p[4]
                c2 = p[2] * p[4] * p[6]
            else:
                c1 = p[0] * p[2] * p[6]
                c2 = p[0] * p[4] * p[6]
            kill = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & (c1 == 0) & (c2 == 0)
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            break
    return img


def trace(skel: np.ndarray, min_px: int = 6) -> list[np.ndarray]:
    """Walk a one-pixel skeleton into long polylines."""
    h, w = skel.shape
    pix = np.argwhere(skel > 0)
    if len(pix) == 0:
        return []

    index = {(int(y), int(x)): i for i, (y, x) in enumerate(pix)}
    adj: list[list[int]] = [[] for _ in range(len(pix))]
    for (y, x), i in index.items():
        for dy, dx in _NEIGHBOURS:
            j = index.get((y + dy, x + dx))
            if j is not None and j > i:
                adj[i].append(j)
                adj[j].append(i)

    used: set[tuple[int, int]] = set()

    def edge(i, j):
        return (i, j) if i < j else (j, i)

    def step_from(i, prev):
        """Pick the unused neighbour that best continues the current heading."""
        best, best_score = -1, -2.0
        if prev >= 0:
            vy = float(pix[i][0] - pix[prev][0])
            vx = float(pix[i][1] - pix[prev][1])
            n = (vx * vx + vy * vy) ** 0.5 or 1.0
            vy, vx = vy / n, vx / n
        for j in adj[i]:
            if edge(i, j) in used:
                continue
            if prev < 0:
                return j
            wy = float(pix[j][0] - pix[i][0])
            wx = float(pix[j][1] - pix[i][1])
            m = (wx * wx + wy * wy) ** 0.5 or 1.0
            score = (wy * vy + wx * vx) / m
            if score > best_score:
                best_score, best = score, j
        return best

    def walk(start, first):
        pts = [start]
        i, prev = start, -1
        j = first
        while j >= 0:
            used.add(edge(i, j))
            pts.append(j)
            prev, i = i, j
            j = step_from(i, prev)
        return pts

    paths: list[np.ndarray] = []
    degree = np.fromiter((len(a) for a in adj), dtype=np.int32, count=len(adj))

    # Endpoints first so open strokes come out whole, then junctions, then
    # whatever closed loops are left over.
    for order in (np.where(degree == 1)[0], np.where(degree > 2)[0],
                  np.arange(len(adj))):
        for i in order:
            i = int(i)
            while True:
                j = step_from(i, -1)
                if j < 0:
                    break
                ids = walk(i, j)
                if len(ids) >= min_px:
                    paths.append(pix[ids][:, ::-1].astype(np.float32))

    return paths


def generate(value, coverage, canvas, pen, cfg, strength: float | None = None,
             features=None, feature_min: float = 0.0, form=None,
             field=None, image_rgb=None):
    """Outline strokes for the current image, in pixel coordinates."""
    px_per_mm = canvas.px_per_mm
    # Called two ways: as an engine in its own right, where the registry
    # passes no strength, and as an overlay, where the pipeline does.  Fall
    # back to the configured value so the control works in both.
    if strength is None:
        strength = getattr(cfg, "outline_strength", 0.5)
    s = float(np.clip(strength, 0.02, 1.0))

    method = getattr(cfg, "line_method", "canny")
    if method == "learned":
        from .. import learned_lines
        if not learned_lines.available():
            method = "coherent"        # graceful fallback without torch

    if method == "learned":
        # A network trained to draw the lines an illustrator would keep (see
        # learned_lines.py).  Its output is already a drawing, so the only
        # decision left is how faint a line is still worth a stroke.
        from .. import learned_lines
        rgb = image_rgb if image_rgb is not None else (
            np.dstack([value] * 3) * 255).astype(np.uint8)
        dark = learned_lines.draw(rgb, (value.shape[1], value.shape[0]),
                                  style=getattr(cfg, "line_style", "contour"))
        mask = dark > (0.62 - 0.42 * s)
        edges = (mask.astype(np.uint8) * 255)
        k = 1.0
        feature_min = 0.0              # the network already chose the landmarks
    elif method == "coherent":
        # Flow-based DoG: looks only across the local structure, then smooths
        # along it, so what comes back is connected line work rather than a
        # confetti of fragments traced up both sides of every stroke.
        from .. import flow as _flow
        from ..lines import coherent_lines
        if field is None:
            field = _flow.compute(value, sigma_grad=1.2, sigma_tensor=3.0,
                                  coherence_mix=1.0,
                                  base_angle_deg=cfg.angle_deg)
        mask = coherent_lines(value, field,
                              sigma_c=max(0.6, 0.20 * px_per_mm),
                              sigma_m=max(2.0, 0.55 * px_per_mm),
                              sharpness=30.0 + 180.0 * s)
        edges = (mask.astype(np.uint8) * 255)
        k = 1.0
    else:
        # Work at a coarser scale than the photo.  Texture - fabric weave,
        # film grain, foliage - is edges all the way down, and a detector run
        # at full resolution answers honestly with thousands of them.
        detail_mm = 0.35 + 1.4 * (1.0 - s)
        k = max(1.0, detail_mm * px_per_mm / 2.0)
        small = cv2.resize(value, (max(8, int(value.shape[1] / k)),
                                   max(8, int(value.shape[0] / k))),
                           interpolation=cv2.INTER_AREA)
        small = cv2.bilateralFilter(small.astype(np.float32), 7, 0.10, 7)
        v8 = np.clip(small * 255.0, 0, 255).astype(np.uint8)
        gx = cv2.Sobel(v8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(v8, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        hi = max(12.0, float(np.percentile(mag, 100.0 - 12.0 * s)))
        edges = cv2.Canny(v8, hi * 0.4, hi, L2gradient=True)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    skel = skeleton(edges)
    paths = trace(skel, min_px=4)

    # Length filter in millimetres, applied at the working scale.
    min_len = 1.8 * px_per_mm / k
    out = []
    for pth in paths:
        if len(pth) < 2:
            continue
        if float(np.hypot(*np.diff(pth, axis=0).T).sum()) < min_len:
            continue
        out.append(pth * k)

    if features is not None and feature_min > 0 and out:
        # As an overlay the job is to draw the *landmarks* - a lid line, a lip
        # seam, a nostril - not every edge in the picture.  Fabric weave and
        # foliage are edges too, and drawing those on top of finished shading
        # only muddies it.  Keeping strokes that sit on thin dark structure
        # picks out the lines a person would have drawn.
        fh, fw = features.shape
        kept = []
        for pth in out:
            ix = np.clip(pth[:, 0].astype(np.int32), 0, fw - 1)
            iy = np.clip(pth[:, 1].astype(np.int32), 0, fh - 1)
            if float(np.percentile(features[iy, ix], 70)) >= feature_min:
                kept.append(pth)
        out = kept
    return out


def generate_lineart(value, coverage, canvas, pen, cfg, features=None,
                     form=None, image_rgb=None):
    """Skeleton of the drawn marks themselves - for scans and line art."""
    px_per_mm = canvas.px_per_mm
    v8 = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    block = max(11, int(6 * px_per_mm) | 1)
    mask = cv2.adaptiveThreshold(v8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    skel = skeleton(mask)
    paths = trace(skel, min_px=4)
    min_len = 1.0 * px_per_mm
    return [p for p in paths
            if len(p) >= 2 and float(np.hypot(*np.diff(p, axis=0).T).sum()) >= min_len]
