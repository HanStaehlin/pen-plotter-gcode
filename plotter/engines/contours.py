"""Tonal contour lines - a topographic map of brightness.

Isolines are closed loops by construction, so this engine produces very long
strokes and very few lifts, and the contours themselves trace the shape of
the subject.  Levels are placed on equal steps of *coverage* rather than
equal steps of brightness, which puts the lines where tone is actually
changing instead of bunching them in whatever range happens to dominate.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry import InkBuffer


def generate(value, coverage, canvas, pen, cfg, features=None, form=None):
    px_per_mm = canvas.px_per_mm
    pen_px = max(0.4, pen.width_mm * px_per_mm)
    levels = int(np.clip(cfg.contour_levels, 2, 64))

    # Smooth at the scale of the contour spacing, not the pen.  An isoline
    # of a photographic tone field follows every texture ripple that happens
    # to cross the level, and at photo resolution that is thousands of tiny
    # closed blobs - the drawing fills in solid and takes hours.
    sigma = max(1.5, (cfg.spacing_mm if cfg.spacing_mm > 0 else 1.2) * px_per_mm)
    field = cv2.GaussianBlur(coverage, (0, 0), sigma)

    paths: list[np.ndarray] = []
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-4:
        return paths, InkBuffer(coverage, pen_px, features=features)

    min_len = max(8.0, 4.0 * px_per_mm)
    min_area = (1.5 * px_per_mm) ** 2
    for i in range(1, levels + 1):
        thresh = lo + (hi - lo) * i / (levels + 1.0)
        mask = (field >= thresh).astype(np.uint8)
        found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in found:
            if len(c) < 12 or cv2.arcLength(c, True) < min_len:
                continue
            # Filter on enclosed area too: a long thin sliver hugging a level
            # is arc-length-legal but reads as a scribble on paper.
            if cv2.contourArea(c) < min_area:
                continue
            pts = c[:, 0, :].astype(np.float32)
            paths.append(np.vstack([pts, pts[:1]]))     # close the loop
    ink = InkBuffer(coverage, pen_px, features=features)
    for p in paths:
        ink.deposit(p)
    return paths, ink
