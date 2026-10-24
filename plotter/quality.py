"""How the finished plot will look, judged from the G-code paths themselves.

The ink accumulator's own error only says how well an engine hit the target
it was chasing, before welding, bridging, pruning and the time budget have
touched the strokes.  This scores what actually goes to the machine: every
path is drawn at the pen's real width, both drawing and target are blurred
the way the eye blurs fine lines at arm's length, and the two are compared.

Two numbers come out.  ``likeness`` is SSIM between the viewed drawing and
the viewed tone target - structure, contrast and tone together, 1 = the same
picture.  ``tone_error`` is the mean difference in coverage, which is the
one to watch when a change is meant to make darks darker.  Neither replaces
looking: pen-and-ink deliberately leaves light tones to the paper, so an
illustration scores below a dense tone engine that it beats by eye.  They
are for comparing a change against itself.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Widths OpenCV actually draws, antialiased, for thickness 3, 2 and 1.
#: It only draws odd widths (thickness 4 comes out like 5), plus an
#: antialiasing fringe, so asking for "4 px" overstates ink by over a third.
#: The rendering resolution is chosen instead so that one of these widths is
#: exactly the pen.
_AA_WIDTHS = ((3, 5.70), (2, 3.70), (1, 1.41))
MAX_PX_PER_MM = 16.0
#: Viewing blur, as a Gaussian sigma on paper.  At arm's length the eye
#: resolves around 1-2 mm, which is what makes hatching read as grey.
VIEW_SIGMA_MM = 1.0


def render_scale(pen_width_mm: float) -> tuple[int, float]:
    """(OpenCV thickness, px per mm) that draws the pen at its true width."""
    pen = max(pen_width_mm, 0.05)
    for thick, width in _AA_WIDTHS:
        if width / pen <= MAX_PX_PER_MM:
            return thick, width / pen
    return _AA_WIDTHS[-1][0], _AA_WIDTHS[-1][1] / pen


def render(mm_paths, canvas, pen_width_mm: float) -> tuple[np.ndarray, float]:
    """Ink coverage 0..1 over the canvas, from paths in machine millimetres.

    Strokes are drawn into one mask, so where they cross the ink is counted
    once - the union a real pen produces.  Returns the image and its px/mm.
    """
    thick, px_per_mm = render_scale(pen_width_mm)
    k = px_per_mm * canvas.mm_per_px              # render px per canvas px
    h, w = int(round(canvas.h_px * k)), int(round(canvas.w_px * k))
    img = np.zeros((h, w), np.uint8)
    pts = [np.round(canvas.to_px(p) * k * 16).astype(np.int32)
           for p in mm_paths if len(p) >= 2]
    if pts:
        cv2.polylines(img, pts, False, 255, thick, cv2.LINE_AA, shift=4)
    return img.astype(np.float32) / 255.0, px_per_mm


def viewed(coverage: np.ndarray, px_per_mm: float,
           sigma_mm: float = VIEW_SIGMA_MM) -> np.ndarray:
    return cv2.GaussianBlur(coverage.astype(np.float32), (0, 0),
                            max(0.3, sigma_mm * px_per_mm))


def assess(mm_paths, canvas, target: np.ndarray, pen_width_mm: float,
           sigma_mm: float = VIEW_SIGMA_MM) -> dict:
    """Likeness and tone error of a finished plot against its tone target."""
    from skimage.metrics import structural_similarity

    img, px_per_mm = render(mm_paths, canvas, pen_width_mm)
    ink = viewed(img, px_per_mm, sigma_mm)
    # Compare at the target's own resolution; the blur has already removed
    # everything finer than that.
    ink = cv2.resize(ink, (target.shape[1], target.shape[0]),
                     interpolation=cv2.INTER_AREA)
    ref = viewed(np.clip(target, 0.0, 1.0), canvas.px_per_mm, sigma_mm)
    return {
        "likeness": float(structural_similarity(ink, ref, data_range=1.0)),
        "tone_error": float(np.abs(ink - ref).mean()),
        "viewed": ink,
    }
