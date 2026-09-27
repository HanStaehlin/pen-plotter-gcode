"""CLIPasso-style optimisation: strokes fit to recognisability, not tone.

Every other engine in this package matches a coverage target pixel by pixel.
This one does something categorically different: it starts from a handful of
random Bezier curves and repeatedly asks "does this, rendered as a sketch,
still look like the same thing to CLIP" - backpropagating through a
differentiable renderer to move the curves until it does. The result
optimises for recognisable *form*, not tone, and can be far sparser than
anything a coverage-matching engine would produce.

See ``sketch.py`` for the renderer and optimiser, and why it reimplements
CLIPasso's idea with a plain-PyTorch rasteriser instead of depending on the
reference ``diffvg`` (not on PyPI, needs a native build).

This is the slowest engine here by roughly two orders of magnitude - tens of
seconds to a few minutes rather than tens of seconds to a few - because it
is running an optimisation loop, not a single geometric pass. It is also the
only engine whose stroke count is fixed in advance rather than discovered
from the image, and the only one for which "tone error" is not a meaningful
number: it was never asked to match the coverage model the other engines
are judged against, so it is left unreported here rather than shown as
though it means the same thing.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import sketch


def generate(value, coverage, canvas, pen, cfg, features=None, form=None,
            image_rgb=None):
    if not sketch.available():
        raise RuntimeError(
            "the clipasso engine needs torch and open_clip "
            "(pip install torch open_clip_torch)")

    if image_rgb is None:
        # Fall back to the lightness map read as grayscale-as-RGB; CLIP still
        # recognises shape and layout from that, just without colour cues.
        image_rgb = np.dstack([value, value, value]) * 255.0
        image_rgb = image_rgb.astype(np.uint8)

    h, w = value.shape
    # Deliberately *not* resized to the working resolution: strokes are
    # optimised and stored in normalised 0-1 coordinates, so the photo can be
    # used at whatever resolution it actually arrived in.  Upscaling a small
    # source photo up to the working resolution first, as an earlier version
    # of this did, ran edge detection on interpolation-blurred pixels instead
    # of the photo's own detail, which visibly degraded the stroke seeding
    # for no benefit - only the final polyline sampling below needs (w, h).
    edges = cv2.Canny(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), 60, 150) > 0

    # Stroke width during optimisation is a visual-weight parameter, not a
    # literal preview of the pen: too thin and the soft renderer struggles to
    # get any gradient signal (a thin line barely changes CLIP's read of the
    # image no matter where it moves), and testing found the picture stays
    # legible at a width far below what tying it to the physical pen and page
    # size would give for a typical A4 plot. The real line weight is set once
    # by the pen at G-code time regardless of what was used to fit the
    # geometry here.
    width_norm = float(cfg.clipasso_width)

    result = sketch.run(
        image_rgb, edges,
        n_strokes=int(cfg.clipasso_strokes),
        iters=int(cfg.clipasso_iters),
        canvas=224,
        width=width_norm,
    )

    paths = sketch.strokes_to_polylines(result, w, h)
    # Degenerate curves (all four control points collapsed to one spot,
    # which a stroke the optimiser gave up on tends toward) draw nothing and
    # only cost a pointless pen lift.
    min_len = max(1.0, 0.3 * canvas.px_per_mm)
    paths = [p for p in paths
             if float(np.hypot(*np.diff(p, axis=0).T).sum()) >= min_len]
    return paths
