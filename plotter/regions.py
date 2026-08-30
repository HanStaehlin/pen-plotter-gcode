"""Region-adaptive routing: segment the image, let each style draw what it's good at.

``hatch_coherence`` has no single value that suits both a feathered
photograph and a face (see crosshatch.py) - that turned out to be a property
of the *image*, not a tuning miss: real texture (feathers, fur, weave) wants
high coherence to stay legible, smooth shading (skin, sky) wants low
coherence or it rings into concentric contours. This module sidesteps the
choice by classifying regions and letting the pipeline route each to the
engine suited to it, rather than asking one global setting to serve both.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from skimage.segmentation import slic
except ImportError:                                   # pragma: no cover
    slic = None


def texture_energy(value: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Local variance of lightness, normalised to roughly 0-1.

    This is the same diagnostic used to work out why Hilbert lost the owl's
    feather barring: a window of local variance answers "is there real
    texture here" independent of how light or dark the region is, which
    plain edge strength or coverage cannot - a smooth, brightly lit cheek has
    edges too.
    """
    radius_px = max(3, int(round(1.2 * px_per_mm)))    # texture scale ~1.2 mm
    k = radius_px | 1
    mean = cv2.blur(value, (k, k))
    var = cv2.blur((value - mean) ** 2, (k, k))
    ref = float(np.percentile(var, 99.0))
    if ref < 1e-9:
        return np.zeros_like(var, dtype=np.float32)
    return np.clip(var / ref, 0.0, 1.0).astype(np.float32)


def _drop_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Flip connected components of True smaller than ``min_area`` to False."""
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    keep = counts >= min_area
    keep[0] = False                     # background label
    return keep[labels]


def classify(value: np.ndarray, coverage: np.ndarray, px_per_mm: float,
            image_rgb: np.ndarray | None = None,
            n_segments: int = 400, compactness: float = 8.0,
            threshold: float = 0.30) -> np.ndarray:
    """Boolean mask, True where a region should be treated as textured.

    Classification is decided per *superpixel*, not per pixel: a per-pixel
    threshold on noisy variance would fragment one feather into hundreds of
    disconnected slivers, each triggering an engine switch, and the seam
    between styles would show up as visible clutter rather than a clean
    boundary between two drawing techniques.
    """
    energy = texture_energy(value, px_per_mm)
    if slic is None:
        return energy > threshold

    # SLIC's distance metric works in colour (it converts to Lab internally);
    # handed a plain single-channel array with no channel axis it has no
    # colour signal to cluster on, spatial compactness wins outright, and the
    # result is a perfect grid that ignores the image entirely.  Falling back
    # to a stacked grayscale still beats that, but real colour tracks the
    # actual feather and fur boundaries far more tightly.
    if image_rgb is not None:
        h, w = value.shape
        if image_rgb.shape[:2] != (h, w):
            image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
        source = image_rgb
    else:
        source = np.dstack([value, value, value])

    segments = slic(source, n_segments=n_segments, compactness=compactness,
                    channel_axis=-1, start_label=0)

    # Paper-white background carries no texture by definition; including it
    # would drag every segment that merely touches the subject's silhouette
    # toward "smooth", misclassifying real texture right at the edge.
    ink = coverage > 0.02
    n = int(segments.max()) + 1
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    flat_seg = segments.ravel()
    flat_energy = energy.ravel()
    flat_ink = ink.ravel()
    np.add.at(sums, flat_seg[flat_ink], flat_energy[flat_ink])
    np.add.at(counts, flat_seg[flat_ink], 1.0)

    seg_mean = np.divide(sums, np.maximum(counts, 1.0))
    # Segments with no ink at all (pure background) default to "smooth" so
    # they are simply skipped rather than routed anywhere.
    textured_segments = seg_mean > threshold
    mask = textured_segments[segments]

    # Clean up small islands and holes.  Every boundary the mask has is a
    # place a stroke gets cut short and a second one has to start to finish
    # the job, so a speckled mask does not just look messy - it inflates the
    # drawn length for no visual gain, since the reason to split styles at
    # all is legibility, and dozens of thumbnail-sized style switches buy
    # none.
    min_area = max(64, int(round((1.5 * px_per_mm) ** 2)))
    mask = _drop_small_regions(mask, min_area)          # drop small textured islands
    mask = ~_drop_small_regions(~mask, min_area)        # fill small smooth holes
    return mask
