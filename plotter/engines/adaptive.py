"""Region-adaptive shading.

Cross-hatch where the image is smooth, flow-traced strokes where it is
textured, chosen automatically per region rather than by one global setting.
See ``regions.py`` for why a single ``hatch_coherence`` cannot serve both.
"""

from __future__ import annotations

from ..geometry import InkBuffer
from ..regions import classify
from . import crosshatch, flowlines


def generate(value, coverage, canvas, pen, cfg, features=None, form=None,
             image_rgb=None):
    px_per_mm = canvas.px_per_mm
    pen_px = max(0.4, pen.width_mm * px_per_mm)

    textured = classify(value, coverage, px_per_mm, image_rgb=image_rgb,
                        n_segments=cfg.region_segments,
                        threshold=cfg.region_threshold)
    smooth = ~textured

    cov_smooth = coverage.copy()
    cov_smooth[textured] = 0.0
    cov_textured = coverage.copy()
    cov_textured[smooth] = 0.0

    paths: list = []
    if cov_smooth.max() > 0:
        p, _ = crosshatch.generate(value, cov_smooth, canvas, pen, cfg,
                                   features=features, form=form)
        paths.extend(p)
    if cov_textured.max() > 0:
        p, _ = flowlines.generate(value, cov_textured, canvas, pen, cfg,
                                  features=features, form=form)
        paths.extend(p)

    ink = InkBuffer(coverage, pen_px, features=features)
    for p in paths:
        ink.deposit(p)
    return paths, ink
