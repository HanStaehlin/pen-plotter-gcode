"""Illustration: learned line drawing over a lighter layer of shading.

The layered way illustrators work.  Lines carry form and identity - the
contour of a jaw, the rim of a headlight - and a separate, deliberately
lighter shading layer carries tone.  Every other engine here asks one set of
strokes to do both jobs, which is why form kept losing to tone: a coverage
target has no notion of which dark pixels are *the* edge of the nose.

The line layer comes from Informative Drawings (see learned_lines.py); the
shading layer is any existing tone engine, run on a reduced coverage target
with the line pixels removed so the two layers don't double-ink the same
place.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry import InkBuffer


def generate(value, coverage, canvas, pen, cfg, features=None, form=None,
             image_rgb=None):
    from . import get, outline

    px_per_mm = canvas.px_per_mm
    pen_px = max(0.4, pen.width_mm * px_per_mm)

    lines = outline.generate(value, coverage, canvas, pen, cfg,
                             strength=cfg.outline_strength, form=form,
                             image_rgb=image_rgb)

    # Rasterise the line layer and keep shading off it, with a margin of a
    # pen width so hatching doesn't pile up alongside every contour.
    h, w = coverage.shape
    line_mask = np.zeros((h, w), np.uint8)
    if lines:
        cv2.polylines(line_mask, [p.astype(np.int32) for p in lines], False, 255,
                      max(1, int(round(pen_px * 2))))
    # Pen-and-ink convention: lines carry the form, the paper carries the
    # light and mid tones, and hatching goes only where the shadows are.
    # Hatching everything, the way a tone engine on its own does, buries the
    # line layer under strokes of exactly the same weight.
    t0 = float(np.clip(cfg.illustration_shadow, 0.0, 0.95))
    shade_cov = np.clip((coverage - t0) / (1.0 - t0), 0.0, None) * coverage
    shade_cov *= float(np.clip(cfg.illustration_tone, 0.0, 1.0))

    # Spend the shading on the subject.  A blurred background - grass behind
    # a dog, a wall behind a sitter - still has mid-dark tone, and hatching it
    # faithfully produces a field of confetti that competes with the subject.
    # Depth separates the two far more reliably than tone does.
    focus = float(np.clip(cfg.illustration_focus, 0.0, 1.0))
    if focus > 0 and image_rgb is not None:
        from .. import depth as depth_mod
        if depth_mod.available():
            d = depth_mod.estimate(image_rgb, size=(w, h))
            near = np.clip((d - 0.25) / 0.45, 0.0, 1.0)
            near = near * near * (3.0 - 2.0 * near)            # smoothstep
            shade_cov *= (1.0 - focus) + focus * near
    shade_cov[line_mask > 0] = 0.0

    shading = []
    name = cfg.illustration_shading
    if name and name != "none" and shade_cov.max() > 0:
        produced = get(name)(value, shade_cov, canvas, pen, cfg,
                             features=features, form=form, image_rgb=image_rgb)
        shading = produced[0] if isinstance(produced, tuple) else produced

    # A hatch stroke shorter than a few millimetres reads as a speck, not as
    # tone, and each one costs a full pen lift.
    min_len = 3.0 * px_per_mm
    shading = [p for p in shading
               if len(p) > 1 and float(np.hypot(*np.diff(p, axis=0).T).sum()) >= min_len]
    paths = list(lines) + list(shading)
    ink = InkBuffer(coverage, pen_px, features=features)
    for p in paths:
        ink.deposit(p)
    return paths, ink
