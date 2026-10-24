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

    # Form shading, after fishdraw: a band of hatching along the side of the
    # subject facing away from the light.  fishdraw finds it as the part of
    # each closed shape not covered by a copy shifted toward the light; a
    # photo's line work is never closed (one leaky region covered 70-80% of
    # the test portraits), so here the same band comes from relighting the
    # depth estimate instead.  It reads as roundness even where the
    # photograph's own tone is flat.
    depth_map = None
    rim = float(np.clip(cfg.illustration_rim, 0.0, 1.0))
    focus = float(np.clip(cfg.illustration_focus, 0.0, 1.0))
    if (rim > 0 or focus > 0) and image_rgb is not None:
        from .. import depth as depth_mod
        if depth_mod.available():
            depth_map = depth_mod.estimate(image_rgb, size=(w, h))
    if rim > 0 and depth_map is not None:
        crescent = _shadow_side(depth_map, px_per_mm, cfg)
        lift = rim * (0.5 + 0.5 * np.clip(coverage / 0.6, 0.0, 1.0))
        shade_cov = np.maximum(shade_cov, crescent * lift
                               * float(np.clip(cfg.illustration_tone, 0.0, 1.0)))
        rim_mask = crescent
    else:
        rim_mask = None
    # Spend the shading on the subject.  A blurred background - grass behind
    # a dog, a wall behind a sitter - still has mid-dark tone, and hatching it
    # faithfully produces a field of confetti that competes with the subject.
    # Depth separates the two far more reliably than tone does.
    if focus > 0 and depth_map is not None:
        near = np.clip((depth_map - 0.25) / 0.45, 0.0, 1.0)
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
    shading = _fade(shading, cfg, rim_mask)
    paths = list(lines) + list(shading)
    ink = InkBuffer(coverage, pen_px, features=features)
    for p in paths:
        ink.deposit(p)
    return paths, ink


def _light(cfg) -> np.ndarray:
    """Unit vector toward the light, in image coordinates (y down)."""
    a = np.radians(cfg.illustration_light_deg)
    return np.array([np.cos(a), -np.sin(a)])


def _shadow_side(depth_map, px_per_mm, cfg) -> np.ndarray:
    """0..1: how far the subject's surface turns away from a virtual light.

    Normals come from the depth estimate; a surface facing away from the
    light gets shading in proportion.  Depth is smoothed at the band width
    first, which is what makes the band that wide - a hard depth edge at the
    silhouette otherwise yields a hairline.
    """
    sigma = max(1.0, 0.5 * cfg.illustration_rim_mm * px_per_mm)
    z = cv2.GaussianBlur(depth_map.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=5)
    near = depth_map > 0.35
    mag = np.hypot(gx, gy)
    ref = float(np.percentile(mag[near], 90)) if near.any() else float(mag.max())
    gx /= ref + 1e-9
    gy /= ref + 1e-9
    # Depth here is nearness, so the surface rises toward the viewer where it
    # grows; the outward normal therefore points down its gradient.
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    elev = np.radians(35.0)
    light = _light(cfg)
    L = np.array([light[0] * np.cos(elev), light[1] * np.cos(elev), np.sin(elev)])
    away = np.clip((np.sin(elev) - n @ L) / np.sin(elev), 0.0, 1.0)
    return (away * np.clip((depth_map - 0.25) / 0.45, 0.0, 1.0)).astype(np.float32)


def _fade(paths, cfg, rim_mask):
    """Start each hatch stroke a random way in from its lit end.

    A hatch block with every stroke ending on the same line looks stamped
    on; letting strokes begin at staggered points makes the shading thin out
    toward the light instead of stopping at an edge.  fishdraw trims up to
    half of each stroke this way.
    """
    fade = float(np.clip(cfg.illustration_fade, 0.0, 0.9))
    if fade <= 0 or not paths:
        return paths
    light = _light(cfg)
    rng = np.random.default_rng(31)
    out = []
    for p in paths:
        p = np.asarray(p, np.float32)
        if (p[-1] - p[0]) @ light > 0:          # make p[0] the lit end
            p = p[::-1]
        seg = np.hypot(*np.diff(p, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        cut = rng.uniform(0.0, fade) * cum[-1]
        k = int(np.searchsorted(cum, cut))
        if k >= len(p) - 1:
            continue
        t = (cut - cum[k - 1]) / max(seg[k - 1], 1e-9) if k > 0 else 0.0
        start = p[k - 1] + (p[k] - p[k - 1]) * t if k > 0 else p[0]
        out.append(np.vstack([start, p[k:]]).astype(np.float32))
    return out
