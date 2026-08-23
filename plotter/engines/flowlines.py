"""Flow-guided strokes.

Long strokes are traced through the image's own orientation field and stop
when the local ink target has been met.  Because density is discovered by
chasing a coverage deficit rather than assigned by thresholding, tone comes
out continuous instead of banded, and because strokes follow structure they
describe the subject instead of screening over it.
"""

from __future__ import annotations

import math

import numpy as np

from .. import flow
from ..geometry import InkBuffer


def build_field(value, cfg):
    """The orientation field the strokes will follow."""
    return flow.compute(value,
                        sigma_grad=1.2,
                        sigma_tensor=max(0.8, cfg.flow_smooth),
                        coherence_mix=cfg.flow_coherence,
                        base_angle_deg=cfg.angle_deg)


def generate(value, coverage, canvas, pen, cfg, field=None, features=None):
    px_per_mm = canvas.px_per_mm
    pen_px = max(0.4, pen.width_mm * px_per_mm)
    if field is None:
        field = build_field(value, cfg)

    # Cap how sparse shading may get: past ~26 pen widths a "line every
    # 10 mm" stops reading as tone and starts reading as a stray stroke.
    ink = InkBuffer(coverage, pen_px, max_spacing_px=26.0, features=features)
    paths = fill(field, ink, canvas, pen, cfg)
    return paths, ink


def fill(field, ink, canvas, pen, cfg, seed=12345):
    """Lay strokes along ``field`` until the ink accumulator is satisfied.

    Shared with the cross-hatch engine, which runs this once per angle with a
    rotated field and a ceiling on how much tone the pass may take.
    """
    px_per_mm = canvas.px_per_mm
    # Image extent in pixels, recovered from the accumulator's own geometry.
    w = int(round(ink.bw * ink.pen_px))
    h = int(round(ink.bh * ink.pen_px))

    step = max(0.6, min(1.6, ink.pen_px * 0.5))
    min_steps = max(2, int(cfg.stroke_min_mm * px_per_mm / step))
    max_steps = max(min_steps + 1, int(cfg.stroke_max_mm * px_per_mm / step))

    sampler = _Sampler(field)
    rng = np.random.default_rng(seed)

    paths: list[np.ndarray] = []
    # A stroke deposits very nearly a full `target` of coverage where it
    # lands, so it is only worth laying down if at least half of one is
    # missing - the same round-to-nearest rule error diffusion uses.  Seeding
    # on any leftover deficit instead over-inks the whole image by ~50%.
    seed_frac = 0.5
    keep_frac = 0.25          # a stroke may run on through thinner deficit
    floor = 0.01

    for _ in range(14):
        seeds = _seed_order(ink, seed_frac, floor, rng)
        if seeds.size == 0:
            break
        drawn = 0
        for by, bx in seeds:
            # Re-checked here because earlier strokes in this very pass may
            # already have covered this cell.
            tgt = ink.target[by, bx]
            if ink.cap is not None and tgt > ink.cap:
                tgt = ink.cap
            if tgt - ink.ink[by, bx] < max(floor, seed_frac * tgt):
                continue
            sx = (bx + rng.random()) * ink.pen_px
            sy = (by + rng.random()) * ink.pen_px
            if not (0 <= sx < w and 0 <= sy < h):
                continue

            pts = _trace(sx, sy, sampler, ink, step, max_steps, w, h,
                         keep_frac, floor)
            if len(pts) < min_steps:
                continue
            arr = np.asarray(pts, dtype=np.float32)
            ink.deposit(arr)
            paths.append(arr)
            drawn += 1
        if drawn == 0:
            break

    return paths


def _seed_order(ink: InkBuffer, seed_frac: float, floor: float, rng) -> np.ndarray:
    """Cells still short of ink, darkest first.

    Working darkest-first means the long structural strokes through the
    subject get laid down while the field is still empty, so they run their
    full length instead of being chopped up by shading strokes already in
    the way.
    """
    deficit = ink.deficit_map()
    threshold = np.maximum(floor, seed_frac * ink.target)
    ys, xs = np.nonzero(deficit > threshold)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    # Break ties randomly, otherwise raster order prints a visible grain.
    key = deficit[ys, xs] + rng.random(ys.size) * 0.02
    order = np.argsort(-key)
    return np.stack([ys[order], xs[order]], axis=1)


def _trace(sx, sy, sampler, ink, step, max_steps, w, h, keep_frac, floor):
    """Follow the tangent field out of a seed in both directions."""
    half = max_steps // 2
    forward = _walk(sx, sy, sampler, ink, step, half, w, h, +1, keep_frac, floor)
    backward = _walk(sx, sy, sampler, ink, step, half, w, h, -1, keep_frac, floor)
    if not forward and not backward:
        return []
    return backward[::-1] + [(sx, sy)] + forward


def _walk(x, y, sampler, ink, step, max_steps, w, h, sign, keep_frac, floor):
    pts = []
    tx, ty = sampler.at(x, y)
    tx *= sign
    ty *= sign
    turned = 0.0
    half = step * 0.5

    for _ in range(max_steps):
        # Midpoint (RK2) integration: a plain Euler step drifts off curved
        # ridges badly enough to show up as strokes crossing their neighbours.
        mx, my = x + tx * half, y + ty * half
        if not (0 <= mx < w and 0 <= my < h):
            break
        nx, ny = sampler.at(mx, my)
        if nx * tx + ny * ty < 0:
            nx, ny = -nx, -ny

        x2, y2 = x + nx * step, y + ny * step
        if not (0 <= x2 < w and 0 <= y2 < h):
            break
        # Continuing costs less than starting a new stroke, so a stroke is
        # allowed to run on through thinner deficit than would justify a
        # fresh seed.  Longer strokes mean fewer pen lifts.
        if not ink.wants_ink(x2, y2, keep_frac, floor):
            break

        turned += abs(nx * ty - ny * tx)      # |sin| of the turn per step
        if turned > 12.0:                      # runaway spiral in noisy flow
            break

        pts.append((x2, y2))
        x, y, tx, ty = x2, y2, nx, ny

    return pts


class _Sampler:
    """Fast scalar bilinear read of the double-angle orientation field.

    The tracer takes a few hundred thousand samples per image, so this goes
    through flat memoryviews rather than numpy indexing - same arithmetic,
    roughly an order of magnitude less interpreter overhead.
    """

    __slots__ = ("c2", "s2", "w", "h", "_keep")

    def __init__(self, field: flow.FlowField):
        c2 = np.ascontiguousarray(field.c2, dtype=np.float32).reshape(-1)
        s2 = np.ascontiguousarray(field.s2, dtype=np.float32).reshape(-1)
        self._keep = (c2, s2)          # memoryviews do not own the buffers
        self.c2 = c2.data
        self.s2 = s2.data
        self.h, self.w = field.h, field.w

    def at(self, x: float, y: float) -> tuple[float, float]:
        w, h = self.w, self.h
        if x < 0.0:
            x = 0.0
        elif x > w - 1.001:
            x = w - 1.001
        if y < 0.0:
            y = 0.0
        elif y > h - 1.001:
            y = h - 1.001

        x0 = int(x)
        y0 = int(y)
        fx = x - x0
        fy = y - y0
        i = y0 * w + x0
        j = i + w

        c2, s2 = self.c2, self.s2
        gx = 1.0 - fx
        gy = 1.0 - fy
        w00 = gx * gy
        w10 = fx * gy
        w01 = gx * fy
        w11 = fx * fy

        c = c2[i] * w00 + c2[i + 1] * w10 + c2[j] * w01 + c2[j + 1] * w11
        s = s2[i] * w00 + s2[i + 1] * w10 + s2[j] * w01 + s2[j + 1] * w11
        t = 0.5 * math.atan2(s, c)
        return math.cos(t), math.sin(t)
