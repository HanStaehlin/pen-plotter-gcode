"""Pixel space <-> page millimetres, and the ink accumulator."""

from __future__ import annotations

import cv2
import numpy as np

from .config import PageConfig


class Canvas:
    """Maps image pixels onto the printable area, preserving aspect ratio."""

    def __init__(self, w_px: int, h_px: int, page: PageConfig):
        self.w_px, self.h_px = int(w_px), int(h_px)
        self.page = page
        self.mm_per_px = min(page.draw_w / self.w_px, page.draw_h / self.h_px)
        self.width_mm = self.w_px * self.mm_per_px
        self.height_mm = self.h_px * self.mm_per_px
        self.origin_x = page.margin_mm + (page.draw_w - self.width_mm) / 2.0
        self.origin_y = page.margin_mm + (page.draw_h - self.height_mm) / 2.0

    @property
    def px_per_mm(self) -> float:
        return 1.0 / self.mm_per_px

    def to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        """(N,2) pixel coords -> (N,2) machine millimetres."""
        pts = np.asarray(pts_px, dtype=np.float64)
        out = np.empty_like(pts)
        out[:, 0] = self.origin_x + pts[:, 0] * self.mm_per_px
        y = self.origin_y + pts[:, 1] * self.mm_per_px
        out[:, 1] = -y if self.page.negative_y else y
        return out

    def to_px(self, pts_mm: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_mm, dtype=np.float64)
        out = np.empty_like(pts)
        out[:, 0] = (pts[:, 0] - self.origin_x) * self.px_per_mm
        y = -pts[:, 1] if self.page.negative_y else pts[:, 1]
        out[:, 1] = (y - self.origin_y) * self.px_per_mm
        return out

    def paths_to_mm(self, paths_px) -> list[np.ndarray]:
        return [self.to_mm(p) for p in paths_px if len(p) >= 2]


class InkBuffer:
    """Tracks how much ink has landed where, so engines can chase a target.

    This is what replaces threshold banding.  Instead of deciding "this pixel
    is in darkness band 3, draw 3 hatch layers", an engine keeps laying down
    strokes wherever the deposited ink still falls short of the target
    coverage.  Tone then comes out continuous, and the line spacing that
    produces it is discovered rather than dialled in.

    Ink is accumulated at roughly pen-width resolution and spread across a
    band ``smear`` cells wide.  That band is the neighbourhood over which
    "how dark is it here" gets measured - the perpendicular spread is exactly
    what stops a new stroke from being laid alongside an existing one.
    """

    def __init__(self, coverage: np.ndarray, pen_px: float,
                 max_spacing_px: float = 30.0,
                 features: np.ndarray | None = None):
        h, w = coverage.shape
        self.pen_px = max(0.35, float(pen_px))
        self.scale = 1.0 / self.pen_px               # image px -> buffer px
        self.bw = max(8, int(round(w * self.scale)))
        self.bh = max(8, int(round(h * self.scale)))
        target = cv2.resize(coverage, (self.bw, self.bh),
                            interpolation=cv2.INTER_AREA).astype(np.float32)
        # A touch of blur stops single noisy pixels from attracting strokes.
        self.target = cv2.GaussianBlur(target, (0, 0), 0.8)
        self.ink = np.zeros((self.bh, self.bw), dtype=np.float32)
        self._scratch = np.zeros((self.bh, self.bw), dtype=np.float32)
        self.max_width = max(3.0, 2.0 * float(max_spacing_px))
        #: Optional ceiling on how much coverage the *current* pass may fill.
        #: Cross-hatching uses it to hand the remaining tone to the next angle
        #: instead of letting one direction darken everything on its own.
        self.cap: float | None = None
        # Acuity: how tightly tone should be measured here.  1 means "this is
        # a thin dark landmark, judge it on its own"; 0 means "ordinary
        # shading, average it over the full spacing".
        if features is None:
            self.acuity = None
        else:
            self.acuity = cv2.resize(features, (self.bw, self.bh),
                                     interpolation=cv2.INTER_AREA).astype(np.float32)

    def deficit(self, x_px: float, y_px: float) -> float:
        """Target minus deposited at an image-pixel location."""
        bx = int(x_px * self.scale)
        by = int(y_px * self.scale)
        if bx < 0 or by < 0 or bx >= self.bw or by >= self.bh:
            return -1.0
        target = self.target[by, bx]
        if self.cap is not None and target > self.cap:
            target = self.cap
        return float(target - self.ink[by, bx])

    def wants_ink(self, x_px: float, y_px: float,
                  frac: float, floor: float) -> bool:
        """Is at least ``frac`` of a stroke's worth of coverage still missing?"""
        bx = int(x_px * self.scale)
        by = int(y_px * self.scale)
        if bx < 0 or by < 0 or bx >= self.bw or by >= self.bh:
            return False
        target = self.target[by, bx]
        if self.cap is not None and target > self.cap:
            target = self.cap
        return bool(target - self.ink[by, bx] > max(floor, frac * target))

    def deficit_map(self) -> np.ndarray:
        target = self.target if self.cap is None else np.minimum(self.target, self.cap)
        return target - self.ink

    def deposit(self, pts_px: np.ndarray) -> None:
        """Record a drawn polyline as local ink density.

        The band a stroke is credited over is set by the tone it is drawing:
        to read as coverage ``c`` a 1-unit-wide pen must sit ``1/c`` units from
        its neighbour, so that is exactly how far the stroke's influence has
        to reach.  A fixed band cannot do this - too narrow and light areas
        fill in solid because the gaps between bands still read as empty; too
        wide and fine dark detail gets averaged away.  The profile tapers, so
        the spacing that emerges is smooth rather than banded.
        """
        if len(pts_px) < 2:
            return
        pts = np.asarray(pts_px, dtype=np.float32) * self.scale

        bx = np.clip(pts[:, 0].astype(np.int32), 0, self.bw - 1)
        by = np.clip(pts[:, 1].astype(np.int32), 0, self.bh - 1)
        c = float(np.median(self.target[by, bx]))
        spacing = 1.0 / max(c, 1e-3)
        # Support spans two spacings, not one.  A triangular kernel of
        # half-width s sampled every s sums to a constant - the partition of
        # unity behind linear interpolation.  Halve that support and the
        # midpoint between two strokes reads as under-inked, which pulls in a
        # third stroke, and tone comes out two to three times too dark.
        width = min(self.max_width, 2.0 * spacing)
        if self.acuity is not None:
            # A feature narrower than the band it is measured over gets its
            # ink budget spread into the skin around it and disappears.  A
            # lip seam is 0.8 mm tall and the band judging it is 2 mm wide,
            # so it loses every time.  Tighten the band where detail lives.
            a = float(np.median(self.acuity[by, bx]))
            width = max(1.5, width * (1.0 - 0.85 * a))

        pad = int(width) + 2
        x0 = max(0, int(pts[:, 0].min()) - pad)
        y0 = max(0, int(pts[:, 1].min()) - pad)
        x1 = min(self.bw, int(pts[:, 0].max()) + pad)
        y1 = min(self.bh, int(pts[:, 1].max()) + pad)
        if x1 <= x0 or y1 <= y0:
            return

        view = self._scratch[y0:y1, x0:x1]
        local = [np.round((pts - (x0, y0)) * 16).astype(np.int32)]

        # Four nested passes build a triangular cross-section.  polylines
        # overwrites rather than accumulates, so they run widest-first with
        # a ramped value and each pass leaves the running total behind it.
        # The weight is chosen so the ink laid down totals one pen width per
        # unit of stroke length, whatever the band width turned out to be.
        fractions = (1.0, 0.75, 0.5, 0.25)
        thicknesses = [max(1, int(round(width * f)) | 1) for f in fractions]
        for step, t in enumerate(thicknesses, start=1):
            cv2.polylines(view, local, False, float(step), t,
                          cv2.LINE_8, shift=4)

        # Normalise the stamp instead of trusting nominal thickness: the
        # rasteriser's real footprint runs wide, and end caps add more on
        # short strokes.  Scaling to the measured sum makes the accumulator
        # conserve ink exactly, which is what keeps tone honest.
        stamp = float(view.sum())
        if stamp > 1e-9:
            length = float(np.hypot(*np.diff(pts, axis=0).T).sum())
            view *= length / stamp
            self.ink[y0:y1, x0:x1] += view
        view[:] = 0.0

    def coverage_error(self) -> float:
        """Mean absolute tone error, in coverage units - a quality readout."""
        return float(np.abs(np.clip(self.ink, 0, 1) - self.target).mean())
