"""Modulated serpentine ("wave") shading.

One boustrophedon sweep covers the whole page, so the pen barely lifts.  Tone
comes from how far the line detours: a straight run lays down the least ink
per unit of travel, and the wave earns extra darkness by getting longer
without going anywhere.

The modulation is solved rather than guessed.  To reach coverage ``c`` with a
pen of width ``p`` on rows spaced ``s`` apart, a unit of horizontal travel
must carry ``c*s/p`` units of drawn line; that ratio is inverted through the
arc length of a sine to recover the amplitude, so the tone that comes out is
the tone that was asked for.
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import InkBuffer

# Arc-length ratio of y = A*sin(2*pi*x/L) as a function of k = 2*pi*A/L.
# rho(k) = mean over one period of sqrt(1 + k^2 cos^2 t)
_K = np.linspace(0.0, 60.0, 2400)
_T = np.linspace(0.0, math.pi, 512)
_RHO = np.sqrt(1.0 + (_K[:, None] ** 2) * np.cos(_T) ** 2).mean(axis=1)


def _amplitude_ratio(rho: np.ndarray) -> np.ndarray:
    """Invert rho(k): how hard must the wave swing to stretch by this much?"""
    return np.interp(np.clip(rho, 1.0, _RHO[-1]), _RHO, _K)


def generate(value, coverage, canvas, pen, cfg, features=None, form=None):
    h, w = coverage.shape
    px_per_mm = canvas.px_per_mm
    pen_px = max(0.4, pen.width_mm * px_per_mm)

    spacing_mm = cfg.spacing_mm if cfg.spacing_mm > 0 else 6.0 * pen.width_mm
    s = max(2.0 * pen_px, spacing_mm * px_per_mm)
    period = max(4.0 * pen_px, cfg.wave_period_mm * px_per_mm)

    theta = math.radians(cfg.angle_deg)
    ux, uy = math.cos(theta), math.sin(theta)      # along a row
    vx, vy = -math.sin(theta), math.cos(theta)     # across rows

    cx, cy = w / 2.0, h / 2.0
    diag = math.hypot(w, h)
    n_rows = int(diag / s) + 2
    n_samples = max(8, int(diag / max(0.8, pen_px * 0.6)))
    t = np.linspace(-diag / 2.0, diag / 2.0, n_samples)
    dt = float(t[1] - t[0])
    rng = np.random.default_rng(7)

    # Light tone carries less ink than even a straight line lays down, so
    # some of it has to be left undrawn.  The shortfall is diffused *down the
    # rows* rather than along them: a light area drops whole rows and keeps
    # the ones it draws unbroken, where dashing along a row would shred one
    # long sweep into dozens of strokes and dozens of pen lifts.
    #
    # The decision is taken per block, not per sample.  Diffusing each column
    # independently lets neighbouring columns drift out of step wherever tone
    # varies, and the row then breaks into thousands of specks; a block-wide
    # decision guarantees every run is at least one block long.
    block = max(1, int(round(6.0 * px_per_mm / max(dt, 1e-6))))
    edges = np.arange(0, n_samples, block)
    carry = rng.random() + 0.02 * rng.random(len(edges))

    paths: list[np.ndarray] = []
    run: list[tuple[float, float]] = []

    for row in range(n_rows):
        off = (row - (n_rows - 1) / 2.0) * s
        bx = cx + vx * off
        by = cy + vy * off
        xs = bx + ux * t
        ys = by + uy * t

        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        cov = np.zeros(n_samples, dtype=np.float32)
        ix = np.clip(xs.astype(np.int32), 0, w - 1)
        iy = np.clip(ys.astype(np.int32), 0, h - 1)
        cov[inside] = coverage[iy[inside], ix[inside]]

        rho = cov * s / pen_px
        k = _amplitude_ratio(rho)
        amp = k * period / (2.0 * math.pi)

        # Amplitude is capped at half the row spacing so neighbouring rows do
        # not collide; past that point extra darkness has to come from a
        # shorter wavelength instead.
        amp_max = 0.5 * s
        over = amp > amp_max
        local_period = np.full(n_samples, period, dtype=np.float64)
        if np.any(over):
            amp[over] = amp_max
            local_period[over] = 2.0 * math.pi * amp_max / np.maximum(k[over], 1e-6)

        duty = np.clip(rho, 0.0, 1.0) * inside
        counts = np.maximum(np.add.reduceat(inside.astype(np.float64), edges), 1.0)
        carry += np.add.reduceat(duty, edges) / counts
        keep = carry >= 1.0
        carry[keep] -= 1.0
        draw = inside & (cov > 0) & np.repeat(keep, np.diff(np.append(edges, n_samples)))

        phase = np.cumsum(2.0 * math.pi * dt / local_period)
        wob = amp * np.sin(phase + rng.random() * 2.0 * math.pi)
        px = xs + vx * wob
        py = ys + vy * wob

        if row % 2:                                 # serpentine
            px, py, draw = px[::-1], py[::-1], draw[::-1]

        for i in range(n_samples):
            if draw[i]:
                run.append((px[i], py[i]))
            elif run:
                if len(run) >= 2:
                    paths.append(np.asarray(run, dtype=np.float32))
                run = []

    if len(run) >= 2:
        paths.append(np.asarray(run, dtype=np.float32))

    ink = InkBuffer(coverage, pen_px, features=features)
    for p in paths:
        ink.deposit(p)
    return paths, ink
