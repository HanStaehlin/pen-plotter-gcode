"""Cross-hatching that follows the form.

Layered hatching is how engravers built tone long before halftones: light
areas get one direction, darker areas get a second laid across it, darker
still a third.  The old converter imitated that with fixed darkness bands,
which is why its gradients came out as four flat steps.

Here each angle is a full pass of the flow-line filler with a *ceiling* on how
much tone it may take.  A pass stops when it has filled its share and hands
what is left to the next angle, so the number of directions visible at any
point rises smoothly with darkness instead of snapping between bands.

The angles are offsets from the local flow direction rather than from the
page, so the hatching wraps around a cheek or a shoulder the way an engraver
would cut it.  Setting ``flow_coherence`` to 0 pins the field to a constant
angle and gives the classic straight cross-hatch instead.
"""

from __future__ import annotations

import numpy as np

from ..flow import FlowField
from ..geometry import InkBuffer
from . import flowlines


def generate(value, coverage, canvas, pen, cfg, features=None, form=None):
    px_per_mm = canvas.px_per_mm
    pen_px = max(0.4, pen.width_mm * px_per_mm)

    layers = int(np.clip(cfg.hatch_layers, 1, 5))
    offsets = _offsets(layers, cfg.hatch_spread_deg)

    base = flow_field(value, cfg, form)
    ink = InkBuffer(coverage, pen_px, max_spacing_px=26.0, features=features)

    # How much coverage one direction may contribute before the next takes
    # over.  Too high and the first angle darkens everything on its own and
    # no cross-hatch appears; too low and even pale skin gets four layers.
    share = float(np.clip(cfg.hatch_share, 0.05, 1.0))

    paths: list[np.ndarray] = []
    for i, offset in enumerate(offsets, start=1):
        ink.cap = min(1.0, i * share)
        rotated = _rotate(base, offset)
        paths.extend(flowlines.fill(rotated, ink, canvas, pen, cfg,
                                    seed=9000 + i * 137))
    ink.cap = None
    return paths, ink


def flow_field(value, cfg, form=None):
    """Hatching direction: mostly steady, only faintly tone-following.

    Tone coherence is kept low on purpose - following the image's own
    contours rings every bright patch - but a *depth*-derived form field is
    passed through at full weight, because that one describes the surface
    rather than the lighting.
    """
    from copy import copy
    sub = copy(cfg)
    sub.flow_coherence = cfg.hatch_coherence
    return flowlines.build_field(value, sub, form)


def _offsets(layers: int, spread_deg: float) -> list[float]:
    """Angle offsets for each pass, in degrees.

    Evenly spaced directions would put layer three back on top of layer one
    at four layers; the classic engraver's set is offset so each new angle
    cuts the ones before it at a wide angle.
    """
    if layers == 1:
        return [0.0]
    spread = float(np.clip(spread_deg, 20.0, 90.0))
    if layers == 2:
        return [0.0, spread]
    if layers == 3:
        return [0.0, spread, -spread]
    if layers == 4:
        return [0.0, spread, -spread, 90.0]
    return [0.0, spread, -spread, 90.0, spread * 0.5]


def _rotate(field: FlowField, degrees: float) -> FlowField:
    """Turn an orientation field by an angle.

    Orientation is carried as (cos 2t, sin 2t), so turning by t means turning
    the stored vector by 2t - the usual rotation, at double the angle.
    """
    if abs(degrees) < 1e-9:
        return field
    a = np.radians(degrees) * 2.0
    ca, sa = np.cos(a), np.sin(a)
    return FlowField(field.c2 * ca - field.s2 * sa,
                     field.c2 * sa + field.s2 * ca,
                     field.coherence)
