"""Spending plot time where it shows: stroke scores, pruning, time budget.

Plot time is what the user actually waits for, so it is treated as a budget
rather than a readout.  Every stroke gets a score for how much it adds to
the picture; strokes that add almost nothing are dropped outright (the
delete step of Turk & Banks 1996), and with a time limit the best strokes
are kept until the minutes run out.

Lines and shading share one scale.  Measured on the test photos, the line
layer is roughly nine tenths of the plot time, so "lines first, then
hatching" would leave no room to choose: a faint 3 mm squiggle has to be
able to lose to the hatching in the deepest shadow.
"""

from __future__ import annotations

import numpy as np

from . import gcode


def line_score(strength: float, length_mm: float) -> float:
    """A line is worth its drawn darkness, discounted when it is short.

    Every stroke costs a lift and two dwells however short it is, and a
    fragment of a couple of millimetres reads as a speck rather than as part
    of a contour, so length matters up to about a centimetre.
    """
    return float(strength) * (1.0 - np.exp(-length_mm / LINE_LENGTH_MM))


def shade_score(target: float) -> float:
    """A hatch stroke is worth the darkness of the shadow it is shading."""
    return SHADE_WEIGHT * float(target)


LINE_LENGTH_MM = 4.0
SHADE_WEIGHT = 1.5


def default_scores(px_paths, coverage) -> list[float]:
    """Scores for engines that don't rank their own strokes: darkness under them."""
    h, w = coverage.shape
    out = []
    for p in px_paths:
        p = np.asarray(p)
        ix = np.clip(p[:, 0].astype(np.int32), 0, w - 1)
        iy = np.clip(p[:, 1].astype(np.int32), 0, h - 1)
        out.append(float(coverage[iy, ix].mean()) if len(p) else 0.0)
    return out


def lighten(draw, first, cfg, finish, minutes: float, note=lambda m: None):
    """Fit ``minutes`` by redrawing everything lighter, for unranked engines.

    ``draw(ink)`` re-runs the engine on ``ink`` x the tone target and returns
    millimetre paths; ``first`` is its output at full ink.  Plot time grows
    roughly in proportion to ink, so each guess scales by target / time and
    the bracket keeps it honest.  Returns (finished paths, info dict).
    """
    target = minutes * 60.0
    out = finish(first)
    t = gcode.estimate_time(out, cfg)["total_s"]
    info = {"pruned": 0, "budget_cut": 0, "ink": 1.0}
    if t <= target:
        return out, info
    lo, hi, ink = 0.0, 1.0, 1.0
    best = None
    for _ in range(7):
        guess = ink * target / max(t, 1e-9) * 0.98
        ink = float(np.clip(guess, lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)))
        note(f"fitting {minutes:g} min budget (ink {ink:.2f})")
        out = finish(draw(ink))
        t = gcode.estimate_time(out, cfg)["total_s"]
        if t <= target:
            lo, best = ink, (out, ink)
            if t >= 0.95 * target:
                break
        else:
            hi = ink
        if hi - lo < 0.01:
            break
    if best is None:
        # Nothing the engine draws fits - whatever is left over (an outline
        # overlay) already takes longer than the limit.  The caller falls
        # back to ranking strokes.
        return None, info
    info["ink"] = best[1]
    return best[0], info


def select(mm_paths, scores, cfg, finish, minutes: float = 0.0,
           prune_below: float = 0.0, note=lambda m: None):
    """Prune weak strokes, then keep the best that fit in ``minutes``.

    ``finish`` is the weld/order/bridge pass; the budget is checked on its
    output because welding and bridging remove lifts and change the real
    cost a lot.  Each stroke is first priced on its own, the best prefix
    that fits is finished and timed, and the ratio between the two corrects
    the next guess.  Returns (finished paths, info dict).
    """
    scores = np.asarray(scores, float)
    keep = np.flatnonzero(scores >= prune_below)
    info = {"pruned": int(len(mm_paths) - len(keep)), "budget_cut": 0}
    if minutes <= 0:
        return finish([mm_paths[i] for i in keep]), info

    order = keep[np.argsort(-scores[keep], kind="stable")]
    m = cfg.machine
    dwell = m.pen_up_dwell + m.pen_down_dwell
    v = m.feed_draw / 60.0
    cost = np.array([gcode._path_time(np.asarray(mm_paths[i], float), v,
                                      max(1.0, m.accel_mm_s2 * m.accel_efficiency),
                                      m.junction_deviation)[1] + dwell
                     for i in order])
    cum = np.cumsum(cost)
    target = minutes * 60.0

    ratio, best = 1.0, None
    lo, hi = 0, len(order)            # n known to fit / known not to fit (+1)
    n = len(order)
    for _ in range(8):
        note(f"fitting {minutes:g} min budget")
        out = finish([mm_paths[i] for i in np.sort(order[:n])])
        t = gcode.estimate_time(out, cfg)["total_s"]
        if t <= target:
            best, lo = (out, n), n
            if n == len(order) or t >= 0.97 * target:
                break
        else:
            hi = n - 1
        ratio = t / max(cum[n - 1], 1e-9) if n else 1.0
        guess = int(np.searchsorted(cum, target / ratio, side="right"))
        if t > target:
            guess = min(guess, n - max(1, n // 50))
        n = int(np.clip(guess, lo + 1 if lo < hi else lo, max(hi, lo)))
        if best is not None and n == best[1]:
            break
    if best is None:
        best = (finish([]), 0)
    info["budget_cut"] = int(len(order) - best[1])
    return best[0], info
