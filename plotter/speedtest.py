"""A test sheet for finding how fast the plotter can safely go.

Acceleration ($120/$121) and junction deviation ($11) are the settings that
decide plot time on photo-derived paths, and the limit is mechanical: past a
point the steppers skip.  Skipped steps cannot be seen while plotting, but
they move everything drawn afterwards.  So each block draws a reference
square, works the machine hard - short dashes, a tight maze, circles, fast
pen-up moves - and draws the same square again.  Where the second square no
longer lands on the first, the block before is the limit.

Each part of each block is streamed and timed on its own, so the sheet also
measures what the time estimate misses (mostly a fixed cost per pen lift).
"""

from __future__ import annotations

import numpy as np

from . import gcode
from .config import Config

#: (acceleration mm/s^2, junction deviation mm), gentlest first
LEVELS = ((600, 0.01), (1000, 0.02), (1500, 0.04), (2000, 0.06), (3000, 0.10))
BLOCK_MM = 42.0
PITCH_MM = 56.0
LEFT_MM = 12.0
TOP_MM = 20.0          # below the labels


def _square(x0, y0, s):
    return np.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s], [x0, y0]])


def _dashes(x0, y0, w, h, n_x=8, n_y=6, length=2.0):
    """Many short strokes: costs one pen lift each and little else."""
    out = []
    for j in range(n_y):
        for i in range(n_x):
            x = x0 + (i + 0.5) * w / n_x
            y = y0 + (j + 0.5) * h / n_y
            out.append(np.array([[x - length / 2, y], [x + length / 2, y]]))
    return out


def _maze(x0, y0, side, order=4):
    """A Hilbert curve: nothing but sharp corners, the hardest case."""
    from .engines.hilbert import _hilbert_d2xy
    n = 1 << order
    hx, hy = _hilbert_d2xy(n, np.arange(n * n, dtype=np.int64))
    cell = side / n
    return [np.stack([x0 + (hx + 0.5) * cell, y0 + (hy + 0.5) * cell], axis=1)]


def _circles(x0, y0, side, r=4.0):
    t = np.linspace(0.0, 2.0 * np.pi, 73)
    out = []
    for cx in (x0 + side * 0.25, x0 + side * 0.75):
        for cy in (y0 + side * 0.25, y0 + side * 0.75):
            out.append(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1))
    return out


def _sprint(x0, y0, s):
    """Fast pen-up moves corner to corner: acceleration on the rapids."""
    corners = [(x0, y0), (x0 + s, y0 + s), (x0 + s, y0), (x0, y0 + s)] * 3
    return corners


def _lettering(text, centre_x, top_y, height=4.0):
    from .procedural.naming import lettering
    return lettering(text, height, centre_x, top_y)


def _machine(paths, cfg):
    """Page millimetres (y down) -> machine coordinates."""
    flip = -1.0 if cfg.page.negative_y else 1.0
    return [np.stack([p[:, 0], flip * p[:, 1]], axis=1) for p in paths]


def _program(paths, cfg, rapids=()):
    """G-code for pen-down paths, optionally followed by pen-up rapids."""
    text = gcode.write(_machine(paths, cfg), cfg) if paths else ""
    if rapids:
        flip = -1.0 if cfg.page.negative_y else 1.0
        moves = [cfg.machine.pen_up_cmd, "G90"] + [
            f"G0 X{x:.3f} Y{flip * y:.3f}" for x, y in rapids] + ["G0 X0 Y0"]
        text += "\n".join(moves) + "\n"
    return text


def build(cfg: Config | None = None, levels=LEVELS) -> list[dict]:
    """The test as a list of blocks, each a list of separately timed parts."""
    cfg = cfg or Config()
    blocks = []
    for i, (accel, jd) in enumerate(levels):
        x0 = LEFT_MM + i * PITCH_MM
        y0 = TOP_MM
        s = BLOCK_MM
        inner = s - 6.0
        half = inner / 2.0
        ix, iy = x0 + 3.0, y0 + 3.0
        est = Config()
        est.machine.accel_mm_s2 = accel
        est.machine.junction_deviation = jd
        est.machine.lift_overhead_s = 0.0      # what the test is measuring
        parts = []
        for name, paths, rapids in (
            ("label", _lettering(f"{accel} / {jd:g}", x0 + s / 2, y0 - 8.0), ()),
            ("square", [_square(x0, y0, s)], ()),
            ("dashes", _dashes(ix, iy, inner, half - 1.0), ()),
            ("maze", _maze(ix, iy + half + 1.0, half - 1.0), ()),
            ("circles", _circles(ix + half + 1.0, iy + half + 1.0, half - 1.0), ()),
            ("sprint", [], _sprint(x0, y0, s)),
            ("square again", [_square(x0, y0, s)], ()),
        ):
            prog = _program(paths, cfg, rapids)
            mm = _machine(paths, cfg)
            e = gcode.estimate_time(mm, est) if mm else {"total_s": 0.0, "pen_lifts": 0}
            parts.append({"name": name, "program": prog,
                          "estimate_s": e["total_s"], "lifts": e["pen_lifts"]})
        blocks.append({"accel": accel, "junction_deviation": jd, "parts": parts})
    return blocks


def summarise(results: list[dict]) -> dict:
    """Per-level times, and the per-lift cost the estimate is missing.

    The dashes are almost all pen lifts, so their excess over the estimate,
    divided by their count, is the lift overhead.
    """
    levels = {}
    overheads = []
    for r in results:
        lv = levels.setdefault((r["accel"], r["junction_deviation"]),
                               {"accel": r["accel"], "junction_deviation": r["junction_deviation"],
                                "actual_s": 0.0, "estimate_s": 0.0})
        lv["actual_s"] += r["actual_s"]
        lv["estimate_s"] += r["estimate_s"]
        if r["name"] == "dashes" and r["lifts"]:
            overheads.append((r["actual_s"] - r["estimate_s"]) / r["lifts"])
    return {"levels": list(levels.values()),
            "lift_overhead_s": float(np.median(overheads)) if overheads else None}
