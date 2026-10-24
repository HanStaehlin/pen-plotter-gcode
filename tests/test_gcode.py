"""G-code output and the plot-time estimate."""

import numpy as np
import pytest

from plotter import gcode
from plotter.config import Config


def _line(length):
    return np.array([[10.0, -10.0], [10.0 + length, -10.0]])


def test_header_brackets_cannot_end_the_comment_early():
    # GRBL ends a comment at the first ")"; "Bettina (frog)" once left a
    # stray ")" that stopped every plot on line 1.
    text = gcode.write([_line(5)], Config(), "Bettina (frog)")
    assert text.splitlines()[0] == "(Bettina [frog])"


def test_every_stroke_is_framed_by_pen_down_and_up():
    cfg = Config()
    paths = [_line(5), _line(8) + [0, -5], _line(3) + [0, -10]]
    lines = gcode.write(paths, cfg).splitlines()
    down = [i for i, ln in enumerate(lines) if ln == cfg.machine.pen_down_cmd]
    up = [i for i, ln in enumerate(lines) if ln == cfg.machine.pen_up_cmd]
    assert len(down) == len(paths)
    # pen goes up before anything else, after every stroke, then home
    assert up[0] < down[0] and len(up) == len(paths) + 1
    assert lines[-2].startswith("G0 X0 Y0") or lines[-1].startswith("G0 X0 Y0")
    for d in down:
        nxt_up = min(u for u in up if u > d)
        assert all(ln.startswith(("G1", "G4")) for ln in lines[d + 1:nxt_up])


def test_straight_line_time_is_the_trapezoid():
    cfg = Config()
    m = cfg.machine
    v = m.feed_draw / 60.0
    a = m.accel_mm_s2 * m.accel_efficiency
    length = 100.0
    # two ramps of v/a each covering v^2/2a, cruise over the rest
    expected = 2 * v / a + (length - v * v / a) / v
    got = gcode.estimate_time([_line(length)], cfg)["draw_s"]
    assert got == pytest.approx(expected, rel=0.01)


def test_corners_cost_time():
    cfg = Config()
    straight = _line(60)
    xs = np.arange(0, 61, 2.0)
    zigzag = np.stack([10 + xs, -10 - 2 * (np.arange(len(xs)) % 2)], axis=1)
    t_straight = gcode.estimate_time([straight], cfg)["draw_s"]
    t_zig = gcode.estimate_time([zigzag], cfg)["draw_s"]
    assert t_zig > 2 * t_straight


def test_each_lift_pays_the_measured_overhead():
    a, b = Config(), Config()
    b.machine.lift_overhead_s = a.machine.lift_overhead_s + 0.5
    paths = [_line(5) + [0, -3 * i] for i in range(10)]
    ta = gcode.estimate_time(paths, a)
    tb = gcode.estimate_time(paths, b)
    assert ta["pen_lifts"] == 10
    assert tb["total_s"] - ta["total_s"] == pytest.approx(10 * 0.5)


def test_lower_effective_acceleration_means_longer_plots():
    fast, slow = Config(), Config()
    slow.machine.accel_efficiency = 0.5 * fast.machine.accel_efficiency
    paths = [np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]], float) * 3 - [0, 20]]
    assert gcode.estimate_time(paths, slow)["total_s"] > gcode.estimate_time(paths, fast)["total_s"]
