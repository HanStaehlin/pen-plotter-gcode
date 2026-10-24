"""The rendered-plot score, the canvas and the ink accumulator."""

import numpy as np
import pytest

from plotter import quality
from plotter.config import Config
from plotter.geometry import Canvas, InkBuffer


@pytest.fixture
def canvas():
    return Canvas(1000, 700, Config().page)


@pytest.mark.parametrize("pen", [0.4, 0.2, 0.1])
@pytest.mark.parametrize("spacing", [4.0, 1.6, 0.8])
def test_render_draws_the_pen_at_its_true_width(canvas, pen, spacing):
    # OpenCV draws thick lines wider than asked; the renderer must not, or
    # every plot looks darker than it will on paper.
    ppm = canvas.px_per_mm
    rows = [canvas.to_mm(np.array([[50 * ppm, y * ppm], [200 * ppm, y * ppm]]))
            for y in np.arange(20, 150, spacing)]
    img, k = quality.render(rows, canvas, pen)
    patch = img[int(40 * k):int(120 * k), int(80 * k):int(170 * k)]
    assert patch.mean() == pytest.approx(pen / spacing, rel=0.06)


def test_assess_scores_an_empty_plot_by_the_ink_it_misses(canvas):
    target = np.full((canvas.h_px, canvas.w_px), 0.4, np.float32)
    look = quality.assess([], canvas, target, 0.4)
    assert look["tone_error"] == pytest.approx(0.4, abs=0.01)


def test_canvas_round_trip(canvas):
    pts = np.array([[0, 0], [123.5, 456.25], [999, 699]], float)
    assert np.allclose(canvas.to_px(canvas.to_mm(pts)), pts)
    mm = canvas.to_mm(pts)
    assert (mm[:, 1] <= 0).all()               # negative-Y machine convention


def _hatch(buf, horizontal, n=100):
    for i in range(1, n, 2):
        line = np.array([[2, i], [n - 2, i]], np.float32)
        buf.deposit(line if horizontal else line[:, ::-1].copy())


def test_crossing_layers_combine_as_a_union():
    # Two 50% layers cover 75% of the paper, not 100%.
    target = np.full((100, 100), 0.5, np.float32)
    union = InkBuffer(target, 1.0)
    _hatch(union, True)
    union.new_layer()
    _hatch(union, False)
    linear = InkBuffer(target, 1.0)
    _hatch(linear, True)
    _hatch(linear, False)
    inner = (slice(10, 90), slice(10, 90))
    one_layer = InkBuffer(target, 1.0)
    _hatch(one_layer, True)
    assert one_layer.ink[inner].mean() == pytest.approx(0.5, abs=0.06)
    assert union.ink[inner].mean() == pytest.approx(0.75, abs=0.06)
    assert linear.ink[inner].mean() == pytest.approx(1.0, abs=0.08)
