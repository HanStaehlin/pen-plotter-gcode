"""Text set in single-stroke lettering."""

import numpy as np
import pytest

from plotter import draw_text, lettering
from plotter.config import Config

POEM = """# Sonnet XVIII

Shall I compare thee to a summer's day?
  Thou art more lovely and more temperate:"""


def _all(strokes):
    return np.vstack(strokes)


def test_accents_are_composed_from_base_letter_and_mark():
    assert lettering._decompose("ä") == ("a", ["̈"])
    assert lettering._decompose("é") == ("e", ["́"])
    assert lettering._decompose("ß") == ("ss", [])
    plain, _ = lettering.compose("Grauen", "technical", size_mm=6)
    umlaut, _ = lettering.compose("Gräuen", "technical", size_mm=6)
    assert len(umlaut) == len(plain) + 2            # the two dots of the ä


@pytest.mark.parametrize("style", list(lettering.STYLES))
def test_text_stays_inside_the_box(style):
    strokes, size = lettering.compose(POEM, style, width=200, height=120)
    pts = _all(strokes)
    assert pts[:, 0].min() >= -0.5 and pts[:, 0].max() <= 200.5
    assert pts[:, 1].min() >= -0.5 and pts[:, 1].max() <= 120.5
    assert lettering.MIN_BODY_MM <= size <= lettering.MAX_BODY_MM


def test_long_text_shrinks_and_wraps_to_fit():
    long = " ".join(["Shall I compare thee to a summer's day?"] * 12)
    strokes, size = lettering.compose(long, "book", width=150, height=100)
    assert size < lettering.MAX_BODY_MM
    assert _all(strokes)[:, 0].max() <= 150.5


def test_headings_are_larger_than_body():
    rows, _ = lettering._layout(lettering._parse(POEM), "calligraphy", 6.0, 250)
    assert rows[0][0].size == pytest.approx(6.0 * lettering.HEADING_SCALE)
    assert rows[1][0].size == pytest.approx(6.0)
    assert rows[2][2] > rows[1][2]                  # the indented line starts further in


def test_nib_passes_are_one_continuous_stroke():
    stroke = np.array([[0.0, 0.0], [10.0, 0.0]])
    out = lettering.nib([stroke], width_mm=1.2, pen_mm=0.4)
    assert len(out) == 1                            # no extra pen lifts
    # wide across the nib angle, so the passes spread the stroke
    assert np.ptp(out[0][:, 1]) > 0.5


def test_draw_text_makes_gcode_on_the_page():
    cfg = Config()
    r = draw_text(POEM, cfg, style="calligraphy")
    pts = np.vstack(r.paths)
    assert pts[:, 0].min() >= cfg.page.margin_mm - 0.5
    assert pts[:, 0].max() <= cfg.page.width_mm - cfg.page.margin_mm + 0.5
    assert r.stats["pen_lifts"] > 10 and r.stats["body_mm"] > 0
    assert r.gcode("Grüezi (test)").splitlines()[0] == "(Gruezi [test])"


def test_empty_text_is_refused():
    with pytest.raises(ValueError):
        draw_text("   \n  ", Config())


def test_lines_shrink_before_they_wrap():
    # one long line on a narrow page: whole at a smaller size, not broken
    line = "Rough winds do shake the darling buds of May, and summer's lease"
    rows_whole = len(lettering._parse(line))
    strokes, size = lettering.compose(line, "book", width=180, height=150)
    size_w, wrap = lettering.fit_size(line, "book", 180, 150)
    assert not wrap and size == size_w and rows_whole == 1
    assert size < lettering.MAX_BODY_MM
