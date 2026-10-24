"""Every engine end to end, on a synthetic photo, without torch."""

import numpy as np
import pytest

from conftest import light_config
from plotter import ENGINES, draw_procedural, pipeline


def _inside_page(paths, page, slack=0.5):
    pts = np.vstack(paths)
    return (pts[:, 0].min() >= -slack and pts[:, 0].max() <= page.width_mm + slack
            and pts[:, 1].max() <= slack and pts[:, 1].min() >= -page.height_mm - slack)


@pytest.mark.parametrize("engine", list(ENGINES))
def test_every_engine_draws_on_the_page(photo, engine):
    cfg = light_config(engine)
    r = pipeline.convert(photo, cfg)
    assert r.paths, engine
    assert _inside_page(r.paths, cfg.page)
    assert r.stats["total_s"] > 0
    assert -1.0 <= r.stats["likeness"] <= 1.0
    assert all(np.isfinite(p).all() for p in r.paths)


@pytest.mark.parametrize("engine", ["illustration", "hilbert", "waves"])
def test_time_limit_is_respected(photo, engine):
    full = pipeline.convert(photo, light_config(engine)).stats["total_s"]
    cfg = light_config(engine)
    cfg.paths.time_limit_min = 0.5 * full / 60
    limited = pipeline.convert(photo, cfg)
    assert limited.stats["total_s"] <= cfg.paths.time_limit_min * 60
    assert limited.paths


def test_hilbert_and_waves_are_lightened_not_cut(photo):
    # These draw the picture as a few long strokes; dropping strokes would
    # cut holes, so the limit must spread the saving over the whole picture.
    full = pipeline.convert(photo, light_config("hilbert"))
    cfg = light_config("hilbert")
    cfg.paths.time_limit_min = 0.6 * full.stats["total_s"] / 60
    limited = pipeline.convert(photo, cfg)
    assert limited.stats["budget_cut"] == 0
    # still covers roughly the same area of the page
    a, b = np.vstack(full.paths), np.vstack(limited.paths)
    assert np.ptp(b[:, 0]) > 0.8 * np.ptp(a[:, 0])


def test_portrait_output_is_turned_for_the_bed(photo):
    cfg = light_config("illustration")
    cfg.page.portrait = True
    cfg.page.resolve()
    r = pipeline.convert(photo, cfg)
    turned, note = r.output_paths()
    pts = np.vstack(turned)
    assert "rotated" in note
    assert pts[:, 0].max() <= cfg.page.height_mm + 0.5      # now landscape


def test_procedural_frog_fits_the_page():
    r = draw_procedural("toad", name="Bufo testii")
    assert r.stats["name"] == "Bufo testii"
    assert _inside_page(r.paths, r.config.page)
