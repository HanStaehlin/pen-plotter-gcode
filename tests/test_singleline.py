"""The single-line engine's building blocks, without the line network."""

import numpy as np
import pytest

from plotter import singleline


def _ring(cx, cy, r, n=80):
    t = np.linspace(0, 2 * np.pi, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def test_chained_contours_are_one_evenly_spaced_path():
    path = singleline.chain_contours([_ring(40, 40, 15), _ring(110, 40, 15)], spacing_px=2.0)
    step = np.hypot(*np.diff(path, axis=0).T)
    assert path.ndim == 2 and len(path) > 50
    # evenly resampled: no step much longer than the spacing except the one jump
    assert np.median(step) == pytest.approx(2.0, rel=0.3)
    assert (step > 10).sum() <= 1


def test_selection_drops_texture_and_isolated_marks():
    dark = np.zeros((200, 200), np.float32)
    dark[100, 30:170] = 1.0                       # a long feature line
    dark[110, 30:170] = 1.0
    dark[20:23, 20:23] = 1.0                      # a speck of texture
    dark[190, 185:195] = 1.0                      # a short mark far from everything
    kept = singleline.select_contours(dark, detail=1.0, min_frac=0.03)
    lengths = [np.hypot(*np.diff(p, axis=0).T).sum() for p in kept]
    assert kept and min(lengths) > 0.03 * 200


def test_contour_target_skips_the_jumps():
    path = np.vstack([_ring(40, 40, 15), _ring(110, 40, 15)])
    dark = np.zeros((80, 160), np.float32)
    tgt = singleline.contour_target(path, dark, jump_px=4.0, network_weight=0.0)
    assert tgt[40, 75] == 0.0                     # nothing drawn along the jump between rings
    assert tgt[40, 25:28].max() > 0.3             # but the rings are there


def test_fit_keeps_one_open_line_near_the_target():
    pytest.importorskip("torch")
    path = singleline.chain_contours([_ring(50, 50, 25)], spacing_px=2.5)
    target = singleline.contour_target(path, np.zeros((100, 100), np.float32))
    line = singleline.fit(target, init=path, iterations=40)
    assert line.ndim == 2 and len(line) > len(path)
    r = np.hypot(line[:, 0] - 50, line[:, 1] - 50)
    assert np.median(np.abs(r - 25)) < 3.0


def test_far_groups_are_dropped_to_avoid_long_jumps():
    near_a = [_ring(40, 40, 12), _ring(70, 40, 12)]          # the subject: close together
    far = [_ring(180, 180, 5)]                                # small and far away
    groups = singleline._proximity_groups(near_a + far, reach=12.0)
    assert groups[0] == groups[1] != groups[2]
