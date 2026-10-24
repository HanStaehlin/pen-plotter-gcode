"""Welding, ordering, bridging and simplifying strokes."""

import numpy as np

from plotter import paths as P


def seg(a, b):
    return np.array([a, b], dtype=float)


def test_merge_welds_ends_within_tolerance_only():
    a = seg((0, 0), (10, 0))
    b = seg((10.4, 0), (20, 0))         # 0.4 mm gap
    c = seg((30, 0), (40, 0))           # 10 mm gap
    merged = P.merge([a, b, c], tolerance=1.0)
    assert len(merged) == 2
    assert sum(len(p) for p in merged) == 6


def test_merge_may_reverse_a_stroke_to_join_it():
    a = seg((0, 0), (10, 0))
    b = seg((20, 0), (10.2, 0))          # its end meets a's end
    assert len(P.merge([a, b], tolerance=1.0)) == 1


def test_drop_short():
    kept = P.drop_short([seg((0, 0), (0.2, 0)), seg((0, 0), (5, 0))], 0.4)
    assert len(kept) == 1 and P.path_length(kept[0]) == 5


def test_sorting_cuts_travel_and_keeps_every_stroke():
    rng = np.random.default_rng(3)
    strokes = [seg(p, p + rng.normal(0, 2, 2)) for p in rng.uniform(0, 200, (60, 2))]
    ordered = P.sort_paths(strokes)
    assert len(ordered) == len(strokes)
    assert P.travel_length(ordered) < 0.5 * P.travel_length(strokes)
    assert np.isclose(P.total_length(ordered), P.total_length(strokes))


def test_bridge_only_where_the_sampler_says_ink():
    a, b = seg((0, 0), (10, 0)), seg((12, 0), (20, 0))
    dark = lambda pts: np.ones(len(pts))
    light = lambda pts: np.zeros(len(pts))
    assert len(P.bridge([a, b], 3.0, dark, 0.3)) == 1
    assert len(P.bridge([a, b], 3.0, light, 0.3)) == 2
    assert len(P.bridge([a, b], 1.0, dark, 0.3)) == 2     # gap too long


def test_simplify_drops_collinear_points_and_keeps_corners():
    line = np.stack([np.linspace(0, 10, 50), np.zeros(50)], axis=1)
    assert len(P.simplify(line, 0.05)) == 2
    corner = np.array([[0, 0], [5, 0], [5, 5]], float)
    assert len(P.simplify(corner, 0.05)) == 3
