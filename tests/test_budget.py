"""Stroke scores, pruning and the time limit."""

import numpy as np

from plotter import budget, gcode
from plotter.config import Config


def test_line_score_rewards_dark_and_long():
    assert budget.line_score(0.9, 20) > budget.line_score(0.5, 20)
    assert budget.line_score(0.9, 20) > budget.line_score(0.9, 1)
    # a 1 mm fragment is worth well under half a long line
    assert budget.line_score(0.9, 1) < 0.5 * budget.line_score(0.9, 30)


def _strokes(n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x, y = rng.uniform(10, 280), -rng.uniform(10, 200)
        out.append(np.array([[x, y], [x + rng.uniform(3, 30), y]]))
    return out, list(rng.uniform(0, 1, n))


def test_pruning_drops_only_weak_strokes():
    strokes, scores = _strokes()
    kept, info = budget.select(strokes, scores, Config(), finish=lambda ps: ps,
                               prune_below=0.5)
    assert len(kept) == sum(s >= 0.5 for s in scores)
    assert info["pruned"] == sum(s < 0.5 for s in scores)


def test_time_limit_is_met_by_dropping_the_weakest():
    cfg = Config()
    strokes, scores = _strokes(120)
    full = gcode.estimate_time(strokes, cfg)["total_s"]
    minutes = 0.4 * full / 60
    kept, info = budget.select(strokes, scores, cfg, finish=lambda ps: ps,
                               minutes=minutes)
    assert gcode.estimate_time(kept, cfg)["total_s"] <= minutes * 60
    assert info["budget_cut"] > 0
    # the best stroke survives, the worst does not
    keep = {tuple(p[0]) for p in kept}
    assert tuple(strokes[int(np.argmax(scores))][0]) in keep
    assert tuple(strokes[int(np.argmin(scores))][0]) not in keep


def test_lighten_redraws_the_whole_picture_lighter():
    cfg = Config()
    strokes, _ = _strokes(200)

    def draw(ink):                         # an engine: less ink, fewer strokes
        return strokes[: int(len(strokes) * ink)]

    full = gcode.estimate_time(strokes, cfg)["total_s"]
    out, info = budget.lighten(draw, draw(1.0), cfg, lambda ps: ps, 0.5 * full / 60)
    assert out is not None
    assert gcode.estimate_time(out, cfg)["total_s"] <= 0.5 * full
    assert 0.2 < info["ink"] < 0.6


def test_lighten_says_so_when_nothing_fits():
    cfg = Config()
    strokes, _ = _strokes(50)
    fixed = strokes[:40]                   # an overlay that never gets lighter
    out, _ = budget.lighten(lambda ink: fixed, strokes, cfg, lambda ps: ps, 0.01)
    assert out is None
