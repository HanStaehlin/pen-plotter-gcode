"""The subject mask's decisions, with the segmentation model stubbed out."""

import numpy as np
import pytest

from plotter import subject


@pytest.fixture
def fake_mask(monkeypatch):
    def use(m):
        monkeypatch.setattr(subject, "raw_mask", lambda img: m)
    return use


def test_a_clear_subject_gives_a_soft_mask(fake_mask):
    m = np.zeros((200, 300), np.float32)
    m[50:150, 100:200] = 1.0
    fake_mask(m)
    out = subject.mask(np.zeros((200, 300, 3), np.uint8))
    assert out is not None and out.shape == (200, 300)
    assert out[100, 150] > 0.95 and out[10, 10] < 0.05
    assert 0.2 < out[100, 100] < 0.8               # feathered edge


@pytest.mark.parametrize("share", [0.01, 0.95])
def test_no_single_subject_means_draw_everything(fake_mask, share):
    m = np.zeros((100, 100), np.float32)
    m.flat[: int(share * m.size)] = 1.0            # a landscape's rock, or the whole frame
    fake_mask(m)
    assert subject.mask(np.zeros((100, 100, 3), np.uint8)) is None


def test_backdrop_lines_are_dropped_silhouette_kept():
    near = np.zeros((100, 100), np.float32)
    near[:, 50:] = 1.0
    on = np.array([[70, 10], [70, 90]], float)
    off = np.array([[10, 10], [10, 90]], float)
    edge = np.array([[49, 10], [51, 90]], float)
    lines, scores = subject.keep_lines([on, off, edge], [1, 2, 3], near, focus=1.0)
    assert scores == [1, 3]


def test_weight_keeps_subject_and_scales_backdrop():
    w = subject.weight(np.array([0.0, 1.0], np.float32), background=0.2)
    assert np.allclose(w, [0.2, 1.0])
