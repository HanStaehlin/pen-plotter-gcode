"""Shared fixtures.

Nothing here downloads a model or needs a GPU: pipeline tests use the
classical line tracer and switch depth off, and plotter tests run against
tools/fake_grbl.py, a pretend GRBL 0.9i on a pseudo-terminal.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from plotter.config import Config  # noqa: E402


def light_config(engine: str = "illustration", resolution: int = 300) -> Config:
    """A configuration that runs in well under a second without torch."""
    cfg = Config()
    cfg.render.engine = engine
    cfg.tone.resolution = resolution
    cfg.render.form_weight = 0.0
    cfg.render.subject_focus = 0.0
    cfg.render.line_method = "coherent"
    cfg.render.illustration_rim = 0.0
    cfg.render.illustration_focus = 0.0
    cfg.render.tsp_points = 1500
    return cfg


@pytest.fixture
def photo() -> np.ndarray:
    """A small synthetic photograph: a gradient, a dark disc, a few lines."""
    h, w = 240, 320
    y, x = np.mgrid[0:h, 0:w]
    img = (40 + 200 * x / w).astype(np.float32)
    img[(x - 200) ** 2 + (y - 120) ** 2 < 60 ** 2] = 25
    img[60:64, 20:150] = 10
    img[150:200, 40:44] = 10
    rgb = np.stack([img, img * 0.9, img * 0.8], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


@pytest.fixture(autouse=True)
def _private_machine_profile(monkeypatch, tmp_path):
    """Never read or write the real machine.json from a test."""
    from plotter import grbl
    monkeypatch.setattr(grbl, "PROFILE", str(tmp_path / "machine.json"))
    monkeypatch.setattr(grbl, "BOOT_S", 0.1)


@pytest.fixture
def fake():
    from fake_grbl import FakeGrbl
    return FakeGrbl(seconds_per_line=0.0002)


@pytest.fixture
def machine(fake):
    from plotter.grbl import Grbl
    g = Grbl()
    g.connect(fake.path)
    yield g
    g.disconnect()


def wait_job(g, timeout: float = 20.0) -> dict:
    """Block until the machine's current job leaves the running state."""
    import time
    end = time.time() + timeout
    while g.job and g.job["state"] == "running" and time.time() < end:
        time.sleep(0.02)
    return g.job
