"""Procedurally generated subjects, drawn without a source photograph."""

import numpy as np

from . import frog, naming

#: subjects the app and CLI offer; "frog" means any family
SUBJECTS = ("frog",) + frog.FAMILIES


def generate(kind: str, name: str | None = None, seed: int | None = None):
    """Draw one named frog: returns (polylines in mm with y down, name).

    The name is lettered under the animal, fishdraw-style, and typing the
    same name back reproduces the same drawing.
    """
    if kind not in SUBJECTS:
        raise ValueError(f"unknown subject {kind!r}; expected one of {SUBJECTS}")
    shown, draw_seed, family = naming.resolve(kind, frog.FAMILIES, frog.WEIGHTS,
                                              name=name, seed=seed)
    lines = frog.build(draw_seed, family)
    pts = np.vstack(lines)
    width = pts[:, 0].max() - pts[:, 0].min()
    caption = naming.lettering(shown + ".", height=0.045 * width,
                               centre_x=(pts[:, 0].min() + pts[:, 0].max()) / 2,
                               top_y=pts[:, 1].max() + 0.07 * width)
    return lines + caption, shown
