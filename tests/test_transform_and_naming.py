"""Turning portrait output for the bed, and frog names."""

import numpy as np
import pytest

from plotter import transform
from plotter.procedural import frog, naming


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


TRIANGLE = np.array([[20, -30], [80, -40], [50, -150]], float)


@pytest.mark.parametrize("clockwise", [False, True])
def test_rotation_is_a_rotation_not_a_mirror(clockwise):
    # Swapping X and Y would mirror the drawing - unreadable lettering.
    (turned,), w, h = transform.rotate_paths([TRIANGLE], 210, 297, clockwise)
    assert (w, h) == (297, 210)
    assert np.sign(_signed_area(turned)) == np.sign(_signed_area(TRIANGLE))


def test_rotating_there_and_back_is_the_identity():
    (turned,), w, h = transform.rotate_paths([TRIANGLE], 210, 297, clockwise=False)
    (back,), _, _ = transform.rotate_paths([turned], w, h, clockwise=True)
    assert np.allclose(back, TRIANGLE)


def test_rotate_gcode_writes_both_coordinates_on_every_move():
    # Coordinates are modal; after a turn, a move that set only X changes Y too.
    text = "G90\nG0 X10 Y-20\nG1 X30\nG1 Y-40\n"
    out, info = transform.rotate_gcode(text, 210, 297)
    moves = [ln for ln in out.splitlines() if ln.startswith(("G0", "G1"))]
    assert all("X" in ln and "Y" in ln for ln in moves)


def test_a_name_is_its_own_seed():
    assert naming.seed_from_name("Bufo hannesii") == naming.seed_from_name("  bufo HANNESII. ")
    assert naming.seed_from_name("Bufo hannesii") != naming.seed_from_name("Bufo hannesia")


@pytest.mark.parametrize("genus,family", [("Glyphobufo", "toad"), ("Leptobates", "dartfrog"),
                                          ("Euryhyla", "treefrog"), ("Calorana", "truefrog"),
                                          ("Hydroceras", "hornedfrog"), ("Scaphopus", "clawedfrog")])
def test_genus_ending_names_the_family(genus, family):
    assert naming.family_of(genus + " x", frog.FAMILIES, frog.WEIGHTS) == family


def test_an_invented_name_reproduces_its_frog():
    shown, seed, fam = naming.resolve("frog", frog.FAMILIES, frog.WEIGHTS, seed=7)
    again = naming.resolve("frog", frog.FAMILIES, frog.WEIGHTS, name=shown)
    assert again == (shown, seed, fam)


def test_frogs_are_deterministic_and_varied():
    a = frog.build(5, "toad")
    b = frog.build(5, "toad")
    c = frog.build(6, "toad")
    assert a and len(a) == len(b) and all(np.allclose(p, q) for p, q in zip(a, b))
    assert len(a) != len(c) or not all(np.allclose(p, q) for p, q in zip(a, c))


def test_caption_is_single_stroke_lettering():
    strokes = naming.lettering("Bufo hannesii.", height=5.0, centre_x=0.0, top_y=0.0)
    assert len(strokes) > 10
    ys = np.vstack(strokes)[:, 1]
    assert ys.max() - ys.min() == pytest.approx(5.0, rel=0.05)
