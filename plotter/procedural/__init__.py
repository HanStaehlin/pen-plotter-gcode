"""Procedurally generated subjects, drawn without a source photograph."""

from functools import partial

from . import frog

#: name -> fn(seed) -> list of polylines in millimetres, y down
GENERATORS = {
    "frog": partial(frog.build, species="frog"),
    "toad": partial(frog.build, species="toad"),
    "treefrog": partial(frog.build, species="treefrog"),
    "frog-above": frog.build_topdown,
}
