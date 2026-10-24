"""Procedurally generated subjects, drawn without a source photograph."""

from functools import partial

from . import frog

#: name -> fn(seed) -> list of polylines in millimetres, y down.
#: "frog" picks a family at random; the rest pin one family.
GENERATORS = {"frog": frog.build}
GENERATORS.update({f: partial(frog.build, family=f) for f in frog.FAMILIES})
