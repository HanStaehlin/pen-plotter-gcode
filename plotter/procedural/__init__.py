"""Procedurally generated subjects, drawn without a source photograph."""

from . import frog

#: name -> fn(seed) -> list of polylines in millimetres, y down
GENERATORS = {
    "frog": frog.build,
}
