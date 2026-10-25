"""Photograph to pen-plotter G-code.

    from plotter import Config, convert, load_image

    cfg = Config()
    cfg.render.engine = "flowlines"
    result = convert(load_image("photo.jpg"), cfg)
    open("photo.gcode", "w").write(result.gcode())
"""

from .config import (Config, MachineConfig, PageConfig, PAPER_SIZES,
                     PathConfig, PenConfig, RenderConfig, ToneConfig)
from .engines import ENGINES
from .geometry import Canvas, InkBuffer
from .pipeline import Result, convert, draw_procedural, draw_text
from .tone import cutout, decode_image, load_image

__all__ = [
    "Config", "MachineConfig", "PageConfig", "PAPER_SIZES", "PathConfig",
    "PenConfig", "RenderConfig", "ToneConfig", "ENGINES", "Canvas",
    "InkBuffer", "Result", "convert", "draw_procedural", "draw_text", "cutout", "decode_image", "load_image",
]
