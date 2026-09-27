"""Learned line drawing: Informative Drawings (Chan, Durand & Isola, CVPR 2022).

Every other line source in this package is hand-designed - Canny, flow-guided
DoG - and asks "is there an edge here".  Artists do not draw every edge; they
draw the ones that describe shape and identity, and leave out the rest.  This
network was trained to make exactly that choice: its loss rewards drawings
from which the photo's geometry (via depth) and meaning (via CLIP) can be
recovered, so it keeps an eye and drops a shadow boundary.

One forward pass of a small ResNet generator - well under a second on Metal.
Weights are downloaded on first use from the authors' Hugging Face Space and
cached; torch and huggingface_hub are optional dependencies, and
``available()`` reports False without them rather than failing on import.
Code and weights: https://github.com/carolineec/informative-drawings (MIT).
"""

from __future__ import annotations

import importlib.util

import cv2
import numpy as np

STYLES = {"contour": "model.pth", "sketch": "model2.pth"}
_CACHE: dict = {}


def available() -> bool:
    return (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("huggingface_hub") is not None)


def _build():
    import torch.nn as nn

    norm = nn.InstanceNorm2d

    class Residual(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), norm(c), nn.ReLU(True),
                nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), norm(c))

        def forward(self, x):
            return x + self.conv_block(x)

    class Generator(nn.Module):
        # Layer names mirror the published checkpoint so it loads unchanged.
        def __init__(self, blocks=3):
            super().__init__()
            self.model0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7),
                                        norm(64), nn.ReLU(True))
            down, c = [], 64
            for _ in range(2):
                down += [nn.Conv2d(c, c * 2, 3, stride=2, padding=1), norm(c * 2),
                         nn.ReLU(True)]
                c *= 2
            self.model1 = nn.Sequential(*down)
            self.model2 = nn.Sequential(*[Residual(c) for _ in range(blocks)])
            up = []
            for _ in range(2):
                up += [nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1,
                                          output_padding=1), norm(c // 2), nn.ReLU(True)]
                c //= 2
            self.model3 = nn.Sequential(*up)
            self.model4 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 1, 7),
                                        nn.Sigmoid())

        def forward(self, x):
            return self.model4(self.model3(self.model2(self.model1(self.model0(x)))))

    return Generator()


def _device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load(style: str = "contour"):
    if style in _CACHE:
        return _CACHE[style]
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("carolineec/informativedrawings", STYLES[style],
                           repo_type="space")
    net = _build()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    dev = _device()
    net.to(dev).eval()
    _CACHE[style] = (net, dev)
    return net, dev


def draw(image_rgb: np.ndarray, size: tuple[int, int], style: str = "contour",
         long_edge: int = 1024) -> np.ndarray:
    """Line drawing as 0..1 darkness (1 = full ink) at ``size`` = (w, h).

    The network is fully convolutional, so it runs at any size; the scale it
    sees sets how much detail it considers worth a line.  It was trained on
    images a few hundred pixels across, and on much larger inputs it starts
    drawing texture as if it were structure, hence the fixed working size.
    """
    import torch

    net, dev = load(style)
    h, w = image_rgb.shape[:2]
    s = long_edge / max(h, w)
    # Multiples of 4: two stride-2 downsamples must come back to the same size.
    iw, ih = max(4, int(round(w * s / 4)) * 4), max(4, int(round(h * s / 4)) * 4)
    x = cv2.resize(image_rgb, (iw, ih), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1) / 255.0)[None].to(dev)
    with torch.no_grad():
        out = net(x)[0, 0].float().cpu().numpy()      # 1 = paper, 0 = ink
    dark = np.clip(1.0 - out, 0.0, 1.0)
    return cv2.resize(dark, size, interpolation=cv2.INTER_AREA).astype(np.float32)
