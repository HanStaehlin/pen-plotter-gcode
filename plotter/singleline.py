"""One smooth line through the whole picture, fitted by optimisation.

After Magne et al., "Single-Line Drawing Generation via Semantics-Driven
Optimization" (Computer Graphics Forum 2026, MIT-licensed code at
github.com/tanguymagne/SLDgen): start from a TSP tour, make it one closed
spline, and move its control points by gradient descent until the drawing
matches a target, with terms that keep the line from bunching up or
crossing itself.  The paper steers the curve with Stable Diffusion 3.5,
which needs a 24 GB GPU; here the target is a darkness map - the learned
line drawing plus the photo's tone, or any image of line art - and the
whole fit runs in about a minute on a laptop GPU.

The rendering is what makes the gradients useful.  Points sampled densely
along the curve are splatted into an ink image, each carrying ink in
proportion to the length of curve it stands for, and both that image and the
target are blurred the way the eye blurs fine lines.  A curve that wanders
off the target then sees a smooth pull back toward it, rather than nothing
until it happens to touch a dark pixel.
"""

from __future__ import annotations

import numpy as np

SAMPLES_PER_SPAN = 4
#: Catmull-Rom basis: rows are t^0..t^3, columns the four control points.
_CR = 0.5 * np.array([[0, 2, 0, 0], [-1, 0, 1, 0], [2, -5, 4, -1], [-1, 3, -3, 1]], float)


def _device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _gauss_kernel(sigma):
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def initial_tour(target: np.ndarray, n_points: int, seed: int = 0) -> np.ndarray:
    """Points spread by darkness and joined into one closed tour (px, x/y)."""
    from .engines import tsp
    np.random.seed(seed)
    pts = tsp._stipple(target.astype(np.float32), n_points, 8)
    if len(pts) < 8:
        return pts
    return pts[tsp._refine(pts, tsp._greedy_tour(pts))].astype(np.float32)


def select_contours(dark: np.ndarray, near: np.ndarray | None = None, detail: float = 1.0,
                    threshold: float = 0.22, min_frac: float = 0.03, isolated_frac: float = 0.08):
    """The contours a continuous-line drawing should trace, most important first.

    A continuous-line artist abstracts: the silhouette and the features in
    long strokes, never texture.  Fur, hair and fabric come out of the line
    network as hundreds of short marks, and a single line chasing them all
    scribbles, so contours shorter than ``min_frac`` of the picture are
    dropped and the rest ranked by darkness times length - weighted toward
    the subject when a depth map says where it is - until the length budget
    (``detail`` x 14 picture widths) is spent.

    The kept contours are then grouped by proximity, and groups far from
    the main one are dropped: reaching them would take a long straight jump
    across the page, the one mark that gives a single-line drawing away.
    """
    from .engines import outline
    h, w = dark.shape
    size = max(h, w)
    mask = outline.fill_holes(((dark > threshold) * 255).astype(np.uint8), 4)
    strokes = [p.astype(np.float64) for p in outline.trace(outline.skeleton(mask), min_px=3)
               if len(p) > 2]
    if not strokes:
        return []
    lens = np.array([np.hypot(*np.diff(p, axis=0).T).sum() for p in strokes])
    strength = np.array([outline._mean_along(dark, p) for p in strokes])
    subject = (np.array([outline._mean_along(near, p) for p in strokes])
               if near is not None else np.ones(len(strokes)))
    score = strength * lens * (0.05 + 0.95 * subject ** 2)
    score[lens < min_frac * size] = 0.0
    order = [i for i in np.argsort(-score) if score[i] > 0]
    budget, kept, used = detail * 14.0 * size, [], 0.0
    for i in order:
        if used + lens[i] > budget:
            break
        kept.append(i)
        used += lens[i]
    chosen = [strokes[i] for i in kept]
    if len(chosen) < 2:
        return chosen
    group = _proximity_groups(chosen, isolated_frac * size)
    totals = np.bincount(group, weights=score[kept])
    main = totals >= MAIN_GROUP_SHARE * totals.max()
    return [p for p, g in zip(chosen, group) if main[g]]


#: A group of contours is kept if it carries at least this share of the
#: biggest group's importance; smaller ones far away are not worth the jump.
MAIN_GROUP_SHARE = 0.35


def _proximity_groups(strokes, reach: float) -> np.ndarray:
    """Group label per stroke: strokes within ``reach`` of each other share one."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pts = np.vstack(strokes)
    owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(strokes)])
    pairs = cKDTree(pts).query_pairs(reach, output_type="ndarray")
    n = len(strokes)
    if len(pairs) == 0:
        return np.arange(n)
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def face_weight(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    """1 over faces (with hair and chin), falling off around them; None if none.

    In a portrait the face is what the single line must get right, and
    depth alone does not know that: behind the sitter a door frame can be as
    near and as high-contrast as the face.  OpenCV's stock frontal-face
    detector is enough to say where the face is.
    """
    import cv2
    W, H = size
    gray = cv2.cvtColor(cv2.resize(image_rgb, (W, H), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_RGB2GRAY)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                 minSize=(max(24, W // 12), max(24, W // 12)))
    if len(faces) == 0:
        return None
    m = np.zeros((H, W), np.float32)
    for x, y, fw, fh in faces:
        # the detector boxes eyes to mouth; widen to forehead, hair and chin
        cv2.ellipse(m, (int(x + fw / 2), int(y + fh * 0.45)),
                    (int(fw * 0.85), int(fh * 1.05)), 0, 0, 360, 1.0, -1)
        # neck and shoulders: the sitter, not the room behind them
        cv2.ellipse(m, (int(x + fw / 2), int(y + fh * 2.4)),
                    (int(fw * 1.5), int(fh * 1.1)), 0, 0, 360, 0.7, -1)
    return cv2.GaussianBlur(m, (0, 0), max(2.0, 0.08 * W))


def contour_target(path: np.ndarray, dark: np.ndarray, jump_px: float = 4.0,
                   network_weight: float = 0.15) -> np.ndarray:
    """What the fit aims at: the smoothed contours themselves, plus a little
    of the line network's drawing.

    Aiming at the network's raw drawing pulls the line back onto its pixel
    staircase; aiming at the smoothed contours keeps them smooth and leaves
    the fit free to do what it is for - bend the straight jumps between
    contours into the drawing and space out lines that bunch.
    """
    import cv2
    h, w = dark.shape
    own = np.zeros((h, w), np.float32)
    seg = np.hypot(*np.diff(path, axis=0).T)
    for piece in np.split(path, np.flatnonzero(seg > jump_px) + 1):
        if len(piece) > 1:
            cv2.polylines(own, [np.round(piece * 16).astype(np.int32)], False, 1.0, 1,
                          cv2.LINE_AA, shift=4)
    return np.clip((1 - network_weight) * own + network_weight * dark * (dark > 0.22),
                   0.0, 1.0).astype(np.float32)


def chain_contours(strokes, spacing_px: float = 2.5, smooth_px: float = 3.0) -> np.ndarray:
    """Contour strokes joined into one closed path, connectors as short as possible.

    This is how continuous-line drawings are made by hand - trace a feature,
    move on to the nearest next one without lifting - and it gives the fit a
    start that already follows the subject, where a tour through scattered
    points only follows tone.
    """
    from . import paths as P
    strokes = [np.asarray(s, float) for s in strokes if len(s) >= 2]
    if not strokes:
        return np.zeros((0, 2), np.float32)
    start = np.vstack(strokes).mean(axis=0)
    ordered = P.sort_paths(strokes, home=tuple(start), refine=True)
    path = np.vstack(ordered)
    # Even spacing: the fit moves control points, and evenly spaced ones
    # give it the same freedom everywhere.
    seg = np.hypot(*np.diff(path, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(8, int(t[-1] / spacing_px))
    u = np.linspace(0.0, t[-1], n)
    out = np.stack([np.interp(u, t, path[:, 0]), np.interp(u, t, path[:, 1])], 1)
    # Traced contours step along the pixel grid; a hand-drawn line does not.
    # Smoothing along the path removes the staircase before the fit starts.
    if smooth_px > 0 and len(out) > 8:
        from scipy.ndimage import gaussian_filter1d
        out = gaussian_filter1d(out, smooth_px / spacing_px, axis=0, mode="nearest")
    return out.astype(np.float32)


def fit(target: np.ndarray, n_points: int = 1500, iterations: int = 150,
        line_px: float = 0.8, view_sigma_px: float = 2.5, seed: int = 0,
        progress=None, init: np.ndarray | None = None, lr: float = 0.2,
        smooth_weight: float = 5e-2) -> np.ndarray:
    """Fit one open curve to ``target`` (H x W darkness, 0..1).

    ``line_px`` is the pen width in target pixels: it sets how much ink a unit
    of curve lays down, and so how long the line must be to carry the tone.
    The curve is open: a closed one has to come home from wherever it ends,
    and that last stretch is a line ruled straight across the drawing.
    Returns the curve as a polyline in target pixels (x, y).
    """
    import torch
    import torch.nn.functional as F

    dev = _device()
    h, w = target.shape
    tour = init if init is not None else initial_tour(target, n_points, seed)
    if len(tour) < 8:
        return tour
    ctrl = torch.tensor(tour, dtype=torch.float32, device=dev, requires_grad=True)
    basis = np.stack([[1, t, t * t, t ** 3] for t in np.arange(SAMPLES_PER_SPAN) / SAMPLES_PER_SPAN]) @ _CR
    basis = torch.tensor(basis, dtype=torch.float32, device=dev)          # S x 4

    k = torch.tensor(_gauss_kernel(view_sigma_px), device=dev)
    kx, ky = k.view(1, 1, 1, -1), k.view(1, 1, -1, 1)
    pad = k.numel() // 2

    def blur(img):
        img = F.conv2d(F.pad(img, (pad, pad, 0, 0), mode="replicate"), kx)
        return F.conv2d(F.pad(img, (0, 0, pad, pad), mode="replicate"), ky)

    tgt = blur(torch.tensor(target, dtype=torch.float32, device=dev).view(1, 1, h, w))

    def curve(c):
        # Open Catmull-Rom: the end points are repeated so the curve starts
        # and ends exactly on them.
        q = torch.cat([c[:1], c, c[-1:]])
        quad = torch.stack([q[:-3], q[1:-2], q[2:-1], q[3:]], 1)
        spans = torch.einsum("sk,nkd->nsd", basis, quad).reshape(-1, 2)
        return torch.cat([spans, c[-1:]])

    def seg_lengths(p):
        return torch.linalg.norm(p[1:] - p[:-1], dim=1)

    def render(p):
        seg = F.pad(seg_lengths(p), (1, 1))
        ink = 0.5 * (seg[:-1] + seg[1:]) * line_px          # ink each sample stands for
        x = p[:, 0].clamp(0, w - 1.001)
        y = p[:, 1].clamp(0, h - 1.001)
        x0, y0 = x.floor(), y.floor()
        fx, fy = x - x0, y - y0
        x0, y0 = x0.long(), y0.long()
        img = torch.zeros(h * w, device=dev)
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            img = img.index_add(0, (y0 + dy) * w + (x0 + dx), ink * wt)
        return blur(img.view(1, 1, h, w))

    # Repulsion between parts of the line that are far apart along it but
    # close on the page: the line may pass near itself, not pile up on itself.
    rep_radius = 2.2 * line_px + 0.6
    window = 3 * SAMPLES_PER_SPAN

    def near_pairs(sub):
        """Pairs close on the page but far apart along the line.

        Only these can repel, so they are looked up with a k-d tree (every
        few steps; the line moves little in between) rather than comparing
        every point with every other, which needs gigabytes on a long line.
        """
        from scipy.spatial import cKDTree
        pts = sub.detach().cpu().numpy()
        pairs = cKDTree(pts).query_pairs(2.0 * rep_radius, output_type="ndarray")
        if len(pairs) == 0:
            return None
        pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > window // 2]
        return torch.tensor(pairs, dtype=torch.long, device=dev) if len(pairs) else None

    pairs = None

    # Match where the ink goes, not how much: a line of fixed length cannot
    # carry the target's total ink, and asked to, it crumples into worms
    # to cover more area.
    with torch.no_grad():
        have = render(curve(ctrl)).sum()
        tgt = tgt * (have / tgt.sum().clamp_min(1e-6))

    opt = torch.optim.Adam([ctrl], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations, eta_min=0.05)
    n_ctrl = len(tour)
    for it in range(iterations):
        opt.zero_grad()
        p = curve(ctrl)
        img_loss = ((render(p) - tgt) ** 2).mean() * 1e3
        d2 = ctrl[2:] - 2 * ctrl[1:-1] + ctrl[:-2]
        smooth = (d2 ** 2).sum(1).mean() * smooth_weight
        sub = p[::2]
        if it % 25 == 0:
            pairs = near_pairs(sub)
        if pairs is None:
            repulse = torch.zeros((), device=dev)
        else:
            d = torch.linalg.norm(sub[pairs[:, 0]] - sub[pairs[:, 1]], dim=1)
            repulse = (torch.relu(rep_radius - d) ** 2).sum() / n_ctrl * 0.05
        length = seg_lengths(p).sum() / (h + w) * 1e-3
        loss = img_loss + smooth + repulse + length
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            ctrl[:, 0].clamp_(0, w - 1)
            ctrl[:, 1].clamp_(0, h - 1)
        if progress and it % 50 == 0:
            progress(it, iterations, float(loss.detach()))
    with torch.no_grad():
        return curve(ctrl).cpu().numpy()
