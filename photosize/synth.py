"""Synthetic scenes with exact ground truth.

A light sheet lies on a darker table. The sheet carries the reference marker and one or more
objects (rectangles or ellipses) of known size. The scene is drawn top-down, then photographed
through a camera homography. Each photo pixel averages the scene over its own footprint, as a
sensor does (4x supersampled warp, then area downsampling). Optional degradation adds lens blur,
uneven light, sensor noise and JPEG compression. Because the scene is generated, the true sizes
are known exactly: this is the oracle the measurement is scored against.

Conventions: pixel centres sit at integer coordinates. Sheet pixels are mapped to photo pixels by
`Scene.homography`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from .marker import Board, marker_image

Shape = Literal["rect", "ellipse"]

SHEET_MM = (200.0, 150.0)
MARKER_MM = 40.0
MARKER_POS_MM = (15.0, 15.0)  # top-left corner of the marker on the sheet
PX_PER_MM = 8.0  # resolution of the top-down drawing
SHEET_GREY = 235
TABLE_GREY = 90
TABLE_MARGIN_MM = 30.0  # table drawn around the sheet, so objects can cross the sheet edge
CORNER_BOARD = Board(30.0, {0: (8.0, 8.0), 1: (162.0, 8.0), 2: (8.0, 112.0), 3: (162.0, 112.0)})  # 4 markers


@dataclass(frozen=True)
class ObjectSpec:
    shape: Shape
    cx_mm: float  # centre, sheet coordinates
    cy_mm: float
    length_mm: float  # long side / major axis
    width_mm: float  # short side / minor axis
    angle_deg: float = 0.0
    grey: int = 40


@dataclass(frozen=True, eq=False)
class Scene:
    image: NDArray[np.uint8]  # BGR photo
    objects: tuple[ObjectSpec, ...]
    marker_mm: float
    homography: NDArray[np.float64]  # sheet px -> photo px
    marker_corners_px: NDArray[np.float64]  # true outer corners of the first marker: TL, TR, BR, BL
    board: Board  # marker positions in sheet mm
    all_corners_px: dict[int, NDArray[np.float64]]  # true outer corners of every marker, by id

    @property
    def length_mm(self) -> float:
        return self.objects[0].length_mm

    @property
    def width_mm(self) -> float:
        return self.objects[0].width_mm


def paint(
    canvas: NDArray[np.uint8], shape: Shape, cx: float, cy: float,
    half_l: float, half_w: float, angle_deg: float, *, grey: int, ss: int = 4,
) -> None:
    """Blend a rectangle or ellipse into `canvas` by exact pixel coverage (ss x ss samples per pixel)."""
    r = int(np.ceil(np.hypot(half_l, half_w))) + 2  # half-diagonal: corners must fit in the patch
    x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, canvas.shape[1])
    y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, canvas.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    offsets = (np.arange(ss) + 0.5) / ss - 0.5
    xs = (np.arange(x0, x1)[:, None] + offsets[None, :]).ravel()
    ys = (np.arange(y0, y1)[:, None] + offsets[None, :]).ravel()
    X, Y = np.meshgrid(xs - cx, ys - cy)
    t = np.deg2rad(angle_deg)
    u = X * np.cos(t) + Y * np.sin(t)
    v = -X * np.sin(t) + Y * np.cos(t)
    if shape == "rect":
        inside = (np.abs(u) <= half_l) & (np.abs(v) <= half_w)
    else:
        inside = (u / half_l) ** 2 + (v / half_w) ** 2 <= 1.0
    cover = inside.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3))
    patch = canvas[y0:y1, x0:x1].astype(np.float64)
    canvas[y0:y1, x0:x1] = np.round(patch * (1 - cover) + grey * cover).astype(np.uint8)


def extent_mm(obj: ObjectSpec) -> tuple[float, float, float, float]:
    """Axis-aligned bounds (x0, y0, x1, y1) of the object on the sheet."""
    t = np.deg2rad(obj.angle_deg)
    c, s = abs(np.cos(t)), abs(np.sin(t))
    hl, hw = obj.length_mm / 2, obj.width_mm / 2
    if obj.shape == "rect":
        hx, hy = hl * c + hw * s, hl * s + hw * c
    else:
        hx, hy = float(np.hypot(hl * c, hw * s)), float(np.hypot(hl * s, hw * c))
    return obj.cx_mm - hx, obj.cy_mm - hy, obj.cx_mm + hx, obj.cy_mm + hy


def radial_distort(image: NDArray[np.uint8], k1: float) -> NDArray[np.uint8]:
    """Bend a photo like a lens with radial distortion: each output pixel shows the scene at
    r (1 + k1 r^2) from the centre, r normalised by the image width (k1 > 0: barrel)."""
    h, w = image.shape[:2]
    f, cx, cy = float(w), (w - 1) / 2.0, (h - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = (xs - cx) / f, (ys - cy) / f
    s = 1 + k1 * (xn * xn + yn * yn)
    mx, my = (xn * s * f + cx).astype(np.float32), (yn * s * f + cy).astype(np.float32)
    return np.asarray(cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), np.uint8)


def default_quad(out_size: tuple[int, int] = (1280, 960)) -> NDArray[np.float64]:
    """Where the sheet's corners land in the photo for a mild, fixed tilt."""
    ow, oh = out_size
    return np.array([[0.09 * ow, 0.09 * oh], [0.92 * ow, 0.12 * oh],
                     [0.90 * ow, 0.92 * oh], [0.10 * ow, 0.90 * oh]], np.float64)


def render_scene(
    objects: list[ObjectSpec],
    *,
    sheet_quad_px: NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
    out_size: tuple[int, int] = (1280, 960),
    marker_mm: float = MARKER_MM,
    table_grey: int = TABLE_GREY,
    supersample: int = 4,
    degrade: bool = True,
    with_marker: bool = True,
    board: Board | None = None,
) -> Scene:
    """Draw the sheet, markers and objects, and photograph them.

    `board` places the markers (sheet mm); by default one `marker_mm` marker at MARKER_POS_MM.

    `supersample=1` warps directly with bilinear interpolation (no area integration). It is kept
    only to reproduce the aliasing artefact described in LOG.md.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    quad = sheet_quad_px if sheet_quad_px is not None else default_quad(out_size)
    p, m = PX_PER_MM, TABLE_MARGIN_MM
    sw, sh = int(SHEET_MM[0] * p), int(SHEET_MM[1] * p)
    mpx = int(m * p)
    canvas = np.full((sh + 2 * mpx, sw + 2 * mpx), table_grey, np.uint8)
    canvas[mpx : mpx + sh, mpx : mpx + sw] = SHEET_GREY

    board = board if board is not None else Board(marker_mm, {0: MARKER_POS_MM})
    side = int(round(board.side_mm * p))
    placed: dict[int, tuple[int, int]] = {}
    for marker_id, (bx, by) in board.markers.items():
        mx, my = int(round(bx * p)), int(round(by * p))
        placed[marker_id] = (mx, my)
        if with_marker:
            canvas[mpx + my : mpx + my + side, mpx + mx : mpx + mx + side] = marker_image(side, marker_id)
    for o in objects:
        paint(canvas, o.shape, (m + o.cx_mm) * p, (m + o.cy_mm) * p,
              o.length_mm * p / 2, o.width_mm * p / 2, o.angle_deg, grey=o.grey)

    src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
    H = cv2.getPerspectiveTransform(src, quad.astype(np.float32)).astype(np.float64)
    canvas_to_sheet = np.array([[1, 0, -mpx], [0, 1, -mpx], [0, 0, 1]], np.float64)
    k = supersample
    # Photo pixel x covers supersampled pixels k*x .. k*x+k-1, whose mean centre is k*x + (k-1)/2.
    to_big = np.array([[k, 0, (k - 1) / 2], [0, k, (k - 1) / 2], [0, 0, 1]], np.float64)
    ow, oh = out_size
    big = cv2.warpPerspective(canvas, to_big @ H @ canvas_to_sheet, (ow * k, oh * k),
                              flags=cv2.INTER_LINEAR, borderValue=table_grey)
    photo = cv2.resize(big, out_size, interpolation=cv2.INTER_AREA) if k > 1 else big

    if degrade:  # lens blur, uneven light, sensor noise, JPEG
        photo = cv2.GaussianBlur(photo, (0, 0), float(rng.uniform(0.4, 1.2)))
        gradient = np.linspace(rng.uniform(-25, 0), rng.uniform(0, 25), ow)[None, :]
        noisy = photo.astype(np.float64) + gradient + rng.normal(0.0, 3.0, photo.shape)
        photo = np.clip(noisy, 0, 255).astype(np.uint8)
        ok, jpg = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        decoded = cv2.imdecode(jpg, cv2.IMREAD_GRAYSCALE)
        assert decoded is not None
        photo = decoded
    image = np.asarray(cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR), np.uint8)

    all_px: dict[int, NDArray[np.float64]] = {}
    for marker_id, (mx, my) in placed.items():
        corners = np.array([[mx - 0.5, my - 0.5], [mx + side - 0.5, my - 0.5],
                            [mx + side - 0.5, my + side - 0.5], [mx - 0.5, my + side - 0.5]], np.float64)
        all_px[marker_id] = np.asarray(cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H), np.float64).reshape(-1, 2)
    first = all_px[min(all_px)]
    return Scene(image, tuple(objects), board.side_mm, H, first, board, all_px)


def make_scene(
    rng: np.random.Generator,
    *,
    out_size: tuple[int, int] = (1280, 960),
    max_tilt: float = 0.18,
    shape: Shape | None = None,
    supersample: int = 4,
    degrade: bool = True,
    board: Board | None = None,
) -> Scene:
    """One random dark object, fully on the sheet and at least 5 mm from every marker.

    Without `board`: one 40 mm marker at MARKER_POS_MM and the object to its right (the set-up
    every published number before the board was added). With `board`: the object anywhere in
    the middle of the sheet.
    """
    sw, sh = SHEET_MM
    layout = board if board is not None else Board(MARKER_MM, {0: MARKER_POS_MM})
    boxes = [(x - 5, y - 5, x + layout.side_mm + 5, y + layout.side_mm + 5) for x, y in layout.markers.values()]
    cx_range, cy_range = ((115.0, 145.0), (65.0, 85.0)) if board is None else ((70.0, 130.0), (55.0, 95.0))
    while True:
        kind: Shape = shape or ("rect" if rng.random() < 0.5 else "ellipse")
        length = float(rng.uniform(30.0, 90.0))
        obj = ObjectSpec(kind, float(rng.uniform(*cx_range)), float(rng.uniform(*cy_range)),
                         length, float(rng.uniform(15.0, length)), float(rng.uniform(0.0, 180.0)))
        x0, y0, x1, y1 = extent_mm(obj)
        on_sheet = x0 >= 5 and y0 >= 5 and x1 <= sw - 5 and y1 <= sh - 5
        clear = all(x0 > bx1 or y0 > by1 or x1 < bx0 or y1 < by0 for bx0, by0, bx1, by1 in boxes)
        if on_sheet and clear:
            break

    ow, oh = out_size
    margin = np.array([0.08 * ow, 0.08 * oh])
    base = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], np.float64)
    inward = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], np.float64)
    quad = base + inward * (margin + rng.uniform(0.0, max_tilt, size=(4, 2)) * np.array([ow, oh]))
    return render_scene([obj], sheet_quad_px=quad, rng=rng, out_size=out_size,
                        supersample=supersample, degrade=degrade, board=board)
