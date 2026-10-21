"""Synthetic scenes with exact ground truth.

A flat sheet (mm coordinates) holds an ArUco marker of known size and one dark
object (rectangle or ellipse) of known size. The sheet is rendered top-down and
then warped through a random camera homography, blurred and given noise and an
uneven light gradient. Because the scene is generated, the true object size is
known exactly: this is the oracle the measurement code is scored against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0

Shape = Literal["rect", "ellipse"]


@dataclass(frozen=True)
class Scene:
    image: NDArray[np.uint8]  # BGR photo
    length_mm: float  # true long side / major axis
    width_mm: float  # true short side / minor axis
    shape: Shape
    marker_mm: float
    homography: NDArray[np.float64]  # sheet px -> photo px


def _marker(side_px: int) -> NDArray[np.uint8]:
    dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
    return np.asarray(cv2.aruco.generateImageMarker(dictionary, MARKER_ID, side_px), np.uint8)


def _paint(
    sheet: NDArray[np.uint8], shape: Shape, cx: float, cy: float,
    half_l: float, half_w: float, angle_deg: float, *, value: int, ss: int = 4,
) -> None:
    """Blend a rectangle or ellipse into `sheet` by exact pixel coverage (pixel centres at integers)."""
    r = int(np.ceil(np.hypot(half_l, half_w))) + 2  # half-diagonal: corners must fit in the patch
    x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, sheet.shape[1])
    y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, sheet.shape[0])
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
    patch = sheet[y0:y1, x0:x1].astype(np.float64)
    sheet[y0:y1, x0:x1] = np.round(patch * (1 - cover) + value * cover).astype(np.uint8)


def make_scene(
    rng: np.random.Generator,
    *,
    sheet_mm: tuple[float, float] = (200.0, 150.0),
    marker_mm: float = 40.0,
    px_per_mm: float = 8.0,
    out_size: tuple[int, int] = (1280, 960),
    max_tilt: float = 0.18,
    shape: Shape | None = None,
    degrade: bool = True,
) -> Scene:
    sw, sh = sheet_mm
    w, h = int(sw * px_per_mm), int(sh * px_per_mm)
    sheet = np.full((h, w), 235, np.uint8)

    # Marker in the top-left area, with a white quiet zone around it.
    m_px = int(round(marker_mm * px_per_mm))
    mx, my = int(15 * px_per_mm), int(15 * px_per_mm)
    sheet[my : my + m_px, mx : mx + m_px] = _marker(m_px)

    # Object in the right part of the sheet, never overlapping the marker.
    shape = shape or ("rect" if rng.random() < 0.5 else "ellipse")
    length = float(rng.uniform(30.0, 90.0))
    width = float(rng.uniform(15.0, length))
    angle = float(rng.uniform(0.0, 180.0))
    cx = float(rng.uniform(115.0, 145.0)) * px_per_mm
    cy = float(rng.uniform(65.0, 85.0)) * px_per_mm
    # Analytic coverage with 4x4 supersampling. cv2.fillPoly / cv2.ellipse with
    # LINE_AA grow shapes by ~0.6 px per edge, which made the oracle itself wrong.
    _paint(sheet, shape, cx, cy, length * px_per_mm / 2, width * px_per_mm / 2, angle, value=40)

    # Camera: map sheet corners to a jittered quadrilateral inside the photo.
    ow, oh = out_size
    margin = np.array([0.08 * ow, 0.08 * oh])
    base = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], np.float64)
    inward = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], np.float64)
    jitter = rng.uniform(0.0, max_tilt, size=(4, 2)) * np.array([ow, oh])
    dst = base + inward * (margin + jitter)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))

    photo = cv2.warpPerspective(sheet, H, out_size, flags=cv2.INTER_AREA, borderValue=90)
    if degrade:  # lens blur, uneven light, sensor noise
        photo = cv2.GaussianBlur(photo, (0, 0), float(rng.uniform(0.4, 1.2)))
        gradient = np.linspace(rng.uniform(-25, 0), rng.uniform(0, 25), ow)[None, :]
        noisy = photo.astype(np.float64) + gradient + rng.normal(0.0, 3.0, photo.shape)
        photo = np.clip(noisy, 0, 255).astype(np.uint8)
    image = np.asarray(cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR), np.uint8)

    return Scene(image, length, width, shape, marker_mm, H.astype(np.float64))
