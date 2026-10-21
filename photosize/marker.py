"""The reference: ArUco 4x4 markers printed (or shown) at known positions and size."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_ID = 0


@dataclass(frozen=True)
class Board:
    """Markers lying on one flat sheet: id -> top-left corner (x, y) in mm, all `side_mm` wide."""

    side_mm: float
    markers: dict[int, tuple[float, float]] = field(default_factory=lambda: {MARKER_ID: (0.0, 0.0)})

    def corners_mm(self, marker_id: int) -> NDArray[np.float64]:
        """The marker's outer corners in mm, in ArUco order: top-left, top-right, bottom-right, bottom-left."""
        x, y = self.markers[marker_id]
        s = self.side_mm
        return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], np.float64)

    def subset(self, ids: list[int]) -> Board:
        return Board(self.side_mm, {i: self.markers[i] for i in ids})


def single(side_mm: float) -> Board:
    """One marker; its top-left corner is the origin."""
    return Board(side_mm)


def marker_image(side_px: int, marker_id: int = MARKER_ID) -> NDArray[np.uint8]:
    """The marker as a square image, black outer border included."""
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return np.asarray(cv2.aruco.generateImageMarker(dictionary, marker_id, side_px), np.uint8)
