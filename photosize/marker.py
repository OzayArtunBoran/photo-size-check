"""The reference: one ArUco 4x4 marker, id 0, printed at a known size."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_ID = 0


def marker_image(side_px: int) -> NDArray[np.uint8]:
    """The marker as a square image, black outer border included."""
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return np.asarray(cv2.aruco.generateImageMarker(dictionary, MARKER_ID, side_px), np.uint8)
