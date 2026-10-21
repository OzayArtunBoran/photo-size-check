"""Measure a flat object from a single photo, using an ArUco marker of known size.

Steps:
1. Detect exactly one marker (id 0) with sub-pixel corners.
2. Homography from the marker's image corners to its true corners in mm.
3. Segment the object in the photo (marker masked out), take its contour.
4. Map the contour to mm through the homography and fit a minimum-area box.
5. Re-run step 4 with the marker corners jittered by the detector's expected
   error, so every result carries an uncertainty.

Every check that fails is reported in `problems`; a result with problems must
not be shown as a measurement.

Limit: the object must lie flat on the same plane as the marker. A single photo
cannot recover the height of a 3D object, and nothing here pretends to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from .marker import DICTIONARY as MARKER_DICT, MARKER_ID

CORNER_SIGMA_PX = 0.11  # per-axis sd of refined corners, measured on calibration seed 1000 (see LOG.md)
MIN_MARKER_SIDE_PX = 40.0
MIN_OBJECT_AREA_PX = 800.0
MAX_RELATIVE_SD = 0.03
UNCERTAINTY_SAMPLES = 32


@dataclass
class Measurement:
    length_mm: float = float("nan")
    width_mm: float = float("nan")
    length_sd_mm: float = float("nan")
    width_sd_mm: float = float("nan")
    problems: list[str] = field(default_factory=list)
    contour_px: NDArray[np.float32] | None = None
    marker_px: NDArray[np.float32] | None = None

    @property
    def ok(self) -> bool:
        return not self.problems


def _detect_marker(gray: NDArray[np.uint8]) -> tuple[NDArray[np.float32] | None, str | None]:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, "no reference marker found"
    hits = [c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if i == MARKER_ID]
    if len(hits) != 1:
        return None, f"expected exactly one marker id {MARKER_ID}, found {len(hits)}"
    return hits[0].astype(np.float32), None


def _refine_marker(gray: NDArray[np.uint8], corners: NDArray[np.float32]) -> NDArray[np.float32]:
    """Corners from straight lines fitted to the marker's four outer edges.

    A corner refined on its own uses a few pixels around it. Each side of the marker
    gives dozens of sub-pixel edge points; a line through them and the intersection
    of neighbouring lines is far less noisy. Perspective keeps straight edges straight.
    """
    img = gray.astype(np.float32)
    centre = corners.mean(axis=0)
    offsets = np.arange(-4.0, 4.01, 0.25, dtype=np.float32)
    lines = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([d[1], -d[0]], np.float32)
        if np.dot(n, (a + b) / 2 - centre) < 0:
            n = -n  # outward
        ts = np.linspace(0.15, 0.85, 40, dtype=np.float32)
        base = a[None, :] + ts[:, None] * (b - a)[None, :]
        grid = base[:, None, :] + offsets[None, :, None] * n[None, None, :]
        prof = cv2.remap(img, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR)
        pts = []
        for row, p in zip(prof, base):
            level = (row.min() + row.max()) / 2.0
            above = np.nonzero(row >= level)[0]  # dark border inside, light quiet zone outside
            if len(above) == 0 or above[0] == 0:
                continue
            j = int(above[0])
            s = offsets[j - 1] + (level - row[j - 1]) / (row[j] - row[j - 1]) * (offsets[j] - offsets[j - 1])
            pts.append(p + s * n)
        if len(pts) < 10:
            return corners
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for i in range(4):  # corner i = line (i-1) ∩ line i
        (p1, v1), (p2, v2) = lines[i - 1], lines[i]
        A = np.array([v1, -v2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return corners
        t = np.linalg.solve(A, p2 - p1)
        out.append(p1 + t[0] * v1)
    return np.array(out, np.float32)


def _box_mm(contour_px: NDArray[np.float32], H: NDArray[np.float64]) -> tuple[float, float]:
    pts = cv2.perspectiveTransform(contour_px.reshape(-1, 1, 2), H).reshape(-1, 2)
    (_, _), (a, b), _ = cv2.minAreaRect(pts.astype(np.float32))
    return max(a, b), min(a, b)


def _refine_edge(
    gray: NDArray[np.uint8], contour: NDArray[np.int32], inside: NDArray[np.uint8], up: int = 4
) -> NDArray[np.float32]:
    """Sub-pixel object outline.

    The coarse Otsu mask puts the edge wherever the global threshold falls, which is not
    halfway between this object's own dark level and the sheet around it, and a pixel
    contour is quantised to whole pixels. Here: threshold at the local midpoint, on a 4x
    upsampled crop around the object only (cheap: the crop, not the photo).
    """
    ih, iw = gray.shape
    x, y, w, h = cv2.boundingRect(contour)
    pad = 6
    x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, iw), min(y + h + pad, ih)
    coarse = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.drawContours(coarse, [contour - np.array([x0, y0])], -1, 255, cv2.FILLED)
    crop = gray[y0:y1, x0:x1]
    k = np.ones((5, 5), np.uint8)
    core = cv2.erode(coarse, k, iterations=2) > 0
    ring = (cv2.dilate(coarse, k, iterations=2) > 0) & ~(cv2.dilate(coarse, k) > 0) & (inside[y0:y1, x0:x1] > 0)
    if core.sum() < 20 or ring.sum() < 20:
        return contour.reshape(-1, 2).astype(np.float32)
    level = (float(np.median(crop[core])) + float(np.median(crop[ring]))) / 2.0

    big = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
    near = cv2.resize(cv2.dilate(coarse, k), None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    fine = ((big < level) & (near > 0)).astype(np.uint8)
    found, _ = cv2.findContours(fine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not found:
        return contour.reshape(-1, 2).astype(np.float32)
    pts = max(found, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    # Upsampled pixel centre -> original pixel coordinates (cv2.resize keeps centres aligned).
    return ((pts + 0.5) / up - 0.5 + np.array([x0, y0], np.float32)).astype(np.float32)


def measure(image: NDArray[np.uint8], marker_mm: float, *, seed: int = 0) -> Measurement:
    result = Measurement()
    gray = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image, np.uint8)

    marker, problem = _detect_marker(gray)
    if marker is None:
        result.problems.append(problem or "marker detection failed")
        return result
    marker = _refine_marker(gray, marker)
    result.marker_px = marker

    sides = np.linalg.norm(marker - np.roll(marker, -1, axis=0), axis=1)
    if sides.min() < MIN_MARKER_SIDE_PX:
        result.problems.append(f"marker too small in photo ({sides.min():.0f}px side)")

    true_corners = np.array([[0, 0], [marker_mm, 0], [marker_mm, marker_mm], [0, marker_mm]], np.float32)
    H = cv2.getPerspectiveTransform(marker, true_corners).astype(np.float64)

    # Dark object on a light sheet. Otsu adapts to exposure. Only dark pixels inside
    # the sheet that carries the marker count: the table around the sheet is dark too.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, light = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    centre = marker.mean(axis=0)
    grown = centre + (marker - centre) * 1.35  # marker plus its border
    cv2.fillPoly(light, [np.round(grown).astype(np.int32)], 255)
    sheets, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet = next(
        (s for s in sheets if cv2.pointPolygonTest(s, (float(centre[0]), float(centre[1])), False) > 0),
        None,
    )
    if sheet is None:
        result.problems.append("could not find the light surface the marker lies on")
        return result
    inside = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(inside, [sheet], -1, 255, cv2.FILLED)
    mask = cv2.bitwise_and(inside, cv2.bitwise_not(light))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        result.problems.append("no object found")
        return result
    contour = np.asarray(max(contours, key=cv2.contourArea), np.int32)
    if cv2.contourArea(contour) < MIN_OBJECT_AREA_PX:
        result.problems.append("object too small in photo")
    x, y, w, h = cv2.boundingRect(contour)
    ih, iw = gray.shape
    if x <= 1 or y <= 1 or x + w >= iw - 1 or y + h >= ih - 1:
        result.problems.append("object touches the photo edge (may be cut off)")

    contour_px = _refine_edge(gray, contour, inside)
    result.contour_px = contour_px
    result.length_mm, result.width_mm = _box_mm(contour_px, H)

    # Uncertainty: how much does the answer move if the corners are off by their expected error?
    # minAreaRect depends only on the convex hull, so jitter the hull (tens of points), not the
    # full contour (thousands): same answer, far less work per call.
    hull = np.asarray(cv2.convexHull(contour_px), np.float32).reshape(-1, 2)
    rng = np.random.default_rng(seed)  # per-call generator: no shared RNG state between threads
    samples = []
    for _ in range(UNCERTAINTY_SAMPLES):
        jittered = (marker + rng.normal(0.0, CORNER_SIGMA_PX, marker.shape)).astype(np.float32)
        Hj = cv2.getPerspectiveTransform(jittered, true_corners).astype(np.float64)
        samples.append(_box_mm(hull, Hj))
    sd = np.std(np.array(samples), axis=0)
    result.length_sd_mm, result.width_sd_mm = float(sd[0]), float(sd[1])
    if result.length_sd_mm / result.length_mm > MAX_RELATIVE_SD:
        result.problems.append(
            f"too uncertain: ±{result.length_sd_mm:.1f} mm on {result.length_mm:.1f} mm"
        )
    return result


def draw(image: NDArray[np.uint8], m: Measurement) -> NDArray[np.uint8]:
    """Overlay for looking at the actual output, not just the numbers."""
    out = image.copy()
    if m.marker_px is not None:
        cv2.polylines(out, [np.round(m.marker_px).astype(np.int32)], True, (255, 160, 0), 2)
    if m.contour_px is not None:
        cv2.polylines(out, [np.round(m.contour_px).astype(np.int32)], True, (0, 200, 0), 2)
    text = (
        f"{m.length_mm:.1f} x {m.width_mm:.1f} mm (+/-{m.length_sd_mm:.1f})"
        if m.ok else "REJECTED: " + "; ".join(m.problems)
    )
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out
