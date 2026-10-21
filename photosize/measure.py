"""Measure a flat object from one photo, using an ArUco marker of known size.

1. Find exactly one marker (id 0). Refine its corners by fitting a straight line to each of its
   four outer edges and intersecting neighbouring lines.
2. Homography from the marker's corners in the photo to its true corners in mm.
3. Find the light sheet that carries the marker.
4. Inside the sheet, mark what is clearly darker than the sheet itself. The sheet's brightness is
   modelled as a smooth surface, so uneven light does not matter, and the threshold adapts to the
   object's own grey level.
5. Refuse anything ambiguous (see `Code`): a result with problems must not be shown as a size.
6. Sub-pixel outline: threshold halfway between the object's and the sheet's local grey levels on
   a 4x upsampled crop. Map the outline's convex hull to mm; the minimum-area box gives
   length x width.
7. Uncertainty from three sources: the marker corners (propagated to length and width through
   the Jacobian of steps 2 and 6, by central differences), the outline's own edge error (scaled
   by the local mm per pixel), and how sharply the box orientation is defined (near-circles).
   Deterministic: no random sampling.

Limit: the object must lie flat on the marker's plane. One photo cannot measure height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from .marker import DICTIONARY, MARKER_ID

# Both sigmas are measured by scripts/calibrate.py on calibration scenes (seed 1000), never on
# the scenes used for evaluation.
CORNER_SIGMA_PX = 0.053  # per-axis error of a refined marker corner
EDGE_SIGMA_PX = 0.118  # error of one side of the object's outline, along its normal
MAX_RELATIVE_2SD = 0.03  # reject when 2 sd is more than 3% of the length or the width
MIN_MARKER_SIDE_PX = 40.0
MIN_OBJECT_AREA_PX = 800.0
MARKER_MARGIN_PX = 4.0  # the marker's black border spills this far into the sheet (blur, JPEG)
MIN_CONTRAST = 25.0  # grey levels between the object and the sheet right around it
MIN_DARKNESS = 6.0  # a darkness threshold below this is sensor noise: nothing stands out
MAX_EDGE_ON_LIMIT = 0.05  # share of the fine outline allowed on the limit of its search band


class Code(StrEnum):
    NO_MARKER = "no_marker"
    MARKER_COUNT = "marker_count"
    MARKER_TOO_SMALL = "marker_too_small"
    NO_SHEET = "no_sheet"
    NO_OBJECT = "no_object"
    OBJECT_TOO_SMALL = "object_too_small"
    SEVERAL_OBJECTS = "several_objects"
    TOUCHES_MARKER = "touches_marker"
    CROSSES_EDGE = "crosses_edge"
    LOW_CONTRAST = "low_contrast"
    UNCLEAR_EDGE = "unclear_edge"
    TOO_UNCERTAIN = "too_uncertain"


@dataclass(frozen=True)
class Problem:
    code: Code
    detail: str


@dataclass
class Measurement:
    length_mm: float = float("nan")
    width_mm: float = float("nan")
    length_sd_mm: float = float("nan")
    width_sd_mm: float = float("nan")
    problems: list[Problem] = field(default_factory=list)
    contour_px: NDArray[np.float32] | None = None
    marker_px: NDArray[np.float32] | None = None

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def codes(self) -> set[Code]:
        return {p.code for p in self.problems}

    def flag(self, code: Code, detail: str) -> Measurement:
        self.problems.append(Problem(code, detail))
        return self


def _detect_marker(gray: NDArray[np.uint8]) -> tuple[NDArray[np.float32] | None, Problem | None]:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, Problem(Code.NO_MARKER, "no reference marker found")
    hits = [c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if i == MARKER_ID]
    if len(hits) != 1:
        return None, Problem(Code.MARKER_COUNT, f"expected exactly one marker id {MARKER_ID}, found {len(hits)}")
    return hits[0].astype(np.float32), None


def _refine_marker(gray: NDArray[np.uint8], corners: NDArray[np.float32]) -> NDArray[np.float32]:
    """Corners from straight lines fitted to the marker's four outer edges.

    A corner refined on its own uses a few pixels around it. Each side of the marker gives
    dozens of sub-pixel edge points; a line through them, intersected with its neighbours, is
    far less noisy. Perspective keeps straight edges straight.
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
        prof = cv2.remap(img, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        pts = []
        for row, p in zip(prof, base):
            if row.max() - row.min() < MIN_CONTRAST:
                continue  # no clear edge in this profile
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


def _grow(quad: NDArray[np.float32], px: float) -> NDArray[np.int32]:
    """The marker quad pushed outward so each side moves about `px` pixels."""
    centre = quad.mean(axis=0)
    out = quad - centre
    unit = out / np.linalg.norm(out, axis=1, keepdims=True)
    return np.asarray(np.round(quad + unit * px * np.sqrt(2.0)), np.int32)


def _sheet_brightness(
    blurred: NDArray[np.uint8], sample: NDArray[np.bool_]
) -> NDArray[np.float32] | None:
    """Quadratic surface fitted to the sheet's own grey level, refitted once without outliers."""
    h, w = blurred.shape
    ys, xs = np.nonzero(sample[::8, ::8])
    if len(xs) < 50:
        return None
    ys, xs = ys * 8, xs * 8
    v = blurred[ys, xs].astype(np.float64)
    xn, yn = xs / w, ys / h
    A = np.column_stack([np.ones_like(xn), xn, yn, xn * xn, yn * yn, xn * yn])
    coef = np.linalg.lstsq(A, v, rcond=None)[0]
    r = v - A @ coef
    mad = float(np.median(np.abs(r - np.median(r))))
    keep = np.abs(r) <= max(3 * 1.4826 * mad, 2.0)
    coef = np.linalg.lstsq(A[keep], v[keep], rcond=None)[0]
    x = (np.arange(w) / w)[None, :]
    y = (np.arange(h) / h)[:, None]
    surface = coef[0] + coef[1] * x + coef[2] * y + coef[3] * x * x + coef[4] * y * y + coef[5] * x * y
    return np.asarray(surface, np.float32)


def _box_mm(pts_px: NDArray[np.float32], H: NDArray[np.float64]) -> tuple[float, float]:
    pts = cv2.perspectiveTransform(pts_px.reshape(-1, 1, 2), H).reshape(-1, 2)
    (_, _), (a, b), _ = cv2.minAreaRect(pts.astype(np.float32))
    return max(a, b), min(a, b)


def _refine_edge(
    gray: NDArray[np.uint8], contour: NDArray[np.int32], region: NDArray[np.uint8], up: int = 4
) -> tuple[NDArray[np.float32], float, float]:
    """Sub-pixel outline, the object's contrast, and the share of the outline on its search limit.

    Thresholds halfway between this object's own dark level and the sheet right around it, on a
    4x upsampled crop around the object only (cheap: the crop, not the photo). The search stays
    within 2 px of the coarse outline; if the fine outline runs along that limit, the coarse
    outline was wrong and the edge is unclear.
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
    ring = (cv2.dilate(coarse, k, iterations=2) > 0) & ~(cv2.dilate(coarse, k) > 0) & (region[y0:y1, x0:x1] > 0)
    fallback = contour.reshape(-1, 2).astype(np.float32)
    if core.sum() < 20 or ring.sum() < 20:
        return fallback, 0.0, 1.0
    dark, light = float(np.median(crop[core])), float(np.median(crop[ring]))
    level = (dark + light) / 2.0

    big = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
    near = cv2.resize(cv2.dilate(coarse, k), None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    fine = ((big < level) & (near > 0)).astype(np.uint8)
    found, _ = cv2.findContours(fine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not found:
        return fallback, light - dark, 1.0
    pts = max(found, key=cv2.contourArea).reshape(-1, 2)
    limit = (near > 0) & ~(cv2.erode(near, np.ones((3, 3), np.uint8)) > 0)
    on_limit = float(np.mean(limit[pts[:, 1], pts[:, 0]]))
    # Upsampled pixel centre -> original pixel coordinates (cv2.resize keeps centres aligned).
    fine_px = (pts.astype(np.float32) + 0.5) / up - 0.5 + np.array([x0, y0], np.float32)
    return fine_px.astype(np.float32), light - dark, on_limit


def _dims(corners: NDArray[np.float64], hull_px: NDArray[np.float32], marker_mm: float) -> NDArray[np.float64]:
    true_corners = np.array([[0, 0], [marker_mm, 0], [marker_mm, marker_mm], [0, marker_mm]], np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), true_corners).astype(np.float64)
    return np.array(_box_mm(hull_px, H), np.float64)


def local_scale(H: NDArray[np.float64], point: NDArray[np.float32]) -> float:
    """mm per photo pixel around `point`, from the homography H (photo px -> mm)."""
    c = point.astype(np.float64)
    probe = np.array([c, c + [1.0, 0.0], c + [0.0, 1.0]], np.float64).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(probe, H).reshape(-1, 2)
    return float(np.sqrt(np.linalg.norm(q[1] - q[0]) * np.linalg.norm(q[2] - q[0])))


def corner_sd(marker: NDArray[np.float32], hull_px: NDArray[np.float32], marker_mm: float,
              sigma_px: float = CORNER_SIGMA_PX) -> NDArray[np.float64]:
    """sd of (length, width) in mm caused by marker-corner error, by linear propagation.

    J is the 2x8 Jacobian of (length, width) with respect to the 8 corner coordinates, by central
    differences (16 box evaluations). With independent corner errors of sd sigma, the variance
    is sigma^2 * sum of squares of each row of J. Deterministic: the same input always gives the
    same sd, and nothing is shared between calls.
    """
    flat = marker.astype(np.float64).reshape(-1)
    step = 0.05  # px
    J = np.zeros((2, 8))
    for k in range(8):
        d = np.zeros(8)
        d[k] = step
        J[:, k] = (_dims((flat + d).reshape(4, 2), hull_px, marker_mm)
                   - _dims((flat - d).reshape(4, 2), hull_px, marker_mm)) / (2 * step)
    return np.asarray(sigma_px * np.sqrt(np.sum(J * J, axis=1)), np.float64)


def orientation_spread(hull_mm: NDArray[np.float64], area_tol: float) -> NDArray[np.float64]:
    """Range of (length, width) over box angles whose area is within `area_tol` of the minimum.

    For a rectangle the minimum-area box has one sharp angle and the range is ~0. For a shape
    close to a circle the area barely changes with angle, so outline noise alone can turn the box
    and trade length for width; the range says by how much.
    """
    angles = np.deg2rad(np.arange(0.0, 90.0, 0.25))
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    x, y = hull_mm[:, 0][None, :], hull_mm[:, 1][None, :]
    u, v = x * c + y * s, -x * s + y * c
    du, dv = u.max(axis=1) - u.min(axis=1), v.max(axis=1) - v.min(axis=1)
    area = du * dv
    near = area <= area.min() + area_tol
    lengths, widths = np.maximum(du, dv)[near], np.minimum(du, dv)[near]
    return np.array([lengths.max() - lengths.min(), widths.max() - widths.min()], np.float64)


def _uncertainty(marker: NDArray[np.float32], hull_px: NDArray[np.float32], marker_mm: float,
                 H: NDArray[np.float64], dims: tuple[float, float]) -> tuple[float, float]:
    """sd of (length, width) in mm: marker corners, the outline's edge error, and how well the
    box orientation is defined."""
    corner = corner_sd(marker, hull_px, marker_mm)
    edge_mm = EDGE_SIGMA_PX * local_scale(H, hull_px.mean(axis=0))
    edge = np.sqrt(2.0) * edge_mm  # two sides per dimension
    hull_mm = cv2.perspectiveTransform(hull_px.reshape(-1, 1, 2), H).reshape(-1, 2).astype(np.float64)
    length, width = dims
    area_tol = 2 * np.sqrt(2.0) * edge_mm * float(np.hypot(length, width))  # 2 sd of the box area
    spread = orientation_spread(hull_mm, area_tol) / np.sqrt(12.0)  # sd of a uniform range
    sd = np.sqrt(corner**2 + edge**2 + spread**2)
    return float(sd[0]), float(sd[1])


def measure(image: NDArray[np.uint8], marker_mm: float) -> Measurement:
    """Length and width in mm of the one dark object lying flat next to the marker.

    `marker_mm` is the printed marker's outer side, measured on the print. Always check `ok`:
    when it is False, `problems` says why and the numbers must not be used.
    """
    gray = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image, np.uint8)
    marker, problem = _detect_marker(gray)
    if marker is None:
        return Measurement(problems=[problem or Problem(Code.NO_MARKER, "marker detection failed")])
    return _measure_from_marker(gray, _refine_marker(gray, marker), marker_mm)


def _measure_from_marker(gray: NDArray[np.uint8], marker: NDArray[np.float32], marker_mm: float) -> Measurement:
    """Everything after marker detection. Calibration feeds the true corners in here."""
    result = Measurement()
    result.marker_px = marker
    sides = np.linalg.norm(marker - np.roll(marker, -1, axis=0), axis=1)
    if sides.min() < MIN_MARKER_SIDE_PX:
        result.flag(Code.MARKER_TOO_SMALL, f"marker side is {sides.min():.0f}px in the photo")
    true_corners = np.array([[0, 0], [marker_mm, 0], [marker_mm, marker_mm], [0, marker_mm]], np.float32)
    H = cv2.getPerspectiveTransform(marker, true_corners).astype(np.float64)

    # The light sheet that carries the marker (the marker itself counts as sheet).
    blurred = np.asarray(cv2.GaussianBlur(gray, (5, 5), 0), np.uint8)
    _, light = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    zone = np.zeros_like(gray)
    cv2.fillPoly(zone, [_grow(marker, MARKER_MARGIN_PX)], 255)
    light[zone > 0] = 255
    centre = marker.mean(axis=0)
    sheets, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet = next(
        (s for s in sheets if cv2.pointPolygonTest(s, (float(centre[0]), float(centre[1])), False) > 0),
        None,
    )
    if sheet is None:
        return result.flag(Code.NO_SHEET, "could not find the light surface the marker lies on")
    inside = np.zeros_like(gray)
    cv2.drawContours(inside, [sheet], -1, 255, cv2.FILLED)

    # A flat sheet seen by a camera is convex, and so is its part inside the photo. A dark region
    # cutting into its outline is something crossing the edge of the sheet or of the photo.
    hull = np.zeros_like(gray)
    cv2.fillConvexPoly(hull, cv2.convexHull(sheet), 255)
    k5 = np.ones((5, 5), np.uint8)
    notch = cv2.morphologyEx(cv2.bitwise_and(hull, cv2.bitwise_not(inside)), cv2.MORPH_OPEN, k5)
    notches, _ = cv2.findContours(notch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if any(cv2.contourArea(c) >= MIN_OBJECT_AREA_PX for c in notches):
        result.flag(Code.CROSSES_EDGE, "something dark crosses the edge of the sheet or the photo")

    # Objects: clearly darker than the sheet's own brightness at that spot.
    region = np.asarray(cv2.bitwise_and(cv2.erode(inside, np.ones((7, 7), np.uint8)), cv2.bitwise_not(zone)), np.uint8)
    surface = _sheet_brightness(blurred, (region > 0) & (light > 0))
    if surface is None:
        return result.flag(Code.NO_SHEET, "too little visible sheet to model its brightness")
    darkness = surface - blurred.astype(np.float32)
    values = np.clip(darkness[region > 0], 0, 255).astype(np.uint8).reshape(-1, 1)
    otsu, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cut = max(float(otsu), MIN_DARKNESS)
    mask = ((darkness > cut) & (region > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
    blobs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = sorted((b for b in blobs if cv2.contourArea(b) >= MIN_OBJECT_AREA_PX), key=cv2.contourArea, reverse=True)
    if not big:
        if blobs:
            return result.flag(Code.OBJECT_TOO_SMALL, "only small dark spots on the sheet")
        return result.flag(Code.NO_OBJECT, "nothing on the sheet is darker than the sheet")
    if len(big) > 1:
        areas = ", ".join(f"{cv2.contourArea(b):.0f}" for b in big[:4])
        result.flag(Code.SEVERAL_OBJECTS, f"{len(big)} dark objects on the sheet (areas px: {areas})")
    contour = np.asarray(big[0], np.int32)

    filled = np.zeros_like(gray)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    ring = cv2.bitwise_and(cv2.dilate(zone, np.ones((7, 7), np.uint8)), cv2.bitwise_not(zone))
    if cv2.countNonZero(cv2.bitwise_and(filled, ring)):
        result.flag(Code.TOUCHES_MARKER, "the object touches the marker; part of it may be hidden")

    contour_px, contrast, on_limit = _refine_edge(gray, contour, region)
    result.contour_px = contour_px
    if contrast < MIN_CONTRAST:
        result.flag(Code.LOW_CONTRAST, f"object is only {contrast:.0f} grey levels darker than the sheet")
    if on_limit > MAX_EDGE_ON_LIMIT:
        result.flag(Code.UNCLEAR_EDGE, f"{100 * on_limit:.0f}% of the outline was not found near the coarse edge")

    # minAreaRect depends only on the convex hull, so the hull (tens of points) is what gets
    # mapped to mm, here and in every uncertainty evaluation, instead of the full outline.
    hull_px = np.asarray(cv2.convexHull(contour_px), np.float32).reshape(-1, 2)
    result.length_mm, result.width_mm = _box_mm(hull_px, H)
    result.length_sd_mm, result.width_sd_mm = _uncertainty(
        marker, hull_px, marker_mm, H, (result.length_mm, result.width_mm))
    worst = max(2 * result.length_sd_mm / result.length_mm, 2 * result.width_sd_mm / result.width_mm)
    if worst > MAX_RELATIVE_2SD:
        result.flag(Code.TOO_UNCERTAIN, f"2 sd is {100 * worst:.1f}% of the size")
    return result


def draw(image: NDArray[np.uint8], m: Measurement) -> NDArray[np.uint8]:
    """Overlay for looking at the actual output, not just the numbers."""
    out = image.copy()
    if m.marker_px is not None:
        cv2.polylines(out, [np.round(m.marker_px).astype(np.int32)], True, (255, 160, 0), 2)
    if m.contour_px is not None:
        colour = (0, 200, 0) if m.ok else (0, 0, 255)
        cv2.polylines(out, [np.round(m.contour_px).astype(np.int32)], True, colour, 2)
    text = (
        f"{m.length_mm:.1f} x {m.width_mm:.1f} mm (2 sd: {2 * m.length_sd_mm:.2f} x {2 * m.width_sd_mm:.2f})"
        if m.ok else "REJECTED: " + ", ".join(sorted(m.codes))
    )
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out
