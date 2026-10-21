"""Measure a flat object from one photo, using ArUco markers of known size and position.

1. Find the board's markers: one marker, or several at known positions on the same sheet.
   Refine each marker's corners by fitting a straight line to each of its four outer edges and
   intersecting neighbouring lines.
2. Homography from all marker corners in the photo to their true positions in mm: exact for one
   marker, least squares for several. With several markers, how badly the corners fit one flat
   plane measures how far the photo is from an ideal projection (lens distortion, a curled
   sheet, a wrong board).
3. Find the light sheet that carries the markers.
4. Inside the sheet, mark what is clearly darker than the sheet right there. A first pass with
   one smooth brightness surface finds the clear objects; a second pass estimates the sheet's
   own local brightness without them, so dim edges and reflections are not "dark".
5. Refuse anything ambiguous (see `Code`): a result with problems must not be shown as a size.
6. Sub-pixel outline: threshold halfway between the object's and the sheet's local grey levels on
   a 4x upsampled crop. Map the outline's convex hull to mm; the minimum-area box gives
   length x width.
7. Uncertainty from four sources: the marker corners (through the Jacobian of steps 2 and 6, by
   central differences), the outline's own edge error, how sharply the box orientation is
   defined (near-circles), and, with several markers, the misfit of their corners to one plane.
   Deterministic: no random sampling.

Limit: the object must lie flat on the markers' plane. One photo cannot measure height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from .marker import DICTIONARY, Board, single

# Both sigmas are measured by scripts/calibrate.py on calibration scenes (seed 1000), never on
# the scenes used for evaluation.
CORNER_SIGMA_PX = 0.053  # per-axis error of a refined marker corner
EDGE_SIGMA_PX = 0.118  # error of one side of the object's outline, along its normal
MAX_RELATIVE_2SD = 0.03  # reject when 2 sd is more than 3% of the length or the width
MAX_MISFIT_MM = 1.0  # reject when the markers' corners miss one flat plane by more than this (rms)
MAX_RADIAL_K = 0.25  # search range of the radial distortion estimated from several markers
MIN_MARKER_SIDE_PX = 40.0
MIN_OBJECT_AREA_PX = 800.0
MARKER_MARGIN_PX = 4.0  # a marker's black border spills this far into the sheet (blur, JPEG)
MIN_CONTRAST = 25.0  # grey levels between the object and the sheet right around it
MIN_DARKNESS = 6.0  # a darkness threshold below this is sensor noise: nothing stands out
MAX_EDGE_ON_LIMIT = 0.05  # share of the fine outline allowed on the limit of its search band
WORK_PX_PER_MM = 5.0  # photos are reduced to about this many pixels per mm at the markers
CROSS_DEPTH_MM = 5.0  # a dark region touching the sheet's outline counts as crossing it this deep
SPUR_MM = 12.0  # bulges thinner than this on the sheet's outline (reflections on a frame) are ignored;
                # 12 mm was chosen on the first real photo set (LOG.md) and must hold on new photos


class Code(StrEnum):
    NO_MARKER = "no_marker"
    MARKER_COUNT = "marker_count"
    MARKER_TOO_SMALL = "marker_too_small"
    MARKERS_DISAGREE = "markers_disagree"
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
    marker_px: NDArray[np.float32] | None = None  # corners of the board's first marker
    markers_px: dict[int, NDArray[np.float32]] = field(default_factory=dict)
    misfit_mm: float = float("nan")  # rms distance of the marker corners from one flat plane
    radial_k: float = 0.0  # lens distortion estimated from the markers (0 with one marker)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def codes(self) -> set[Code]:
        return {p.code for p in self.problems}

    def flag(self, code: Code, detail: str) -> Measurement:
        self.problems.append(Problem(code, detail))
        return self


def _detect_markers(gray: NDArray[np.uint8], board: Board) -> tuple[
        dict[int, NDArray[np.float32]] | None, list[NDArray[np.float32]], Problem | None]:
    """The board's markers by id, every other marker in view (to mask, never to measure), or why not."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, [], Problem(Code.NO_MARKER, "no reference marker found")
    found: dict[int, list[NDArray[np.float32]]] = {}
    others: list[NDArray[np.float32]] = []
    for c, i in zip(corners, ids.ravel()):
        quad = c.reshape(4, 2).astype(np.float32)
        if int(i) in board.markers:
            found.setdefault(int(i), []).append(quad)
        else:
            others.append(quad)
    if not found:
        return None, others, Problem(Code.NO_MARKER, f"none of the board's markers {sorted(board.markers)} found")
    missing = sorted(i for i in board.markers if i not in found)
    doubled = sorted(i for i, cs in found.items() if len(cs) > 1)
    if missing or doubled:
        return None, others, Problem(Code.MARKER_COUNT, f"markers missing: {missing}; seen more than once: {doubled}")
    return {i: cs[0] for i, cs in found.items()}, others, None


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


def _fit(corners: dict[int, NDArray[np.float32]] | dict[int, NDArray[np.float64]], board: Board) -> tuple[NDArray[np.float64], float]:
    """Homography from photo px to board mm, and the rms misfit of the corners in mm.

    One marker gives exactly four points and no misfit. Several give a least-squares fit; the
    misfit is what one flat, ideally projected sheet cannot explain.
    """
    ids = sorted(corners)
    src = np.concatenate([np.asarray(corners[i], np.float64) for i in ids])
    dst = np.concatenate([board.corners_mm(i) for i in ids])
    exact = np.asarray(cv2.getPerspectiveTransform(src[:4].astype(np.float32), dst[:4].astype(np.float32)), np.float64)
    if len(ids) == 1:
        return exact, 0.0
    fitted, _ = cv2.findHomography(src, dst, 0)
    H: NDArray[np.float64] = exact if fitted is None else np.asarray(fitted, np.float64)
    mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, float(np.sqrt(np.mean(np.sum((mapped - dst) ** 2, axis=1))))


def _undistort(pts: NDArray[np.float32], k: float, centre: NDArray[np.float32], norm: float) -> NDArray[np.float32]:
    """One-parameter radial model: ideal = observed * (1 + k r^2), r normalised by the image width."""
    if k == 0.0:
        return pts
    d = (pts - centre) / norm
    r2 = np.sum(d * d, axis=-1, keepdims=True)
    return np.asarray(centre + d * (1.0 + k * r2) * norm, np.float32)


def _estimate_radial(corners: dict[int, NDArray[np.float32]], board: Board, centre: NDArray[np.float32],
                     norm: float) -> float:
    """Radial distortion k that lets the markers' corners fit one flat plane best.

    One marker gives four points, which any homography fits exactly: no information about k.
    Several markers spread over the photo do; a homography cannot bend straight lines, the
    lens does. Golden-section search of the rms misfit over k in [-MAX_RADIAL_K, MAX_RADIAL_K].
    """
    if len(corners) < 2:
        return 0.0

    def misfit(k: float) -> float:
        return _fit({i: _undistort(c, k, centre, norm) for i, c in corners.items()}, board)[1]

    lo, hi = -MAX_RADIAL_K, MAX_RADIAL_K
    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = misfit(a), misfit(b)
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = misfit(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = misfit(b)
    return float((lo + hi) / 2.0)


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


def _dark_mask(darkness: NDArray[np.float32], stats_region: NDArray[np.uint8],
               where: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pixels clearly darker than the sheet: Otsu on the darkness over the sheet, opened."""
    values = np.clip(darkness[stats_region > 0], 0, 255).astype(np.uint8).reshape(-1, 1)
    otsu, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cut = max(float(otsu), MIN_DARKNESS)
    mask = ((darkness > cut) & (where > 0)).astype(np.uint8) * 255
    return np.asarray(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.uint8)


def _interior(mask: NDArray[np.uint8], region: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """The parts of `mask` that do not touch the edge of `region`."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    strict = cv2.erode(region, np.ones((3, 3), np.uint8)) > 0
    out = np.zeros_like(mask)
    for i in range(1, n):
        part = labels == i
        if stats[i, cv2.CC_STAT_AREA] >= MIN_OBJECT_AREA_PX and bool(np.all(strict[part])):
            out[part] = 255
    return out


def _local_brightness(blurred: NDArray[np.uint8], sample: NDArray[np.bool_], cell: int = 8) -> NDArray[np.float32]:
    """The sheet's own grey level around every pixel.

    Mean of the sampled sheet pixels in cells of `cell` px (about 1.6 mm at the working scale),
    cells without samples (objects, outside the sheet) filled in from their neighbours, smoothed.
    Unlike one smooth surface, it follows sharp shading near the sheet's edges and reflections.
    """
    h, w = blurred.shape
    gw, gh = max(1, w // cell), max(1, h // cell)
    m = sample.astype(np.float32)
    total = cv2.resize(blurred.astype(np.float32) * m, (gw, gh), interpolation=cv2.INTER_AREA)
    count = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)
    known = count > 0.2
    grid = np.where(known, total / np.maximum(count, 1e-6), 0.0)
    filled = cv2.inpaint(np.clip(np.round(grid), 0, 255).astype(np.uint8), (~known).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    smooth = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 1.0)
    return np.asarray(cv2.resize(smooth, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)


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


def local_scale(H: NDArray[np.float64], point: NDArray[np.float32]) -> float:
    """mm per photo pixel around `point`, from the homography H (photo px -> mm)."""
    c = point.astype(np.float64)
    probe = np.array([c, c + [1.0, 0.0], c + [0.0, 1.0]], np.float64).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(probe, H).reshape(-1, 2)
    return float(np.sqrt(np.linalg.norm(q[1] - q[0]) * np.linalg.norm(q[2] - q[0])))


def _dims(flat: NDArray[np.float64], ids: list[int], hull_px: NDArray[np.float32], board: Board) -> NDArray[np.float64]:
    """(length, width) in mm when the markers' corners are at `flat` (ids in order, 8 values each)."""
    corners = {i: flat[8 * n : 8 * n + 8].reshape(4, 2) for n, i in enumerate(ids)}
    H, _ = _fit(corners, board)
    return np.array(_box_mm(hull_px, H), np.float64)


def corner_sd(corners: dict[int, NDArray[np.float32]], hull_px: NDArray[np.float32], board: Board,
              sigma_px: float = CORNER_SIGMA_PX) -> NDArray[np.float64]:
    """sd of (length, width) in mm caused by marker-corner error, by linear propagation.

    J is the Jacobian of (length, width) with respect to every corner coordinate, by central
    differences (2 box evaluations per coordinate). With independent corner errors of sd sigma,
    the variance is sigma^2 * sum of squares of each row of J. Deterministic: the same input
    always gives the same sd, and nothing is shared between calls.
    """
    ids = sorted(corners)
    flat = np.concatenate([np.asarray(corners[i], np.float64) for i in ids]).reshape(-1)
    step = 0.05  # px
    J = np.zeros((2, flat.size))
    for k in range(flat.size):
        d = np.zeros(flat.size)
        d[k] = step
        J[:, k] = (_dims(flat + d, ids, hull_px, board) - _dims(flat - d, ids, hull_px, board)) / (2 * step)
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


def _uncertainty(corners: dict[int, NDArray[np.float32]], hull_px: NDArray[np.float32], board: Board,
                 H: NDArray[np.float64], dims: tuple[float, float], misfit_mm: float) -> tuple[float, float]:
    """sd of (length, width) in mm: marker corners, the outline's edge error, how well the box
    orientation is defined, and the markers' misfit to one plane beyond what corner noise explains."""
    corner = corner_sd(corners, hull_px, board)
    edge_mm = EDGE_SIGMA_PX * local_scale(H, hull_px.mean(axis=0))
    edge = np.sqrt(2.0) * edge_mm  # two sides per dimension
    hull_mm = cv2.perspectiveTransform(hull_px.reshape(-1, 1, 2), H).reshape(-1, 2).astype(np.float64)
    length, width = dims
    area_tol = 2 * np.sqrt(2.0) * edge_mm * float(np.hypot(length, width))  # 2 sd of the box area
    spread = orientation_spread(hull_mm, area_tol) / np.sqrt(12.0)  # sd of a uniform range
    n_points = 4 * len(corners)
    misfit = 0.0
    if n_points > 4:
        centre = np.concatenate(list(corners.values())).mean(axis=0).astype(np.float32)
        noise = CORNER_SIGMA_PX * local_scale(H, centre) * np.sqrt(2.0 - 8.0 / n_points)  # expected rms
        misfit = float(np.sqrt(max(0.0, misfit_mm**2 - noise**2)))  # mapping error, about one per dimension
    sd = np.sqrt(corner**2 + edge**2 + spread**2 + misfit**2)
    return float(sd[0]), float(sd[1])


def measure(image: NDArray[np.uint8], reference: float | Board) -> Measurement:
    """Length and width in mm of the one dark object lying flat on the markers' sheet.

    `reference` is the printed marker's outer side in mm (one marker, id 0), or a `Board` with
    several markers at known positions. Sizes must be measured on the print. Always check `ok`:
    when it is False, `problems` says why and the numbers must not be used.

    A large photo is first reduced so the markers have about WORK_PX_PER_MM pixels per mm. Every
    pixel-based setting then means the same physical size in any photo, and the work is cheaper.
    Corners and outlines in the result are in the original photo's pixels.
    """
    board = reference if isinstance(reference, Board) else single(float(reference))
    full = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image, np.uint8)
    found, others, problem = _detect_markers(full, board)
    if found is None:
        return Measurement(problems=[problem or Problem(Code.NO_MARKER, "marker detection failed")])
    side = float(np.mean([np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).mean() for c in found.values()]))
    f = min(1.0, WORK_PX_PER_MM * board.side_mm / side)
    if f == 1.0:
        return _measure_from_markers(full, {i: _refine_marker(full, c) for i, c in found.items()}, board, others)
    h, w = full.shape
    size = (max(1, round(w * f)), max(1, round(h * f)))
    scale = np.array([size[0] / w, size[1] / h], np.float32)  # exact per-axis factors after rounding
    gray = np.asarray(cv2.resize(full, size, interpolation=cv2.INTER_AREA), np.uint8)
    refined = {i: _refine_marker(gray, ((c + 0.5) * scale - 0.5).astype(np.float32)) for i, c in found.items()}
    result = _measure_from_markers(gray, refined, board, [((c + 0.5) * scale - 0.5).astype(np.float32) for c in others])
    def back(pts: NDArray[np.float32]) -> NDArray[np.float32]:
        return ((pts + 0.5) / scale - 0.5).astype(np.float32)

    result.markers_px = {i: back(c) for i, c in result.markers_px.items()}
    if result.marker_px is not None:
        result.marker_px = back(result.marker_px)
    if result.contour_px is not None:
        result.contour_px = back(result.contour_px)
    return result


def _measure_from_markers(gray: NDArray[np.uint8], corners: dict[int, NDArray[np.float32]], board: Board,
                          others: list[NDArray[np.float32]] | None = None) -> Measurement:
    """Everything after marker detection. Calibration feeds the true corners in here.

    `others` are markers in view that are not part of the board: masked like the board's
    markers, never used for measuring.
    """
    result = Measurement()
    ids = sorted(corners)
    result.marker_px = corners[ids[0]]
    result.markers_px = dict(corners)
    smallest = min(float(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).min()) for c in corners.values())
    if smallest < MIN_MARKER_SIDE_PX:
        result.flag(Code.MARKER_TOO_SMALL, f"a marker side is {smallest:.0f}px in the photo")
    # Geometry happens in ideal (undistorted) pixel coordinates; masks stay in photo pixels.
    h, w = gray.shape
    centre_px = np.array([(w - 1) / 2.0, (h - 1) / 2.0], np.float32)
    k = _estimate_radial(corners, board, centre_px, float(w))
    result.radial_k = k
    ideal = {i: _undistort(c, k, centre_px, float(w)) for i, c in corners.items()}
    H, misfit = _fit(ideal, board)
    result.misfit_mm = misfit
    if misfit > MAX_MISFIT_MM:
        result.flag(Code.MARKERS_DISAGREE,
                    f"the markers' corners miss one flat plane by {misfit:.2f} mm rms: wrong board, curled sheet or lens distortion")

    # The light sheet that carries the markers (the markers themselves count as sheet).
    blurred = np.asarray(cv2.GaussianBlur(gray, (5, 5), 0), np.uint8)
    _, light = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    zone = np.zeros_like(gray)
    for c in [*corners.values(), *(others or [])]:
        cv2.fillPoly(zone, [_grow(c, MARKER_MARGIN_PX)], 255)
    light[zone > 0] = 255
    centre = corners[ids[0]].mean(axis=0)
    sheets, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet = next(
        (s for s in sheets if cv2.pointPolygonTest(s, (float(centre[0]), float(centre[1])), False) > 0),
        None,
    )
    if sheet is None:
        return result.flag(Code.NO_SHEET, "could not find the light surface the markers lie on")
    inside = np.zeros_like(gray)
    cv2.drawContours(inside, [sheet], -1, 255, cv2.FILLED)
    # The sheet's convex outline, without thin bulges: a reflection that joins the sheet across a
    # dark frame would otherwise stretch the outline over the frame, which then looks like a dark
    # region crossing the sheet. Opening removes bulges only; a real notch stays.
    spur = max(3, int(round(SPUR_MM * WORK_PX_PER_MM)) | 1)
    opened = cv2.morphologyEx(inside, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spur, spur)))
    body = cv2.findNonZero(opened)
    hull = np.zeros_like(gray)
    cv2.fillConvexPoly(hull, cv2.convexHull(body if body is not None else sheet), 255)
    region = np.asarray(cv2.bitwise_and(cv2.erode(inside, np.ones((7, 7), np.uint8)), cv2.bitwise_not(zone)), np.uint8)

    # Pass 1: one smooth brightness surface over the whole sheet finds the clear objects, so they
    # can be left out when the sheet's own local brightness is estimated in pass 2.
    surface = _sheet_brightness(blurred, (region > 0) & (light > 0))
    if surface is None:
        return result.flag(Code.NO_SHEET, "too little visible sheet to model its brightness")
    rough = _dark_mask(surface - blurred.astype(np.float32), region, region)
    clear = _interior(rough, region)
    keep_out = cv2.dilate(clear, np.ones((11, 11), np.uint8))
    # Pass 2: the sheet's local brightness follows its own shading (dim edges, glare, vignetting),
    # so only what is darker than the sheet right there counts as dark. Search the sheet as well
    # as its convex outline: the opening above rounds the outline's corners, more so on the far
    # side of a strongly tilted photo. A precaution: no scene has shown a difference yet (LOG.md).
    local = _local_brightness(blurred, (region > 0) & (light > 0) & (keep_out == 0))
    search = cv2.bitwise_and(cv2.bitwise_or(hull, inside), cv2.bitwise_not(zone))
    dark = _dark_mask(local - blurred.astype(np.float32), region, np.asarray(search, np.uint8))

    # Dark parts strictly inside the sheet are objects, and so are parts against a marker. A dark
    # part touching the sheet's outline (or the photo's edge) is an object crossing it if it
    # reaches CROSS_DEPTH_MM into the sheet's convex outline; thinner slivers there are shading or
    # reflections along the edge.
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    reach = cv2.distanceTransform(hull, cv2.DIST_L2, 5)
    strict = cv2.erode(region, np.ones((3, 3), np.uint8)) > 0
    ring = cv2.bitwise_and(cv2.dilate(zone, np.ones((7, 7), np.uint8)), cv2.bitwise_not(zone)) > 0
    objects: list[int] = []
    small = False
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_OBJECT_AREA_PX:
            small = True
            continue
        part = labels == i
        if bool(np.all(strict[part])) or bool(np.any(ring[part])):
            objects.append(i)
            continue
        depth_mm = float(reach[part].max()) * local_scale(H, centroids[i].astype(np.float32))
        if depth_mm >= CROSS_DEPTH_MM and Code.CROSSES_EDGE not in result.codes:
            result.flag(Code.CROSSES_EDGE, f"a dark region crosses the edge of the sheet or the photo ({depth_mm:.0f} mm deep)")
    if not objects:
        if small:
            return result.flag(Code.OBJECT_TOO_SMALL, "only small dark spots on the sheet")
        return result.flag(Code.NO_OBJECT, "nothing on the sheet is darker than the sheet")
    objects.sort(key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    if len(objects) > 1:
        areas = ", ".join(str(int(stats[i, cv2.CC_STAT_AREA])) for i in objects[:4])
        result.flag(Code.SEVERAL_OBJECTS, f"{len(objects)} dark objects on the sheet (areas px: {areas})")
    if bool(np.any(ring[labels == objects[0]])):
        result.flag(Code.TOUCHES_MARKER, "the object touches a marker; part of it may be hidden")
    outlines, _ = cv2.findContours((labels == objects[0]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = np.asarray(max(outlines, key=cv2.contourArea), np.int32)

    contour_px, contrast, on_limit = _refine_edge(gray, contour, region)
    result.contour_px = contour_px
    if contrast < MIN_CONTRAST:
        result.flag(Code.LOW_CONTRAST, f"object is only {contrast:.0f} grey levels darker than the sheet")
    if on_limit > MAX_EDGE_ON_LIMIT:
        result.flag(Code.UNCLEAR_EDGE, f"{100 * on_limit:.0f}% of the outline was not found near the coarse edge")

    # minAreaRect depends only on the convex hull, so the hull (tens of points) is what gets
    # mapped to mm, here and in every uncertainty evaluation, instead of the full outline.
    hull_px = np.asarray(cv2.convexHull(_undistort(contour_px, k, centre_px, float(w))), np.float32).reshape(-1, 2)
    result.length_mm, result.width_mm = _box_mm(hull_px, H)
    result.length_sd_mm, result.width_sd_mm = _uncertainty(
        ideal, hull_px, board, H, (result.length_mm, result.width_mm), misfit)
    worst = max(2 * result.length_sd_mm / result.length_mm, 2 * result.width_sd_mm / result.width_mm)
    if worst > MAX_RELATIVE_2SD:
        result.flag(Code.TOO_UNCERTAIN, f"2 sd is {100 * worst:.1f}% of the size")
    return result


def draw(image: NDArray[np.uint8], m: Measurement) -> NDArray[np.uint8]:
    """Overlay for looking at the actual output, not just the numbers."""
    out = image.copy()
    quads = list(m.markers_px.values()) or ([m.marker_px] if m.marker_px is not None else [])
    for q in quads:
        cv2.polylines(out, [np.round(q).astype(np.int32)], True, (255, 160, 0), 2)
    if m.contour_px is not None:
        colour = (0, 200, 0) if m.ok else (0, 0, 255)
        cv2.polylines(out, [np.round(m.contour_px).astype(np.int32)], True, colour, 2)
    text = (
        f"{m.length_mm:.1f} x {m.width_mm:.1f} mm (2 sd: {2 * m.length_sd_mm:.2f} x {2 * m.width_sd_mm:.2f})"
        if m.ok else "REJECTED: " + ", ".join(sorted(m.codes))
    )
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out
