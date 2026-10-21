"""Measure the two error sizes the uncertainty uses, on calibration scenes only.

corner: error of each refined marker corner against the true corner, per axis, in photo px.
edge:   error of length and width when the true marker corners are fed in, so only the object's
        outline is at fault; converted to px per side with the local mm-per-px scale.

Calibration uses SeedSequence(1000); evaluate.py and the tests use other seeds.
Usage: python scripts/calibrate.py [--n 300]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photosize import make_scene  # noqa: E402
from photosize.marker import single  # noqa: E402
from photosize.measure import _detect_markers, _measure_from_markers, _refine_marker, local_scale  # noqa: E402

CALIBRATION_SEED = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    corner_errs, edge_px = [], []
    for seq in np.random.SeedSequence(CALIBRATION_SEED).spawn(args.n):
        scene = make_scene(np.random.default_rng(seq))
        gray = np.asarray(cv2.cvtColor(scene.image, cv2.COLOR_BGR2GRAY), np.uint8)
        board = single(scene.marker_mm)
        found, _, _ = _detect_markers(gray, board)
        if found is None:
            continue
        corner_errs.append(_refine_marker(gray, found[0]) - scene.marker_corners_px)

        true_corners = scene.marker_corners_px.astype(np.float32)
        m = _measure_from_markers(gray, {0: true_corners}, board)
        if not m.ok or m.contour_px is None:
            continue
        tc = np.array([[0, 0], [scene.marker_mm, 0], [scene.marker_mm, scene.marker_mm], [0, scene.marker_mm]], np.float32)
        H = cv2.getPerspectiveTransform(true_corners, tc).astype(np.float64)
        scale = local_scale(H, m.contour_px.mean(axis=0))
        for err in (m.length_mm - scene.length_mm, m.width_mm - scene.width_mm):
            edge_px.append(err / (np.sqrt(2.0) * scale))  # each dimension has two sides

    c = np.concatenate(corner_errs)
    e = np.array(edge_px)
    print(f"calibration scenes: {args.n} (seed {CALIBRATION_SEED})")
    print(f"corner error px: mean {c.mean(axis=0).round(4)}  sd per axis {c.std(axis=0).round(4)}"
          f"  -> CORNER_SIGMA_PX = {np.sqrt(np.mean(c ** 2)):.3f} (rms)")
    print(f"edge error px per side: mean {e.mean():+.4f}  sd {e.std():.4f}"
          f"  -> EDGE_SIGMA_PX = {np.sqrt(np.mean(e ** 2)):.3f} (rms)")


if __name__ == "__main__":
    main()
