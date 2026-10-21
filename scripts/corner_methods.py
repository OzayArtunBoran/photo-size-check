"""Marker-corner error against the true corners: OpenCV's refinements vs the line fit.

Runs on both renderers: supersample=1 is the first version's bilinear warp (it aliases the
marker edges), supersample=4 is the current area-integrated camera.
Usage: python scripts/corner_methods.py [--n 150]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photosize import make_scene  # noqa: E402
from photosize.marker import DICTIONARY, MARKER_ID  # noqa: E402
from photosize.measure import _refine_marker  # noqa: E402

METHODS = {
    "SUBPIX": cv2.aruco.CORNER_REFINE_SUBPIX,
    "CONTOUR": cv2.aruco.CORNER_REFINE_CONTOUR,
    "APRILTAG": cv2.aruco.CORNER_REFINE_APRILTAG,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    for supersample in (1, 4):
        errors: dict[str, list[float]] = {name: [] for name in [*METHODS, "SUBPIX + line fit"]}
        for seq in np.random.SeedSequence(1000).spawn(args.n):
            scene = make_scene(np.random.default_rng(seq), supersample=supersample)
            gray = np.asarray(cv2.cvtColor(scene.image, cv2.COLOR_BGR2GRAY), np.uint8)
            for name, method in METHODS.items():
                params = cv2.aruco.DetectorParameters()
                params.cornerRefinementMethod = method
                corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
                if ids is None or MARKER_ID not in ids.ravel():
                    continue
                found = corners[[int(i) for i in ids.ravel()].index(MARKER_ID)].reshape(4, 2)
                errors[name] += list(np.linalg.norm(found - scene.marker_corners_px, axis=1))
                if name == "SUBPIX":
                    fitted = _refine_marker(gray, found.astype(np.float32))
                    errors["SUBPIX + line fit"] += list(np.linalg.norm(fitted - scene.marker_corners_px, axis=1))
        label = "area-integrated (current)" if supersample > 1 else "bilinear (first version)"
        print(f"renderer: {label}, {args.n} scenes, seed 1000")
        for name, e in errors.items():
            a = np.array(e)
            print(f"  {name:18s} corner error px: mean {a.mean():.3f}  median {np.median(a):.3f}  p95 {np.percentile(a, 95):.3f}")
        sub, fit = np.mean(errors["SUBPIX"]), np.mean(errors["SUBPIX + line fit"])
        print(f"  line fit vs SUBPIX: {sub / fit:.1f}x smaller mean error")


if __name__ == "__main__":
    main()
