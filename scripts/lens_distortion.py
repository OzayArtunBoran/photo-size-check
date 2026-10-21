"""How much radial lens distortion moves the result, and whether the uncertainty notices.

One marker cannot see lens distortion: a single homography has no term for it. This bends the
synthetic photos with r' = r (1 + k1 r^2) (r normalised by the image width) and re-measures.
Usage: python scripts/lens_distortion.py [--n 60]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photosize import make_scene, measure  # noqa: E402


def distort(img: NDArray[np.uint8], k1: float) -> NDArray[np.uint8]:
    h, w = img.shape[:2]
    f, cx, cy = float(w), w / 2, h / 2
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = (xs - cx) / f, (ys - cy) / f
    s = 1 + k1 * (xn * xn + yn * yn)
    mx, my = (xn * s * f + cx).astype(np.float32), (yn * s * f + cy).astype(np.float32)
    return np.asarray(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    scenes = [make_scene(np.random.default_rng(q)) for q in np.random.SeedSequence(21).spawn(args.n)]
    for k1 in (0.0, 0.02, 0.05, 0.10):
        rel, inside, ok = [], 0, 0
        for s in scenes:
            m = measure(distort(s.image, k1), s.marker_mm)
            if not m.ok:
                continue
            ok += 1
            e = m.length_mm - s.length_mm
            rel.append(e / s.length_mm)
            inside += int(abs(e) <= 2 * m.length_sd_mm)
        r = np.array(rel)
        print(f"k1={k1:.2f}: accepted {ok}/{args.n}  median length error {100 * np.median(r):+.2f}%"
              f"  max |error| {100 * np.abs(r).max():.2f}%  inside 2 sd {100 * inside / max(ok, 1):.0f}%")


if __name__ == "__main__":
    main()
