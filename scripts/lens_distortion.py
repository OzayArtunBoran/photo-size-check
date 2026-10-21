"""How radial lens distortion moves the result, with one marker and with four.

A single homography has no term for lens distortion, and one marker gives only four points, so
nothing can be estimated. With four markers spread over the sheet, measure() estimates the
radial distortion from how the 16 corners fit one plane and removes it. This bends synthetic
photos with r' = r (1 + k1 r^2) (r normalised by the image width) and re-measures both ways.
Note: the bending uses the same one-parameter radial form that measure() estimates; a real
lens can differ, which only real photos can show.
Usage: python scripts/lens_distortion.py [--n 60]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photosize import make_scene, measure  # noqa: E402
from photosize.synth import CORNER_BOARD, radial_distort  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    scenes = [make_scene(np.random.default_rng(q), board=CORNER_BOARD) for q in np.random.SeedSequence(21).spawn(args.n)]
    for k1 in (0.0, 0.02, 0.05, 0.10):
        for name, ref in (("one marker", CORNER_BOARD.subset([0])), ("four markers", CORNER_BOARD)):
            rel, inside, ok = [], 0, 0
            for s in scenes:
                m = measure(radial_distort(s.image, k1), ref, camera_edge_mm=0.0)
                if not m.ok:
                    continue
                ok += 1
                e = m.length_mm - s.length_mm
                rel.append(e / s.length_mm)
                inside += int(abs(e) <= 2 * m.length_sd_mm)
            r = np.abs(np.array(rel)) if rel else np.array([np.nan])
            print(f"k1={k1:.2f} {name:12s}: accepted {ok}/{args.n}  median |error| {100 * np.median(r):.2f}%"
                  f"  max {100 * np.max(r):.2f}%  true value inside 2 sd {100 * inside / max(ok, 1):.0f}%")


if __name__ == "__main__":
    main()
