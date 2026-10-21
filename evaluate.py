"""Score the measurement against generated scenes with known true sizes.

Usage: python evaluate.py [--n 300] [--seed 1] [--workers 4]
Prints the error statistics and writes overlays of the worst results to debug/, so they can be
looked at. Only this process writes files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from photosize import draw, make_scene, measure
from photosize.evaluation import run, summary
from photosize.synth import CORNER_BOARD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--four", action="store_true", help="four markers in the sheet's corners")
    args = ap.parse_args()

    results = run(args.n, args.seed, args.workers, args.four)
    print("\n".join(summary(results)))

    out = Path("debug")
    out.mkdir(exist_ok=True)
    children = np.random.SeedSequence(args.seed).spawn(args.n)
    ok = [r for r in results if not r.problems]
    for r in sorted(ok, key=lambda r: -abs(r.length - r.true_length))[:5]:
        scene = make_scene(np.random.default_rng(children[r.index]), board=CORNER_BOARD if args.four else None)
        img = draw(scene.image, measure(scene.image, scene.board, camera_edge_mm=0.0))
        label = f"true {scene.length_mm:.1f} x {scene.width_mm:.1f} mm ({scene.objects[0].shape})"
        cv2.putText(img, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(str(out / f"worst_{r.index:03d}_{r.length - r.true_length:+.2f}mm.png"), img)


if __name__ == "__main__":
    main()
