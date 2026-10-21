"""Score the measurement against generated scenes with known true sizes.

Usage: python evaluate.py [--n 300] [--seed 1] [--workers 4]

Each scene gets its own child seed from one SeedSequence, so the result does not
depend on worker count or on which worker finishes first. Workers only compute;
the parent process alone writes files, so there are no concurrent writes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from photosize import draw, make_scene, measure


@dataclass(frozen=True)
class Result:
    index: int
    true_length: float
    true_width: float
    length: float
    width: float
    length_sd: float
    problems: tuple[str, ...]


def _one(job: tuple[int, np.random.SeedSequence]) -> Result:
    index, seq = job
    scene = make_scene(np.random.default_rng(seq))
    m = measure(scene.image, scene.marker_mm)
    return Result(index, scene.length_mm, scene.width_mm, m.length_mm, m.width_mm,
                  m.length_sd_mm, tuple(m.problems))


def run(n: int, seed: int, workers: int = 1) -> list[Result]:
    jobs = list(enumerate(np.random.SeedSequence(seed).spawn(n)))
    if workers <= 1:
        return [_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, jobs, chunksize=8))  # map keeps input order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    results = run(args.n, args.seed, args.workers)
    ok = [r for r in results if not r.problems]
    rejected = [r for r in results if r.problems]
    el = np.array([r.length - r.true_length for r in ok])
    ew = np.array([r.width - r.true_width for r in ok])
    rel = np.abs(el) / np.array([r.true_length for r in ok])
    within = np.mean(np.abs(el) <= 2 * np.array([r.length_sd for r in ok]))

    print(f"scenes {args.n} | measured {len(ok)} | rejected {len(rejected)}")
    print(f"length error mm: median {np.median(np.abs(el)):.2f}  p95 {np.percentile(np.abs(el), 95):.2f}  mean(signed) {el.mean():+.2f}")
    print(f"width  error mm: median {np.median(np.abs(ew)):.2f}  p95 {np.percentile(np.abs(ew), 95):.2f}  mean(signed) {ew.mean():+.2f}")
    print(f"length relative error: median {100 * np.median(rel):.2f}%  max {100 * rel.max():.2f}%")
    print(f"true length inside ±2 reported sd: {100 * within:.0f}%")
    for r in rejected[:10]:
        print(f"  rejected scene {r.index}: {'; '.join(r.problems)}")

    # Look at the actual output: re-render the worst scenes (deterministic) and save overlays.
    out = Path("debug")
    out.mkdir(exist_ok=True)
    children = np.random.SeedSequence(args.seed).spawn(args.n)
    for r in sorted(ok, key=lambda r: -abs(r.length - r.true_length))[:5]:
        scene = make_scene(np.random.default_rng(children[r.index]))
        img = draw(scene.image, measure(scene.image, scene.marker_mm))
        cv2.putText(img, f"true {scene.length_mm:.1f} x {scene.width_mm:.1f} mm ({scene.objects[0].shape})",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(str(out / f"worst_{r.index:03d}_{r.length - r.true_length:+.2f}mm.png"), img)


if __name__ == "__main__":
    main()
