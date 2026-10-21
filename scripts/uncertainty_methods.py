"""Why the corner uncertainty is propagated with a Jacobian instead of Monte Carlo jitter.

For the same scenes, compare:
- reference: Monte Carlo with 4,000 jittered corner sets (the slow, accurate answer);
- Monte Carlo with 32 samples (the first version), repeated with 20 different seeds;
- the Jacobian (16 box evaluations, deterministic).

Usage: python scripts/uncertainty_methods.py [--n 30]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photosize import make_scene, measure, single  # noqa: E402
from photosize.measure import CORNER_SIGMA_PX, _dims, corner_sd  # noqa: E402


def monte_carlo(marker: NDArray[np.float32], hull: NDArray[np.float32], marker_mm: float,
                samples: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    board = single(marker_mm)
    dims = [_dims((marker.astype(np.float64) + rng.normal(0.0, CORNER_SIGMA_PX, (4, 2))).reshape(-1), [0], hull, board)
            for _ in range(samples)]
    return np.asarray(np.std(np.array(dims), axis=0), np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    ratio, spread, t_jac, t_mc = [], [], 0.0, 0.0
    for seq in np.random.SeedSequence(42).spawn(args.n):
        scene = make_scene(np.random.default_rng(seq))
        m = measure(scene.image, scene.marker_mm, camera_edge_mm=0.0)
        if not m.ok or m.contour_px is None or m.marker_px is None:
            continue
        hull = np.asarray(cv2.convexHull(m.contour_px), np.float32).reshape(-1, 2)
        ref = monte_carlo(m.marker_px, hull, scene.marker_mm, 4000, seed=0)
        t0 = time.perf_counter()
        jac = corner_sd({0: m.marker_px}, hull, single(scene.marker_mm))
        t_jac += time.perf_counter() - t0
        runs = []
        for s in range(20):
            t0 = time.perf_counter()
            runs.append(monte_carlo(m.marker_px, hull, scene.marker_mm, 32, seed=s))
            t_mc += time.perf_counter() - t0
        r = np.array(runs)
        ratio.append(jac / ref)
        spread.append(r.std(axis=0) / r.mean(axis=0))
    ratio_a, spread_a = np.array(ratio), np.array(spread)
    n = len(ratio)
    print(f"scenes: {n}")
    print(f"Jacobian / reference MC (4000 samples): median {np.median(ratio_a):.3f}  range [{ratio_a.min():.3f}, {ratio_a.max():.3f}]")
    print(f"32-sample MC, spread of its own sd across 20 seeds: median {100 * np.median(spread_a):.1f}%  max {100 * spread_a.max():.1f}%")
    print(f"time per estimate: Jacobian {1000 * t_jac / n:.2f} ms, 32-sample MC {1000 * t_mc / (20 * n):.2f} ms")


if __name__ == "__main__":
    main()
