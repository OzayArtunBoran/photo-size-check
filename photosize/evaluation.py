"""Score measure() against generated scenes whose true sizes are known.

`run` spreads scenes over worker processes. Each scene gets its own child seed from one
SeedSequence, so the results do not depend on the number of workers or on which one finishes
first. Workers only compute and return values; nothing is written from a worker.

Workers are started with "spawn" on every platform. Forking a process whose OpenCV thread pool
is already running is unsafe: on Linux CI the pool hung, on macOS the workers crashed (LOG.md).
With workers > 1, call `run` from under `if __name__ == "__main__":`, as evaluate.py does.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from .measure import measure
from .synth import CORNER_BOARD, make_scene


@dataclass(frozen=True)
class Result:
    index: int
    true_length: float
    true_width: float
    length: float
    width: float
    length_sd: float
    width_sd: float
    distance: float  # object centre to marker centre in the photo, in marker sides
    problems: tuple[str, ...]  # problem codes; empty when the size was accepted


def score_scene(job: tuple[int, np.random.SeedSequence, bool]) -> Result:
    index, seq, four = job
    scene = make_scene(np.random.default_rng(seq), board=CORNER_BOARD if four else None)
    m = measure(scene.image, scene.board, camera_edge_mm=0.0)  # synthetic: no camera edge error
    distance = float("nan")
    if m.marker_px is not None and m.contour_px is not None:
        side = float(np.linalg.norm(m.marker_px[1] - m.marker_px[0]))
        distance = float(np.linalg.norm(m.contour_px.mean(axis=0) - m.marker_px.mean(axis=0))) / side
    return Result(index, scene.length_mm, scene.width_mm, m.length_mm, m.width_mm,
                  m.length_sd_mm, m.width_sd_mm, distance, tuple(sorted(p.code.value for p in m.problems)))


def _one_opencv_thread() -> None:
    cv2.setNumThreads(1)  # one OpenCV thread per worker process: workers do not compete for cores


def run(n: int, seed: int, workers: int = 1, four: bool = False) -> list[Result]:
    """Score n scenes; `four` puts four markers in the sheet's corners instead of one."""
    jobs = [(i, seq, four) for i, seq in enumerate(np.random.SeedSequence(seed).spawn(n))]
    if workers <= 1:
        return [score_scene(j) for j in jobs]
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=spawn, initializer=_one_opencv_thread) as pool:
        return list(pool.map(score_scene, jobs, chunksize=8))  # map keeps input order


def summary(results: list[Result]) -> list[str]:
    ok = [r for r in results if not r.problems]
    rejected: dict[str, int] = {}
    for r in results:
        for code in r.problems:
            rejected[code] = rejected.get(code, 0) + 1
    lines = [f"scenes {len(results)} | measured {len(ok)} | rejected {len(results) - len(ok)} {rejected or ''}".rstrip()]
    if not ok:
        return lines + ["nothing was measured, so there are no errors to report"]

    el = np.array([r.length - r.true_length for r in ok])
    ew = np.array([r.width - r.true_width for r in ok])
    zl = el / np.array([r.length_sd for r in ok])
    zw = ew / np.array([r.width_sd for r in ok])
    rel = np.abs(el) / np.array([r.true_length for r in ok])
    for name, e in (("length", el), ("width ", ew)):
        lines.append(f"{name} error mm: median {np.median(np.abs(e)):.2f}  p95 {np.percentile(np.abs(e), 95):.2f}"
                     f"  max {np.abs(e).max():.2f}  mean(signed) {e.mean():+.3f}")
    lines.append(f"length relative error: median {100 * np.median(rel):.2f}%  max {100 * rel.max():.2f}%")
    lines.append(f"true value inside ±2 reported sd: length {100 * np.mean(np.abs(zl) <= 2):.1f}%"
                 f"  width {100 * np.mean(np.abs(zw) <= 2):.1f}%   (error/sd has sd {zl.std():.2f} / {zw.std():.2f})")
    d = np.array([r.distance for r in ok])
    if len(ok) >= 30 and np.isfinite(d).all():
        edges = np.quantile(d, [0, 1 / 3, 2 / 3, 1])
        parts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (d >= lo) & (d <= hi)
            parts.append(f"{lo:.1f}-{hi:.1f}: {100 * np.mean(np.abs(zl[sel]) <= 2):.0f}%")
        lines.append("length coverage by distance from the marker (marker sides): " + ", ".join(parts))
    return lines
