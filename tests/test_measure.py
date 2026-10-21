from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from evaluate import run
from photosize import Code, Measurement, make_scene, measure
from photosize.synth import paint


def same(a: float, b: float) -> bool:
    """Equality that treats NaN as equal to NaN (a rejected result has NaN sizes)."""
    return (math.isnan(a) and math.isnan(b)) or a == b


@pytest.mark.parametrize("shape", ["rect", "ellipse"])
def test_oracle_draws_the_true_area(shape: str) -> None:
    # The ground truth is checked too: a wrong oracle makes every score meaningless.
    canvas = np.zeros((900, 900), np.uint8)
    paint(canvas, shape, 450.3, 449.8, 300.0, 120.0, 33.0, grey=255)  # type: ignore[arg-type]
    true = 4 * 300 * 120 if shape == "rect" else np.pi * 300 * 120
    assert abs(canvas.sum() / 255 - true) / true < 1e-3


def test_oracle_marker_corners_have_no_offset() -> None:
    # A pixel-centre mistake in the oracle shows up as a constant offset between true and found corners.
    errs = []
    for s in range(10):
        scene = make_scene(np.random.default_rng(s), degrade=False)
        m = measure(scene.image, scene.marker_mm)
        assert m.marker_px is not None
        errs.append(m.marker_px - scene.marker_corners_px)
    assert np.abs(np.mean(np.concatenate(errs), axis=0)).max() < 0.02  # px


def test_accuracy_on_held_out_scenes() -> None:
    results = run(40, seed=7)
    errs = np.array([r.length - r.true_length for r in results if not r.problems])
    assert len(errs) >= 38
    assert np.median(np.abs(errs)) < 0.2  # mm
    assert abs(errs.mean()) < 0.1  # no systematic bias


def test_rejects_object_cut_by_frame() -> None:
    scene = make_scene(np.random.default_rng(3))
    full = measure(scene.image, scene.marker_mm)
    assert full.ok and full.contour_px is not None
    cut_x = int(full.contour_px[:, 0].mean())  # crop through the middle of the object
    cut = measure(np.ascontiguousarray(scene.image[:, :cut_x]), scene.marker_mm)
    assert Code.CROSSES_EDGE in cut.codes


def test_same_answer_at_lower_resolution() -> None:
    scene = make_scene(np.random.default_rng(11))
    full = measure(scene.image, scene.marker_mm)
    smaller = np.asarray(cv2.resize(scene.image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA), np.uint8)
    small = measure(smaller, scene.marker_mm)
    assert full.ok and small.ok
    assert abs(full.length_mm - small.length_mm) < 0.3  # mm
    assert abs(full.width_mm - small.width_mm) < 0.3


def _key(m: Measurement) -> tuple[tuple[float, ...], tuple[str, ...]]:
    return (m.length_mm, m.width_mm, m.length_sd_mm, m.width_sd_mm), tuple(sorted(m.codes))


def test_concurrent_calls_match_sequential() -> None:
    scenes = [make_scene(np.random.default_rng(s)) for s in range(8)]
    seq = [_key(measure(s.image, s.marker_mm)) for s in scenes]
    with ThreadPoolExecutor(max_workers=8) as pool:
        par = list(pool.map(lambda s: _key(measure(s.image, s.marker_mm)), scenes * 4))
    for i, (sizes, codes) in enumerate(par):
        ref_sizes, ref_codes = seq[i % 8]
        assert codes == ref_codes
        assert all(same(a, b) for a, b in zip(sizes, ref_sizes))


def test_evaluation_does_not_depend_on_worker_count() -> None:
    one, four = run(16, seed=5, workers=1), run(16, seed=5, workers=4)
    assert [r.index for r in one] == [r.index for r in four]
    for a, b in zip(one, four):
        assert a.problems == b.problems
        assert all(same(x, y) for x, y in zip((a.length, a.width, a.length_sd), (b.length, b.width, b.length_sd)))
