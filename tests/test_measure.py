from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from evaluate import run
from photosize import make_scene, measure
from photosize.synth import paint


@pytest.mark.parametrize("shape", ["rect", "ellipse"])
def test_oracle_draws_the_true_area(shape: str) -> None:
    # The ground truth is checked too: a wrong oracle makes every score meaningless.
    sheet = np.zeros((900, 900), np.uint8)
    paint(sheet, shape, 450.3, 449.8, 300.0, 120.0, 33.0, grey=255)  # type: ignore[arg-type]
    true = 4 * 300 * 120 if shape == "rect" else np.pi * 300 * 120
    assert abs(sheet.sum() / 255 - true) / true < 1e-3


def test_accuracy_on_held_out_scenes() -> None:
    results = run(40, seed=7)
    errs = np.array([r.length - r.true_length for r in results if not r.problems])
    assert len(errs) >= 38
    assert np.median(np.abs(errs)) < 0.4  # mm
    assert abs(errs.mean()) < 0.2  # no systematic bias


def test_rejects_photo_without_marker() -> None:
    scene = make_scene(np.random.default_rng(3))
    img = scene.image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[gray < 60] = 235  # paint over the marker's black cells (and the object)
    assert not measure(img, scene.marker_mm).ok


def test_rejects_object_cut_by_frame() -> None:
    scene = make_scene(np.random.default_rng(3))
    m_full = measure(scene.image, scene.marker_mm)
    assert m_full.ok and m_full.contour_px is not None
    cut_x = int(m_full.contour_px[:, 0].mean())  # crop through the middle of the object
    m_cut = measure(np.ascontiguousarray(scene.image[:, :cut_x]), scene.marker_mm)
    assert not m_cut.ok


def test_same_answer_at_lower_resolution() -> None:
    scene = make_scene(np.random.default_rng(11))
    full = measure(scene.image, scene.marker_mm)
    small = measure(cv2.resize(scene.image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA),
                    scene.marker_mm)
    assert full.ok and small.ok
    assert abs(full.length_mm - small.length_mm) < 3 * max(full.length_sd_mm, small.length_sd_mm)


def test_concurrent_calls_match_sequential() -> None:
    scenes = [make_scene(np.random.default_rng(s)) for s in range(8)]
    seq = [measure(s.image, s.marker_mm) for s in scenes]
    with ThreadPoolExecutor(max_workers=8) as pool:
        par = list(pool.map(lambda s: measure(s.image, s.marker_mm), scenes * 4))
    for i, m in enumerate(par):
        ref = seq[i % 8]
        assert (m.length_mm, m.width_mm, m.length_sd_mm) == (ref.length_mm, ref.width_mm, ref.length_sd_mm)


def test_evaluation_does_not_depend_on_worker_count() -> None:
    assert run(16, seed=5, workers=1) == run(16, seed=5, workers=4)
