"""Scenes from the independent review. Each one used to return a wrong size marked as ok.

The property under test: a result is either right (within 0.5 mm) or rejected with a reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from photosize import Code, Measurement, ObjectSpec, measure, render_scene

TOL_MM = 0.5


def rect(cx: float, cy: float, length: float = 60.0, width: float = 30.0,
         angle: float = 0.0, grey: int = 40) -> ObjectSpec:
    return ObjectSpec("rect", cx, cy, length, width, angle, grey)


def shoot(objects: list[ObjectSpec], seed: int = 1, **kw: object) -> Measurement:
    scene = render_scene(objects, rng=np.random.default_rng(seed), **kw)  # type: ignore[arg-type]
    return measure(scene.image, scene.marker_mm)


def right_or_rejected(m: Measurement, length: float, width: float) -> bool:
    return not m.ok or (abs(m.length_mm - length) < TOL_MM and abs(m.width_mm - width) < TOL_MM)


@pytest.mark.parametrize("gap_mm", [0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
def test_object_next_to_the_marker(gap_mm: float) -> None:
    # The marker covers x 15..55 mm and y 15..55 mm of the sheet.
    m = shoot([rect(55.0 + gap_mm + 30.0, 35.0)])
    assert right_or_rejected(m, 60.0, 30.0), (m.length_mm, m.problems)
    if gap_mm == 0.0:
        assert Code.TOUCHES_MARKER in m.codes
    if gap_mm >= 2.0:
        assert m.ok, m.problems


def test_several_objects_are_rejected() -> None:
    m = shoot([rect(130.0, 105.0), ObjectSpec("ellipse", 150.0, 45.0, 70.0, 40.0, 20.0)])
    assert Code.SEVERAL_OBJECTS in m.codes


def test_object_crossing_the_sheet_edge_is_rejected() -> None:
    m = shoot([rect(185.0, 75.0)])  # spans x 155..215 mm on a 200 mm sheet
    assert Code.CROSSES_EDGE in m.codes


def test_object_cut_by_the_photo_frame_is_rejected() -> None:
    # The sheet runs out of the photo on the right; the big object is cut by the frame and a
    # small one stays fully visible. The small one must not be reported as "the" object.
    quad = np.array([[100, 100], [1500, 130], [1480, 850], [120, 830]], np.float64)
    m = shoot([rect(160.0, 75.0, 70.0, 30.0), rect(95.0, 115.0, 25.0, 12.0)], sheet_quad_px=quad)
    assert not m.ok
    assert Code.CROSSES_EDGE in m.codes


@pytest.mark.parametrize("grey", [40, 100, 140, 156, 158, 170, 190])
def test_grey_objects_are_measured(grey: int) -> None:
    for seed in range(3):
        m = shoot([rect(130.0, 80.0, angle=20.0, grey=grey)], seed=seed)
        assert m.ok, (grey, seed, m.problems)
        assert right_or_rejected(m, 60.0, 30.0), (grey, seed, m.length_mm, m.width_mm)


def test_faint_object_is_rejected_not_guessed() -> None:
    m = shoot([rect(130.0, 80.0, grey=220)])  # 15 grey levels below the sheet
    assert not m.ok
    assert m.codes & {Code.LOW_CONTRAST, Code.NO_OBJECT}


def test_empty_sheet() -> None:
    assert Code.NO_OBJECT in shoot([]).codes


def test_no_marker() -> None:
    assert Code.NO_MARKER in shoot([rect(130.0, 80.0)], with_marker=False).codes


@pytest.mark.parametrize("corner", [(184.0, 139.0), (16.0, 139.0), (184.0, 11.0)])
def test_object_in_a_sheet_corner_keeps_its_size(corner: tuple[float, float]) -> None:
    # A 30 x 20 mm object 1 mm from two edges of the sheet (200 x 150 mm) is measured in full.
    cx, cy = corner
    m = shoot([rect(cx, cy, 30.0, 20.0)])
    assert right_or_rejected(m, 30.0, 20.0), (m.length_mm, m.width_mm, m.problems)
    assert m.ok, m.problems
