"""The optional segmentation model (photosize/segment_model.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from photosize import make_scene, measure
from photosize import segment_model as sm
from photosize.synth import CORNER_BOARD


def test_a_file_with_the_wrong_checksum_is_refused(tmp_path: Path) -> None:
    fake = tmp_path / "u2netp.onnx"
    fake.write_bytes(b"not the model")
    with pytest.raises(RuntimeError, match="SHA-256"):
        sm.ensure_model(fake)


needs_model = pytest.mark.skipif(not sm.model_path().exists(),
                                 reason="model not downloaded; CI and scripts/compare_segmentation.py fetch it")


@needs_model
def test_the_model_finds_the_object_not_the_markers() -> None:
    # Before the markers were painted out, the model segmented the four markers instead of the
    # object and 12 of 40 scenes were refused (LOG.md).
    model = sm.U2NetP()
    for seed in range(5):
        scene = make_scene(np.random.default_rng(100 + seed), board=CORNER_BOARD)
        m = measure(scene.image, scene.board, camera_edge_mm=0.0, object_mask=model.object_mask)
        assert m.ok, (seed, m.problems)
        assert abs(m.length_mm - scene.length_mm) < 0.5 and abs(m.width_mm - scene.width_mm) < 0.5
