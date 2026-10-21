"""Overlays for the README: one accepted measurement and two refusals.

Usage: python scripts/figures.py  (writes docs/examples.png)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photosize import ObjectSpec, draw, measure, render_scene  # noqa: E402


def panel(objects: list[ObjectSpec], title: str, quad: np.ndarray | None = None) -> np.ndarray:
    scene = render_scene(objects, sheet_quad_px=quad, rng=np.random.default_rng(4))
    m = measure(scene.image, scene.marker_mm, camera_edge_mm=0.0)
    img = np.asarray(cv2.resize(draw(scene.image, m), (640, 480), interpolation=cv2.INTER_AREA), np.uint8)
    img[:34] = 255
    img[-34:] = 255
    result = (f"{m.length_mm:.1f} x {m.width_mm:.1f} mm, 2 sd {2 * m.length_sd_mm:.2f} x {2 * m.width_sd_mm:.2f}"
              if m.ok else "rejected: " + ", ".join(sorted(m.codes)))
    colour = (0, 140, 0) if m.ok else (0, 0, 200)
    cv2.putText(img, result, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)
    cv2.putText(img, title, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 2, cv2.LINE_AA)
    return img


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "docs"
    out.mkdir(exist_ok=True)
    cut_quad = np.array([[100, 100], [1500, 130], [1480, 850], [120, 830]], np.float64)
    panels = [
        panel([ObjectSpec("rect", 130.0, 85.0, 70.0, 35.0, 25.0)], "true 70.0 x 35.0 mm"),
        panel([ObjectSpec("rect", 86.0, 35.0, 60.0, 30.0)], "1 mm from the marker"),
        panel([ObjectSpec("rect", 160.0, 75.0, 70.0, 30.0), ObjectSpec("rect", 95.0, 115.0, 25.0, 12.0)],
              "cut by the photo frame", cut_quad),
    ]
    cv2.imwrite(str(out / "examples.png"), np.hstack(panels))
    print("wrote", out / "examples.png")


if __name__ == "__main__":
    main()
