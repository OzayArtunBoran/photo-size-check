"""A4 sheet with the reference marker (50 mm) and a 100 mm scale bar, as vector graphics.

Print at 100% ("actual size", not "fit to page"), then measure the marker's outer black side
and the scale bar with a ruler: printers rescale, and every size inherits that error.
Usage: python scripts/make_marker_sheet.py  (writes real/marker_sheet.html; print it to PDF)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photosize.marker import DICTIONARY, MARKER_ID  # noqa: E402

MARKER_MM = 50.0


def main() -> None:
    cells = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICTIONARY), MARKER_ID, 6)
    n = cells.shape[0]  # 4x4 data cells plus a one-cell black border
    cell = MARKER_MM / n
    rects = "".join(
        f'<rect x="{x * cell:.4f}" y="{y * cell:.4f}" width="{cell:.4f}" height="{cell:.4f}"/>'
        for y in range(n) for x in range(n) if cells[y, x] < 128
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 0; }}
body {{ margin: 0; font-family: Helvetica, Arial, sans-serif; font-size: 3.2mm; color: #555; }}
.marker {{ position: absolute; left: 20mm; top: 20mm; width: {MARKER_MM}mm; height: {MARKER_MM}mm; }}
.bar {{ position: absolute; left: 20mm; top: 280mm; width: 100mm; height: 0; border-top: 0.2mm solid #999; }}
.tick {{ position: absolute; top: 277mm; width: 0; height: 6mm; border-left: 0.2mm solid #999; }}
.note {{ position: absolute; left: 20mm; top: 268mm; }}
</style></head><body>
<svg class="marker" viewBox="0 0 {MARKER_MM} {MARKER_MM}" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">{rects}</svg>
<div class="note">ArUco 4x4 id 0, {MARKER_MM:.0f} mm. Print at 100%. Measure the marker's black side and the bar below.</div>
<div class="bar"></div><div class="tick" style="left:20mm"></div><div class="tick" style="left:120mm"></div>
</body></html>"""
    out = Path(__file__).resolve().parents[1] / "real" / "marker_sheet.html"
    out.write_text(html)
    print("wrote", out)


if __name__ == "__main__":
    main()
