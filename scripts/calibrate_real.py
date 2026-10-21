"""Size of the edge error real photos add, from a calibration round of photos only.

The synthetic oracle does not show it: on the first real photos the objects' outlines sat
0.2-0.5 mm inside their true edges while the markers' edges did not (LOG.md). Its cause is not
established, so it is not corrected; it is measured and carried as uncertainty.

For each accepted board measurement in the calibration CSV, the camera term c (mm, per side)
is chosen so that the errors divided by sqrt(sd^2 + 2 c^2) have an rms of 1: the reported sd
then matches the errors actually seen. Never run this on the test round.

Usage: python scripts/calibrate_real.py real/results/v2_B/results.csv
"""

from __future__ import annotations

import csv
import sys

import numpy as np


def main() -> None:
    rows = [r for r in csv.DictReader(open(sys.argv[1]))
            if r["reference"] == "board" and r["expected"] == "measure" and r["ok"] == "True"]
    err = np.array([float(r[k]) - float(r["true_" + k]) for r in rows for k in ("length", "width")])
    sd = np.array([float(r[k + "_sd"]) for r in rows for k in ("length", "width")])

    def rms_z(c: float) -> float:
        return float(np.sqrt(np.mean((err / np.sqrt(sd**2 + 2 * c * c)) ** 2)))

    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rms_z(mid) > 1.0 else (lo, mid)
    c = (lo + hi) / 2
    total = np.sqrt(sd**2 + 2 * c * c)
    print(f"calibration measurements: {len(rows)} photos, {len(err)} sizes")
    print(f"errors mm: mean {err.mean():+.3f}, rms {np.sqrt(np.mean(err ** 2)):.3f}, max |error| {np.abs(err).max():.3f}")
    print(f"without the camera term: rms(error/sd) {rms_z(0.0):.1f}, inside 2 sd {100 * np.mean(np.abs(err) <= 2 * sd):.0f}%")
    print(f"CAMERA_EDGE_SIGMA_MM = {c:.3f}  ->  inside 2 sd {100 * np.mean(np.abs(err) <= 2 * total):.0f}% on this round")


if __name__ == "__main__":
    main()
