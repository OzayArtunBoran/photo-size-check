"""Measure real photos of the screen slides against their exact sizes.

Each photo is measured with the whole board and, for comparison, with its first marker alone.
Photos are matched to slides by the order given (`--order`, one slide number per photo in
file-name order); the measured size is printed next to the expected one so a wrong order is
visible. Overlays go to --out/overlays (kept out of git: they show the room around the screen).

Usage:
  python scripts/measure_photos.py --photos real/raw --slides real/slides.json \
      --order 8,8,8,7,7,7,6,6,6,5,5,5,4,4,4,3,3,3,2,2,2,1,1,1 --out real/results/v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photosize import Board, Measurement, draw, measure  # noqa: E402


def board_of(slide: dict[str, object]) -> Board:
    b = slide["board"]
    assert isinstance(b, dict)
    return Board(float(b["side_mm"]), {int(k): (float(v[0]), float(v[1])) for k, v in b["markers"].items()})


def row_for(name: str, slide: dict[str, object], label: str, m: Measurement) -> dict[str, object]:
    obj = slide["objects"][0]  # type: ignore[index]
    true_l, true_w = float(obj["length_mm"]), float(obj["width_mm"])
    return {
        "photo": name, "slide": slide["slide"], "title": slide["title"], "expected": slide["expected"],
        "reference": label, "ok": m.ok, "codes": " ".join(sorted(c.value for c in m.codes)),
        "true_length": true_l, "true_width": true_w,
        "length": round(m.length_mm, 3), "width": round(m.width_mm, 3),
        "length_sd": round(m.length_sd_mm, 3), "width_sd": round(m.width_sd_mm, 3),
        "misfit_mm": round(m.misfit_mm, 3), "radial_k": round(m.radial_k, 4),
    }


def summarise(rows: list[dict[str, object]], label: str) -> None:
    mine = [r for r in rows if r["reference"] == label]
    measure_rows = [r for r in mine if r["expected"] == "measure"]
    refuse_rows = [r for r in mine if r["expected"] != "measure"]
    ok = [r for r in measure_rows if r["ok"]]
    print(f"\n[{label}] should measure: {len(ok)}/{len(measure_rows)} measured; "
          f"should refuse: {sum(not r['ok'] for r in refuse_rows)}/{len(refuse_rows)} refused")
    if not ok:
        return
    el = np.array([float(r["length"]) - float(r["true_length"]) for r in ok])  # type: ignore[arg-type]
    ew = np.array([float(r["width"]) - float(r["true_width"]) for r in ok])  # type: ignore[arg-type]
    rl = el / np.array([float(r["true_length"]) for r in ok])  # type: ignore[arg-type]
    rw = ew / np.array([float(r["true_width"]) for r in ok])  # type: ignore[arg-type]
    zl = el / np.array([float(r["length_sd"]) for r in ok])  # type: ignore[arg-type]
    zw = ew / np.array([float(r["width_sd"]) for r in ok])  # type: ignore[arg-type]
    print(f"  error mm: length median {np.median(el):+.2f} (|max| {np.abs(el).max():.2f}), width median {np.median(ew):+.2f} (|max| {np.abs(ew).max():.2f})")
    print(f"  error %:  length median {100 * np.median(rl):+.2f}%, width median {100 * np.median(rw):+.2f}%")
    print(f"  true value inside ±2 reported sd: length {100 * np.mean(np.abs(zl) <= 2):.0f}%, width {100 * np.mean(np.abs(zw) <= 2):.0f}%"
          f"   (error/sd: median |z| {np.median(np.abs(zl)):.1f} / {np.median(np.abs(zw)):.1f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--order", required=True, help="slide number per photo, in file-name order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    slides = {int(s["slide"]): s for s in json.loads(Path(args.slides).read_text())}
    photos = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    order = [int(x) for x in args.order.split(",")]
    if len(order) != len(photos):
        raise SystemExit(f"{len(photos)} photos but {len(order)} slide numbers in --order")
    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for photo, number in zip(photos, order):
        slide = slides[number]
        board = board_of(slide)
        read = cv2.imread(str(photo))  # EXIF is dropped: nothing written below carries it
        if read is None:
            raise SystemExit(f"cannot read {photo}")
        image = np.asarray(read, np.uint8)
        refs = [("board", board)] + ([("one marker", board.subset([min(board.markers)]))] if len(board.markers) > 1 else [])
        for label, ref in refs:
            m = measure(image, ref)
            rows.append(row_for(photo.name, slide, label, m))
            if label == "board":
                small = cv2.resize(draw(image, m), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / "overlays" / f"{photo.stem}_slide{number}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            got = f"{m.length_mm:7.2f} x {m.width_mm:6.2f}" if m.ok else "refused: " + ",".join(sorted(c.value for c in m.codes))
            obj = slide["objects"][0]
            print(f"{photo.name:24s} slide {number:2d} {label:10s} {got:40s} true {obj['length_mm']:.2f} x {obj['width_mm']:.2f}"
                  f"  ({slide['expected']})")
    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for reference in dict.fromkeys(str(r["reference"]) for r in rows):
        summarise(rows, reference)


if __name__ == "__main__":
    main()
