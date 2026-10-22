"""Step 4 (finding the object) by the classical two-pass threshold vs an open-source model.

Variants, everything else in measure() unchanged:
  classical      the default
  model+refine   U^2-Netp finds the object; the classical sub-pixel outline (step 6) refines it
  model raw      U^2-Netp's own outline, no refinement

Three sets, all with known answers:
  1. held-out synthetic scenes (seed 2, four corner markers), as evaluate.py --four
  2. the independent review's scenes (tests/test_robustness.py): each must be right or refused
  3. real photos of the screen slides, rounds B and C, when the raw photos are present locally
     (they are not in the repository)

Usage: python scripts/compare_segmentation.py [--n 200] [--out real/results/segmentation]
The model is downloaded on first use (photosize/segment_model.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photosize import Board, Code, Measurement, ObjectSpec, measure, render_scene  # noqa: E402
from photosize.segment_model import U2NetP  # noqa: E402
from photosize.synth import CORNER_BOARD, make_scene  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = {  # photo folder, slide number per photo in file-name order (as in real/results/*/results.csv)
    "B": ("real/raw_v2_B", "8,8,6,7,7,6,5,5,4,4,3,3,2,2,1,1,1"),
    "C": ("real/raw_v2_C", "16,16,15,15,14,14,13,13,12,12,12,11,11,10,10,9,9"),
}


def board_of(slide: dict[str, object]) -> Board:  # as in scripts/measure_photos.py
    b = slide["board"]
    assert isinstance(b, dict)
    return Board(float(b["side_mm"]), {int(k): (float(v[0]), float(v[1])) for k, v in b["markers"].items()})


def variants(model: U2NetP) -> dict[str, dict[str, object]]:
    return {
        "classical": {},
        "model+refine": {"object_mask": model.object_mask},
        "model raw": {"object_mask": model.object_mask, "refine_edges": False},
    }


def stats(errors: list[tuple[float, float]], sds: list[tuple[float, float]]) -> str:
    if not errors:
        return "none accepted"
    e = np.abs(np.array(errors)).ravel()
    s = np.array(sds).ravel()
    inside = 100 * float(np.mean(e <= 2 * s))
    return (f"median {np.median(e):.3f} mm, p95 {np.percentile(e, 95):.3f}, max {e.max():.3f}; "
            f"inside ±2 sd {inside:.0f}%")


def synthetic(n: int, kinds: dict[str, dict[str, object]]) -> None:
    print(f"\n1. Held-out synthetic scenes (seed 2, four markers, n={n})")
    for name, kw in kinds.items():
        errors, sds, refused, spent = [], [], 0, 0.0
        for seq in np.random.SeedSequence(2).spawn(n):
            scene = make_scene(np.random.default_rng(seq), board=CORNER_BOARD)
            t = time.perf_counter()
            m = measure(scene.image, scene.board, camera_edge_mm=0.0, **kw)  # type: ignore[arg-type]
            spent += time.perf_counter() - t
            if not m.ok:
                refused += 1
                continue
            errors.append((m.length_mm - scene.length_mm, m.width_mm - scene.width_mm))
            sds.append((m.length_sd_mm, m.width_sd_mm))
        print(f"  {name:13s} accepted {n - refused}/{n}; {stats(errors, sds)}; {1000 * spent / n:.0f} ms per scene")


def rect(cx: float, cy: float, length: float = 60.0, width: float = 30.0, angle: float = 0.0,
         grey: int = 40) -> ObjectSpec:
    return ObjectSpec("rect", cx, cy, length, width, angle, grey)


def review_cases() -> list[tuple[str, list[ObjectSpec], dict[str, object], Callable[[Measurement], bool]]]:
    """The review's scenes with the property each must have (as in tests/test_robustness.py)."""
    def right(length: float, width: float) -> Callable[[Measurement], bool]:
        return lambda m: m.ok and abs(m.length_mm - length) < 0.5 and abs(m.width_mm - width) < 0.5

    def right_or_refused(length: float, width: float) -> Callable[[Measurement], bool]:
        return lambda m: not m.ok or right(length, width)(m)

    def refused(*codes: Code) -> Callable[[Measurement], bool]:
        return lambda m: not m.ok and bool(m.codes & set(codes))

    cases: list[tuple[str, list[ObjectSpec], dict[str, object], Callable[[Measurement], bool]]] = []
    for gap in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        check = right(60.0, 30.0) if gap >= 2.0 else right_or_refused(60.0, 30.0)
        cases.append((f"next to the marker, gap {gap:g} mm", [rect(85.0 + gap, 35.0)], {"seed": 1}, check))
    cases.append(("two objects", [rect(130.0, 105.0), ObjectSpec("ellipse", 150.0, 45.0, 70.0, 40.0, 20.0)],
                  {"seed": 1}, refused(Code.SEVERAL_OBJECTS)))
    cases.append(("crosses the sheet edge", [rect(185.0, 75.0)], {"seed": 1}, refused(Code.CROSSES_EDGE)))
    quad = np.array([[100, 100], [1500, 130], [1480, 850], [120, 830]], np.float64)
    cases.append(("cut by the photo frame", [rect(160.0, 75.0, 70.0, 30.0), rect(95.0, 115.0, 25.0, 12.0)],
                  {"seed": 1, "sheet_quad_px": quad}, refused(Code.CROSSES_EDGE)))
    for grey in (40, 100, 140, 156, 158, 170, 190):
        for seed in range(3):
            cases.append((f"grey {grey}, seed {seed}", [rect(130.0, 80.0, angle=20.0, grey=grey)], {"seed": seed},
                          right(60.0, 30.0)))
    cases.append(("faint object (grey 220)", [rect(130.0, 80.0, grey=220)], {"seed": 1},
                  refused(Code.LOW_CONTRAST, Code.NO_OBJECT)))
    cases.append(("empty sheet", [], {"seed": 1}, refused(Code.NO_OBJECT)))
    return cases


def review(kinds: dict[str, dict[str, object]]) -> None:
    cases = review_cases()
    print(f"\n2. The independent review's scenes ({len(cases)}): right, or refused where refusing is right")
    for name, kw in kinds.items():
        failed = []
        for title, objects, opts, check in cases:
            seed = int(opts.get("seed", 1))  # type: ignore[call-overload]
            extra = {k: v for k, v in opts.items() if k != "seed"}
            scene = render_scene(objects, rng=np.random.default_rng(seed), **extra)  # type: ignore[arg-type]
            m = measure(scene.image, scene.marker_mm, camera_edge_mm=0.0, **kw)  # type: ignore[arg-type]
            if not check(m):
                got = f"{m.length_mm:.2f} x {m.width_mm:.2f}" if m.ok else ",".join(sorted(m.codes))
                failed.append(f"{title} -> {got}")
        print(f"  {name:13s} {len(cases) - len(failed)}/{len(cases)} as required" + "".join(f"\n      FAIL {f}" for f in failed))


def real(kinds: dict[str, dict[str, object]], out: Path) -> None:
    slides = {int(s["slide"]): s for s in json.loads((ROOT / "real/slides_v2.json").read_text())}
    rows: list[dict[str, object]] = []
    for label, (folder, order) in ROUNDS.items():
        photos = sorted(p for p in (ROOT / folder).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        numbers = [int(x) for x in order.split(",")]
        if len(photos) != len(numbers):
            print(f"\n3. Real photos: round {label} skipped ({len(photos)} photos in {folder}, {len(numbers)} expected)")
            continue
        for photo, number in zip(photos, numbers):
            read = cv2.imread(str(photo))
            if read is None:
                raise SystemExit(f"cannot read {photo}")
            slide = slides[number]
            obj = slide["objects"][0]
            for name, kw in kinds.items():
                t = time.perf_counter()
                m = measure(np.asarray(read, np.uint8), board_of(slide), **kw)  # type: ignore[arg-type]
                rows.append({"round": label, "photo": photo.name, "slide": number, "expected": slide["expected"],
                             "variant": name, "ok": m.ok, "codes": " ".join(sorted(c.value for c in m.codes)),
                             "true_length": obj["length_mm"], "true_width": obj["width_mm"],
                             "length": round(m.length_mm, 3), "width": round(m.width_mm, 3),
                             "length_sd": round(m.length_sd_mm, 3), "width_sd": round(m.width_sd_mm, 3),
                             "seconds": round(time.perf_counter() - t, 3)})
    if not rows:
        return
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "real.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for label in dict.fromkeys(str(r["round"]) for r in rows):
        print(f"\n3. Real photos, round {label} (board, camera term as calibrated on round B)")
        for name in kinds:
            mine = [r for r in rows if r["round"] == label and r["variant"] == name]
            should = [r for r in mine if r["expected"] == "measure"]
            ok = [r for r in should if r["ok"]]
            refuse = [r for r in mine if r["expected"] != "measure"]
            errors = [(float(r["length"]) - float(r["true_length"]), float(r["width"]) - float(r["true_width"])) for r in ok]  # type: ignore[arg-type]
            sds = [(float(r["length_sd"]), float(r["width_sd"])) for r in ok]  # type: ignore[arg-type]
            wrongly = [r for r in refuse if r["ok"]]
            seconds = float(np.mean([float(r["seconds"]) for r in mine]))  # type: ignore[arg-type]
            print(f"  {name:13s} measured {len(ok)}/{len(should)}, refused {len(refuse) - len(wrongly)}/{len(refuse)}; "
                  f"{stats(errors, sds)}; {1000 * seconds:.0f} ms per photo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="real/results/segmentation")
    args = ap.parse_args()
    cv2.setNumThreads(1)  # same conditions for every variant
    kinds = variants(U2NetP(passes=1))
    synthetic(args.n, kinds)
    review(kinds)
    real(kinds, ROOT / args.out)


if __name__ == "__main__":
    main()
