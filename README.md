# photo-size-check

Measure a flat object from one photo, using a printed ArUco marker of known size, and refuse to answer when the result cannot be trusted.

I built this with Claude Code. The main point of the repo is how the code is checked:
- Every number is compared with a size that is known to be true.
- The problems those checks found are written down in [LOG.md](LOG.md), including the ones they missed.
- A second, independent review was asked to break the code.

![One accepted measurement and two refusals](docs/examples.png)

*Left: measured, with its uncertainty. Middle and right: refused, with the reason.*

## Read this first

- **All numbers here come from synthetic photos with exactly known sizes. Nothing has been tested on real photos yet.** The synthetic set models blur, uneven light, sensor noise and JPEG compression. It does not model lens distortion, the scale error of a printed marker, shadows, a curled sheet or real textures. See [Known limits](#known-limits).
- The object must be:
  - flat, on the same plane as the marker;
  - darker than the sheet;
  - the only object on the sheet.

## How it measures (`photosize/measure.py`)

1. **Find the marker.** Find exactly one marker (4x4, id 0). Refine its corners by fitting a straight line to dozens of sub-pixel points along each outer edge, then intersecting neighbouring lines.
2. **Map to millimetres.** Build a homography from the marker's corners in the photo to its true corners in mm.
3. **Find the sheet.** Find the light sheet that carries the marker.
4. **Find the object.** Inside the sheet, mark what is clearly darker than the sheet itself. The sheet's brightness is fitted as a smooth surface, so uneven light does not matter. The threshold is set by Otsu on that darkness, so grey objects are found as well as black ones.
5. **Refuse when unsure.** If anything is ambiguous, refuse with a code (table below).
6. **Measure the outline.**
   - Threshold halfway between the object's own grey level and the sheet around it, on a 4x upsampled crop around the object only.
   - Map the outline's convex hull to mm.
   - The minimum-area box gives length x width.
7. **Estimate the uncertainty.** It combines three measured sources:
   - **Marker corners**, propagated through a Jacobian (see the next section).
   - **The outline's own edge error**, scaled by the local mm per pixel.
   - **How sharply the box orientation is defined.** For a near-circle the box can turn and trade length for width.

When `measure()` returns `ok == False`, `problems` lists the reasons and the sizes must not be used:

| code | meaning |
|---|---|
| `no_marker`, `marker_count` | no marker, or not exactly one |
| `marker_too_small` | marker side under 40 px in the photo |
| `no_sheet` | no light surface around the marker |
| `no_object`, `object_too_small` | nothing darker than the sheet, or only specks |
| `several_objects` | more than one object on the sheet |
| `touches_marker` | the object touches the marker, so part of it may be hidden |
| `crosses_edge` | something dark cuts into the sheet's outline: an object crossing the sheet edge or the photo frame |
| `low_contrast` | the object is less than 25 grey levels darker than the sheet |
| `unclear_edge` | the fine outline runs along the limit of its search band |
| `too_uncertain` | 2 sd is more than 3% of the length or the width |

## Why the uncertainty uses a Jacobian, not Monte Carlo

The first version estimated the uncertainty by jittering the marker corners 32 times and taking the sd. It now propagates the corner error through the 2x8 Jacobian of (length, width) with respect to the 8 corner coordinates, using central differences. The comparison below comes from `scripts/uncertainty_methods.py` on 30 scenes:

- **The 32-sample estimate was itself noisy.** Its sd moved by 10% (median; max 13%) when only the random seed changed. Both the "too uncertain" gate and the coverage figure depended on luck.
- **The Jacobian gives the same answer as a slow, accurate Monte Carlo.** Against a 4,000-sample Monte Carlo the ratio has a median of 1.01 and a range of 0.92–1.23.
- **It is deterministic and cheaper.** The same input always gives the same sd. It needs no random generator, so there is nothing to share between threads. It costs 0.16 ms instead of 0.28 ms.
- **Its limit is linearity.** Corner errors are about 0.05 px, so the linear approximation holds.

## How it is checked

`photosize/synth.py` renders photos whose true sizes are known exactly:
- A sheet on a table, with the marker and one or more rectangles or ellipses.
- A random camera homography.
- Each photo pixel averages the scene over its footprint, as a sensor does (a 4x supersampled warp followed by area downsampling).
- Blur, uneven light, noise and JPEG compression.

The oracle is tested too: drawn areas match the analytic areas, and the true marker corners show no constant offset against the detected ones.

Calibration and evaluation never share scenes:
- `scripts/calibrate.py` measures the two error sizes on calibration scenes (seed 1000).
- `evaluate.py` and the tests use other seeds.

```
$ python evaluate.py --n 2000 --seed 2 --workers 4
scenes 2000 | measured 2000 | rejected 0
length error mm: median 0.06  p95 0.26  max 0.83  mean(signed) +0.009
width  error mm: median 0.04  p95 0.18  max 0.80  mean(signed) +0.029
length relative error: median 0.11%  max 1.77%
true value inside ±2 reported sd: length 98.3%  width 98.5%   (error/sd has sd 0.68 / 0.61)
length coverage by distance from the marker (marker sides): 1.5-2.4: 99%, 2.4-2.7: 98%, 2.7-3.8: 98%
```

The uncertainty is a little conservative: a Gaussian ±2 sd would cover 95%, and here it covers about 98%.

The tests (`pytest`, 28 tests, about 7 s) cover the following:
- **The review's scenes** (`tests/test_robustness.py`): an object 0–10 mm from the marker, two objects, an object crossing the sheet edge, an object cut by the photo frame next to a small one, grey objects from 40 to 190, a faint object, an empty sheet, and no marker. Each result must be right within 0.5 mm or refused. On the first version, 8 of these 33 scenes returned a wrong size marked as ok.
- **Accuracy and bias** on held-out scenes, the same answer at 75% resolution, and refusal of an object cut by the frame.
- **Concurrency:** 8 threads calling `measure()` give exactly the sequential results, and the evaluation gives identical results with 1 or 4 worker processes.

CI runs `mypy --strict` (package, CLI, tests and scripts) and the tests on every push.

## Cost and concurrency

- **Cost.**
  - `measure()` takes about 14 ms for a 1280x960 photo on my laptop.
  - It runs on the CPU with OpenCV and NumPy. There is no model and no API call.
  - The uncertainty maps only the convex hull (tens of points) to mm, not the full outline (thousands). The box is the same; I checked this on 300 jittered homographies, where the largest difference was 3e-5 mm.
- **Concurrency.**
  - `measure()` holds no shared state and uses no random numbers.
  - The evaluation gives each scene its own child seed from one `SeedSequence`, so results do not depend on worker count or order.
  - Only the parent process writes files.
  - Each worker runs OpenCV with one thread, so the workers do not compete for cores.

## Known limits

- **Real photos are not tested yet.** The next step is phone photos of objects with known sizes.
- **Printed marker scale.** `marker_mm` must be the size measured on the print. If the marker is printed at 97% ("fit to page"), every size is about 3% too large, and nothing in the photo can show this.
- **Lens distortion is not modelled.** A single marker cannot detect it. The table below comes from `scripts/lens_distortion.py` (r normalised by the image width). Every result was accepted:

  | radial distortion k1 | median length error | true value inside 2 sd |
  |---|---|---|
  | 0.05 | −0.56% | 48% |
  | 0.10 | −1.17% | 13% |

  Calibrating the camera, or covering the sheet with several markers, would fix this.
- **Flat objects only.** One photo cannot measure height, and a thick object's top face is not on the marker's plane.
- **Touching objects** are measured as one object.
- **Width bias.** A small positive bias in width (+0.03 mm) remains from the outline.

## Run

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python evaluate.py                 # error statistics; worst overlays in debug/
.venv/bin/python scripts/calibrate.py        # the corner and edge error sizes
.venv/bin/python scripts/corner_methods.py   # OpenCV corner refinements vs the line fit
.venv/bin/python scripts/uncertainty_methods.py
.venv/bin/python scripts/lens_distortion.py
.venv/bin/python scripts/figures.py          # docs/examples.png
```
