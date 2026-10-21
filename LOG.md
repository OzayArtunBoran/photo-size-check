# Verification log

These are the real problems found while building this repo with Claude Code, in the order they happened, with how each was found. None of them raised an error: each time the code ran and returned a number.

The order of work matters. I built the oracle first: synthetic scenes with an exactly known size, plus an evaluation script that compares the measurement with it. The measurement code came after that.

## 1. It measured the table, not the object

**Symptom.** Every one of 200 scenes was rejected with "object touches the photo edge". The numbers behind the rejection were 362 mm for an 87 mm object.

**Found by.** The frame-edge check, then the overlay image. The green "object" outline ran around the whole photo.

**Cause.** Otsu thresholding separates dark from light. The grey table around the sheet is dark too, so the largest dark region was the table.

**Fix.** Find the light sheet that contains the marker, and look for dark pixels only inside it.

**Lesson.** Without the edge check, this would have printed confident wrong sizes.

## 2. My ground-truth check had a half-pixel error

**Symptom.** The marker corners looked 0.44 px off on average, always in the same direction (mean offset -0.27, -0.26 px).

**Found by.** The error was a constant offset, not noise. A real detector error would not be the same in every image.

**Cause.** The marker covers pixels `mx … mx+side-1`, so its true edge is at `mx - 0.5`, not `mx`. The pixel-centre convention was wrong in the check, not in the product code.

**Fix.** Corrected the convention. The true corner error was 0.30 px.

## 3. The oracle drew shapes about 1 px too large

**Symptom.** After the segmentation fix, every measurement was about 0.35 mm too large, even on clean images without blur or noise.

**Found by.** Comparing the drawn pixel area with the analytic area. `cv2.fillPoly` and `cv2.ellipse` with `LINE_AA` grow a shape by about 0.6 px per edge: +0.9% area for a 400x200 px rectangle.

**Fix.** Draw shapes by exact coverage (4x4 supersampling of an inside test) instead of OpenCV's rasteriser.

## 4. My replacement renderer cut off rectangle corners

**Symptom.** The new renderer drew rectangles 0.32% too small in area. Ellipses were exact.

**Found by.** The same area test, now kept as `test_oracle_draws_the_true_area`.

**Cause.** The drawing patch had a radius of half the length, but a rotated rectangle's corners reach half the diagonal.

**Fix.** Patch radius = half-diagonal, clipped to the sheet. The clipping was added after a crash on a large object near the sheet edge.

## 5. The uncertainty was invented, not measured

**Symptom.** The code reported about ±2.5 mm and rejected 38% of scenes as "too uncertain", while the real median error was 0.45 mm. The true value fell inside ±2 sd 100% of the time, which means the interval was far too wide, not well calibrated.

**Found by.** Comparing the reported sd with the real error on the same scenes.

**Cause.** The corner noise (0.5 px) was a guess.

**Fix.** Three steps, each measured:
- Fed the true marker corners into the measurement. The edge error alone was then 0.05 mm, so the marker corners were the main error source.
- Compared OpenCV's corner refinements against the true corners. SUBPIX was best at 0.28 px; CONTOUR and APRILTAG were 0.65–0.75 px.
- Added a line fit along the marker's edges, which brought the error to 0.10 px.

The noise level was then measured on a separate calibration seed (0.11 px per axis) and checked on held-out scenes. Result: median 0.17 mm, no systematic bias, 0 rejected, 93% coverage.

## Still open

The worst case is 4.6% error, from strong tilt with the object far from a small marker. See the Known limit section in the README.
