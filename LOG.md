# Verification log

These are the real problems found while building this repo with Claude Code, in the order they were found, with how each was found and which commit fixed it.

Two rules for reading it:
- Every number here comes from a script, a test or an evaluation run that is in the repo, or it is marked as measured at the time on the first version.
- "Silently wrong" means `measure()` returned a size marked ok that was more than 0.5 mm off.

How the work started: the synthetic oracle (`synth.py`) and the first measurement code were written in the same step. The evaluation script came right after, before any measured number was trusted. The first time the measurement ran, it ran through the evaluation against the known sizes.

## Part 1: building the first version (commit 77b7f8d)

### 1. It measured the table, not the object
**Symptom.** The first evaluation run rejected every one of 200 scenes; the ones I inspected all said "object touches the photo edge". Then the evaluation script itself crashed with an `IndexError`, because it could not handle an empty result list; that crash was a separate bug, fixed in 7fd718b. The numbers behind the rejections were wrong by a factor of four: 362 mm for an 87 mm object.

**Found by.** The frame-edge check, then the overlay image. The "object" outline ran around the whole photo.

**Cause.** Otsu thresholding separates dark from light. The grey table around the sheet is dark too, so the largest dark region was the table.

**Fix.** Look for dark pixels only inside the light sheet that carries the marker.

### 2. My corner check had a half-pixel error
**Symptom.** The marker corners looked 0.44 px off (median), always in the same direction: the mean offset was (-0.27, -0.26) px.

**Found by.** A real detector error would not be the same in every image.

**Cause.** The marker covers pixels `mx … mx+side-1`, so its true edge is at `mx - 0.5`. The convention was wrong in the analysis, not in the product code. After correcting it, the median corner error was 0.30 px. The oracle now stores the true corners with the right convention, and a test checks that they show no constant offset.

### 3. A bias of about +0.35 mm had three causes, and the oracle was only one of them
**Symptom.** After fix 1, measurements were too large on average: mean signed error +0.34 mm in length and +0.40 mm in width.

**First cause: the oracle.** `cv2.fillPoly` with `LINE_AA` on integer coordinates drew a 400x200 px rectangle with 1.0% too much area, about 0.7 px per edge. I replaced it with exact pixel coverage (4x4 supersampled inside test).

**What remained.** With the new painter, clean images (no blur or noise) were still about +0.2 mm too large. The remaining two causes were in the measurement:
- **The outline.** It sat wherever the global threshold fell. I replaced it with a sub-pixel outline: threshold halfway between the object's and the sheet's local grey levels, on a 4x upsampled crop.
- **The marker corners.** See item 5.

### 4. My replacement painter cut off rectangle corners
**Symptom.** Rectangles came out 0.32% too small in area; ellipses were exact.

**Found by.** The area test, which is now `test_oracle_draws_the_true_area`.

**Cause.** The drawing patch had a radius of half the length, but a rotated rectangle reaches half its diagonal.

**Fix.** Patch radius = half-diagonal. A large object near the sheet edge then crashed with a broadcasting error, so the patch is now also clipped to the canvas.

### 5. The uncertainty was a guess
**Symptom.** The code reported about ±2.5 mm and rejected 77 of 200 scenes (38%) as "too uncertain", while the real median error was 0.45 mm. The true value fell inside ±2 sd 100% of the time: the interval was far too wide.

**Found by.** Comparing the reported sd with the real error on the same scenes.

**Cause.** The corner noise (0.5 px) was guessed, not measured.

**Fix.**
1. I fed the true marker corners into the measurement. The outline error alone was then 0.05 mm (median), so the marker corners were the main error source.
2. I compared OpenCV's corner refinements against the true corners. SUBPIX was best at 0.28 px; CONTOUR and APRILTAG were 0.65–0.75 px.
3. I added a straight-line fit along each marker edge, which brought the corner error to 0.10 px.
4. I measured the noise on calibration scenes (0.11 px).

Result on held-out scenes at the time: median 0.17 mm, 0 rejected, 93% coverage. Item 8 shows why that 0.11 px and that 93% did not hold.

## Part 2: an independent review of the first version

After publishing, I asked a second Claude Code agent, which had no access to my reasoning, to find everything wrong with the repo. I then re-ran its experiments myself before acting on them. It found problems that my own checks had missed.

### 6. Four ways to get a wrong size marked as ok
**What happened.** These are the review's scenes, re-run by me on the first version:
- **An object near the marker was cut short.** At 2 mm from the marker a 60 mm object measured 55.3 mm, marked ok with ±0.2 mm; at 5 mm it measured 58.3 mm, also marked ok. The marker mask, grown by ×1.35, erased a strip of the object, and the overlay showed it.
- **With two objects, the larger one was measured** without any warning.
- **An object cut by the photo frame was dropped, and a small neighbour was measured instead** (25.0 x 12.0 for a 70 x 30 object). The frame check could never fire, because the sheet outline already excluded anything touching the frame. My own "cut by frame" test passed only because it fell through to "no object found".
- **Grey objects near the global threshold were only partly found.** At grey 158, a 60 mm object measured 13 mm, with an sd of 0.04 mm.

**Fix (615f80d).**
- Only the marker plus 4 px is masked, and an object touching the marker is refused.
- More than one object is refused.
- A dark notch in the sheet's convex outline means something crosses the sheet edge or the frame.
- Objects are found by how much darker they are than a fitted sheet-brightness surface.
- Low contrast and unclear outlines are refused.
- Problems carry machine-readable codes.

**Check.** `tests/test_robustness.py` rebuilds each scene. On the first version, 8 of its 33 scenes were silently wrong; now each one is either right or refused.

### 7. The "too uncertain" check never fired
**What happened.** Across 2,000 scenes the relative sd never went above 1.67%, against a 3% limit, and the check only looked at 1 sd of the length.

**Fix.** The limit is now 2 sd, on both length and width.

### 8. The 0.11 px corner sigma described the oracle, not the detector
**What happened.**
- **`cv2.warpPerspective` ignores `INTER_AREA`.** Its output is identical to `INTER_LINEAR`, which I checked. So the first renderer aliased the marker edges, and the 0.11 px was mostly that artefact.
- **A more realistic camera halves the corner error.** I averaged each photo pixel over its footprint, as a sensor does, with a 4x supersampled warp and area downsampling. The line-fit corner error fell from 0.125 to 0.063 px (mean, 150 calibration scenes), and with the old sigma the uncertainty became about 3 times too large.
- **The 93% was also luck.** The first version's 32-sample Monte Carlo sd moves by about 10% with its random seed, so that figure depended on the seed.

**Fix.**
- **The renderer (cdd2318).** Area-integrated camera, JPEG compression, a table around the sheet, and random objects kept on the sheet. Before, about 1 in 1,900 left the sheet and its ground truth was wrong.
- **Measured error sizes (30c4927).** `scripts/calibrate.py` measures the corner error (0.053 px per axis) and the outline error (0.118 px per side) on calibration scenes only.
- **A Jacobian instead of Monte Carlo (30c4927).** It is deterministic, matches a 4,000-sample Monte Carlo (median ratio 1.01), and is cheaper. README has the reasons.

The corner methods were compared again on both renderers (`scripts/corner_methods.py`):
- The line fit has 2.3x (old renderer) and 3.0x (current renderer) less mean error than SUBPIX.
- SUBPIX sits 0.18 px inside the true corner on average; the line fit removes that.
- CONTOUR's error is almost all a constant inward offset (-0.76 px, sd 0.11).
- APRILTAG is noisy (sd 0.48 px).

### 9. Tests that passed for the wrong reason
**What happened.**
- The "cut by frame" test passed via "no object found" (item 6).
- The concurrency test compared floats with `==`, so it would fail on a rejected scene, because NaN is never equal to NaN.
- The resolution test allowed 1.38 mm for an effect of 0.10 mm.
- `mypy --strict` failed once the tests were included, even though the README said it passed.

**Fix.**
- Each test now asserts the specific problem code it expects.
- Comparisons treat NaN as equal to NaN.
- The resolution tolerance is 0.3 mm.
- mypy runs on the package, the tests and the scripts in CI.

### 10. Near-circles traded length for width
**What happened.** After the Jacobian change, the worst held-out result was a 62.2 x 60.1 mm ellipse measured as 61.4 x 60.8, outside its reported 2 sd of 0.30 mm. The outline was right. For a near-circle, the minimum-area box hardly changes area as it turns, so outline noise turns it.

**Fix (d222c1d).** Every box angle whose area is within 2 sd of the minimum now counts as possible, and the spread of length and width over those angles enters the sd. On ellipses with W/L from 0.90 to 1.00 (20 scenes each), ±2 sd now covers 95–100%.

**Later (ec1550b).** A 60 x 58 mm ellipse rendered with no noise at all still measured 58.95 x 58.95: the wider sd covered the error, but the estimate itself sat at the wrong end of the range. When the tight boxes span more than 3 degrees, the longest and shortest sides over them are now reported, which for an ellipse are its axes. With four markers the worst synthetic error then fell from 0.75 to 0.08 mm.

### 11. The first README, LOG and application text said things that were not true
**What happened.** Claude Code wrote them, and I did not check them against the history:
- **The order of work.** They said the evaluation script was written before the measurement code; it was written after.
- **Errors.** They said no problem raised an error; the first evaluation run crashed (item 1).
- **Results without conditions.** They called the uncertainty "calibrated" and gave "93%" and "3x" without saying under which conditions.
- **Type checking.** They said `mypy --strict` passed, but it did not with the tests included.
- **A change that never happened.** My application text said a per-call random generator was added after I asked about race conditions; it had existed from the first version.

**Fix.** This file and the README now state only what a script, a test or a commit shows. The application text is rewritten the same way.

## Part 3: real photos

There was no printer or ruler, so the reference was a 14-inch MacBook Pro screen. At 254 ppi it is exactly 10 px per mm, and the slides draw markers and shapes in millimetres. The phone was a Xiaomi with HDR and AI off. Raw photos are not in the repository; per-photo results are in `real/results/`.

### 12. First real photos: most valid photos were refused
**What happened.** On 24 photos of the one-marker slides (v1), 10 of the 18 photos that should have been measured were refused. Five causes:
- **Dim edges.** The screen dims towards its edges at oblique angles.
- **A reflection.** A reflection on the frame (the user spotted a small blue patch) joined the white area and stretched the sheet's outline over the dark frame.
- **A dim corner.** One dim screen corner looked like a second object.
- **Pixel thresholds.** Every pixel threshold was tuned for the synthetic scale, about 5 px/mm; these photos have 6–12.
- **Exposure.** A close-up against a bright window made the screen the darker class for the photo-wide threshold (found in round B).

**Fix (e49d219).**
- Photos are reduced to about 5 px/mm at the markers.
- The sheet's brightness is estimated locally in a second pass.
- A dark region touching the outline counts as crossing it only if it reaches 5 mm deep.
- The outline ignores bulges thinner than 12 mm.

**Result.** 18/18 measured and 6/6 refused on that set.

**Caveat.** 12 mm was chosen on those photos. The later rounds are the check: they had no false crossing refusals.

### 13. Sizes about 1% small, uncertainty ten times too narrow
**What happened.**
- **Error.** The accepted sizes were 0.5–2.5% too small, while the reported 2 sd was about 0.1 mm.
- **Not the working resolution.** Re-measuring at 3.5, 5, 8 px/mm and full resolution gave the same errors.
- **Growing with distance.** On a two-object slide, the object farther from the marker was off more (-1.3% / -2.6%) than the nearer one (-1.1% / -1.5%). Lens distortion and corner errors were being extrapolated from one 60 mm marker.

**Fix (5ed3413).**
- A `Board` holds several markers at known positions.
- The homography is a least-squares fit to all their corners.
- A one-parameter radial distortion is estimated from how well those corners fit one plane.

**Synthetic result.** With distortion k1 = 0.10, four markers gave 0.03% median error and 100% coverage; one marker gave 0.83% and 50% (`scripts/lens_distortion.py`). The synthetic bending has the same form as the model, so this shows the method works, not that real lenses agree.

### 14. A calibration round and a test round, with the settings locked in between
**Design.** I re-shot 16 new slides with four 30 mm markers.
- Round B is for calibration.
- Round C uses different shapes and is the test.
- The photos were identified by eye: the files came in reverse order, one slide's two shots fell on either side of another slide's, and there were two extra shots.

**Round B results.** 12/13 measured and 4/4 refused.
- The error was -0.9% (length) and -1.0% (width).
- With the first marker alone on the same photos: -3.0% and -3.3%.
- **Four markers fixed most of it, not all.**

**Calibration.** `scripts/calibrate_real.py` chose a real-camera edge term so that the errors divided by the reported sd have rms 1 on round B: 0.392 mm per side. It is carried as uncertainty, not as a correction. The "too uncertain" limit moved from 3% to 5% of the size. Both were committed (8d28101) before round C was measured.

**Round C (5ccc542), measured once.**
- 10/13 measured and 4/4 refused.
- All 20 sizes inside ±2 sd, with errors -0.86% / -1.00% (at most 0.89 mm).
- With the first marker alone: 50% / 40% inside ±2 sd.
- Refused although measurable:
  - a 140 x 20 mm strip twice, because 2 sd is over 5% of 20 mm;
  - one photo focused on the window behind the screen, where two markers were unreadable.

### 15. What the remaining bias is not
The objects' outlines sit 0.2–0.5 mm inside their true edges on real photos. I tested four explanations; none holds.
- **Blur against a pixel-sized search band.** No: changing the working resolution changed nothing (item 13).
- **The display dimming white around large black areas.** No: the white 0.4–0.8 mm outside an object edge equals the white 5 mm away (163 vs 162).
- **Lens blur plus gamma encoding**, which shifts a 50% threshold towards the dark side. Not on its own: the markers' edges, under the same camera, sit where they should (mean -0.03 mm), while an object's edge is at -0.37 mm.
- **The homography.** No: mapping with only the four marker centres gives the same object errors.

The shift is larger across the short ends of a rotated rectangle than along its sides. It shows only on the anti-aliased, rotated shapes, not on the pixel-aligned markers, and an ideal browser render of the same page measures exactly. My guess is something in how the screen or the phone treats those edges, but that is unverified.

### 16. A fix that was reverted
A sheet threshold taken from the markers' own black and white was meant to handle the bright window. It made that photo worse: the window joined the sheet and showed 7 "objects". It also refused a photo from the first set and changed the empty-sheet result. I reverted it (never committed). That photo stays a refusal, which is safe.

### 17. A safeguard no test can show
Smoothing the sheet's outline rounds its corners, so I let the object search cover the sheet itself as well as the outline. A test meant to prove it passed even with the safeguard removed. Objects 1 mm from a corner are not affected at the tested tilts, and at 0.5 mm they touch the edge and are refused either way. It stays as a precaution for strongly tilted photos, and the LOG says so.

### 18. A constant left stale on purpose
`EDGE_SIGMA_PX` is 0.118 px, measured before the two-pass segmentation. The current code measures 0.078 px on the same calibration scenes. I left it, because round B's calibration and round C's locked test used 0.118. Next to the 0.392 mm camera term it changes real-photo sds by less than 0.001 mm. On synthetic photos it only makes the sd more cautious.

### 19. Mistakes in my own process this round
- **A commit with type errors.** I committed while `mypy` reported errors: piping it into `tail` hid the exit code. Amended before anything was pushed; checks now run without pipes.
- **Wrong counts in the README.** "Five of these scenes" should have been 8 of 33; "32 scenes" should have been 33. Both were caught by re-running the count before committing.
- **A timing from older code.** A commit message quoted 6.9 s -> 5.5 s from the earlier renderer. Amended to the measurement on the current code.
- **A heading that disagreed with its list.** The heading said five, the list had four. Fixed.
- **An automated edit that missed two calls.** A regex that added `camera_edge_mm=0.0` to every synthetic test call missed two nested calls. Found by grepping the result, not by trusting it.

## Where it stands

- **Real photos (screen-drawn shapes, one phone), four markers.** On the test round, all 20 sizes were inside ±2 sd, with a 1% bias that is unexplained and carried as uncertainty. Thin objects (< ~22 mm) are refused.
- **Synthetic, 2,000 held-out scenes.**
  - One marker: median 0.06 / 0.04 mm.
  - Four markers: 0.03 / 0.02 mm, max 0.13 mm.
  - ±2 sd covers 98.7–100%.
- **Not tested yet:** printed markers, real objects, other phones.
- **Wrong without a warning:** a wrongly scaled marker print.
