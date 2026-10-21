# photo-size-check

Measure a flat object from one photo, using a printed ArUco marker of known size, and know when not to trust the answer.

I wrote this with Claude Code. The main point of the repo is how the code is checked. Code that runs proves very little, so every number here is compared with a size that is known to be true.

## How it measures

1. Detect one ArUco marker (4x4, id 0) with sub-pixel corners.
2. Fit a straight line to dozens of sub-pixel points along each of the marker's four edges, and take each corner as the intersection of two lines. This gives about 3x less corner error than OpenCV's own refinement, for about 1 ms per call.
3. Build a homography from the marker's corners in the photo to its true corners in mm.
4. Segment the dark object, but only inside the light sheet that carries the marker.
5. Find a sub-pixel outline: threshold halfway between this object's own dark level and the sheet around it, on a 4x upsampled crop of the object only.
6. Map the outline to mm and fit a minimum-area box, which gives length x width.
7. Uncertainty: re-run step 6 with the marker corners jittered by their measured error. Only the convex hull is jittered, which gives the same box for far less work.
8. Every failed check goes into `problems`. A result with problems must not be shown to a user. The checks are: no marker or more than one marker, marker too small, no object, object touching the frame, and uncertainty above 3%.

The object must lie flat on the same plane as the marker. One photo cannot recover the height of a 3D object, and the code does not try to.

## How it is checked

`photosize/synth.py` renders scenes whose true size is known exactly:
- a sheet with the marker and a rectangle or ellipse;
- a random camera tilt;
- blur, uneven light and sensor noise.

`evaluate.py` scores the measurement against those true sizes and saves overlays of the worst cases to `debug/` so they can be looked at.

```
$ python evaluate.py --n 300 --seed 1 --workers 4
scenes 300 | measured 300 | rejected 0
length error mm: median 0.17  p95 0.90  mean(signed) +0.02
width  error mm: median 0.10  p95 0.58  mean(signed) +0.04
length relative error: median 0.32%  max 4.60%
true length inside ±2 reported sd: 93%
```

`tests/` covers the following:
- The ground-truth renderer draws the true area. A wrong oracle makes every score meaningless.
- Accuracy and bias on held-out scenes.
- Rejection when there is no marker, and when the object is cut by the frame.
- The same answer at 75% resolution.
- Concurrent `measure()` calls from 8 threads give exactly the sequential results. The function holds no shared state and uses a per-call RNG.
- The evaluation gives the same result with 1 or 4 worker processes. Each scene gets its own child seed, and only the parent process writes files.

`mypy --strict` passes.

## What the checks caught while building it

These are listed in [LOG.md](LOG.md). In short:
- The first version measured the table instead of the object, and returned 362 mm for an 87 mm object.
- The ground truth itself was wrong twice.
- The first uncertainty estimate was about 5x too large, so it rejected 38% of good measurements.

None of these failed with an error. The code ran and returned plausible-looking numbers each time.

## Known limit

With a 40 mm marker, strong tilt and the object far from the marker, small corner errors are extrapolated. The worst case is 4.6% off.

The jitter model treats the four corners as independent, but the real errors are correlated through the fitted edges. As a result, the reported ±2 sd covers the true value 93% of the time instead of 95%. A larger marker placed next to the object reduces this error. A better uncertainty model would jitter the fitted edge lines instead of the corners.

## Run

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python evaluate.py
```
