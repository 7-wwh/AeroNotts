# Rocket Visual-Motion Feature Extractor

A computer-vision tool that watches how things in a rocket video **grow or
shrink on screen** and turns that into a rich per-frame metrics CSV.

It doesn't bet on a single measurement. It measures **lots of visual-motion
signals** — sparse radial flow, dense flow divergence, camera ego-motion,
image appearance, and more — and writes them all to one CSV per frame. The
philosophy: *you measure many things; the model (or your analysis) decides
which combinations indicate ascent, descent, apogee, or approaching the
ground.*

```
 launch.mp4 ──► rocket_flow.py ──► csv output/launch_metrics.csv   (72 per-frame feature columns)
                 │                    csv output/launch_metrics.png (plot + apogee markers)
                 ▼
            csv output/launch_flow.mp4  (annotated video)
```

Code is split into a `scripts/` package so each feature family is easy to
find, edit, and test on its own (see §8).

---

## 1. What you get (outputs)

| Output | What it is |
|--------|------------|
| `<name>_flow.mp4` | The original video, re-drawn with each tracked point in its own **random color**, a motion **trail** for each point, and an on-screen HUD. |
| `<name>_metrics.csv` | One row per frame with 72 columns: sparse radial flow, dense-flow divergence, magnitude stats, 3×3 flow grid, camera ego-motion, homography, image appearance, horizon, FOE, smoothed velocity/acceleration, state and phase. |
| `<name>_metrics.png` | A plot of expansion rate and acceleration over time with colored state bands and dashed apogee markers. |

---

## 2. The big idea: what is "radial expansion"?

Imagine a camera watching a rocket fly away.

```
   frame 1                 frame 2                  frame 3
      * *                     * * *                     * * * *
      *R*    ------>          * R *       ------>        * R *
      * *                     * * *                     * * * *
   (small rocket)          (bigger rocket)          (even bigger rocket)
```

As the rocket gets closer to the camera it appears **bigger**, and every texture
detail on it slides **outward** away from the center of the image. If the rocket
moves away, everything slides **inward**. So:

> The direction of the "sliding" tells us approach vs. recede.
> How fast it slides tells us the velocity.
> How fast the sliding *changes speed* tells us the acceleration.

This is the same optical effect your brain uses: when you drive forward, the
roadside trees appear to zoom out from a point on the horizon. That point is
called the **Focus of Expansion (FOE)**.

---

## 3. The pipeline, step by step

Every frame the script runs several independent feature families: sparse
tracking (steps 1–4), dense flow (step 5), camera ego-motion (step 6), and
image appearance (step 7), then post-processes everything into smoothed
velocity, state, and flight phase (step 8).

### Step 1 — Pick good "points" to watch (Shi-Tomasi corner detection)

We can't watch every pixel; there are millions and most are boring (flat sky).
Instead, OpenCV's `goodFeaturesToTrack` finds **corners** — spots where the
image changes in two directions at once (e.g. the edge of a cloud, a smoke
swirl, the rocket's paint detail). These are "good" because we can find them
again in the next frame.

Settings: up to `--max-corners 500` corners, each at least `--min-dist 10`
pixels apart.

### Step 2 — Track them to the next frame (Lucas-Kanade optical flow)

`calcOpticalFlowPyrLK` takes each corner from the previous frame and searches a
small window in the *new* frame for the matching corner. The result is a
**flow vector** for each point:

```
displacement = (u, v)   e.g. (3, -2)  meaning "moved 3px right, 2px up"
```

This is *sparse* optical flow: we only compute motion for our chosen corners,
not every pixel.

**Safeguard (back-substitution):** a moving corner can be lost. To catch that,
the script runs the tracker **backwards** too — from the new frame back to the
old one — and only keeps a point if it lands *almost exactly* where it started
(round-trip error < 1 pixel). This filters out most wrong matches. Disable with
`--no-back-sub` if you want.

### Step 3 — Find the Focus of Expansion (FOE)

Here is the clever math part. If all points are expanding/contracting, then
every flow vector points **along a straight line** that passes through one
special point: the FOE, at coordinates `(xF, yF)`.

For a point at `(x, y)` with flow `(u, v)`, "the vector points away from the
FOE" means the flow is parallel to the line joining the point to the FOE.
Parallel lines have **zero cross product**, which gives this neat condition:

```
u*(y − yF) − v*(x − xF) = 0
```

Rearrange it into a form that is *linear* in the unknowns `(xF, yF)`:

```
v*xF − u*yF = v*x − u*y
```

Every tracked point gives us one such equation. With hundreds of points we have
hundreds of equations but only **two** unknowns — so we solve an over-determined
least-squares problem. The script builds a 2×2 system:

```
| Σ w²v²     −Σ w²uv |   | xF |   | Σ w²(vx − uy) v |
|                    | · |    | = |                  |
| −Σ w²uv     Σ w²u² |   | yF |   | −Σ w²(vx − uy) u |
```

where the weight `w = 1/|v|` ensures short, noisy vectors count as much as big
ones, and `Σ` means "sum over all points". Solve with Cramer's rule (the `det`
in the code) and you get the FOE.

To avoid the FOE jumping around frame to frame, it's **exponentially smoothed**:

```
FOE_new = α · FOE_measured + (1 − α) · FOE_old        (α = --foe-alpha, default 0.3)
```

You can skip estimation entirely and force a center with `--center frame`
(middle of image) or `--center 320,240`.

### Step 4 — Compute radial speed for every point

Now that we know the center, split each flow vector into a *radial* part (toward
or away from the FOE) and a *tangential* part (around it). We only care about
the radial part.

Let `d = (dx, dy)` be the vector from the FOE to the point, with length
`r = √(dx² + dy²)`. The **radial speed** is the projection of the flow onto the
unit radial direction:

```
vr = (u·dx + v·dy) / r
```

- `vr > 0`  → the point moves **outward** (expansion)
- `vr < 0`  → the point moves **inward** (contraction)

Per frame we average `vr` over all tracked points to get one clean number.

### Step 5 — Dense flow: divergence, magnitude stats, and spatial grid

On top of the sparse points, the script runs **dense** optical flow
(Farneback) between every frame pair, giving a flow vector for *every* pixel.
From it we get three independent views:

- **Divergence** `div = du/dx + dv/dy` — one number saying whether the *whole
  visual field* is expanding (positive) or contracting (negative). Unlike
  radial speed it doesn't need a known center, so it's robust to the camera
  being tilted. Reported as `div_mean/median/std/p95/pos_frac/max`.
- **Magnitude stats** `flow_median/p95/std/max` — how much the whole image is
  moving, in percentiles.
- **Spatial grid** `grid_flow_00..22` — the mean flow magnitude in each cell of
  a 3×3 grid. This captures the *shape* of the motion field (e.g. everything
  flowing toward one corner vs. expanding from the middle), which one global
  number can't.

### Step 6 — Camera ego-motion: rotation, residual, homography

A wobbly camera pollutes the radial measurement, so we estimate the camera's
own motion and hand the model both the camera motion *and* the motion left
after removing it:

- **Affine model** (RANSAC `estimateAffinePartial2D`): a global
  rotation/scale/translation fit to the tracked points →
  `cam_rotation/scale/tx/ty`.
- **Residual flow**: the per-point flow minus the rigid camera model — what's
  *left* after subtracting rotation/translation. Its magnitude and divergence
  (`residual_flow_mean/p95`, `residual_div`) are the "true approach" signal.
- **Homography** (RANSAC `findHomography`): a planar model whose
  scale/rotation/translation/perspective terms
  (`hom_scale/rotation/tx/ty/persp_x/persp_y/ok`) estimate global zoom
  (approach vs. retreat) even when the scene is roughly a plane.

### Step 7 — Image appearance: edges, texture, sharpness, sky/ground

Cheap per-frame stats that support the motion features:

- **Sharpness** = Laplacian variance; **edge density** = Canny edge fraction;
  **texture** = gray variance; **gradient magnitude** = mean Sobel response.
  As the rocket approaches, ground detail typically increases.
- **Sky/ground fractions** — a crude heuristic (bright/low-saturation pixels =
  sky, darker/textured = ground). Approximate, not a vision model.
- **Horizon** (experimental) — the strongest long straight Hough line, reported
  as angle / normalized position / confidence. The camera may be tilted, so the
  angle is measured rather than assumed horizontal.

### Step 8 — Velocity, acceleration, state, and flight phase

- **Velocity** `= mean(vr) × fps` converts from pixels-per-frame to pixels-per-second.
- The raw per-frame value is noisy, so it's smoothed with a **Savitzky–Golay
  filter** (a smart moving-window polynomial fit — `--smooth 15`).
- **Acceleration** = the derivative of the smoothed velocity (`np.gradient`),
  i.e. pixels/s².
- **State**: compare the mean radial speed against a threshold
  `--thresh` (default 0.1 px/frame):

  ```
  vr >  +thresh  →  ASCEND   (expanding / approaching)
  vr <  −thresh  →  DESCEND  (contracting / receding)
  otherwise      →  STABLE
  ```

  A **median filter** over the last few frames removes flicker (a single noisy
  frame shouldn't flip the label).
- **Phase**: a flight has exactly **one ascent, one apogee, one descent**. The
  script builds an altitude proxy (the cumulative integral of radial velocity)
  and takes its **single global peak** as the apogee. It then splits the flight
  around it:
  - idle frames before liftoff → `STABLE` (on the pad)
  - `ASCEND` up to `--apogee-window` frames before the peak
  - `APOGEE` around the peak (`--apogee-window`, default 3, frames each side)
  - `DESCEND` down to landing
  - idle frames after landing → `STABLE`
  `--idle-min-frames` (default 5) controls how sustained the motion must be to
  count as flying. `state` stays the per-frame direction classifier;
  `phase` is the clean one-of-each trajectory label. Videos that only show part
  of the flight degrade gracefully (no leading/trailing STABLE, or an empty
  ASCEND/DESCEND side).

---

## 4. How to use it

```bash
# Install once (only needed on a fresh machine)
pip install opencv-python numpy scipy matplotlib

# Extract metrics from your video
python3 rocket_flow.py launch.mp4
#   writes launch_flow.mp4, launch_metrics.csv, launch_metrics.png
#   into a "csv output" folder next to the input video

# Optional tweaks
python3 rocket_flow.py launch.mp4 --draw-foe        # draw the expansion center
python3 rocket_flow.py launch.mp4 --scale 0.5       # process at half resolution (faster)
python3 rocket_flow.py launch.mp4 --no-plot         # skip the metrics plot
python3 rocket_flow.py launch.mp4 --no-dense        # skip Farneback divergence/grid (faster)
python3 rocket_flow.py launch.mp4 --no-homography   # skip homography columns
python3 rocket_flow.py launch.mp4 --no-appearance   # skip image-appearance columns
python3 rocket_flow.py launch.mp4 --no-horizon      # skip horizon detection
python3 rocket_flow.py launch.mp4 --apogee-window 5 # widen the APOGEE phase around the peak
python3 rocket_flow.py launch.mp4 --idle-min-frames 10  # longer window to spot idle STABLE pads

# Generate a synthetic test video (expansion then contraction) to check everything
python3 rocket_flow.py --synthetic test.mp4
```

---

## 5. How to read the results

**Annotated video** — each tracked point has a fixed random color. Watch a trail:
if trails point *away* from the center, the state HUD should read `ASCEND`.

**CSV** — one row per frame, 72 columns grouped by feature family. Missing
measurements (e.g. frame 0, or a disabled group) are written as `nan`.

| Group | Columns | Meaning |
|-------|---------|---------|
| Time / labels | `time_s`, `state`, `phase` | seconds; ASCEND/DESCEND/STABLE; flight phase incl. APOGEE |
| Sparse radial | `radial_expansion`, `radial_expansion_median`, `radial_std`, `radial_p95` | signed radial speed of tracked points about the FOE (px/frame, + outward) |
| | `outward_frac`, `inward_frac` | fraction of points expanding / contracting |
| | `foe_x`, `foe_y` | estimated Focus of Expansion |
| Tracking | `pt_displacement_mean/median/std/max` | per-frame displacement of tracked points |
| | `point_count`, `point_density` | how many features are being tracked (density = per-pixel) |
| | `feature_radius_mean/std` | spread of tracked points about the FOE → apparent-scale proxy |
| Flow | `flow_magnitude`, `flow_dir_hist_0..7` | mean magnitude + 8-bin direction histogram (sums to 1) |
| Dense flow | `flow_median/p95/std/max` | global magnitude stats from dense Farneback flow |
| | `div_mean/median/std/p95/pos_frac/max` | flow divergence (whole field expanding/contracting) |
| | `grid_flow_00..22` | mean flow magnitude per cell of a 3×3 grid → motion-field shape |
| Camera | `cam_rotation/scale/tx/ty` | global affine rotation/scale/translation (RANSAC) |
| | `residual_flow_mean/p95`, `residual_div` | flow left after subtracting the rigid camera model |
| | `hom_scale/rotation/tx/ty/persp_x/persp_y/ok` | planar homography decomposition (RANSAC) |
| Appearance | `edge_density`, `texture_var`, `grad_magnitude_mean`, `sharpness` | Canny density, gray variance, Sobel magnitude, Laplacian variance |
| | `sky_fraction`, `ground_fraction` | crude bright/sky vs textured/ground pixel heuristic |
| Horizon | `horizon_angle/pos/conf` | strongest Hough line: angle, normalized y, confidence (experimental) |
| Temporal | `expansion_rate`, `expansion_acceleration` | smoothed radial velocity (px/s) and its derivative |

`frame` is deliberately **not** a column — the rows are in frame order, and the
frame number is just the row index (`time_s` is kept as a convenience).

**Plot** — green bands = ASCEND, red bands = DESCEND, dashed lines = APOGEE.
The velocity line crossing zero is the moment the rocket switches direction.

### The five motion signals, explained

The core sparse signals are the foundation; the dense/camera/appearance groups
let the model see the *same* motion from independent angles:

| Signal | What it captures |
|--------|------------------|
| `radial_expansion` | **How strongly** the image is growing (or shrinking). Positive = objects moving outward (approaching), negative = inward (receding). One number per frame. |
| `flow_magnitude` | **How much** motion there is at all — mean speed of every tracked point. A busy smoke plume scores high even if it's not perfectly radial. |
| `flow_dir_hist_0..7` | **Which directions** the motion points. Pure rocket zoom-in concentrates in the "toward-center" bins; camera shake spreads across all bins. Lets you tell "expansion" from "noise". |
| `div_mean` / `hom_scale` | **Global** expansion: dense-flow divergence and the homography zoom factor are far less dependent on the camera being upright. |
| `cam_rotation` / `residual_flow_mean` | **Camera shake**: how much of the motion is the wobbly camera, and how much is left over after removing it. |

Together: `flow_magnitude` says "something is moving", the histogram says "is it
radial?", `radial_expansion` says "which way", divergence/homography confirm it
globally, and the camera group separates "true approach" from "camera wobble".

---

## 6. Tips for good results on real footage

- **Stable camera.** A shaking camera adds motion the FOE math can't explain.
  If your footage shakes, stabilize it first (OpenCV's
  `cv2.createStabilizer` / `VideoStab`) or reduce `--scale`.
- **Lots of texture.** Flat clear sky has no corners to track. Smoke, clouds,
  horizon, and rocket details are your friends.
- **High frame rate** gives smoother velocity (less jump per frame).
- **Tune the threshold.** If you see states flipping constantly, raise
  `--thresh`. If it stays STABLE forever, lower it.
- **Zooming camera** creates radial motion of its own — be aware of it, and
  consider `--center` as a fixed known center instead of `auto`.

---

## 7. Known limitations

- **Pixels ≠ meters.** Velocity and acceleration are in *pixels per second*,
  not real distance. Converting to real units needs the camera-to-object
  distance and focal length (a pinhole-camera model).
- **Camera ego-motion** is assumed to be small. A rotating or panning camera
  breaks the "everything is radial" assumption.
- **It measures apparent size change, not the rocket directly.** If the rocket
  passes directly *behind* a fixed object (e.g. behind a tower), the tower's
  motion will contaminate the average.
- **FOE instability** when there's very little motion. The `--foe-alpha`
  smoothing and `--state-median` filter help but can't fix nothing-to-look-at.

---

## 8. Where is the code?

`rocket_flow.py` is a thin CLI entry point; all logic lives in the `scripts/`
package so each feature family is isolated and testable.

```
rocket_flow.py                 CLI entry: args, frame loop, CSV/plot/video output
scripts/
  io.py                        video open/write, output paths
  state.py                     classify(), global flight phases (1 apogee), smoothing
  schema.py                    CSV column names + order (single source of truth)
  draw.py                      HUD, trails, random colors, FOE crosshair
  synth.py                     synthetic test-video generator
  plot.py                      metrics plot + state bands + apogee markers
  features/
    sparse.py                  LK tracking, FOE, radial speed, displacement, density
    dense.py                   Farneback divergence, magnitude stats, 3x3 grid
    camera.py                  affine rotation/translation, residual flow, homography
    appearance.py              edges, texture, sharpness, sky/ground heuristic
    horizon.py                 Hough-line horizon detection (experimental)
```

| Idea | Where |
|------|-------|
| Shi-Tomasi corner selection | `scripts/features/sparse.py` (`goodFeaturesToTrack`) |
| Lucas-Kanade tracking | `scripts/features/sparse.py` (`calcOpticalFlowPyrLK`) |
| Backward round-trip check | `scripts/features/sparse.py` |
| FOE least-squares | `scripts/features/sparse.py` (`estimate_foe()` — the 2×2 system) |
| Radial speed | `scripts/features/sparse.py` (`radial_speeds()`) |
| Dense divergence | `scripts/features/dense.py` (`divergence()`, `np.gradient`) |
| 3×3 flow grid | `scripts/features/dense.py` (`grid_flow()`) |
| Camera rotation / residual | `scripts/features/camera.py` (`affine_model()`, `residual_flow()`) |
| Homography decomposition | `scripts/features/camera.py` (`homography_model()`) |
| Appearance features | `scripts/features/appearance.py` |
| Horizon detection | `scripts/features/horizon.py` |
| State / APOGEE / smoothing | `scripts/state.py` (`classify`, `flight_phases`, `savgol_filter`, `medfilt`) |
| Synthetic test generator | `scripts/synth.py` |

The math behind the FOE is a standard computer-vision result called the
**Focus of Expansion / Focus of Contraction** estimation (Horn's optical-flow
works). Nothing here is rocket-science math — just vectors, dot products, and a
least-squares fit. The "rocket science" is only in how you use it.
