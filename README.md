# Rocket Radial-Expansion Tracker

A beginner-friendly machine-learning project that predicts **how high a rocket
is off the ground** by watching how things in the video grow or shrink on
screen.

It works in three stages:

1. **Feature extraction (OpenCV)** — `rocket_flow.py` looks through the video
   with optical flow, tracks hundreds of points, and writes their motion to a
   CSV. This is the "eyes" of the model.
2. **Labeling (you)** — `label_altitude.py` lets you type in the real altitude
   for each frame (from telemetry, a barometer, or knowing when the rocket
   left the pad). This is the "answer key" the model learns from.
3. **Training (XGBoost)** — `train_model.py` learns to map the motion features
   to altitude, then can predict altitude on new footage it has never seen.

```
 launch.mp4 ──► rocket_flow.py ──► launch_metrics.csv ──┐
                                               (features) │
                                                  (you)   ▼
                                         label_altitude.py ├─► launch_labels.csv
                                               (answers)  │        │
                                                          │        ▼
                                                         train_model.py ──► launch_xgb.json (model)
                                                          │
                                                          ▼
                                          predict_altitude.py ──► altitude over time
                                              (new footage / features)
```

---

## 1. What you get (outputs)

### From `rocket_flow.py` (the feature extractor)

| Output | What it is |
|--------|------------|
| `<name>_flow.mp4` | The original video, re-drawn with each tracked point in its own **random color**, a motion **trail** for each point, and an on-screen HUD. |
| `<name>_metrics.csv` | One row per frame: time, radial expansion, flow magnitude, flow-direction histogram, FOE position, expansion rate/acceleration, and state. **This is the feature file for the model.** |
| `<name>_frames.csv` | `frame,time_s` alignment file (one row per frame, same order as the metrics CSV). Used to map labels onto frames — `frame` is deliberately kept out of the model input. |
| `<name>_points.csv` | One row per *tracked point*: frame, point id, x/y position, flow `(u,v)`, radial speed. Useful for diagnostics and custom features. |
| `<name>_metrics.png` | A plot of expansion rate and acceleration over time with colored state bands. |

### From `label_altitude.py` (your ground truth)

| Output | What it is |
|--------|------------|
| `<name>_labels.csv` | `frame,altitude` — the true altitude you entered for each frame. |

### From `train_model.py` (the machine learning)

| Output | What it is |
|--------|------------|
| `<name>_xgb.json` | The trained XGBoost model (load with `xgb.Booster()`). |
| `<name>_train_test.csv` | Every labeled frame with features + true altitude + predicted altitude. |
| `<name>_prediction.png` | True vs predicted altitude over time, with the test split marked. |
| `<name>_importance.png` | Which features the model found most important. |
| `<name>_report.txt` | Test RMSE, MAE, R² — how good the model is. |

### From `predict_altitude.py` (using the model on new footage)

| Output | What it is |
|--------|------------|
| `<name>_predictions.csv` | `frame,time_s,altitude_m` — predicted altitude for every frame. |
| `<name>_predictions.png` | Predicted altitude vs time plot. |

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

The script does five things every frame, in a loop:

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

### Step 5 — Velocity, acceleration, and the verdict

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

---

## 4. How to use it

```bash
# Install once (only needed on a fresh machine)
pip install opencv-python numpy scipy matplotlib xgboost pandas scikit-learn

# ---------- STEP 1: extract features from your video ----------
python3 rocket_flow.py launch.mp4
#   produces launch_flow.mp4, launch_metrics.csv, launch_frames.csv,
#           launch_points.csv, launch_metrics.png
#   (the *_frames.csv file maps frame numbers to time_s and is used to align
#    your labels; frame is deliberately NOT in the model-facing metrics CSV)

# Optional tweaks
python3 rocket_flow.py launch.mp4 --draw-foe        # draw the expansion center
python3 rocket_flow.py launch.mp4 --scale 0.5       # process at half resolution (faster)
python3 rocket_flow.py launch.mp4 --no-points-csv   # skip the per-point CSV

# Generate a synthetic test video (expansion then contraction) to check everything
python3 rocket_flow.py --synthetic test.mp4

# ---------- STEP 2: label the true altitude frame by frame ----------
python3 label_altitude.py launch_metrics.csv
#   interactive: type a number to label the current frame, Enter = repeat
#   previous value, '.' = skip, '100 200 45' = label frames 100-200 as 45m,
#   's 0' = set all frames to 0 (pad), 'save' = write launch_labels.csv
#   (auto-uses launch_frames.csv for frame/time alignment)

# ---------- STEP 3: train the XGBoost altitude model ----------
python3 train_model.py launch_metrics.csv launch_labels.csv
#   produces launch_xgb.json + prediction/importance plots + report.txt
#   (auto-uses launch_frames.csv for alignment)

# ---------- STEP 4 (later): predict on new footage ----------
python3 predict_altitude.py new_video.mp4 --model launch_xgb.json
#   or reuse an existing features CSV (no re-analysis):
python3 predict_altitude.py --features launch_metrics.csv --model launch_xgb.json
```

> **Tip:** label as many frames as you realistically can — a few hundred labeled
> frames makes a far more reliable model than a few dozen. If you have telemetry,
> label dense ranges (`0 90 0` for frames 0–90 = 0 m) instead of frame-by-frame.

---

## 5. How to read the results

**Annotated video** — each tracked point has a fixed random color. Watch a trail:
if trails point *away* from the center, the state HUD should read `ASCEND`.

**CSV** — each row is one frame. Columns:

| Column | Meaning |
|--------|---------|
| `time_s` | seconds from start of video |
| `radial_expansion` | average signed radial speed, px per frame (outward +, inward −) |
| `flow_magnitude` | mean flow speed √(u²+v²) across all tracked points |
| `flow_dir_hist_0..7` | 8-bin distribution of flow directions (0–360°), sums to 1 |
| `foe_x`, `foe_y` | estimated Focus of Expansion |
| `expansion_rate` | smoothed radial velocity (px/s) |
| `expansion_acceleration` | smoothed acceleration (px/s²) |
| `state` | ASCEND / DESCEND / STABLE |

`frame` is **not** in this file — it lives in `<name>_frames.csv` (same row
order) and is used only to align your altitude labels; it is never a model
feature. `time_s` *is* kept in the metrics file; whether the model uses it is
your choice via the `--time-feature` flag.

These are the raw per-frame metrics. When training, `train_model.py` also
**derives extra history features** from them — a running velocity integral
(`vel_cumsum`, since altitude ≈ ∫ velocity), rolling averages, and time-lagged
velocities — because a rocket's height depends on where it *was*, not just where
it is now. The full feature list is embedded in the saved model, and
`predict_altitude.py` automatically uses exactly those columns.

**Plot** — green bands = ASCEND, red bands = DESCEND. The velocity line
crossing zero is the moment the rocket switches direction.

### The five model signals, explained

The metrics CSV feeds the XGBoost model. Each is a different view of the same
motion, so the model can learn to combine them:

| Signal | What it captures |
|--------|------------------|
| `radial_expansion` | **How strongly** the image is growing (or shrinking). Positive = objects moving outward (approaching), negative = inward (receding). One number per frame. |
| `flow_magnitude` | **How much** motion there is at all — mean speed of every tracked point. A busy smoke plume scores high even if it's not perfectly radial. |
| `flow_dir_hist_0..7` | **Which directions** the motion points. Pure rocket zoom-in concentrates in the "toward-center" bins; camera shake spreads across all bins. Lets the model tell "expansion" from "noise". |
| `expansion_rate` | Smoothed radial velocity in px/s — the *speed* of size change. |
| `expansion_acceleration` | How fast the rate of size change itself changes — sharp transitions (liftoff, burnout, parachute deploy) show up here. |

Together: `flow_magnitude` says "something is moving", the histogram says "is it
radial?", `radial_expansion` says "which way", and rate + acceleration describe
the trajectory. That combination is what makes altitude predictable.

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

## 8. Where is the math in the code?

| Idea | Where |
|------|-------|
| Shi-Tomasi corner selection | `cv2.goodFeaturesToTrack` (main loop) |
| Lucas-Kanade tracking | `cv2.calcOpticalFlowPyrLK` (main loop) |
| Backward round-trip check | `cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, ...)` |
| FOE least-squares | `estimate_foe()` — the 2×2 system |
| Radial speed | `radial_speeds()` |
| Flow magnitude + direction histogram | main loop (`np.hypot`, `np.histogram` on `arctan2`) |
| State classification | `classify()` |
| Smoothing / acceleration | `savgol_filter`, `np.gradient` |
| State de-flickering | `scipy.signal.medfilt` |
| Synthetic test generator | `generate_synthetic()` |
| Model history features | `train_model.py` (`vel_cumsum`, rolling, lags) |

The math behind the FOE is a standard computer-vision result called the
**Focus of Expansion / Focus of Contraction** estimation (Horn's optical-flow
works). Nothing here is rocket-science math — just vectors, dot products, and a
least-squares fit. The "rocket science" is only in how you use it.
