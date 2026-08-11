# Rocket Radial-Expansion Tracker

A computer-vision tool that watches how things in a rocket video **grow or
shrink on screen** and turns that into per-frame motion metrics.

It uses optical flow to track hundreds of points across the video, then measures
the rate at which they slide outward (expansion / approaching) or inward
(contraction / receding). The result is a clean per-frame metrics CSV you can
use for whatever comes next — visualization, analysis, or feeding a machine
learning model of your own.

```
 launch.mp4 ──► rocket_flow.py ──► launch_metrics.csv  (per-frame motion features)
                 │                    launch_metrics.png (plot)
                 ▼
            launch_flow.mp4  (annotated video)
```

---

## 1. What you get (outputs)

| Output | What it is |
|--------|------------|
| `<name>_flow.mp4` | The original video, re-drawn with each tracked point in its own **random color**, a motion **trail** for each point, and an on-screen HUD. |
| `<name>_metrics.csv` | One row per frame: time, radial expansion, flow magnitude, flow-direction histogram, FOE position, expansion rate/acceleration, and state. |
| `<name>_metrics.png` | A plot of expansion rate and acceleration over time with colored state bands. |

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
pip install opencv-python numpy scipy matplotlib

# Extract metrics from your video
python3 rocket_flow.py launch.mp4
#   produces launch_flow.mp4, launch_metrics.csv, launch_metrics.png

# Optional tweaks
python3 rocket_flow.py launch.mp4 --draw-foe        # draw the expansion center
python3 rocket_flow.py launch.mp4 --scale 0.5       # process at half resolution (faster)
python3 rocket_flow.py launch.mp4 --no-plot         # skip the metrics plot

# Generate a synthetic test video (expansion then contraction) to check everything
python3 rocket_flow.py --synthetic test.mp4
```

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

`frame` is deliberately **not** a column — the rows are in frame order, and the
frame number is just the row index (`time_s` is kept as a convenience).

**Plot** — green bands = ASCEND, red bands = DESCEND. The velocity line
crossing zero is the moment the rocket switches direction.

### The five motion signals, explained

Each metric is a different view of the same motion:

| Signal | What it captures |
|--------|------------------|
| `radial_expansion` | **How strongly** the image is growing (or shrinking). Positive = objects moving outward (approaching), negative = inward (receding). One number per frame. |
| `flow_magnitude` | **How much** motion there is at all — mean speed of every tracked point. A busy smoke plume scores high even if it's not perfectly radial. |
| `flow_dir_hist_0..7` | **Which directions** the motion points. Pure rocket zoom-in concentrates in the "toward-center" bins; camera shake spreads across all bins. Lets you tell "expansion" from "noise". |
| `expansion_rate` | Smoothed radial velocity in px/s — the *speed* of size change. |
| `expansion_acceleration` | How fast the rate of size change itself changes — sharp transitions (liftoff, burnout, parachute deploy) show up here. |

Together: `flow_magnitude` says "something is moving", the histogram says "is it
radial?", `radial_expansion` says "which way", and rate + acceleration describe
the trajectory.

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

The math behind the FOE is a standard computer-vision result called the
**Focus of Expansion / Focus of Contraction** estimation (Horn's optical-flow
works). Nothing here is rocket-science math — just vectors, dot products, and a
least-squares fit. The "rocket science" is only in how you use it.
