# What's In This CSV? — A Beginner's Guide

This file explains every single column in the `videoplayback_metrics.csv` file.
Think of each row as a single video frame (a single still image from your video).
Each column tells you something different about what's happening in that frame.

There are **72 columns**. They're grouped into 8 categories. Let's walk through them.

---

## Before we start — the big picture

This software watches a video of a rocket launch and writes down, for every frame:
- **What the rocket is doing** (going up? coming down? sitting still?)
- **How the pixels in the image are moving** (which way they're drifting)
- **What the image looks like** (how much detail, how sharp, how much sky)

It does this by tracking hundreds of "dots" (Shi-Tomasi corners) from one frame to the next using a technique called **Lucas-Kanade optical flow**.

---

## 1. Time + What's Happening (3 columns)

### time_s — "When did this frame happen?"
- A number like `0.0417`, `0.0834`, `1.50`, `5.92`...
- It's the timestamp in **seconds**. If the video is 24 frames per second, each frame is 0.0417 seconds apart.
- This is your X-axis if you want to plot anything over time.

### state — "Right now, is the rocket going up, down, or sitting still?"
- A label: either `STABLE`, `ASCEND`, or `DESCEND`.
- This is decided **per frame** by looking at whether the tracked points are spreading out (ASCEND) or coming together (DESCEND), or barely moving (STABLE).
- It's like asking "in this exact moment, what direction is the rocket moving?"
- **Smoothing:** To avoid a single noisy frame flipping the label incorrectly, it applies a median filter.

### phase — "At this point in the whole flight, where are we?"
- A label: `STABLE`, `ASCEND`, `APOGEE`, or `DESCEND`.
- This is a **global** view — the software watches the entire video once, finds the apogee (highest point), and then labels every frame's role in the overall flight.
- `STABLE` = before liftoff or after landing (rocket isn't moving)
- `ASCEND` = rocket is on its way up
- `APOGEE` = the few frames right around the peak (highest point)
- `DESCEND` = rocket is falling back down
- Unlike `state` (which can flip around moment to moment), `phase` tells the clean story: "Here's when it launched, here's the peak, here's when it landed."

**Key difference:** `state` = moment-by-moment. `phase` = the arc of the whole flight.

---

## 2. Radial Expansion (6 columns)

These 6 columns answer: **"Are the tracked points spreading out from the center, or coming together?"**

Imagine the rocket is flying away from you. Points on the rocket (and the ground, and the sky) drift outward — like ripples from a stone dropped in water. The center of all this outward expansion is called the **FOE** (Focus of Expansion — more on that in section 5).

For each tracked point, the software computes a single number called **radial speed** (`vr`):
- Positive + spreading out = rocket going up
- Negative + coming together = rocket coming down
- Near zero = not moving

Then it summarizes this across all tracked points:

### radial_expansion — "Overall, are points flying apart or coming together?"
- The **average** radial speed. Your #1 rocket metric.
- Big positive number → rocket ascending fast.
- Big negative number → rocket descending fast.
- Near zero → rocket is stable (hovering, landed, or stationary).
- The sign and magnitude of this column is the single most important signal for determining if the rocket is ascending or descending.

### radial_expansion_median — "What does the 'typical' point do?"
- The **median** radial speed (ignores weird outliers).
- If `radial_expansion` (mean) feels jumpy but `radial_expansion_median` is smooth, it tells you most points agree but a few are weird.

### radial_std — "Is everyone moving in the same direction at the same speed?"
- The **standard deviation** of radial speeds.
- Low = all points moving similarly (clean expansion).
- High = different points moving at very different speeds (messy flow, maybe the rocket is tumbling or there's bad tracking).

### radial_p95 — "How fast is the fastest 5% of points expanding?"
- The **95th percentile** radial speed.
- Catches brief bursts of very fast expansion (e.g., a flare or fast-moving feature the mean might dilute).

### outward_frac — "What fraction of points are moving away from the center?"
- A ratio from 0.0 to 1.0. For example, `0.6500` means 65% of points are moving outward.
- Should be high (>0.5) during a clean ascent.
- Pairs with `inward_frac`. Together they should roughly sum to 1.0 (every point is either going out or coming in, or barely moving).

### inward_frac — "What fraction of points are moving toward the center?"
- Same idea, opposite direction. High during descent.
- For a clean rocket shot: `outward_frac` is big during ascent, `inward_frac` is big during descent.

---

## 3. How Much Are Points Moving (4 columns)

These look at the **raw pixel distance** each tracked point traveled from one frame to the next (not radial — just plain Euclidean distance).

### pt_displacement_mean — "On average, how many pixels did each point move?"
- The average straight-line distance of all tracked points between consecutive frames. In px/frame.
- Higher = more motion overall in the image (fast rocket movement, fast camera shake).
- Lower = less motion.
- This tells you the overall "agitation" level of the image, regardless of direction.

### pt_displacement_median — "What's the typical point's movement?"
- The median (middle value) of all point displacements. More robust than the mean.
- If `pt_displacement_mean` is skewed by a few super-fast points, this tells you what most points actually did.

### pt_displacement_std — "Are all points moving about the same speed, or is it chaotic?"
- Standard deviation of the displacements.
- Low = most points moved about the same amount (smooth, uniform motion).
- High = points scattered across many different speeds (chaotic flow, likely bad tracking or multiple independent motions).

### pt_displacement_max — "What's the single fastest point doing?"
- The maximum displacement among all points.
- Catches that one point that zoomed across the screen (could be a flare, could be a tracking glitch).

---

## 4. Where Are the Points? (4 columns)

These describe the **spatial distribution** of the tracked point cloud itself.

### point_count — "How many points are we tracking?"
- The actual integer count of tracked points this frame (e.g., `28.0`, `70.0`, `169.0`).
- Starts low, grows as the tracker finds more corners. It should stabilize somewhere below `--max-corners` (default 500).
- This is useful for diagnosing tracking quality: near-zero means the tracker lost everything.

### point_density — "How much of the image is covered by tracked points?"
- `point_count / (image_width × image_height)`.
- A tiny number (e.g., `0.0007` means 0.07% of the image area has tracked points).
- Higher density = better tracking coverage; lower = tracking is sparse.

### feature_radius_mean — "How far from the center are the tracked points, on average?"
- The mean distance from each tracked point to the FOE center.
- If this is large, points are spread out across the whole image.
- If this is small, points are clustered near the center.
- Tells you the "radius" of your tracked cloud.

### feature_radius_std — "How evenly spread is the cloud?"
- Standard deviation of point-to-FOE distances.
- Low = points are roughly the same distance from the center (even ring).
- High = points are patchy (some close, some far).

---

## 5. What Direction Is Everything Flowing? (9 columns)

### flow_magnitude — "How fast is the average point moving?" (simple version)
- Identical to `pt_displacement_mean`. Included so you can directly compare sparse (this) vs. dense (Group 6) flow magnitudes side by side.

### flow_dir_hist_0 through flow_dir_hist_7 — "Which of 8 compass directions is the motion going?"
- The software takes every tracked point's flow vector and figures out its direction (0° to 360°).
- It bins all directions into **8 buckets**, each covering 45°:

```
Bin 0  [  0°– 45° ]  →  East-Northeast
Bin 1  [ 45°– 90°  ]  →  Southeast
Bin 2  [ 90°–135°  ]  →  South-Southeast
Bin 3  [135°–180°  ]  →  South-Southwest
Bin 4  [180°–225°  ]  →  Southwest
Bin 5  [225°–270°  ]  →  West-Southwest
Bin 6  [270°–315°  ]  →  West-Northwest
Bin 7  [315°–360°  ]  →  North-Northeast
```

- Each column is a **ratio** (e.g., `0.2700` = 27% of points flow in that direction).
- All 8 should roughly sum to 1.0.
- If you see one bin with `0.70+` and the rest near 0, motion is highly directional (e.g., everything drifting down-left = likely descent).
- If all 8 bins are near `0.125` (1/8), motion is spread evenly in all directions (no coherent flow — possibly noise or featureless scene).

---

## 6. Focus of Expansion (FOE) (2 columns)

### foe_x, foe_y — "Where in the image does all the expansion seem to come from?"
- The **(x, y) pixel coordinates** (in full original image resolution) of the FOE.
- During **ascent**, all the tracked points' flows radiate **outward** from this point (like spokes on a wheel, pointing away from the hub).
- During **descent**, flows radiate **inward** toward this point (spokes pointing toward the hub).
- Think of it as the "center of the universe" for this frame's motion — the one point that stays stationary while everything else flows away (or toward it).
- It's exponentially smoothed across frames (`--foe-alpha = 0.3`), so it moves smoothly rather than jumping frame-to-frame.
- **NaN** only in the first few frames before a reliable estimate exists.
- These values feed directly into the radial speed calculations in section 2 — `r` in the radial speed formula is the distance of each point from `(foe_x, foe_y)`.

---

## 7. Dense Optical Flow (19 columns)

Up until now, we've only looked at motion at the ~hundreds of tracked points. **Dense flow** (Farneback's algorithm) computes motion at **every single pixel** — all of them, no matter if there's a corner or not.

These 19 columns summarize the full dense flow field. All measured in px/frame (processing resolution).

### Magnitude Statistics — "How much are pixels moving, in general?" (4 columns)

Let `m = sqrt(u² + v²)` for every pixel (the length of each pixel's motion vector).

#### flow_median — "What's the typical pixel motion?"
- The median of `m` across the entire image.
- Robust view of "how much stuff is drifting."

#### flow_p95 — "How fast is the fastest 5% of pixels?"
- 95th percentile of `m`. Catches the extreme movers.

#### flow_std — "Is motion uniform or all over the place?"
- Standard deviation of `m`. Low = uniform motion; high = chaotic.

#### flow_max — "What's the single fastest pixel doing?"
- Absolute maximum of `m`.

### Divergence — "Is the flow field expanding or shrinking?" (6 columns)

Divergence is `∂u/∂x + ∂v/∂y` — basically measuring "are pixels spreading apart or squeezing together locally?"

- **Positive divergence** = pixels spreading apart (expansion = rocket ascending).
- **Negative divergence** = pixels squeezing together (contraction = rocket descending).

This is a second, independent expansion signal (the first being `radial_expansion` from sparse flow).

#### div_mean — "Overall, is the image expanding or contracting?"
- Mean divergence over the whole image.
- Positive during ascent; negative during descent.
- The dense-flow equivalent of `radial_expansion`.

#### div_median — "What does the typical local region do?"
- Median divergence — robust to outliers.

#### div_std — "How uneven is the expansion?"
- Standard deviation of divergence.

#### div_p95 — "Where's the strongest local expansion?"
- 95th percentile divergence.

#### div_pos_frac — "What fraction of the image is locally expanding?"
- Ratio 0–1. `1.0` = every pixel expanding; `0.0` = every pixel contracting.
- A clean ascent should push this toward 1.0; descent pushes it toward 0.0.

#### div_max — "What's the strongest expansion anywhere?"
- Maximum divergence value.

### 3×3 Grid — "Where in the image is motion happening?" (9 columns)

The image is split into a 3×3 grid (9 equal boxes), and the mean flow magnitude is computed in each:

```
 grid_flow_00  grid_flow_01  grid_flow_02     ← TOP row    (sky / rocket upper body)
 grid_flow_10  grid_flow_11  grid_flow_12     ← MID row    (horizon)
 grid_flow_20  grid_flow_21  grid_flow_22     ← BOTTOM row (ground / foreground)
```

- Column naming: `grid_flow_{row}{col}`.
- Example: `grid_flow_11` is the dead center of the image. `grid_flow_00` is top-left.
- Each value is the mean px/frame motion in that 1/9 region.
- Compare top vs. bottom: if `grid_flow_01` (top-center) is much bigger than `grid_flow_21` (bottom-center), motion is concentrated in the upper image (likely the rocket in the sky). If bottom cells are also high, the ground might be moving too (camera shake / landing stage).

---

## 8. Camera Motion (14 columns)

The software tries to figure out: "How is the *camera itself* moving?" Because the rocket isn't the only thing creating motion — the camera might be panning, zooming, or shaking. These columns model the global camera motion, then subtract it out to isolate the rocket.

### Affine Model — Global camera rotation + zoom + shift (4 columns)

The software fits a "partial affine" transform (a 2×3 matrix) to all tracked points using RANSAC. This transform encodes:
- How much the camera rotated,
- How much it zoomed,
- How much it shifted sideways / up-down.

```
Matrix M = [ a   b   tx ]
           [ -b  a   ty ]
```

#### cam_rotation — "How much is the whole camera rotating?"
- In degrees. Positive = counter-clockwise. Negative = clockwise.
- Captures camera pan/tilt/roll.

#### cam_scale — "How much is the camera zooming?"
- A ratio. `1.0` = no zoom.
- `< 1.0` = scene is shrinking (camera pulling back, or rocket climbing away).
- `> 1.0` = scene is growing (zooming in, or rocket approaching).
- For a rocket shot: this should dip below 1.0 during ascent (rocket getting smaller as it flies away).

#### cam_tx — "How far is the camera shifting sideways?"
- Horizontal translation in px (processing resolution).

#### cam_ty — "How far is the camera shifting up/down?"
- Vertical translation in px.

### Residual Flow — "What's left after removing the camera's motion?" (3 columns)

After subtracting the global camera model, whatever motion remains is the **non-rigid** motion — i.e., the rocket itself moving independently of the background.

#### residual_flow_mean — "How much is the rocket moving on its own?"
- Average residual flow magnitude (px/frame).
- ~0 if the camera perfectly explains all motion (static scene, static camera).
- Large = the rocket is doing something the global camera model can't account for.

#### residual_flow_p95 — "What's the worst unexplained motion?"
- 95th percentile residual.

#### residual_div — "Is the rocket's own motion expanding or contracting?"
- Divergence of the residual field: `2 * mean((res · d) / |d|²)`.
- Positive = rocket's residual flow is expanding (pushing outward beyond camera motion).
- Negative = rocket's residual flow is contracting.
- Independent check on `radial_expansion` — should agree in sign.

### Homography Model — Planar scene fit (7 columns)

A more powerful model: it fits a **full projective homography** (8 parameters, 3×3 matrix) that can handle perspective effects (e.g., a flat ground plane tilting). Requires at least 8 points.

```
Matrix H = [ a   b  tx ]
           [ c   d  ty ]
           [ px  py  1 ]
```

#### hom_scale — "How much is the (planar) scene zooming?"
- `sqrt(abs(a·d − b·c))` — the planar zoom factor.

#### hom_rotation — "How much is the planar scene rotating?"
- `degrees(arctan2(H[1,0], H[0,0]))` — planar rotation in degrees.

#### hom_tx, hom_ty — "Where is the planar scene shifting?"
- Planar translation (px).

#### hom_persp_x, hom_persp_y — "Is the plane tilting?"
- `H[2,0]` and `H[2,1]` — perspective drive terms. Near 0 = flat/fronto-parallel plane; nonzero = perspective skew.

#### hom_ok — "How well does the scene fit a flat plane?"
- Ratio 0–1. Fraction of tracked points that are inliers (fit the homography well).
- `≈ 1.0` = scene is flat/planar (everything fits one plane).
- **Lower** = the scene is *not* flat (the rocket is breaking out of the plane — a strong rocket-detection signal!).
- If the rocket rises out of the launch pad plane, `hom_ok` should **decay** as it becomes an outlier.

---

## 9. What Does the Picture Look Like? (6 columns)

These are simple per-frame image statistics — no motion involved, just "what does this frame look like?"

#### edge_density — "How much detail/structure is in the image?"
- Fraction of pixels that are edges (via Canny edge detector). 0.0–1.0.
- High when there's lots of structure (rocket, pad, launch tower).
- Low for textureless regions (smooth sky, uniform smoke).

#### texture_var — "How much variation is there in pixel brightness?"
- `gray.var()`. Measures how 'busy' the image is in terms of pixel value spread.
- High variance = lots of contrast/detail.
- Low variance = flat/muted.

#### grad_magnitude_mean — "On average, how strong are the brightness gradients?"
- Mean Sobel gradient magnitude. Measures edge sharpness across the image.

#### sharpness — "How in-focus is the image?"
- Laplacian variance — a standard blur metric.
- **High** = sharp and in focus.
- **Low** = blurry out of focus (e.g., smoke obscuring the rocket).

#### sky_fraction — "How much of the frame is sky?"
- Fraction of pixels that are bright and low-saturation (HSV heuristic).
- Ratio 0–1.

#### ground_fraction — "How much of the frame is ground/land?"
- Fraction of pixels that are mid-dark and textured (HSV heuristic).
- Ratio 0–1.

---

## 10. Horizon Detection (experimental) (3 columns)

Tries to find the horizon line in the image using Hough transform on edges.

#### horizon_angle — "What angle is the horizon tilted at?"
- Degrees from horizontal. A level horizon = ~0° (or ±180°).
- NaN if no line was found (e.g., overexposed sky, no visible horizon).

#### horizon_pos — "Where vertically is the horizon?"
- A ratio 0–1. `0.0` = horizon at the very top; `1.0` = horizon at the very bottom.
- NaN if no line found.

#### horizon_conf — "How confident are we that this is a real horizon line?"
- A ratio 0–1 (line length / image diagonal). Higher = longer, more trustworthy line.
- 0.0 if no line found. **Never NaN.**

---

## 11. Smooth Velocity Signals (2 columns)

These are the **final, polished** signals used for apogee detection. Computed at the very end after all frames are processed.

#### expansion_rate — "How fast is the rocket expanding the scene, in real units?"
- Units: **px/second** (scaled back to original resolution).
- It's a Savitzky-Golay-smoothed version of `radial_expansion · fps · (1/scale)`.
- **Positive** = rocket getting farther away (ascending).
- **Zero** = at the transition.
- **Negative** = rocket getting closer (descending).
- NaN values in raw `radial_expansion` are filled with 0.0 before smoothing.

#### expansion_acceleration — "Is the rocket speeding up or slowing down?"
- Units: **px/second²**.
- Time derivative of `expansion_rate` (`np.gradient(expansion_rate, time_s)`).
- Tells you the **rate of change** of the rocket's speed.
- Positive = accelerating expansion (rocket speeding away faster).
- Zero-crossing (goes + to −) = peak expansion rate = near the apogee.
- Negative = decelerating expansion then accelerating descent.

---

That's all 72 columns! Summary of the 8 groups:

| Columns | Group | What it tells you |
|---------|-------|-------------------|
| 1–3 | Time + Labels | When + overall flight status |
| 4–9 | Radial Expansion | Net inflow/outflow of tracked points |
| 10–13 | Displacement | How far points are physically moving |
| 14–17 | Cloud Distribution | Where the tracked points are |
| 18–26 | Direction Histogram | Which directions the motion points |
| 27–28 | FOE | Where the expansion is centered |
| 29–47 | Dense Flow | Full-image motion + divergence + grid |
| 48–61 | Camera Motion | Global camera shift + rocket-specific residual |
| 62–67 | Appearance | Image sharpness/detail/sky/ground |
| 68–70 | Horizon | Horizon line angle and position |
| 71–72 | Temporal Signals | Smooth velocity & acceleration for apogee detection |
