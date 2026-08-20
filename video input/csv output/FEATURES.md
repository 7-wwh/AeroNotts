# FEATURES.md — Per-Column Reference

Complete documentation of every column emitted by `rocket_flow.py` into
`*_metrics.csv`. Each section lists the columns in schema order
(`scripts/schema.py` is the single source of truth for names *and* order), the
exact formula, the source function, and units.

For the apogee / phase pipeline that produces the `state` and `phase` columns,
see **apogee_detection.md**.

---

## CSV conventions

- One row per frame, in frame order. You can recover the frame number from its
  position: `row i = frame i`. There is deliberately **no `frame` column**.
- `time_s = i / fps` (`video_props`), put in the first column.
- All floats are written with `%.4f`; missing values are the literal string
  `nan` (e.g. horizon features when no line is found).
- Flow-related columns are computed at processing resolution (`--scale`) and
  rescaled to original resolution where it matters (see `--scale` notes).
- Columns with no NaN are guaranteed every frame: `time_s`, `state`, `phase`,
  `point_count`, `point_density`, `flow_magnitude`-family, dense-flow columns,
  appearance columns, and the three temporal columns.

### Quick column map

| group | columns | source |
|-------|---------|--------|
| time + labels | 3 | `rocket_flow.py` post-processing |
| sparse flow | 23 | `scripts/features/sparse.py` |
| FOE | 2 | `scripts/features/sparse.py::estimate_foe` |
| dense flow | 19 | `scripts/features/dense.py` |
| camera ego-motion | 14 | `scripts/features/camera.py` |
| appearance | 6 | `scripts/features/appearance.py` |
| horizon (experimental) | 3 | `scripts/features/horizon.py` |
| temporal | 2 | `rocket_flow.py` post-processing |
| **total** | **72** | |

---

## 1. Time + labels (3 columns)

| column | description |
|--------|-------------|
| `time_s` | `frame_idx / fps`. |
| `state` | Per-frame class from thresholding mean radial speed: `ASCEND`, `DESCEND`, `STABLE`. Deflickered with a median filter (`--state-median`, default 5). Is **not** inverted — it always means "expansion = ASCEND" (see `APOGEE` note in §2.1 of apogee_detection.md for the inversion semantics of `--descend-is-expansion`). |
| `phase` | Global trajectory label from `flight_phases()`: `STABLE` / `ASCEND` / `APOGEE` / `DESCEND`. Exactly one `APOGEE` band per video. See **apogee_detection.md**. |

---

## 2. Sparse optical-flow features (Shi–Tomasi + Lucas–Kanade) (23)

Tracked points: up to `--max-corners` (default 500) Shi-Tomasi corners clustered
with `--min-dist`, followed by pyramidal LK (`--lk-win` 15, `--max-level` 2,
bidirectional back-substitution by default). Lost points are re-seeded each
frame, so `point_count` measures *current tracked features*.

All geometry is in **processing pixels** (already scaled).

### 2.1 Radial (expansion) family

Computed against the FOE (or fixed center) `c`:

```python
d  = points − c
r  = hypot(d[:,0], d[:,1])
vr = (u·dx + v·dy) / (r + 1e-6)      # per point, px/frame
```

| column | formula | units | meaning |
|--------|---------|-------|---------|
| `radial_expansion` | `mean(vr)` | px/frame | overall net outflow; the headline rocket feature |
| `radial_expansion_median` | `median(vr)` | px/frame | robust central outflow (resists outliers) |
| `radial_std` | `std(vr)` | px/frame | spread of radial speeds across the point cloud |
| `radial_p95` | `percentile(vr, 95)` | px/frame | near-max outflow |
| `outward_frac` | `mean(vr > 0)` | ratio | fraction of points moving away from FOE |
| `inward_frac` | `mean(vr < 0)` | ratio | fraction moving toward FOE (for a rocket these two ≈ 1 − outward) |

### 2.2 Displacement family

```python
mag = hypot(flow[:,0], flow[:,1])    # px between consecutive frames
```

| column | formula | units | meaning |
|--------|---------|-------|---------|
| `pt_displacement_mean` | `mean(mag)` | px/frame | average track length moved between frames |
| `pt_displacement_median` | `median(mag)` | px/frame | robust central displacement |
| `pt_displacement_std` | `std(mag)` | px/frame | how heterogeneous the motion is |
| `pt_displacement_max` | `max(mag)` | px/frame | fastest-moving feature |

### 2.3 Cloud size

| column | formula | units | meaning |
|--------|---------|-------|---------|
| `point_count` | `len(p0)` | count | number of tracked features this frame |
| `point_density` | `count / (sW·sH)` | 1/px² | normalized tracking load |
| `feature_radius_mean` | `mean(hypot(pt − c))` | px | how far, on average, features lie from the FOE |
| `feature_radius_std` | `std(hypot(pt − c))` | px | spatial spread of the tracked cloud |

### 2.4 Flow magnitude + direction histogram

| column | formula | meaning |
|--------|---------|---------|
| `flow_magnitude` | `mean(mag)` | same as `pt_displacement_mean` (kept for symmetry with dense flow) |

`flow_dir_hist_0..7` — 8-bin histogram of flow direction:

```python
angle = degrees(arctan2(v, u)) % 360
hist, _ = np.histogram(angle, bins=8, range=(0, 360))
hist = hist / max(1, len(angles))       # normalized → ratios, sum ≈ 1
```

Bin `i` covers `[i·45°, (i+1)·45°)`. Dominant outward motion during ascent shows
up as a peak near the bin containing the FOE–point radial direction; a
concentric/contracting descent pushes mass into the opposite bins.

---

## 3. FOE (2 columns)

| column | meaning |
|--------|---------|
| `foe_x` | smoothed focus-of-expansion x, original-resolution px |
| `foe_y` | smoothed focus-of-expansion y, original-resolution px |

Fit from the formula `u·(y − yF) − v·(x − xF) = 0` via weighted least squares
(weight `1/|v|²`), min 5 vectors; then exponential smoothing with
`--foe-alpha` (default 0.3). NaN only before the first valid estimate. See
`sparse.py::estimate_foe`.

---

## 4. Dense optical-flow features (Farneback) (19)

Computed from `cv2.calcOpticalFlowFarneback(prev, cur)` with
`pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2`.

Full per-pixel field `F(x,y) = (u(x,y), v(x,y))`, in **processing px/frame**.
Disabled with `--no-dense`.

### 4.1 Magnitude stats

```python
m = hypot(F[...,0], F[...,1])
```

| column | formula | meaning |
|--------|---------|---------|
| `flow_median` | `median(m)` | typical pixel motion |
| `flow_p95` | `percentile(m, 95)` | near-max pixel motion |
| `flow_std` | `std(m)` | variability of motion over the image |
| `flow_max` | `max(m)` | peak pixel motion |

### 4.2 Divergence — the expansion clue

```python
div = ∂u/∂x + ∂v/∂y               # via np.gradient (central differences)
```

| column | formula | meaning |
|--------|---------|---------|
| `div_mean` | `mean(div)` | net expansion of the whole image (rocks → >0) |
| `div_median` | `median(div)` | robust central divergence |
| `div_std` | `std(div)` | how unevenly the field expands/contracts |
| `div_p95` | `percentile(div, 95)` | strongest local expansion |
| `div_pos_frac` | `mean(div > 0)` | fraction of pixels locally expanding |
| `div_max` | `max(div)` | peak local divergence |

Units: `1/frame` (per px-per-frame gradient of px-per-frame field).

### 4.3 Grid magnitudes

`grid_flow_00` … `grid_flow_22` — the image split into a 3×3 grid
(row-major: `grid_flow_00` = top-left, `grid_flow_22` = bottom-right),
each cell's **mean flow magnitude** in px/frame:

```python
cell = m[h·i/3 : h·(i+1)/3 , w·j/3 : w·(j+1)/3]
grid_flow_ij = mean(cell)
```

The upper cells capture the rocket/tracked-motion region; the lower cells the
static foreground. Their ratio is a cheap proxy for where motion concentrates.

---

## 5. Camera ego-motion features (14)

Models the *static scene* wrapping-shift, so the rocket's apparent motion is
isolated in *residual* flow. Disabled pieces: homography with `--no-homography`.

### 5.1 Affine model (RANSAC)

```python
M = cv2.estimateAffinePartial2D(prev, cur, method=RANSAC)   # needs ≥ 6 pts
   [ a  b  tx ]
   [−b  a  ty ]        (partial = rigid: rotation + scale + translation)
```

| column | formula | meaning |
|--------|---------|---------|
| `cam_rotation` | `degrees(arctan2(b, a))` | global rotation (deg) |
| `cam_scale` | `hypot(a, b)` | global zoom (1.0 = no zoom) |
| `cam_tx` | `M[0,2]` | translation x (px) |
| `cam_ty` | `M[1,2]` | translation y (px) |

`cam_scale` < 1 means whole scene shrinking → camera backing away / rocket
climbing away. `cam_*` are NaN when fewer than 6 points or the RANSAC fit fails.

### 5.2 Residual flow (rocket' signature)

After removing the rigid model's contribution, whatever motion *remains* is
non-rigid — i.e. the rocket itself.

```python
model_cur = [prev, 1] @ M.T            # where the scene "should" be
res       = cur − model_cur
mag       = hypot(res[:,0], res[:,1])
```

| column | formula | meaning |
|--------|---------|---------|
| `residual_flow_mean` | `mean(mag)` | avg non-rigid motion left over |
| `residual_flow_p95` | `percentile(mag, 95)` | worst non-rigid outlier |
| `residual_div` | `2 · mean( (res·d) / |d|² )` | divergence of residual field about the FOE |

`residual_div` is NaN when no center is available (`--center auto` with too few
points).

### 5.3 Homography model (planar scene)

```python
H = cv2.findHomography(prev, cur, RANSAC, ransac_reproj=5.0)   # needs ≥ 8 pts
```

| column | formula | meaning |
|--------|---------|---------|
| `hom_scale` | `sqrt(abs(det(H[:2,:2])))` | planar zoom |
| `hom_rotation` | `degrees(arctan2(H[1,0], H[0,0]))` | planar rotation |
| `hom_tx` | `H[0,2]` | planar translation x |
| `hom_ty` | `H[1,2]` | planar translation y |
| `hom_persp_x` | `H[2,0]` | horizontal perspective drive |
| `hom_persp_y` | `H[2,1]` | vertical perspective drive |
| `hom_ok` | `mean(inlier_mask)` | fraction of points fitting the model (0–1) |

`hom_ok` ≈ 1 → the scene is well-modeled as planar; a rocket breaking that planarity
drops `hom_ok`.

---

## 6. Image appearance features (6)

Pure per-frame pixel statistics on the (scaled) frame, no motion. Disabled with
`--no-appearance`.

| column | formula | meaning |
|--------|---------|---------|
| `edge_density` | `mean(Canny(gray,100,200) > 0)` | fraction of edge pixels (rocket+pad = high) |
| `texture_var` | `gray.var()` | pixel-value variance (richness of detail) |
| `grad_magnitude_mean` | `mean(hypot(SobelX, SobelY))` | mean gradient strength |
| `sharpness` | `var(Laplacian(gray))` | Laplacian variance — classic focus/blur measure |
| `sky_fraction` | `mean((V>0.55) & (S<0.35))` | HSV heuristic: bright, low-sat = sky |
| `ground_fraction` | `mean((0.15<V≤0.6) & (S≥0.05))` | mid-dark, textured = ground |

These group together (e.g. sharpness dropping toward landing, `ground_fraction`
growing as the field of view fills with the launch site).

---

## 7. Horizon features (experimental) (3)

Longest straight line found on a `0.5×`-downscaled Canny edge map via
`HoughLinesP(threshold=60, minLineLength=0.15·diag, maxLineGap=20)`.
Disabled with `--no-horizon`. Reported as **NaN** when nothing is found; brief
sections of the flight with no line read `nan`.

| column | formula | meaning |
|--------|---------|---------|
| `horizon_angle` | `degrees(arctan2(Δy, Δx))` of best line | tilt from horizontal (deg) |
| `horizon_pos` | `mid_y / height_small` | normalized line center-row, 0=top 1=bottom |
| `horizon_conf` | `best_len / hypot(h,w)` | confidence: line length vs image diagonal |

Useful as a camera-roll/attitude signal, or to locate the horizon row for
sky/ground context. Treat as exploratory — it cannot handle a featureless or
curved boundary.

---

## 8. Temporal / post-processed (2)

Computed once at the end of `rocket_flow.py` (not per-frame inside the loop).

| column | formula | meaning |
|--------|---------|---------|
| `expansion_rate` | `savgol(radial·fps/scale, win=15, poly=2)` | smoothed radial speed **in px/s, original res** |
| `expansion_acceleration` | `np.gradient(expansion_rate, time_s)` | derivative of the rate, **px/s²** |

- These are the smooth signals used by the apogee/phase detection
  (see **apogee_detection.md**).
- The **peak of `expansion_acceleration`** lags the peak of `expansion_rate`;
  the zero-crossing of `expansion_acceleration` (rate flattening) plus the peak
  of `expansion_rate` bracket the apogee — that is the *shape* the altitude
  proxy exploits, since `expansion_rate` alone plateaus instead of peaking.

---

## Column list (schema order)

```
time_s  state  phase
radial_expansion  radial_expansion_median  radial_std  radial_p95
outward_frac  inward_frac
pt_displacement_mean  pt_displacement_median  pt_displacement_std  pt_displacement_max
point_count  point_density
feature_radius_mean  feature_radius_std
flow_magnitude
flow_dir_hist_0..7                     (8)
foe_x  foe_y
flow_median  flow_p95  flow_std  flow_max
div_mean  div_median  div_std  div_p95  div_pos_frac  div_max
grid_flow_00..22                       (9)
cam_rotation  cam_scale  cam_tx  cam_ty
residual_flow_mean  residual_flow_p95  residual_div
hom_scale  hom_rotation  hom_tx  hom_ty  hom_persp_x  hom_persp_y  hom_ok
edge_density  texture_var  grad_magnitude_mean  sharpness  sky_fraction  ground_fraction
horizon_angle  horizon_pos  horizon_conf
expansion_rate  expansion_acceleration
```

Count: 3 + 23 + 2 + 19 + 14 + 6 + 3 + 2 = **72 columns**.