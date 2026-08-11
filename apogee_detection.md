# Apogee Detection

How AeroNotts finds the rocket's apogee (apex of the trajectory) and splits the
flight into stable / ascend / apogee / descend phases.

The detection is a pure **single-pass numerical procedure over the whole video** —
it is done once after all frames are processed, not incrementally per-frame.
The whole algorithm lives in two places:

- `rocket_flow.py` — builds the radial-speed time series and an altitude proxy.
- `scripts/state.py::flight_phases()` — finds the peak and labels every frame.

Everything below uses the worked example `videoplayback.mp4`
(1802 frames, processed with `--scale 0.5`) so you can reproduce the numbers in
`video input/csv output/videoplayback_metrics.csv`.

---

## 1. Pipeline overview

```
frame N                        frame N+1
───────                        ────────
Shi-Tomasi corners ──► LK track ──► per-point flow vectors (u, v)
                                        │
                                        ▼
                                  FOE (focus of expansion)
                                        │
                                        ▼
                       radial speed per point = (u·dx + v·dy) / |d|
                                        │
                                        ▼
              radial_expansion (mean over points) ── column per frame
                                        │
                                        ▼
              ───────  post-processing (after all frames)  ──────
              vel     = radial · fps · (1/scale)          [px/s]
              vel_s   = savgol(vel)                       [px/s]
              accel   = gradient(vel_s, t)               [px/s²]
              alt     = cumsum(vel_s · ascent_sign · dt) [px]
              apogee  = argmax(alt)                       [frame index]
              phases  = STABLE/ASCEND/APOGEE/DESCEND      [per frame]
```

---

## 2. Raw input: mean radial speed

Every frame yields a per-point optical-flow vector `(u, v)` at each tracked
Shi-Tomasi point, obtained with Lucas–Kanade (pyramidal, bidirectional
back-substitution) in `scripts/features/sparse.py`.

Given an expansion center (the FOE, see below) at `c = (xF, yF)`, each point's
radial speed is the projection of its flow onto the line joining it to the center
(`sparse.py::radial_speeds`):

```python
d = points - center
r = hypot(d[:, 0], d[:, 1])
vr = (flow[:, 0] * d[:, 0] + flow[:, 1] * d[:, 1]) / (r + 1e-6)
```

- **`vr > 0`** → point moves **away** from the FOE (outward).
- **`vr < 0`** → point moves **toward** the FOE (inward).

The per-frame feature `radial_expansion` is the **mean** of `vr` over all tracked
points. For an approaching camera (rocket climbing away from a static camera) the
mean is strongly positive; it peaks as the rocket passes the middle of the view,
then shrinks toward zero as the rocket recedes into the distance.

> This is the single quantity that drives everything downstream: smoothing,
> the altitude proxy, per-frame `state`, and the global `phase`.

---

## 3. Focal point: the FOE

The radial-speed projection needs a center. The FOE is the image point from
which all genuine expansion appears to radiate. It is estimated by a
**weighted least-squares fit** (`sparse.py::estimate_foe`).

Each flow vector must be radial about `(xF, yF)`:

```
u·(y − yF) − v·(x − xF) = 0        →        v·xF − u·yF = v·x − u·y
```

Each vector is weighted by `1/|v|²` so tiny, noisy vectors are not overweighted.
Minimum 5 usable vectors; otherwise no estimate this frame. The raw estimate is
then **exponentially smoothed across frames** to keep it stable
(`rocket_flow.py`, `--foe-alpha`, default 0.3):

```
foe ← α · foe_raw + (1 − α) · foe_prev
```

`foe_x` / `foe_y` are written to the CSV in original-resolution pixels
(contrary to the flow columns, which are computed at the processing scale and
rescaled — see §4).

Useful flags:
- `--center auto` (default) — dynamic FOE as above.
- `--center frame` — fixed at frame center.
- `--center x,y` — fixed at arbitrary `(x, y)` in original resolution.

---

## 4. Velocity in px/s and NaN handling

The raw columns live in **processing pixels** (i.e. `--scale` already applied).
After the loop, `rocket_flow.py` converts times and velocities to a
physical-ish series:

```python
radial = np.array([row["radial_expansion"] for row in rows], dtype=float)
radial = np.nan_to_num(radial, nan=0.0)      # no-tracking frame → no motion
ts     = np.array(times)
dt     = np.diff(ts, prepend=ts[0]); dt[dt<=0] = 1/fps
vel    = radial * fps * inv                   # px/s in ORIGINAL resolution
```

- `inv = 1/scale` rescales back to original-resolution pixels (`--scale 0.5`
  runs → speeds doubled).
- NaN handling: if the tracker found nothing in a frame (no points), there is no
  radial speed; setting it to 0.0 avoids polluting the smoothing and integration
  with undefined values.

---

## 5. Smoothing

A zero-crossing of a noisy speed trace is meaningless, so the velocity is smoothed
with a **Savitzky–Golay filter** (local polynomial fit), `~rocket_flow.py`:

```python
win = args.smooth            # default 15, forced odd
vel_s = savgol_filter(vel, win, 2)          # polynomial order 2
```

All downstream work — altitude proxy, phase boundaries, `expansion_rate`,
`expansion_acceleration` — uses `vel_s`.

---

## 6. Altitude proxy (in pixels)

A single global peak is far more robust than per-frame zero-crossings (see §8),
but the raw speed time series does not have a clean single peak: speed rises,
flattens, then falls as the rocket recedes — a plateau, not a spike. To turn that
into a well-defined **apex**, the speed is integrated into a displacement:

```python
ascent_sign = -1.0 if invert else 1.0       # flip if "descent == expansion"
alt = np.cumsum(vel_s * ascent_sign * dt)   # pixel displacement proxy
```

Why this works:

| quantity        | during ASCENT | at apogee | during DESCENT |
|-----------------|---------------|-----------|----------------|
| vel_s           | > 0           | → 0       | < 0            |
| alt (cumsum)    | rises         | **maximum** | falls |

The cumulative sum of (signed) velocity is just the integral of velocity — a
displacement proxy. At the turn-around point the derivative changes sign, which
is exactly the **global maximum** of `alt`. Because `cumsum` accumulates, small
noise in `vel_s` does not create spurious local maxima the way zero-crossings of
`vel_s` itself would.

Units are **pixels** (integrated px/s × seconds). The peak value is *not* a real
altitude — it is proportional-ish to it for a fixed camera/rocket setup, and it
is the shape (the argmax) that matters.

---

## 7. Apogee = the global maximum

```python
apogee_idx = int(np.argmax(alt))      # scripts/state.py
```

That one index is **the** apogee frame. For `videoplayback.mp4`:

```
apogee frame = 1062
```

Just confirming an intuition: this places the apex near the video's middle
section, right where the raw `radial_expansion` column changes from sustained
positive (expanding) to sustained negative (contracting) in the CSV.

The exact **APOGEE band** is `/` window frames wide on each side
(`--apogee-window`, default 3):

```python
lo = max(0, apogee_idx - apogee_window)
hi = min(n, apogee_idx + apogee_window + 1)   # exclusive
```

So for the example run the APOGEE phase spans frames **1059–1066**:

| phase window   | frames     | alt behaviour            |
|----------------|------------|--------------------------|
| STABLE pad     | 0 – 59     | idle, |vel| < threshold  |
| ASCEND         | 60 – 1058  | alt rising               |
| **APOGEE**     | **1059–1066** | around the top          |
| DESCEND        | 1066–1733  | alt falling              |
| STABLE pad     | 1734 – 1801| landed, |vel| < threshold|

(The top and bottom ends of the band overlap by design — the band is clamped to
`[0, n]` — so one frame can be its own 1-frame APOGEE for `window=0`.)

---

## 8. STABLE pads around liftoff / landing

The apogee splits the flight, but the video usually also contains idle footage
before ignition and after impact. Those caps are found with a **sustained-motion
mask** — frames where speed is large enough to count as "flying":

```python
moving = np.abs(vel_s) >= thresh_px_s          # thresh_px_s = args.thresh * fps
moving = medfilt(moving, kernel_size=idle_min_frames) > 0.5   # pop noise
idx    = np.where(moving)[0]
liftoff = idx[idx <  apogee_idx][0]             # first flying frame before apex
landing = idx[idx >  apogee_idx][-1]            # last  flying frame after apex
```

- Frames `< liftoff` → `STABLE`.
- Frames `> landing` → `STABLE`.
- The median filter (`--idle-min-frames`, default 5) demands motion be
  *sustained*, so a one-frame tracking hiccup does not set liftoff.

The complete label stream is therefore:

```
STABLE ──► ASCEND ──► APOGEE ──► DESCEND ──► STABLE
```

with the two STABLE runs typically the launch pad and the landing spot.

---

## 9. Full worked example

Reproduce it yourself:

```bash
python3 rocket_flow.py "video input/videoplayback.mp4" --scale 0.5
```

| step | value |
|------|-------|
| frames | 1802 |
| `--scale` | 0.5 (processing ½ resolution) |
| `--smooth` | 15 |
| `--apogee-window` | 3 |
| `--idle-min-frames` | 5 |
| peak of `alt` (argmax) | **frame 1062** |
| APOGEE phase | frames 1059–1066 |
| liftoff | frame 60 |
| landing | frame 1733 |
| ASCEND | 60 – 1058 |
| DESCEND | 1066 – 1733 |
| STABLE | 0 – 59 and 1734 – 1801 |

---

## 10. Caveats and knobs

- **Pixels, not meters.** `alt` and `vel` are in image pixels; true distance needs
  extra calibration (focal length, pixel pitch, camera-to-launch-site geometry).
- **Monocular, single camera.** Depth is inferred purely from optical-flow
  geometry. A rocket climbing straight away is well modelled; lateral drift,
  strong wind, camera shake, or panning all degrade the estimate.
- **Why not zero-crossings?** The old per-frame classifier flipped ASCEND/DESCEND
  on every noisy crossing — 37 mini "apogees" in one video. The *global peak of
  the altitude proxy* has exactly one answer and is stable to smoothing.
- **`--thresh` sensitivity.** `state` (per-frame) and the STABLE pads both depend
  on `thresh * fps`: too low, and camera noise counts as motion (pads shrink);
  too high, and slow real motion is ignored.
- **`--smooth` window.** Must be odd and `< frame count`. Too small → jagged
  velocity and a noisy argmax; too large → the argmax shifts/lags the true apex.
- **`--apogee-window`.** Only widens the marker band visually; it does *not* move
  the detected apex, which is always `argmax(alt)`.
- **`--idle-min-frames`.** A 1 disables the median filter entirely (any single
  moving frame sets the pads); larger values demand sustained motion.
- **FOE validity.** If fewer than 5 valid flow vectors exist, `foe` holds the last
  good smoothing value; with `--center auto` and no points at all, radial-speed
  features are NaN (later zeroed) and `foe_x/foe_y` carry forward.