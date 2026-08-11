#!/usr/bin/env python3
"""Track radial image expansion/contraction with sparse optical flow.

Estimates the Focus of Expansion (FOE) from Lucas-Kanade optical flow of
Shi-Tomasi features, then computes per-frame radial expansion velocity and
acceleration to classify ascent (object expanding / approaching) vs descent
(object contracting / receding).

Outputs:
  - annotated video: tracked points drawn with random per-point colors + trails,
    FOE crosshair, and a HUD (radial velocity, acceleration, state)
  - metrics CSV: frame, time, mean/median radial velocity, FOE, accel, state
  - matplotlib plot of velocity & acceleration with state regions overlaid
"""
import argparse
import os
import sys

import numpy as np
import cv2
from collections import deque
from scipy.signal import savgol_filter, medfilt

DESC = "DESCEND"
STAB = "STABLE"
ASC = "ASCEND"

STATE_COLORS = {ASC: (60, 220, 60), STAB: (200, 200, 200), DESC: (60, 60, 240)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="input video file")
    p.add_argument("--out", help="output annotated video (default: <input>_flow.mp4)")
    p.add_argument("--csv", help="output metrics CSV (default: <input>_metrics.csv)")
    p.add_argument("--plot", help="output metrics plot (default: <input>_metrics.png)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the metrics plot (headless / batch use)")
    p.add_argument("--points-csv", help="output per-point CSV (default: <input>_points.csv)")
    p.add_argument("--no-points-csv", action="store_true",
                   help="do not write the per-point CSV")
    p.add_argument("--frames-csv", help="output frames CSV with frame/time "
                                        "alignment (default: <input>_frames.csv)")
    p.add_argument("--synthetic", metavar="OUT.mp4",
                   help="instead of processing, generate a synthetic test video of "
                        "radial expansion then contraction and exit")
    p.add_argument("--syn-dur", type=float, default=6.0, help="synthetic video duration (s)")
    p.add_argument("--syn-fps", type=int, default=30, help="synthetic video fps")
    p.add_argument("--syn-size", type=int, default=480, help="synthetic frame size (square)")

    g = p.add_argument_group("optical flow")
    g.add_argument("--max-corners", type=int, default=500)
    g.add_argument("--quality", type=float, default=0.01)
    g.add_argument("--min-dist", type=float, default=10.0)
    g.add_argument("--lk-win", type=int, default=15)
    g.add_argument("--max-level", type=int, default=2)
    g.add_argument("--no-back-sub", action="store_true",
                   help="disable bidirectional back-substitution verification")
    g.add_argument("--scale", type=float, default=1.0,
                   help="downscale processing resolution (speed-up, e.g. 0.5)")

    g2 = p.add_argument_group("metrics / state")
    g2.add_argument("--center", default="auto",
                    help="expansion center: 'auto' (FOE), 'frame', or 'x,y'")
    g2.add_argument("--thresh", type=float, default=0.1,
                    help="radial speed threshold (px/frame) for state classification")
    g2.add_argument("--descend-is-expansion", action="store_true",
                    help="invert convention: expansion means DESCEND, contraction means ASCEND")
    g2.add_argument("--smooth", type=int, default=15,
                    help="Savitzky-Golay window for velocity smoothing (odd)")
    g2.add_argument("--foe-alpha", type=float, default=0.3,
                    help="exponential smoothing factor for FOE across frames (0-1)")
    g2.add_argument("--state-median", type=int, default=5,
                    help="median window to de-flicker state labels (odd, 1 disables)")
    g2.add_argument("--trail", type=int, default=20, help="max trail length per point")

    g3 = p.add_argument_group("visualization")
    g3.add_argument("--seed", type=int, default=42, help="RNG seed for point colors")
    g3.add_argument("--no-hud", action="store_true", help="disable on-screen HUD text")
    g3.add_argument("--no-arrows", action="store_true", help="disable flow arrows")
    g3.add_argument("--draw-foe", action="store_true", help="draw FOE crosshair")

    return p.parse_args()


def estimate_foe(points, flow):
    """Least-squares Focus of Expansion from flow vectors.

    Each flow vector (u,v) at point (x,y) must be radial w.r.t. the FOE (xF,yF):
        u*(y - yF) - v*(x - xF) = 0  ->  v*xF - u*yF = v*x - u*y
    Weighted by 1/|v| so small, noisy vectors count equally.
    """
    u = flow[:, 0]
    v = flow[:, 1]
    x = points[:, 0]
    y = points[:, 1]
    mag = np.hypot(u, v)
    keep = mag > 1e-3
    u, v, x, y, mag = u[keep], v[keep], x[keep], y[keep], mag[keep]
    if len(u) < 5:
        return None
    w2 = (1.0 / mag) ** 2
    rhs = v * x - u * y
    A00 = float(np.sum(w2 * v * v))
    A01 = float(-np.sum(w2 * u * v))
    A11 = float(np.sum(w2 * u * u))
    b0 = float(np.sum(w2 * rhs * v))
    b1 = float(-np.sum(w2 * rhs * u))
    det = A00 * A11 - A01 * A01
    if abs(det) < 1e-9:
        return None
    xf = (b0 * A11 - A01 * b1) / det
    yf = (A00 * b1 - b0 * A01) / det
    return np.array([xf, yf], dtype=np.float32)


def radial_speeds(points, flow, center):
    d = points - center
    r = np.hypot(d[:, 0], d[:, 1])
    return (flow[:, 0] * d[:, 0] + flow[:, 1] * d[:, 1]) / (r + 1e-6)


def state_code(state):
    return {"ASCEND": 1, "STABLE": 0, "DESCEND": -1}.get(state, 0)


def new_color(rng):
    while True:
        c = rng.integers(0, 256, 3, dtype=int)
        if c.sum() > 120:
            return tuple(int(v) for v in c)


def draw_text_hud(frame, lines, origin=(12, 28), step=26):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (label, value, color) in enumerate(lines):
        pos = (origin[0], origin[1] + i * step)
        cv2.putText(frame, label, pos, font, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, pos, font, 0.6, color, 2, cv2.LINE_AA)
    # bump origin so future text stays below
    return (origin[0], origin[1] + len(lines) * step)


def classify(vr_mean, thresh, invert):
    if vr_mean > thresh:
        return DESC if invert else ASC
    if vr_mean < -thresh:
        return ASC if invert else DESC
    return STAB


def map_state(st, invert):
    if st == ASC:
        return DESC if invert else ASC
    if st == DESC:
        return ASC if invert else DESC
    return STAB


def generate_synthetic(args):
    fps = args.syn_fps
    n_frames = int(args.syn_dur * fps)
    size = args.syn_size
    cx = cy = size // 2
    rng = np.random.default_rng(7)

    n_points = 120
    angles = rng.uniform(0, 2 * np.pi, n_points)
    phase = rng.uniform(0, 2 * np.pi, n_points)

    # radius: expand over first half, contract over second half
    times = np.arange(n_frames) / fps
    half = args.syn_dur / 2.0
    r_smooth = np.where(times < half,
                        50.0 + (times / half) * 180.0,
                        230.0 - ((times - half) / half) * 180.0)
    radii = r_smooth[:, None] * (1.0 + 0.02 * np.sin(phase)[None, :])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.synthetic, fourcc, fps, (size, size))
    if not vw.isOpened():
        print("ERROR: could not open video writer for synthetic output")
        sys.exit(1)
    for f in range(n_frames):
        frame = np.zeros((size, size, 3), np.uint8)
        xs = cx + radii[f] * np.cos(angles)
        ys = cy + radii[f] * np.sin(angles)
        for (x, y) in zip(xs, ys):
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
        vw.write(frame)
    vw.release()
    print(f"wrote synthetic video: {args.synthetic}")
    sys.exit(0)


def main():
    args = parse_args()
    if args.synthetic:
        generate_synthetic(args)

    if not args.input or not os.path.isfile(args.input):
        print("ERROR: provide an input video file", file=sys.stderr)
        sys.exit(1)

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_video = args.out or f"{base}_flow.mp4"
    out_csv = args.csv or f"{base}_metrics.csv"
    out_plot = args.plot or f"{base}_metrics.png"
    out_points = None if args.no_points_csv else (args.points_csv or f"{base}_points.csv")
    out_frames = args.frames_csv or f"{base}_frames.csv"

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: cannot open video {args.input}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = args.scale
    sW, sH = max(1, int(W * scale)), max(1, int(H * scale))
    inv = 1.0 / scale if scale > 0 else 1.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video, fourcc, fps, (sW, sH))
    if not vw.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_video = os.path.splitext(out_video)[0] + ".avi"
        vw = cv2.VideoWriter(out_video, fourcc, fps, (sW, sH))
    if not vw.isOpened():
        print("ERROR: could not open any video writer", file=sys.stderr)
        sys.exit(1)

    # fixed center handling
    if args.center == "frame":
        fixed_center = np.array([sW / 2, sH / 2], np.float32)
    elif "," in args.center:
        fx, fy = (float(t) for t in args.center.split(","))
        fixed_center = np.array([fx * scale, fy * scale], np.float32)
    else:
        fixed_center = None

    rng = np.random.default_rng(args.seed)
    lk_params = dict(winSize=(args.lk_win, args.lk_win),
                     maxLevel=args.max_level,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    trail_len = max(1, args.trail)

    rows = []          # per-frame metric rows
    point_rows = [] if out_points else None   # per-point rows
    states = []        # raw state per frame
    times = []
    prev_gray = None
    p0 = None
    p_ids = None       # ids aligned with p0
    colors = {}        # id -> color
    history = {}       # id -> deque of points
    foe = None
    next_id = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (sW, sH), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t = frame_idx / fps

        mean_vr = 0.0
        med_vr = 0.0
        flow_mag = 0.0
        hist = np.zeros(8, dtype=float)
        foe_c = (fixed_center.copy() if fixed_center is not None else None)

        if prev_gray is not None and p0 is not None and len(p0) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
            keep = st.ravel() == 1
            if not args.no_back_sub:
                p1r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk_params)
                b = np.hypot(p1r[:, 0] - p0[:, 0], p1r[:, 1] - p0[:, 1])
                keep &= (st2.ravel() == 1) & (b < 1.0)
            good_prev = p0[keep]
            good_cur = p1[keep]
            good_ids = np.asarray(p_ids)[keep]

            # drop points that left the frame
            inb = (good_cur[:, 0] >= 0) & (good_cur[:, 0] < sW) & \
                  (good_cur[:, 1] >= 0) & (good_cur[:, 1] < sH)
            good_prev, good_cur, good_ids = good_prev[inb], good_cur[inb], good_ids[inb]

            if len(good_cur) > 0:
                flow = good_cur - good_prev

                # flow magnitude + direction distribution (8-bin histogram)
                mag = np.hypot(flow[:, 0], flow[:, 1])
                flow_mag = float(np.mean(mag)) if len(mag) else 0.0
                angles = np.degrees(np.arctan2(flow[:, 1], flow[:, 0])) % 360.0
                hist, _ = np.histogram(angles, bins=8, range=(0.0, 360.0))
                hist = hist / max(1.0, float(len(angles)))

                if fixed_center is not None:
                    foe_c = fixed_center.copy()
                else:
                    foe_raw = estimate_foe(good_cur, flow)
                    if foe_raw is not None:
                        foe = (foe_raw if foe is None
                               else args.foe_alpha * foe_raw + (1 - args.foe_alpha) * foe)
                    foe_c = foe

                if foe_c is not None:
                    vr = radial_speeds(good_cur, flow, foe_c)
                    mean_vr = float(np.mean(vr))
                    med_vr = float(np.median(vr))
                else:
                    mean_vr = med_vr = 0.0

                state = classify(mean_vr, args.thresh, args.descend_is_expansion)

                # update per-point history & draw
                for i, (pid, cur, prv) in enumerate(zip(good_ids, good_cur, good_prev)):
                    h = history.setdefault(int(pid), deque(maxlen=trail_len))
                    h.append((float(cur[0]), float(cur[1])))
                    col = colors.setdefault(int(pid), new_color(rng))
                    if len(h) >= 2:
                        pts = np.array(h, np.float32).reshape(-1, 1, 2)
                        cv2.polylines(frame, [pts.astype(np.int32)], False, col, 1, cv2.LINE_AA)
                    cv2.circle(frame, (int(cur[0]), int(cur[1])), 3, col, -1, cv2.LINE_AA)
                    if not args.no_arrows:
                        cv2.arrowedLine(frame, (int(prv[0]), int(prv[1])),
                                        (int(cur[0]), int(cur[1])), col, 1, cv2.LINE_AA, tipLength=0.3)
                    if point_rows is not None:
                        point_rows.append((
                            frame_idx, t, int(pid),
                            float(cur[0]) * inv, float(cur[1]) * inv,
                            float(flow[i, 0]) * inv, float(flow[i, 1]) * inv,
                            float(vr[i]) if foe_c is not None else np.nan,
                            float(state_code(state)) if foe_c is not None else 0,
                        ))

                # replace lost points
                p0 = good_cur.astype(np.float32)
                p_ids = list(good_ids)
                if len(p0) < args.max_corners:
                    mask = np.full((sH, sW), 255, np.uint8)
                    for (x, y) in p0:
                        cv2.circle(mask, (int(x), int(y)), int(args.min_dist), 0, -1)
                    newp = cv2.goodFeaturesToTrack(gray, args.max_corners - len(p0),
                                                   args.quality, args.min_dist, mask=mask,
                                                   blockSize=7, useHarrisDetector=False)
                    if newp is not None:
                        for (x, y) in newp.reshape(-1, 2):
                            p0 = np.vstack([p0, [x, y]])
                            pid = next_id
                            next_id += 1
                            p_ids.append(pid)
                            colors[pid] = new_color(rng)
                            history[pid] = deque(maxlen=trail_len)
                            history[pid].append((float(x), float(y)))
            else:
                p0 = None
                p_ids = None
                state = STAB
        else:
            p0 = cv2.goodFeaturesToTrack(gray, args.max_corners, args.quality,
                                         args.min_dist, blockSize=7)
            state = STAB
            if p0 is not None:
                p0 = p0.reshape(-1, 2)
                p_ids = []
                for (x, y) in p0:
                    pid = next_id
                    next_id += 1
                    p_ids.append(pid)
                    colors[pid] = new_color(rng)
                    history[pid] = deque(maxlen=trail_len)
                    history[pid].append((float(x), float(y)))

        # FOE crosshair
        if args.draw_foe and foe_c is not None:
            fx, fy = int(foe_c[0]), int(foe_c[1])
            cv2.circle(frame, (fx, fy), 8, (0, 215, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (fx - 14, fy), (fx + 14, fy), (0, 215, 255), 1)
            cv2.line(frame, (fx, fy - 14), (fx, fy + 14), (0, 215, 255), 1)

        states.append(state)
        times.append(t)
        rows.append((frame_idx, t, mean_vr, flow_mag,
                     hist[0], hist[1], hist[2], hist[3],
                     hist[4], hist[5], hist[6], hist[7],
                     foe_c[0] * inv if foe_c is not None else np.nan,
                     foe_c[1] * inv if foe_c is not None else np.nan))

        if not args.no_hud:
            col = STATE_COLORS[state]
            draw_text_hud(frame, [
                (f"frame {frame_idx}/{n_total}", "", (255, 255, 255)),
                (f"time {t:6.2f}s", "", (255, 255, 255)),
                (f"state: {state}", "", col),
                (f"mean vr {mean_vr:7.2f} px/f", "", (255, 255, 255)),
                (f"points {len(p0) if p0 is not None else 0}", "", (255, 255, 255)),
            ])

        vw.write(frame)
        frame_idx += 1
        prev_gray = gray

    cap.release()
    vw.release()

    if frame_idx == 0:
        print("ERROR: no frames read from video", file=sys.stderr)
        sys.exit(1)

    # ---- post-processing: velocity, acceleration, smoothed state ----
    arr = np.array(rows, dtype=float)
    frames = arr[:, 0].astype(int)
    ts = arr[:, 1]
    vr_px_f = arr[:, 2]
    vel = vr_px_f * fps * inv               # px/s in original resolution
    dt = np.diff(ts, prepend=ts[0])
    dt[dt <= 0] = 1.0 / fps

    win = args.smooth
    if win % 2 == 0:
        win += 1
    win = min(win, len(vel))
    if win >= 3 and len(vel) >= win:
        vel_s = savgol_filter(vel, win, 2)
        accel = np.gradient(vel_s, ts)
    else:
        vel_s = vel.copy()
        accel = np.gradient(vel_s, ts)

    # de-flicker states with median filter
    code = {ASC: 1, STAB: 0, DESC: -1}
    code_arr = np.array([code[s] for s in states], dtype=float)
    mwin = args.state_median
    if mwin > 1 and len(code_arr) >= 3:
        mwin = min(mwin, len(code_arr))
        if mwin % 2 == 0:
            mwin += 1
        code_arr = medfilt(code_arr, kernel_size=mwin)
    code_arr = np.round(code_arr).astype(int)
    inv_map = {1: ASC, 0: STAB, -1: DESC}
    labels = [map_state(inv_map[int(c)], args.descend_is_expansion) for c in code_arr]

    # ---- metrics CSV (model input: no frame) ----
    with open(out_csv, "w") as f:
        f.write("time_s,radial_expansion,flow_magnitude,"
                "flow_dir_hist_0,flow_dir_hist_1,flow_dir_hist_2,flow_dir_hist_3,"
                "flow_dir_hist_4,flow_dir_hist_5,flow_dir_hist_6,flow_dir_hist_7,"
                "foe_x,foe_y,expansion_rate,expansion_acceleration,state\n")
        for i in range(len(rows)):
            r = rows[i]
            f.write(f"{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},"
                    f"{r[4]:.4f},{r[5]:.4f},{r[6]:.4f},{r[7]:.4f},"
                    f"{r[8]:.4f},{r[9]:.4f},{r[10]:.4f},{r[11]:.4f},"
                    f"{r[12]:.2f},{r[13]:.2f},"
                    f"{vel_s[i]:.4f},{accel[i]:.4f},{labels[i]}\n")
    print(f"wrote metrics CSV: {out_csv}")

    # ---- frames CSV (frame alignment, kept out of the model input) ----
    with open(out_frames, "w") as f:
        f.write("frame,time_s\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f}\n")
    print(f"wrote frames CSV: {out_frames}")

    # ---- per-point CSV ----
    if point_rows:
        with open(out_points, "w") as f:
            f.write("frame,time_s,point_id,x,y,flow_u,flow_v,radial_speed,state_code\n")
            for r in point_rows:
                f.write(f"{r[0]},{r[1]:.4f},{r[2]},{r[3]:.2f},{r[4]:.2f},"
                        f"{r[5]:.2f},{r[6]:.2f},{r[7]:.4f},{r[8]}\n")
        print(f"wrote points CSV: {out_points}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        # matplotlib uses RGB in 0-1; STATE_COLORS are BGR in 0-255
        MPL = {s: tuple(int(c) / 255.0 for c in reversed(STATE_COLORS[s]))
               for s in STATE_COLORS}

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        # draw state bands on velocity axis
        prev_s = None
        start = ts[0]
        for i, s in enumerate(labels):
            if s != prev_s:
                if prev_s is not None and prev_s != STAB:
                    ax1.axvspan(start, ts[i], color=MPL[prev_s], alpha=0.25)
                if s != STAB:
                    start = ts[i]
                prev_s = s
        if prev_s is not None and prev_s != STAB:
            ax1.axvspan(start, ts[-1], color=MPL[prev_s], alpha=0.25)

        ax1.plot(ts, vel, color="0.55", lw=0.8, label="velocity raw")
        ax1.plot(ts, vel_s, color="#1f77b4", lw=1.8, label="velocity smoothed")
        ax1.axhline(0, color="k", lw=0.7)
        ax1.set_ylabel("radial velocity (px/s)")
        ax1.set_title(f"Rocket radial expansion metrics — {os.path.basename(args.input)}")
        ax1.legend(loc="best", fontsize=9)
        ax1.grid(alpha=0.3)

        ax2.plot(ts, accel, color="#d62728", lw=1.6, label="acceleration")
        ax2.axhline(0, color="k", lw=0.7)
        ax2.set_ylabel("acceleration (px/s^2)")
        ax2.set_xlabel("time (s)")
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(alpha=0.3)

        handles = [Patch(facecolor=MPL[ASC], alpha=0.5, label=f"ASCEND ({ASC})"),
                   Patch(facecolor=MPL[DESC], alpha=0.5, label=f"DESCEND ({DESC})"),
                   Patch(facecolor=MPL[STAB], alpha=0.5, label="STABLE")]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True,
                   bbox_to_anchor=(0.5, 0.995), fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_plot, dpi=130)
        plt.close(fig)
        print(f"wrote metrics plot: {out_plot}")
    except Exception as e:
        print(f"warning: could not render plot: {e}", file=sys.stderr)

    print(f"wrote annotated video: {out_video}  ({frame_idx} frames)")


if __name__ == "__main__":
    main()
