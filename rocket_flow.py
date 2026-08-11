#!/usr/bin/env python3
"""Rocket visual-motion feature extractor — thin CLI entry point.

The heavy lifting lives in the `scripts/` package. This module only parses
arguments, drives the frame loop, and writes the metrics CSV / plot / video.
"""
import argparse
import os
import sys
import types

import numpy as np
import cv2
from scipy.signal import savgol_filter

from scripts import io, schema, state, draw, synth, plot
from scripts.features import aggregate
from scripts.features import sparse as sparse_mod
from scripts.features import dense as dense_mod


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="input video file")
    p.add_argument("--out", help="output annotated video "
                                "(default: <dir of input>/csv output/<input>_flow.mp4)")
    p.add_argument("--csv", help="output metrics CSV "
                                "(default: <dir of input>/csv output/<input>_metrics.csv)")
    p.add_argument("--plot", help="output metrics plot "
                                "(default: <dir of input>/csv output/<input>_metrics.png)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the metrics plot (headless / batch use)")
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
    g2.add_argument("--apogee-window", type=int, default=3,
                    help="frames either side of the velocity crossing marked APOGEE")
    g2.add_argument("--idle-min-frames", type=int, default=5,
                    help="median window to detect idle STABLE pads at start/end (odd, 1 disables)")
    g2.add_argument("--trail", type=int, default=20, help="max trail length per point")

    g3 = p.add_argument_group("feature groups (all on by default)")
    g3.add_argument("--no-dense", action="store_true",
                    help="skip dense (Farneback) features: divergence, magnitude stats, grid")
    g3.add_argument("--no-homography", action="store_true",
                    help="skip homography features (keeps affine rotation/residual)")
    g3.add_argument("--no-appearance", action="store_true",
                    help="skip image appearance features (edges, texture, sharpness, sky/ground)")
    g3.add_argument("--no-horizon", action="store_true",
                    help="skip experimental horizon-line detection")

    g4 = p.add_argument_group("visualization")
    g4.add_argument("--seed", type=int, default=42, help="RNG seed for point colors")
    g4.add_argument("--no-hud", action="store_true", help="disable on-screen HUD text")
    g4.add_argument("--no-arrows", action="store_true", help="disable flow arrows")
    g4.add_argument("--draw-foe", action="store_true", help="draw FOE crosshair")

    return p.parse_args()


def write_csv(out_csv, rows, ts, vel_s, accel, labels, phases):
    with open(out_csv, "w") as f:
        f.write(",".join(schema.COLUMNS) + "\n")
        for i, row in enumerate(rows):
            row["state"] = labels[i]
            row["phase"] = phases[i]
            row["expansion_rate"] = float(vel_s[i])
            row["expansion_acceleration"] = float(accel[i])
            vals = []
            for c in schema.COLUMNS:
                v = row[c]
                if isinstance(v, float) and np.isnan(v):
                    vals.append("nan")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"wrote metrics CSV: {out_csv}")


def main():
    args = parse_args()
    if args.synthetic:
        synth.generate_synthetic(args)

    if not args.input or not os.path.isfile(args.input):
        print("ERROR: provide an input video file", file=sys.stderr)
        sys.exit(1)

    base, out_video, out_csv, out_plot = io.output_paths(args.input, args)
    cap = io.open_video(args.input)
    fps, W, H, n_total = io.video_props(cap)
    sW, sH = io.scale_size(W, H, args.scale)
    inv = 1.0 / args.scale if args.scale > 0 else 1.0

    vw, out_video = io.open_writer(out_video, fps, (sW, sH))

    # fixed center handling
    if args.center == "frame":
        fixed_center = np.array([sW / 2, sH / 2], np.float32)
    elif "," in args.center:
        fx, fy = (float(t) for t in args.center.split(","))
        fixed_center = np.array([fx * args.scale, fy * args.scale], np.float32)
    else:
        fixed_center = None

    tracker = sparse_mod.SparseFlow(args)
    rng = np.random.default_rng(args.seed)
    colors, history = {}, {}

    rows = []
    states = []
    times = []
    foe = None
    prev_gray = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.scale != 1.0:
            frame = cv2.resize(frame, (sW, sH), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t = frame_idx / fps

        prev = cur = flow = ids = None
        if prev_gray is not None:
            res = tracker.step(prev_gray, gray, sW, sH)
            if res is not None:
                prev, cur, flow, ids = res
        else:
            tracker.init_first(gray)

        # FOE (fixed or smoothed estimate)
        if fixed_center is not None:
            foe_c = fixed_center.copy()
        elif cur is not None and flow is not None and len(cur) >= 5:
            foe_raw = sparse_mod.estimate_foe(cur, flow)
            if foe_raw is not None:
                foe = (foe_raw if foe is None
                       else args.foe_alpha * foe_raw + (1 - args.foe_alpha) * foe)
            foe_c = foe
        else:
            foe_c = None

        dense = dense_mod.dense_flow(prev_gray, gray) if not args.no_dense else None

        # raw per-frame state from mean radial speed
        vr_mean = 0.0
        if cur is not None and flow is not None and len(cur) > 0 and foe_c is not None:
            vr_mean = float(np.mean(sparse_mod.radial_speeds(cur, flow, foe_c)))
        st_raw = state.classify(vr_mean, args.thresh, args.descend_is_expansion)

        draw.annotate(frame, cur, prev, ids, colors, history, args, rng, foe_c)
        draw.draw_hud(frame, frame_idx, n_total, t, st_raw, st_raw,
                      len(tracker.p0) if tracker.p0 is not None else 0, vr_mean, args.no_hud)

        ctx = types.SimpleNamespace(frame=frame, gray=gray, sW=sW, sH=sH, inv=inv,
                                    t=t, tracker=tracker, prev=prev, cur=cur,
                                    flow=flow, ids=ids, dense=dense)
        rows.append(aggregate(ctx, args, foe_c))
        states.append(st_raw)
        times.append(t)

        vw.write(frame)
        frame_idx += 1
        prev_gray = gray

    cap.release()
    vw.release()

    if frame_idx == 0:
        print("ERROR: no frames read from video", file=sys.stderr)
        sys.exit(1)

    # ---- post-processing: velocity, acceleration, de-flickered state, phase ----
    radial = np.array([r["radial_expansion"] for r in rows], dtype=float)
    radial = np.nan_to_num(radial, nan=0.0)   # no-tracking frames = no motion
    ts = np.array(times)
    dt = np.diff(ts, prepend=ts[0])
    dt[dt <= 0] = 1.0 / fps
    vel = radial * fps * inv                     # px/s in original resolution

    win = args.smooth
    if win % 2 == 0:
        win += 1
    win = min(win, len(vel))
    if win >= 3 and len(vel) >= win:
        vel_s = savgol_filter(vel, win, 2)
    else:
        vel_s = vel.copy()
    accel = np.gradient(vel_s, ts)

    clean_states = state.deflicker(states, args.state_median)
    labels = [state.map_state(s, args.descend_is_expansion) for s in clean_states]
    phases = state.flight_phases(vel_s, dt, args.apogee_window,
                                 args.thresh * fps, args.idle_min_frames,
                                 args.descend_is_expansion)

    write_csv(out_csv, rows, ts, vel_s, accel, labels, phases)

    if not args.no_plot:
        plot.render(ts, vel, vel_s, accel, labels, phases, out_plot,
                    os.path.basename(args.input))

    print(f"wrote annotated video: {out_video}  ({frame_idx} frames)")


if __name__ == "__main__":
    main()
