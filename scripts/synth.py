"""Synthetic test-video generator: radial expansion then contraction."""

import sys

import numpy as np
import cv2


def generate_synthetic(args):
    fps = args.syn_fps
    n_frames = int(args.syn_dur * fps)
    size = args.syn_size
    cx = cy = size // 2
    rng = np.random.default_rng(7)

    n_points = 120
    angles = rng.uniform(0, 2 * np.pi, n_points)
    phase = rng.uniform(0, 2 * np.pi, n_points)

    times = np.arange(n_frames) / fps
    half = args.syn_dur / 2.0
    r_smooth = np.where(times < half,
                        50.0 + (times / half) * 180.0,
                        230.0 - ((times - half) / half) * 180.0)
    radii = r_smooth[:, None] * (1.0 + 0.02 * np.sin(phase)[None, :])

    vw = cv2.VideoWriter(args.synthetic,
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
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
