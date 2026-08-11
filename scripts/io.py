"""Video / file I/O helpers."""

import os
import sys

import cv2


def output_paths(input_path, args):
    """Resolve output paths: <dir of input>/csv output/<base>_{flow,metrics,png}."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), "csv output")
    os.makedirs(out_dir, exist_ok=True)
    out_video = args.out or os.path.join(out_dir, f"{base}_flow.mp4")
    out_csv = args.csv or os.path.join(out_dir, f"{base}_metrics.csv")
    out_plot = args.plot or os.path.join(out_dir, f"{base}_metrics.png")
    return base, out_video, out_csv, out_plot


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: cannot open video {path}", file=sys.stderr)
        sys.exit(1)
    return cap


def video_props(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, W, H, n


def open_writer(path, fps, size):
    """Open a video writer, falling back from mp4v to MJPG/avi if needed."""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not vw.isOpened():
        path = os.path.splitext(path)[0] + ".avi"
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not vw.isOpened():
        print("ERROR: could not open any video writer", file=sys.stderr)
        sys.exit(1)
    return vw, path


def scale_size(W, H, scale):
    return max(1, int(W * scale)), max(1, int(H * scale))
