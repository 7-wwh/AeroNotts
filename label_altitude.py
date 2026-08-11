#!/usr/bin/env python3
"""Interactive altitude labeling tool for rocket_flow.py features.

Reads the per-frame metrics CSV produced by rocket_flow.py and lets you type
in the rocket's altitude above the ground (your known ground-truth) for each
frame or for ranges of frames. Output is a CSV the training script can use.

The point of this tool: you know the actual altitude at some times (e.g. from a
barometer, a known rail height, telemetry, or visual estimate). Those known
values become the training labels for the XGBoost model.

Workflow:
    1. python3 rocket_flow.py launch.mp4           # produces launch_metrics.csv
    2. python3 label_altitude.py launch_metrics.csv
    3. python3 train_model.py launch_metrics.csv launch_labels.csv

Commands while labeling (type a line, press Enter):
    <number>              set current frame altitude, advance one frame
    <number> <number>     start-frame end-frame : set altitude for a RANGE
    <empty>               advance one frame keeping the last altitude
    .                     advance one frame WITHOUT a label (leave blank)
    s N                   set all frames to altitude N
    r N                   rewind to first frame, reset, relabel everything as N
    back                  go back one frame
    jump N                jump to frame N
    info                  print current frame info
    save                  write labels and quit
    quit                  quit without saving
"""
import argparse
import csv
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics_csv", help="metrics CSV from rocket_flow.py")
    p.add_argument("--frames", help="frames CSV with frame/time alignment "
                                    "(default: <metrics>_frames.csv)")
    p.add_argument("--out", help="labels CSV output (default: <metrics>_labels.csv)")
    p.add_argument("--resume", help="existing labels CSV to keep and extend")
    return p.parse_args()


def frames_path_for(metrics_path, override=None):
    if override:
        return override
    name = os.path.basename(metrics_path)
    for suffix in ("_metrics.csv", ".csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return os.path.join(os.path.dirname(os.path.abspath(metrics_path)),
                        name + "_frames.csv")


def load_frames(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["frame"]), float(r["time_s"])))
    if not rows:
        sys.exit(f"ERROR: no frames in {path}")
    return rows


def load_labels(path):
    labels = {}
    if path and os.path.isfile(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                labels[int(r["frame"])] = float(r["altitude"])
    return labels


def save_labels(labels, path, frames):
    if not frames:
        return
    with open(path, "w") as f:
        f.write("frame,altitude\n")
        for fi, _t in frames:
            if fi in labels:
                f.write(f"{fi},{labels[fi]:.2f}\n")
    print(f"saved {len(labels)} labels to {path}")


def default_out(metrics_path):
    name = os.path.basename(metrics_path)
    for suffix in ("_metrics.csv", ".csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name + "_labels.csv"


def main():
    args = parse_args()
    frames_path = frames_path_for(args.metrics_csv, args.frames)
    if not os.path.isfile(frames_path):
        sys.exit(f"ERROR: frames CSV not found: {frames_path}\n"
                 f"(run rocket_flow.py first - it writes the *_frames.csv file)")
    frames = load_frames(frames_path)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.metrics_csv)),
                                   default_out(args.metrics_csv))
    labels = load_labels(args.resume)
    if args.resume:
        print(f"resumed {len(labels)} existing labels")

    idx = 0
    n = len(frames)
    last_alt = None
    last_was_labeled = False

    print(f"{n} frames loaded from {frames_path}")
    print("type a number to label the current frame; Enter alone copies the "
          "previous label; '.' skips without labeling.")
    print("commands: 's <alt>' set all | 'a b alt' range | 'back' | 'jump N' | 'save' | 'quit'")

    while True:
        if idx >= n:
            print(f"\nreached end of {n} frames")
            save_labels(labels, out, frames)
            return
        fi, t = frames[idx]
        cur = labels.get(fi)
        cur_s = f"{cur:.2f}" if cur is not None else "-"
        try:
            inp = input(f"[{idx:5d}/{n}] frame {fi:5d} t={t:7.2f}s  "
                        f"alt={cur_s:>9s} -> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        cmd = inp.strip()
        low = cmd.lower()

        if low in ("", " ", "y"):
            if last_alt is not None and last_was_labeled:
                labels[fi] = last_alt
            idx += 1
            continue
        if low == ".":
            idx += 1
            last_was_labeled = False
            continue
        if low == "save":
            save_labels(labels, out, frames)
            return
        if low == "quit":
            print("quit without saving")
            return
        if low == "info":
            print(f"frame {fi} t={t:.2f}s labelled={cur_s} total_labels={len(labels)}")
            continue
        if low == "back":
            idx = max(0, idx - 1)
            continue
        if low.startswith("jump"):
            try:
                idx = min(n - 1, max(0, int(cmd.split()[1])))
            except (ValueError, IndexError):
                print("  usage: jump N")
            continue
        if low.startswith("s "):
            try:
                v = float(cmd.split()[1])
                for fi2, _t in frames:
                    labels[fi2] = v
                last_alt = v
                print(f"  set ALL {n} frames to {v:.2f}")
            except (ValueError, IndexError):
                print("  usage: s <alt>")
            continue
        if low == "r":
            labels.clear()
            idx = 0
            last_alt = None
            print("  reset")
            continue
        if low.startswith("r "):
            try:
                v = float(cmd.split()[1])
                for fi2, _t in frames:
                    labels[fi2] = v
                idx = 0
                last_alt = v
                print(f"  reset and set all frames to {v:.2f}")
            except (ValueError, IndexError):
                print("  usage: r <alt>")
            continue

        # range: "A B alt"
        parts = cmd.split()
        if len(parts) == 3:
            try:
                a, b, v = int(parts[0]), int(parts[1]), float(parts[2])
                a, b = min(a, b), max(a, b)
                count = 0
                for fi2, _t in frames:
                    if a <= fi2 <= b:
                        labels[fi2] = v
                        count += 1
                print(f"  set {count} frames [{a}..{b}] to {v:.2f}")
                idx = min(n - 1, b + 1)
                last_alt = v
                continue
            except ValueError:
                pass

        # single number
        try:
            v = float(cmd)
            labels[fi] = v
            last_alt = v
            last_was_labeled = True
            idx += 1
            continue
        except ValueError:
            pass

        print("  ? unknown command or value")

    save_labels(labels, out, frames)


if __name__ == "__main__":
    main()
