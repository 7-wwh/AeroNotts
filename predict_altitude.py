#!/usr/bin/env python3
"""Predict rocket altitude on NEW footage using a trained XGBoost model.

Two ways to use it:

  1. From a video file (runs rocket_flow.py's feature extraction first):
       python3 predict_altitude.py new_video.mp4 --model launch_xgb.json

  2. From an existing features CSV (no re-analysis needed):
       python3 predict_altitude.py --features launch_metrics.csv --model launch_xgb.json

The feature list and order must match what train_model.py used (same defaults:
radial_expansion, flow_magnitude, flow_dir_hist_*, foe_x, foe_y, expansion_rate,
expansion_acceleration + cumulative/rolling/lag history features). If you
trained with --time-feature or --state-feature, the CSV needs those columns and
they are added automatically here when present.

Outputs:
    <name>_predictions.csv   frame, time, altitude prediction per frame
    <name>_predictions.png   altitude vs time plot
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import xgboost as xgb

from train_model import FEATURES


def frames_path_for(features_path, override=None):
    if override:
        return override
    name = os.path.basename(features_path)
    for suffix in ("_metrics.csv", ".csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return os.path.join(os.path.dirname(os.path.abspath(features_path)),
                        name + "_frames.csv")


def build_history_features(df, feat_cols):
    """Rebuild exactly the history features train_model.py adds."""
    fps = float(np.median(np.diff(df["time_s"]))) if len(df) > 1 else 30.0
    vel = df["expansion_rate"].values
    df = df.copy()
    df["vel_cumsum"] = np.cumsum(vel) / max(fps, 1e-9)
    df["abs_cumsum"] = np.cumsum(np.abs(vel)) / max(fps, 1e-9)
    for w in (3, 5, 10):
        df[f"vel_roll{w}"] = df["expansion_rate"].rolling(w, min_periods=1).mean()
        df[f"acc_roll{w}"] = df["expansion_acceleration"].rolling(w, min_periods=1).mean()
    for lag in (1, 2, 5, 10):
        df[f"vel_lag{lag}"] = df["expansion_rate"].shift(lag).fillna(0.0)
    feat_cols = list(feat_cols)
    feat_cols += ["vel_cumsum", "abs_cumsum",
                  "vel_roll3", "vel_roll5", "vel_roll10",
                  "acc_roll3", "acc_roll5", "acc_roll10",
                  "vel_lag1", "vel_lag2", "vel_lag5", "vel_lag10"]
    if "time_s" in df.columns and "time_s" not in feat_cols:
        feat_cols.append("time_s")  # model may have been trained with it
    if "state" in df.columns:
        df["state_cat"] = df["state"].map({"ASCEND": 2, "STABLE": 1, "DESCEND": 0})
        if "state_cat" not in feat_cols:
            feat_cols.append("state_cat")
    return df, feat_cols


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("video", nargs="?", help="input video to analyze")
    g.add_argument("--features", help="existing metrics CSV (skip video analysis)")
    p.add_argument("--model", required=True, help="trained model JSON from train_model.py")
    p.add_argument("--frames", help="frames CSV with frame/time alignment "
                                    "(default: <features>_frames.csv)")
    p.add_argument("--out", help="output name prefix (default: <video|features>_predictions)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.features:
        metrics_path = args.features
        frames_path = args.frames or frames_path_for(metrics_path)
        if not os.path.isfile(frames_path):
            sys.exit(f"ERROR: frames CSV not found: {frames_path}")
    else:
        if not os.path.isfile(args.video):
            sys.exit(f"ERROR: no such video: {args.video}")
        tmpdir = tempfile.mkdtemp(prefix="predict_")
        metrics_path = os.path.join(tmpdir, "m.csv")
        frames_path = os.path.join(tmpdir, "m_frames.csv")
        print(f"extracting features from {args.video} ...")
        r = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "rocket_flow.py"),
             args.video, "--csv", metrics_path, "--frames-csv", frames_path,
             "--no-points-csv", "--no-plot", "--out", os.path.join(tmpdir, "out.mp4")],
            capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"rocket_flow.py failed:\n{r.stderr}")

    df = pd.read_csv(metrics_path).sort_index().reset_index(drop=True)
    frm = pd.read_csv(frames_path).sort_values("frame").reset_index(drop=True)
    if len(frm) != len(df):
        sys.exit("ERROR: frames CSV row count does not match metrics CSV")
    df["frame"] = frm["frame"].values
    bst = xgb.Booster()
    bst.load_model(args.model)

    # the trained model records its exact feature list - use it
    model_feats = list(bst.feature_names) if bst.feature_names else None
    if model_feats is None:
        sys.exit("ERROR: model has no embedded feature list (retrain with the "
                 "current train_model.py)")
    df, _ = build_history_features(df, FEATURES)
    missing = [c for c in model_feats if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: model features missing from data: {missing}")
    X = df[model_feats].values.astype(np.float32)
    pred = bst.predict(xgb.DMatrix(X, feature_names=model_feats))

    def default_out():
        src = metrics_path if args.features else args.video
        name = os.path.basename(src)
        for suffix in ("_metrics.csv", ".mp4", ".csv"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name + "_predictions"

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(
        metrics_path if args.features else args.video)), default_out())
    out_csv = out + "_predictions.csv"
    pd.DataFrame({
        "frame": df["frame"],
        "time_s": df["time_s"],
        "altitude_m": np.round(pred, 2),
    }).to_csv(out_csv, index=False)
    print(f"wrote predictions: {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(df["time_s"], pred, color="#d62728", lw=1.6,
                label="predicted altitude")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("altitude (m)")
        ax.set_title(f"Predicted altitude - {os.path.basename(args.model)}")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out + "_predictions.png", dpi=130)
        plt.close(fig)
        print(f"wrote plot: {out}_predictions.png")
    except Exception as e:
        print(f"warning: could not render plot: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
