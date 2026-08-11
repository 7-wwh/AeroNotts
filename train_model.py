#!/usr/bin/env python3
"""Train an XGBoost model to predict rocket altitude from optical-flow features.

Inputs:
    features CSV  - produced by rocket_flow.py  (<video>_metrics.csv)
    labels CSV    - produced by label_altitude.py (<video>_labels.csv)

Workflow:
    1. python3 rocket_flow.py launch.mp4          # features
    2. python3 label_altitude.py launch_metrics.csv   # ground-truth labels
    3. python3 train_model.py launch_metrics.csv launch_labels.csv

Outputs:
    <name>_xgb.json           trained XGBoost model (load with xgb.Booster())
    <name>_train_test.csv     features + true + predicted altitude for every row
    <name>_prediction.png     true vs predicted altitude over time
    <name>_importance.png     XGBoost feature importance bar chart
    <name>_report.txt         RMSE / MAE / R2 metrics

IMPORTANT: frames are ordered in time, so this script splits chronologically
(first 80% train, last 20% test) instead of randomly, to avoid "leaking" the
answer across similar neighboring frames. Use --plot-test-split to see a plot
of the split on the altitude timeline.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# frame/time_s are intentionally excluded: they are ID columns used to align
# labels, and time lets the model memorize the time->altitude curve of ONE
# video instead of learning from the motion.
# Use --time-feature to add time_s back if you only ever analyse one launch.
FEATURES = [
    "radial_expansion",
    "flow_magnitude",
    "flow_dir_hist_0",
    "flow_dir_hist_1",
    "flow_dir_hist_2",
    "flow_dir_hist_3",
    "flow_dir_hist_4",
    "flow_dir_hist_5",
    "flow_dir_hist_6",
    "flow_dir_hist_7",
    "foe_x",
    "foe_y",
    "expansion_rate",
    "expansion_acceleration",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features_csv", help="metrics CSV from rocket_flow.py")
    p.add_argument("labels_csv", help="labels CSV from label_altitude.py")
    p.add_argument("--frames", help="frames CSV with frame/time alignment "
                                    "(default: <features>_frames.csv)")
    p.add_argument("--out", help="output name prefix (default: <features>_xgb)")
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="fraction of the LAST frames held out as test (default 0.2)")
    p.add_argument("--state-feature", action="store_true",
                   help="include the state column as a categorical feature")
    p.add_argument("--time-feature", action="store_true",
                   help="also include time_s as a feature (memorizes one video's curve)")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--early-stopping", type=int, default=30,
                   help="rounds without improvement before stopping")
    return p.parse_args()


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


def main():
    args = parse_args()

    feats = pd.read_csv(args.features_csv)
    labels = pd.read_csv(args.labels_csv)
    frames_path = frames_path_for(args.features_csv, args.frames)
    if not os.path.isfile(frames_path):
        sys.exit(f"ERROR: frames CSV not found: {frames_path}\n"
                 f"(run rocket_flow.py first - it writes the *_frames.csv file)")
    frm = pd.read_csv(frames_path)
    frm = frm.sort_values("frame").reset_index(drop=True)
    if len(frm) != len(feats):
        sys.exit(f"ERROR: frames CSV has {len(frm)} rows but metrics CSV has "
                 f"{len(feats)} rows - they must match (same video, same flags)")

    # frame is alignment metadata, NOT a model feature: attach it for merging
    # labels and chronological ordering, but never add it to feat_cols.
    feats = feats.copy()
    feats["frame"] = frm["frame"].values

    # keep only labeled frames, aligned by frame number
    df = feats.merge(labels, on="frame", how="inner")
    if len(df) == 0:
        sys.exit("ERROR: no overlapping frames between features and labels")
    df = df.dropna(subset=FEATURES + ["altitude"]).sort_values("frame").reset_index(drop=True)
    print(f"{len(df)} labeled frames (of {len(feats)} total)")
    if len(df) < 50:
        print("WARNING: only %d labeled frames - a bigger labelled set gives a "
              "much more reliable model" % len(df))

    # Position is the integral of velocity, so a model needs motion HISTORY,
    # not just this frame's value. Build cumulative + rolling + lag features.
    fps = float(np.median(np.diff(df["time_s"]))) if len(df) > 1 else 30.0
    vel = df["expansion_rate"].values
    df["vel_cumsum"] = np.cumsum(vel) / max(fps, 1e-9)      # running integral (px)
    df["abs_cumsum"] = np.cumsum(np.abs(vel)) / max(fps, 1e-9)
    for w in (3, 5, 10):
        df[f"vel_roll{w}"] = df["expansion_rate"].rolling(w, min_periods=1).mean()
        df[f"acc_roll{w}"] = df["expansion_acceleration"].rolling(w, min_periods=1).mean()
    for lag in (1, 2, 5, 10):
        df[f"vel_lag{lag}"] = df["expansion_rate"].shift(lag).fillna(0.0)
    # features will already be all finite after fillna above

    feat_cols = list(FEATURES)
    feat_cols += ["vel_cumsum", "abs_cumsum",
                  "vel_roll3", "vel_roll5", "vel_roll10",
                  "acc_roll3", "acc_roll5", "acc_roll10",
                  "vel_lag1", "vel_lag2", "vel_lag5", "vel_lag10"]
    if args.time_feature:
        feat_cols.append("time_s")
    if args.state_feature and "state" in df.columns:
        df["state_cat"] = df["state"].map({"ASCEND": 2, "STABLE": 1, "DESCEND": 0})
        feat_cols.append("state_cat")

    X = df[feat_cols].values.astype(np.float32)
    y = df["altitude"].values.astype(np.float32)
    frames = df["frame"].values

    # chronological split: train on first (1-test_frac), test on last
    n = len(df)
    n_test = int(n * args.test_frac)
    if n_test < 1:
        n_test = 1
    if n - n_test < 20:
        sys.exit("ERROR: too few labeled frames to leave a test set")
    X_tr, X_te = X[: n - n_test], X[n - n_test:]
    y_tr, y_te = y[: n - n_test], y[n - n_test:]

    print(f"train: {len(X_tr)} frames | test: {len(X_te)} frames "
          f"(frames {frames[n-n_test]}-{frames[-1]})")

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te, label=y_te)
    params = {
        "objective": "reg:squarederror",
        "max_depth": args.max_depth,
        "eta": args.lr,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    evals = [(dtrain, "train"), (dtest, "test")]
    bst = xgb.train(params, dtrain, num_boost_round=args.n_estimators,
                    evals=evals, early_stopping_rounds=args.early_stopping,
                    verbose_eval=False)
    best_iter = bst.best_iteration if hasattr(bst, "best_iteration") else args.n_estimators

    pred_tr = bst.predict(dtrain)
    pred_te = bst.predict(dtest)

    rmse = float(np.sqrt(mean_squared_error(y_te, pred_te)))
    mae = float(mean_absolute_error(y_te, pred_te))
    r2 = float(r2_score(y_te, pred_te))
    print(f"\nTEST  RMSE={rmse:.2f} m   MAE={mae:.2f} m   R2={r2:.3f}   "
          f"(best_iter={best_iter})")

    def default_out(features_path):
        name = os.path.basename(features_path)
        for suffix in ("_metrics.csv", ".csv"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name + "_xgb"

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.features_csv)),
                                   default_out(args.features_csv))
    bst.feature_names = feat_cols          # embed feature list so prediction matches
    bst.save_model(out + ".json")
    print(f"saved model: {out}.json")

    # full prediction CSV (features + true + predicted on every labeled frame)
    full = df.copy()
    full["predicted_altitude"] = bst.predict(xgb.DMatrix(X))
    full[["frame", "altitude", "predicted_altitude"] + feat_cols].to_csv(
        out + "_train_test.csv", index=False)
    print(f"saved predictions: {out}_train_test.csv")

    # report
    with open(out + "_report.txt", "w") as f:
        f.write(f"rows labeled        : {n}\n")
        f.write(f"train frames        : {len(X_tr)} (frames 0-{frames[n-n_test-1]})\n")
        f.write(f"test frames         : {len(X_te)} (frames {frames[n-n_test]}-{frames[-1]})\n")
        f.write(f"test RMSE (m)       : {rmse:.3f}\n")
        f.write(f"test MAE (m)        : {mae:.3f}\n")
        f.write(f"test R2             : {r2:.3f}\n")
        f.write(f"best boosting iter  : {best_iter}\n")
        f.write(f"features            : {feat_cols}\n")
    print(f"saved report: {out}_report.txt")

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(df["time_s"], df["altitude"], "o-", ms=3, lw=1,
                color="#1f77b4", label="true altitude")
        ax.plot(df["time_s"], full["predicted_altitude"], ".-", ms=3,
                color="#d62728", label="predicted (all frames)")
        ax.axvline(df["time_s"].iloc[n - n_test], color="k", ls="--",
                   label="test split")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("altitude (m)")
        ax.set_title(f"True vs predicted altitude - {os.path.basename(args.features_csv)}")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out + "_prediction.png", dpi=130)
        plt.close(fig)
        print(f"saved plot: {out}_prediction.png")

        importances = bst.get_score(importance_type="gain")
        if importances:
            names = [k for k, _ in sorted(importances.items(), key=lambda x: -x[1])]
            vals = [v for _, v in sorted(importances.items(), key=lambda x: -x[1])]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(names, vals)
            ax.invert_yaxis()
            ax.set_title("XGBoost feature importance (gain)")
            fig.tight_layout()
            fig.savefig(out + "_importance.png", dpi=130)
            plt.close(fig)
            print(f"saved plot: {out}_importance.png")
    except Exception as e:
        print(f"warning: could not render plots: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
