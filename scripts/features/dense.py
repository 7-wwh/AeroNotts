"""Dense optical flow (Farneback): divergence, magnitude stats, 3x3 grid."""

import numpy as np
import cv2

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def dense_flow(prev_gray, gray):
    """Farneback dense flow between two grayscale frames, or None if no prev."""
    if prev_gray is None:
        return None
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK_PARAMS)


def magnitude(flow):
    return np.hypot(flow[..., 0], flow[..., 1])


def magnitude_stats(flow):
    m = magnitude(flow)
    return {
        "flow_median": float(np.median(m)),
        "flow_p95": float(np.percentile(m, 95)),
        "flow_std": float(np.std(m)),
        "flow_max": float(np.max(m)),
    }


def divergence(flow):
    """div(F) = du/dx + dv/dy via finite differences."""
    u = flow[..., 0]
    v = flow[..., 1]
    du, _ = np.gradient(u)
    _, dv = np.gradient(v)
    return du + dv


def divergence_stats(flow):
    d = divergence(flow)
    return {
        "div_mean": float(np.mean(d)),
        "div_median": float(np.median(d)),
        "div_std": float(np.std(d)),
        "div_p95": float(np.percentile(d, 95)),
        "div_pos_frac": float(np.mean(d > 0)),
        "div_max": float(np.max(d)),
    }


def grid_flow(flow, cells=3):
    """Mean flow magnitude per image cell (rows x cols)."""
    m = magnitude(flow)
    h, w = m.shape
    out = {}
    for i in range(cells):
        for j in range(cells):
            cell = m[i * h // cells:(i + 1) * h // cells,
                     j * w // cells:(j + 1) * w // cells]
            out[f"grid_flow_{i}{j}"] = float(np.mean(cell))
    return out
