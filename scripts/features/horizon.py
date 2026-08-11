"""Experimental horizon-line detection via Hough on downscaled edges."""

import numpy as np
import cv2

_scale = 0.5


def horizon_features(gray):
    """Detect the strongest long straight line and report angle/position/conf.

    Angles are degrees from horizontal; position is the normalized center y of
    the line (0=top, 1=bottom); confidence is line length / image diagonal.
    Returns NaN values when no line is found.
    """
    h, w = gray.shape
    sm = cv2.resize(gray, (max(1, int(w * _scale)), max(1, int(h * _scale))),
                    interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(sm, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=0.15 * max(sm.shape), maxLineGap=20)
    if lines is None:
        return {"horizon_angle": np.nan, "horizon_pos": np.nan, "horizon_conf": 0.0}

    best = None
    best_len = 0.0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = np.hypot(x2 - x1, y2 - y1)
        if length > best_len:
            best_len = length
            best = (x1, y1, x2, y2)
    if best is None:
        return {"horizon_angle": np.nan, "horizon_pos": np.nan, "horizon_conf": 0.0}

    x1, y1, x2, y2 = best
    angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    mid_y = float((y1 + y2) / 2.0 / sm.shape[0])
    conf = float(best_len / np.hypot(sm.shape[0], sm.shape[1]))
    return {"horizon_angle": angle, "horizon_pos": mid_y, "horizon_conf": conf}
