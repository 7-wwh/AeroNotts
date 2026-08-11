"""Per-frame image appearance: edges, texture, sharpness, sky/ground split."""

import numpy as np
import cv2


def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return float(np.mean(edges > 0))


def texture_var(gray):
    return float(gray.var())


def grad_magnitude_mean(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(gx, gy)))


def sharpness(gray):
    """Laplacian variance: standard focus/sharpness measure."""
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


def sky_ground_fractions(frame):
    """Crude heuristic: sky = bright & low-saturation, ground = mid-dark & textured."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    sky = (val > 0.55) & (sat < 0.35)
    ground = (val <= 0.6) & (val > 0.15) & (sat >= 0.05)
    return {
        "sky_fraction": float(np.mean(sky)),
        "ground_fraction": float(np.mean(ground)),
    }
