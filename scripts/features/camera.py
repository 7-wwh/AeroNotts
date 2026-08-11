"""Camera ego-motion: affine rotation/translation, residual flow, homography."""

import numpy as np
import cv2


def affine_model(prev, cur):
    """Estimate global rotation/scale/translation between matched point sets.

    Returns a dict with cam_* features plus the 2x3 matrix M, or None if the
    fit fails or there are too few points.
    """
    if prev is None or len(prev) < 6:
        return None
    M, _ = cv2.estimateAffinePartial2D(prev, cur, method=cv2.RANSAC)
    if M is None:
        return None
    a, b = M[0, 0], M[0, 1]
    return {
        "M": M,
        "cam_rotation": float(np.degrees(np.arctan2(b, a))),
        "cam_scale": float(np.hypot(a, b)),
        "cam_tx": float(M[0, 2]),
        "cam_ty": float(M[1, 2]),
    }


def residual_flow(prev, cur, M, center):
    """Flow remaining after subtracting the rigid (rotation+translation) model.

    Returns a dict of residual statistics. divergence is estimated from the
    sparse residual field evaluated at the current point positions.
    """
    if M is None:
        return {}
    ones = np.ones((len(prev), 1))
    model_cur = np.hstack([prev, ones]) @ M.T
    res = cur - model_cur
    mag = np.hypot(res[:, 0], res[:, 1])
    out = {
        "residual_flow_mean": float(np.mean(mag)),
        "residual_flow_p95": float(np.percentile(mag, 95)),
    }
    out["residual_div"] = divergence_sparse(res, cur, center) if center is not None else np.nan
    return out


def divergence_sparse(residual, positions, center):
    """2 * mean((residual . d) / |d|^2): divergence of a radial-ish field.

    `d` is the point's position relative to the expansion center; `residual`
    is the flow vector at that point.
    """
    d = positions - center
    r2 = np.sum(d * d, axis=1) + 1e-6
    k = np.sum(residual * d, axis=1) / r2
    return float(2.0 * np.mean(k))


def homography_model(prev, cur, ransac_reproj=5.0):
    """Planar homography between matched point sets (RANSAC)."""
    if prev is None or len(prev) < 8:
        return None
    H, mask = cv2.findHomography(prev, cur, cv2.RANSAC, ransac_reproj)
    if H is None:
        return None
    scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
    rot = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
    return {
        "hom_scale": scale,
        "hom_rotation": rot,
        "hom_tx": float(H[0, 2]),
        "hom_ty": float(H[1, 2]),
        "hom_persp_x": float(H[2, 0]),
        "hom_persp_y": float(H[2, 1]),
        "hom_ok": float(np.mean(mask.ravel() == 1)),
    }
