"""Sparse optical flow: LK tracking, FOE, radial speed, displacement, density.

Owns the tracked point set (Shi-Tomasi corners followed by Lucas-Kanade) and
derives per-frame radial / displacement / density features from it.
"""

import numpy as np
import cv2


def estimate_foe(points, flow):
    """Least-squares Focus of Expansion from flow vectors.

    Each flow vector (u,v) at point (x,y) must be radial w.r.t. the FOE (xF,yF):
        u*(y - yF) - v*(x - xF) = 0  ->  v*xF - u*yF = v*x - u*y
    Weighted by 1/|v| so small, noisy vectors count equally.
    """
    u = flow[:, 0]
    v = flow[:, 1]
    x = points[:, 0]
    y = points[:, 1]
    mag = np.hypot(u, v)
    keep = mag > 1e-3
    u, v, x, y, mag = u[keep], v[keep], x[keep], y[keep], mag[keep]
    if len(u) < 5:
        return None
    w2 = (1.0 / mag) ** 2
    rhs = v * x - u * y
    A00 = float(np.sum(w2 * v * v))
    A01 = float(-np.sum(w2 * u * v))
    A11 = float(np.sum(w2 * u * u))
    b0 = float(np.sum(w2 * rhs * v))
    b1 = float(-np.sum(w2 * rhs * u))
    det = A00 * A11 - A01 * A01
    if abs(det) < 1e-9:
        return None
    xf = (b0 * A11 - A01 * b1) / det
    yf = (A00 * b1 - b0 * A01) / det
    return np.array([xf, yf], dtype=np.float32)


def radial_speeds(points, flow, center):
    d = points - center
    r = np.hypot(d[:, 0], d[:, 1])
    return (flow[:, 0] * d[:, 0] + flow[:, 1] * d[:, 1]) / (r + 1e-6)


class SparseFlow:
    """Owns the tracked point set (positions + ids) and steps it frame to frame."""

    def __init__(self, args):
        self.max_corners = args.max_corners
        self.quality = args.quality
        self.min_dist = args.min_dist
        self.lk_params = dict(
            winSize=(args.lk_win, args.lk_win),
            maxLevel=args.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.no_back_sub = args.no_back_sub
        self.p0 = None        # current points (Nx2 float32)
        self.p_ids = None     # ids aligned with p0
        self.next_id = 0
        self.prev_points = None  # matched points from previous frame

    def init_first(self, gray):
        """Detect the initial point set on the first frame."""
        p0 = cv2.goodFeaturesToTrack(gray, self.max_corners, self.quality,
                                     self.min_dist, blockSize=7)
        if p0 is None:
            p0 = np.zeros((0, 2), np.float32)
        else:
            p0 = p0.reshape(-1, 2)
        self.p0 = p0.astype(np.float32)
        self.p_ids = list(range(self.next_id, self.next_id + len(p0)))
        self.next_id += len(p0)
        self.prev_points = None

    def step(self, prev_gray, gray, sW, sH):
        """Track current points into `gray`.

        Returns (prev, cur, flow, ids) aligned by row, or None if nothing tracked.
        Lost points are replenished afterwards.
        """
        if self.p0 is None or len(self.p0) == 0:
            self.replenish(gray, sW, sH)
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.p0, None, **self.lk_params)
        keep = st.ravel() == 1
        if not self.no_back_sub:
            p1r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **self.lk_params)
            back = np.hypot(p1r[:, 0] - self.p0[:, 0], p1r[:, 1] - self.p0[:, 1])
            keep &= (st2.ravel() == 1) & (back < 1.0)
        prev = self.p0[keep]
        cur = p1[keep]
        ids = np.asarray(self.p_ids)[keep]
        inb = (cur[:, 0] >= 0) & (cur[:, 0] < sW) & (cur[:, 1] >= 0) & (cur[:, 1] < sH)
        prev, cur, ids = prev[inb], cur[inb], ids[inb]
        if len(cur) == 0:
            self.p0 = None
            self.p_ids = None
            self.prev_points = None
            self.replenish(gray, sW, sH)
            return None
        flow = cur - prev
        self.prev_points = prev
        self.p0 = cur.astype(np.float32)
        self.p_ids = list(ids)
        self.replenish(gray, sW, sH)
        return prev, cur, flow, ids

    def replenish(self, gray, sW, sH):
        """Add new corners where the scene is under-tracked."""
        if self.p0 is None:
            self.p0 = np.zeros((0, 2), np.float32)
            self.p_ids = []
        if len(self.p0) >= self.max_corners:
            return
        mask = np.full((sH, sW), 255, np.uint8)
        for (x, y) in self.p0:
            cv2.circle(mask, (int(x), int(y)), int(self.min_dist), 0, -1)
        newp = cv2.goodFeaturesToTrack(gray, self.max_corners - len(self.p0),
                                       self.quality, self.min_dist, mask=mask,
                                       blockSize=7, useHarrisDetector=False)
        if newp is not None:
            newp = newp.reshape(-1, 2)
            self.p0 = np.vstack([self.p0, newp]).astype(np.float32)
            for _ in newp:
                self.p_ids.append(self.next_id)
                self.next_id += 1


def sparse_features(tracker, cur, flow, center, sW, sH):
    """All sparse-flow columns for one frame. `center` may be None."""
    n = len(tracker.p0) if tracker.p0 is not None else 0
    d = {
        "point_count": float(n),
        "point_density": float(n / (sW * sH)) if sW * sH else np.nan,
    }
    if cur is None or flow is None or len(flow) == 0:
        return d

    mag = np.hypot(flow[:, 0], flow[:, 1])
    d.update({
        "pt_displacement_mean": float(np.mean(mag)),
        "pt_displacement_median": float(np.median(mag)),
        "pt_displacement_std": float(np.std(mag)),
        "pt_displacement_max": float(np.max(mag)),
        "flow_magnitude": float(np.mean(mag)),
    })
    angles = np.degrees(np.arctan2(flow[:, 1], flow[:, 0])) % 360.0
    hist, _ = np.histogram(angles, bins=8, range=(0.0, 360.0))
    hist = hist / max(1.0, float(len(angles)))
    for i in range(8):
        d[f"flow_dir_hist_{i}"] = float(hist[i])

    if center is not None:
        vr = radial_speeds(cur, flow, center)
        d.update({
            "radial_expansion": float(np.mean(vr)),
            "radial_expansion_median": float(np.median(vr)),
            "radial_std": float(np.std(vr)),
            "radial_p95": float(np.percentile(vr, 95)),
            "outward_frac": float(np.mean(vr > 0)),
            "inward_frac": float(np.mean(vr < 0)),
        })
        dd = cur - center
        rr = np.hypot(dd[:, 0], dd[:, 1])
        d["feature_radius_mean"] = float(np.mean(rr))
        d["feature_radius_std"] = float(np.std(rr))
    return d
