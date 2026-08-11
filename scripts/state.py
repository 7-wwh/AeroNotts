"""Flight-state classification: direction labels + APOGEE phase detection."""

import numpy as np
from scipy.signal import medfilt

DESC = "DESCEND"
STAB = "STABLE"
ASC = "ASCEND"
APO = "APOGEE"

# BGR (OpenCV) colors
STATE_COLORS = {
    ASC: (60, 220, 60),
    STAB: (200, 200, 200),
    DESC: (60, 60, 240),
    APO: (0, 180, 255),
}

_CODE = {ASC: 1, STAB: 0, DESC: -1}
_INV_CODE = {1: ASC, 0: STAB, -1: DESC}


def classify(vr_mean, thresh, invert):
    """Map a frame's mean radial speed to ASCEND/DESCEND/STABLE."""
    if vr_mean > thresh:
        return DESC if invert else ASC
    if vr_mean < -thresh:
        return ASC if invert else DESC
    return STAB


def map_state(st, invert):
    """Flip ASC<->DESC if --descend-is-expansion was used."""
    if st == ASC:
        return DESC if invert else ASC
    if st == DESC:
        return ASC if invert else DESC
    return STAB


def deflicker(states, window):
    """Median-filter the state sequence so a single noisy frame can't flip labels."""
    arr = np.array([_CODE[s] for s in states], dtype=float)
    if window > 1 and len(arr) >= 3:
        w = min(window, len(arr))
        if w % 2 == 0:
            w += 1
        arr = medfilt(arr, kernel_size=w)
    arr = np.round(arr).astype(int)
    return [_INV_CODE[int(c)] for c in arr]


def flight_phases(vel_s, dt, apogee_window, thresh_px_s, idle_min_frames, invert):
    """Global flight-trajectory phases: STABLE pads + one ASCEND / APOGEE / DESCEND.

    A real rocket launch has exactly one ascent, one apogee, one descent, with
    idle STABLE pads before liftoff and after landing. Instead of marking every
    velocity zero-crossing (noisy), we find the single peak of an altitude proxy
    (the cumulative integral of radial velocity) and split the flight around it.

    Returns a list of phase labels ('STABLE'/'ASCEND'/'APOGEE'/'DESCEND').
    """
    n = len(vel_s)
    if n == 0:
        return []

    ascent_sign = -1.0 if invert else 1.0
    alt = np.cumsum(vel_s * ascent_sign * dt)
    apogee_idx = int(np.argmax(alt))

    # sustained-motion mask -> liftoff / landing boundaries for the STABLE pads
    moving = np.abs(vel_s) >= thresh_px_s
    if idle_min_frames > 1 and n >= 3:
        w = min(idle_min_frames, n)
        if w % 2 == 0:
            w += 1
        moving = medfilt(moving.astype(float), kernel_size=w) > 0.5
    idx = np.where(moving)[0]
    pre = idx[idx < apogee_idx]
    post = idx[idx > apogee_idx]
    liftoff = int(pre[0]) if len(pre) else 0
    landing = int(post[-1]) if len(post) else n - 1

    lo = max(0, apogee_idx - apogee_window)
    hi = min(n, apogee_idx + apogee_window + 1)

    phases = []
    for i in range(n):
        if i < liftoff:
            phases.append(STAB)
        elif i < lo:
            phases.append(ASC)
        elif i < hi:
            phases.append(APO)
        elif i <= landing:
            phases.append(DESC)
        else:
            phases.append(STAB)
    return phases
