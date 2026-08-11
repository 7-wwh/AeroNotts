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


def detect_apogee(vel_s, apogee_window):
    """Frames where smoothed velocity crosses from positive to negative.

    Apogee is the peak of the flight: expansion stops and contraction begins.
    Mark the crossing frame plus `apogee_window` frames on either side.
    """
    n = len(vel_s)
    apogee = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if vel_s[i - 1] > 0 and vel_s[i] <= 0:
            lo = max(0, i - apogee_window)
            hi = min(n, i + apogee_window + 1)
            apogee[lo:hi] = True
    return apogee


def assign_phases(states, vel_s, apogee_window):
    """Add the APOGEE phase on top of the ASCEND/DESCEND/STABLE states."""
    apogee = detect_apogee(vel_s, apogee_window)
    return [APO if a else s for a, s in zip(apogee, states)]
