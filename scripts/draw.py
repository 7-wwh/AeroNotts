"""Drawing: random point colors, trails, arrows, FOE crosshair, HUD text."""

import cv2
import numpy as np
from collections import deque

from . import state


def new_color(rng):
    while True:
        c = rng.integers(0, 256, 3, dtype=int)
        if c.sum() > 120:
            return tuple(int(v) for v in c)


def draw_text_hud(frame, lines, origin=(12, 28), step=26):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (label, value, color) in enumerate(lines):
        pos = (origin[0], origin[1] + i * step)
        cv2.putText(frame, label, pos, font, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, pos, font, 0.6, color, 2, cv2.LINE_AA)


def draw_foe(frame, foe_c):
    fx, fy = int(foe_c[0]), int(foe_c[1])
    cv2.circle(frame, (fx, fy), 8, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (fx - 14, fy), (fx + 14, fy), (0, 215, 255), 1)
    cv2.line(frame, (fx, fy - 14), (fx, fy + 14), (0, 215, 255), 1)


def annotate(frame, cur, prev, ids, colors, history, args, rng, foe_c):
    """Draw tracked points, trails, arrows and the optional FOE crosshair."""
    if args.draw_foe and foe_c is not None:
        draw_foe(frame, foe_c)
    if cur is None:
        return
    trail_len = max(1, args.trail)
    for i, (pid, c, p) in enumerate(zip(ids, cur, prev)):
        h = history.setdefault(int(pid), deque(maxlen=trail_len))
        h.append((float(c[0]), float(c[1])))
        col = colors.setdefault(int(pid), new_color(rng))
        if len(h) >= 2:
            pts = np.array(h, np.float32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts.astype(np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.circle(frame, (int(c[0]), int(c[1])), 3, col, -1, cv2.LINE_AA)
        if not args.no_arrows:
            cv2.arrowedLine(frame, (int(p[0]), int(p[1])),
                            (int(c[0]), int(c[1])), col, 1, cv2.LINE_AA, tipLength=0.3)


def draw_hud(frame, frame_idx, n_total, t, st, ph, n_points, vr_mean, no_hud):
    if no_hud:
        return
    col = state.STATE_COLORS.get(st, (255, 255, 255))
    pcol = state.STATE_COLORS.get(ph, (255, 255, 255))
    draw_text_hud(frame, [
        (f"frame {frame_idx}/{n_total}", "", (255, 255, 255)),
        (f"time {t:6.2f}s", "", (255, 255, 255)),
        (f"state: {st}", "", col),
        (f"phase: {ph}", "", pcol),
        (f"mean vr {vr_mean:7.2f} px/f", "", (255, 255, 255)),
        (f"points {n_points}", "", (255, 255, 255)),
    ])
