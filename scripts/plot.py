"""Metrics plot: velocity/acceleration with state bands and apogee markers."""

import numpy as np

from . import state as st


def render(ts, vel_raw, vel_s, accel, states, phases, out_plot, input_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as e:
        print(f"warning: could not import matplotlib: {e}")
        return

    MPL = {s: tuple(int(c) / 255.0 for c in reversed(st.STATE_COLORS[s]))
           for s in st.STATE_COLORS}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # state bands on velocity axis
    prev_s = None
    start = ts[0]
    for i, s in enumerate(states):
        if s != prev_s:
            if prev_s is not None and prev_s != st.STAB:
                ax1.axvspan(start, ts[i], color=MPL[prev_s], alpha=0.25)
            if s != st.STAB:
                start = ts[i]
            prev_s = s
    if prev_s is not None and prev_s != st.STAB:
        ax1.axvspan(start, ts[-1], color=MPL[prev_s], alpha=0.25)

    # apogee markers at the middle of each APOGEE phase run
    in_apo = False
    run_start = 0
    for i, ph in enumerate(phases):
        if ph == st.APO and not in_apo:
            in_apo = True
            run_start = i
        elif ph != st.APO and in_apo:
            ax1.axvline(ts[(run_start + i) // 2], color="k", ls="--", lw=1.2, alpha=0.7)
            in_apo = False
    if in_apo:
        ax1.axvline(ts[(run_start + len(phases)) // 2], color="k", ls="--", lw=1.2, alpha=0.7)

    ax1.plot(ts, vel_raw, color="0.55", lw=0.8, label="velocity raw")
    ax1.plot(ts, vel_s, color="#1f77b4", lw=1.8, label="velocity smoothed")
    ax1.axhline(0, color="k", lw=0.7)
    ax1.set_ylabel("radial velocity (px/s)")
    ax1.set_title(f"Rocket radial expansion metrics — {input_name}")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(ts, accel, color="#d62728", lw=1.6, label="acceleration")
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_ylabel("acceleration (px/s^2)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3)

    handles = [Patch(facecolor=MPL[st.ASC], alpha=0.5, label=f"ASCEND ({st.ASC})"),
               Patch(facecolor=MPL[st.DESC], alpha=0.5, label=f"DESCEND ({st.DESC})"),
               Patch(facecolor=MPL[st.STAB], alpha=0.5, label="STABLE"),
               Patch(facecolor="none", label="--- apogee")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 0.995), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_plot, dpi=130)
    plt.close(fig)
    print(f"wrote metrics plot: {out_plot}")
