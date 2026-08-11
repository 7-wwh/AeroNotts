"""Feature aggregation: merge per-group feature dicts into one per-frame row."""

import numpy as np

from .. import schema
from . import sparse as _sparse
from . import dense as _dense
from . import camera as _camera
from . import appearance as _appearance
from . import horizon as _horizon


def empty_row():
    return {k: np.nan for k in schema.COLUMNS}


def aggregate(ctx, args, foe_c):
    """Build one row (dict of schema.COLUMNS -> value) for the current frame.

    `ctx` is a SimpleNamespace holding the shared, already-computed data:
      frame, gray, sW, sH, inv, t, tracker, prev, cur, flow, ids, dense
    Feature groups that are disabled (--no-*) or impossible this frame leave
    their columns as NaN.
    """
    row = empty_row()
    row["time_s"] = ctx.t

    cur, flow = ctx.cur, ctx.flow

    # ---- sparse group (always on; NaN on frames with no tracking) ----
    row.update(_sparse.sparse_features(ctx.tracker, cur, flow, foe_c, ctx.sW, ctx.sH))
    row["foe_x"] = float(foe_c[0] * ctx.inv) if foe_c is not None else np.nan
    row["foe_y"] = float(foe_c[1] * ctx.inv) if foe_c is not None else np.nan

    # ---- dense group ----
    if not args.no_dense and ctx.dense is not None:
        row.update(_dense.magnitude_stats(ctx.dense))
        row.update(_dense.divergence_stats(ctx.dense))
        row.update(_dense.grid_flow(ctx.dense))

    # ---- camera group (affine + residual always; homography optional) ----
    if cur is not None and ctx.prev is not None and len(cur) >= 6:
        am = _camera.affine_model(ctx.prev, cur)
        if am is not None:
            M = am.pop("M")
            row.update(am)
            row.update(_camera.residual_flow(ctx.prev, cur, M, foe_c))
        if not args.no_homography:
            hm = _camera.homography_model(ctx.prev, cur)
            if hm is not None:
                row.update(hm)

    # ---- appearance group ----
    if not args.no_appearance:
        row.update({
            "edge_density": _appearance.edge_density(ctx.gray),
            "texture_var": _appearance.texture_var(ctx.gray),
            "grad_magnitude_mean": _appearance.grad_magnitude_mean(ctx.gray),
            "sharpness": _appearance.sharpness(ctx.gray),
        })
        row.update(_appearance.sky_ground_fractions(ctx.frame))

    # ---- horizon group (experimental) ----
    if not args.no_horizon:
        row.update(_horizon.horizon_features(ctx.gray))

    return row
