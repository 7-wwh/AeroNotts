"""AeroNotts rocket visual-analysis package.

Split into small, single-concern modules so each piece is easy to maintain:

  io            - video open/write, output paths, scale handling
  state         - flight-state classification (ASCEND/DESCEND/STABLE) + APOGEE phase
  schema        - single source of truth for the metrics CSV columns
  draw          - HUD, trails, random point colors, FOE crosshair
  synth         - synthetic test-video generator
  plot          - metrics plot with state bands and apogee markers
  features      - per-frame feature extraction (sparse / dense / camera / appearance / horizon)
"""
