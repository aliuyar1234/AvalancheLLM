# PHASE 5 ANALYSIS METRICS FIGURES

## Purpose
Phase 5 computes the mechanistic metrics and quasi-critical signatures on Dataset A across the configured gain grid, under rate-matched thresholds and marginals-preserving null controls. It exports the main tables and figures used by the paper.

This phase MUST be inference and analysis only. It MUST NOT train any model parameters.

## Prerequisites
- Phase 1 calibration completed and produced per-layer mu and sigma and baseline rate stats.
- Phase 2 gain grid completed and produced branching metrics versus gain on Dataset A.
- Phase 3 rasters extracted for Dataset A across gains (or at minimum the subset required for analysis).
- Phase 4 null rasters generated, including within-layer time permutation and token-shuffled input null, for the same conditions as the real rasters.

Dependency artifacts required:
- rasters_token_layer.npz from the Phase 3 or Phase 4 producing run
- rasters_nulls.npz from Phase 4 producing run
- metrics.parquet and avalanches.parquet inputs may be produced in this phase, but upstream rasters must exist

## Commands
Execute both commands (INSTRUCT then BASE):
- CANON.CLI.CMD.PHASE5_ANALYZE_AND_EXPORT
- CANON.CLI.CMD.PHASE5_ANALYZE_AND_EXPORT_BASE

Each command MUST:
- Load the resolved pipeline config.
- Locate upstream run directories via dependencies recorded in run_record.json or via config dependency section.
- Produce a new Phase 5 run directory under runs.

## Expected outputs
A new run directory MUST be created under runs. Inside it:

Required run metadata:
- run_record.json
- config_resolved.yaml

Required machine-readable artifacts in results:
- results/metrics.parquet
- results/avalanches.parquet

Required paper-facing tables in tables:
- tables/table_T01_SUMMARY.csv
- tables/table_T01_SUMMARY.parquet
- tables/appendix/table_T04_TAIL_FITS.csv and tables/appendix/table_T04_TAIL_FITS.parquet
- tables/appendix/table_T05_CRACKLING_DIAGNOSTICS.csv and tables/appendix/table_T05_CRACKLING_DIAGNOSTICS.parquet
- tables/appendix/table_T06_ABLATIONS.csv and tables/appendix/table_T06_ABLATIONS.parquet

Required figures in figs (both PDF and PNG):
- figs/fig_F01_RASTER_EXAMPLE.pdf and figs/fig_F01_RASTER_EXAMPLE.png
- figs/fig_F02_RATE_MATCH_CHECK.pdf and figs/fig_F02_RATE_MATCH_CHECK.png
- figs/fig_F03_BRANCHING_CURVES.pdf and figs/fig_F03_BRANCHING_CURVES.png
- figs/fig_F04_NULL_DELTAB.pdf and figs/fig_F04_NULL_DELTAB.png
- figs/fig_F05_GSTAR_SELECTION.pdf and figs/fig_F05_GSTAR_SELECTION.png
- figs/fig_F08_SPIKEDEF_ROBUST.pdf and figs/fig_F08_SPIKEDEF_ROBUST.png
- figs/appendix/fig_F09_CHI_CURVES.pdf and figs/appendix/fig_F09_CHI_CURVES.png
- figs/appendix/fig_F10_NULL_COMPARE.pdf and figs/appendix/fig_F10_NULL_COMPARE.png
- figs/appendix/fig_F11_ABLATIONS.pdf and figs/appendix/fig_F11_ABLATIONS.png

## Definition of Done
All of the following are true:
- The table schema for T01_SUMMARY passes the checks in spec/18_RESULTS_TABLE_SKELETONS.md.
- All Phase 5 figures pass the checks in spec/19_FIGURES_TABLES_PLAN.md.
- The run_record.json artifact_index contains an entry for every required table and figure with sha256 and file size.
- Rate-matching acceptance check holds:
  - achieved_rate_max_abs_err is less than or equal to CANON.CONST.RATE_MATCH_TOL_ABS for every condition.

## Acceptance tests
Run these checks after the command completes:

1. File existence checks:
- Verify required results parquet files exist.
- Verify required table CSV and Parquet exist.
- Verify all required figure PDFs and PNGs exist.

2. Table schema validation:
- Validate required columns and row coverage for T01_SUMMARY using spec/18.

3. Marginals-preserving null sanity check:
- Confirm that the within-layer permutation null preserves per-layer event counts exactly for each sequence, as recorded in logs or as a computed check in metrics.parquet.

4. Determinism spot check:
- Re-run Phase 5 with the same config and upstream dependencies and confirm that table hashes match, unless allow_resume is false and the system refuses to overwrite.

## Common failures and fixes
1. Missing upstream rasters.
   Detect: Phase 5 fails with missing rasters file.
   Fix: run Phase 3 and Phase 4, then rerun Phase 5.
2. Table schema mismatch.
   Detect: missing required columns or wrong row count.
   Fix: update the table writer to match spec/18 and rerun Phase 5.
3. Missing figure exports.
   Detect: one or more fig files not created.
   Fix: implement figure exporter per spec/19 and ensure it uses CANON.OUTPUT filenames.
4. Run overwrite attempt.
   Detect: program refuses because run directory exists.
   Fix: do not delete prior runs. Change run_id_mode or modify config so content hash differs.

## Cross references
- spec/18_RESULTS_TABLE_SKELETONS.md
- spec/19_FIGURES_TABLES_PLAN.md
- spec/06_BRANCHING_METRICS_AND_DELTAB.md
- spec/07_NULL_MODELS_AND_CONTROLS.md
