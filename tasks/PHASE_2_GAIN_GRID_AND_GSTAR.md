# PHASE 2 GAIN GRID AND GSTAR

## Purpose
Phase 2 runs the gain grid on Dataset A and selects the mechanistic calibration gain gstar using branching:
- Run branching metrics across CANON.CONST.GAIN_GRID_DEFAULT (under rate matching).
- Select gstar such that b_tot is closest to one (spec/08), without using task performance.

## Prerequisites
- Phase 1 calibration completed and produced:
  - results/CANON.OUTPUT.CALIBRATION_STATS_NPZ_BASENAME
  - results/CANON.OUTPUT.RATE_TARGETS_JSON_BASENAME

## Dependency graph
- Inputs:
  - Phase 1 run artifacts
  - configs/pipeline.yaml (gain grid, target rates)
- Produces:
  - gstar artifact consumed by Phase 3 and Phase 6

## Commands
Run, in order:
1. CANON.CLI.CMD.PHASE2_GAIN_GRID
2. CANON.CLI.CMD.PHASE2_SELECT_GSTAR
3. CANON.CLI.CMD.PHASE2_GAIN_GRID_BASE
4. CANON.CLI.CMD.PHASE2_SELECT_GSTAR_BASE

Each command MUST produce its own run directory and run_record.json entry.

## Expected outputs
### From PHASE2_GAIN_GRID run
- results/CANON.OUTPUT.METRICS_PARQUET_BASENAME
This metrics file MUST include rows for every combination of:
- spike_def_id
- target_rate
- gain in CANON.CONST.GAIN_GRID_DEFAULT
with computed b_time, b_depth, b_tot and delta_b equivalents when available.

### From PHASE2_SELECT_GSTAR run
- results/CANON.OUTPUT.GSTAR_JSON_BASENAME
This JSON MUST include gstar for each (spike_def_id, target_rate) and the tie-break rule result.

## Definition of Done
- Gain grid metrics exist and include all configured gains.
- gstar.json exists and is derived only from branching metrics on Dataset A.

## Acceptance tests
1. Gain grid completeness:
   - metrics.parquet contains all gains in CANON.CONST.GAIN_GRID_DEFAULT for Dataset A
2. gstar selection sanity:
   - gstar is one of the configured gains
   - if two gains are tied, gstar is the one closest to CANON.CONST.GAIN_BASELINE
3. Provenance:
   - gstar run_record.json lists the gain-grid run_id as a dependency

## Common failures and fixes
1. Missing gain grid rows.
   Detect: some gains absent in metrics.parquet.
   Fix: iterate exactly over CANON.CONST.GAIN_GRID_DEFAULT and log condition ids.
2. gstar computed using performance.
   Detect: logs show accuracy or loss used for selection.
   Fix: enforce selection rule in spec/08 and hard-fail if task metrics are referenced.
3. Drift in rate matching.
   Detect: per-layer rate error exceeds CANON.CONST.RATE_MATCH_TOL_ABS.
   Fix: recompute tau_l(g) correctly per spec/04 and rerun.

## Cross references
- spec/04_RATE_MATCHING_PROTOCOL.md
- spec/06_BRANCHING_METRICS_AND_DELTAB.md
- spec/08_GAIN_INTERVENTION_AND_GSTAR.md
- spec/18_RESULTS_TABLE_SKELETONS.md
