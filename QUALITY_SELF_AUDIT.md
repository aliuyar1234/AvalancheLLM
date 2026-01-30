# QUALITY SELF AUDIT

This document is a release gate that summarizes how the pack meets the anti-drift requirements.

## Global gates
The following conditions MUST hold before creating a release zip:
- Forbidden marker scan passes: repository contains no to-do marker, no to-be-determined marker, and no three consecutive dots.
- SDR scan passes: literals for IDs, paths, CLI commands, and constants listed in CANON do not appear outside spec/00_CANONICAL.md.
- Each producing phase writes run_record.json that validates against spec/run_record.schema.json.
- Figures and tables match the filenames in CANON.OUTPUT mappings.

## File-by-file evidence of richness

### spec/00_CANONICAL.md
- Example: CANON.EXAMPLE.RUN_DIR_S05 and canonical artifact filenames.
- Failure modes: missing key, hardcoded literal, filename drift.
- Acceptance: YAML parses and filenames match CANON mappings.

### spec/04_RATE_MATCHING_PROTOCOL.md
- Example: numeric z distribution and quantile-derived tau_l(g).
- Failure modes: achieved rate outside tolerance, insufficient samples, non-deterministic quantiles.
- Acceptance: achieved_rate_max_abs_err within CANON.CONST.RATE_MATCH_TOL_ABS.

### spec/05_AVALANCHE_DEFINITION_AND_STATS.md
- Example: toy raster and expected connected components.
- Failure modes: adjacency mismatch, boundary errors, excessive component count.
- Acceptance: toy component stats match expected values.

### spec/06_BRANCHING_METRICS_AND_DELTAB.md
- Example: toy raster with computed b_time and b_depth.
- Failure modes: denominator mismatch at boundaries, null handling errors, NaN metrics.
- Acceptance: metrics stable and delta-b computed as defined.

### spec/07_NULL_MODELS_AND_CONTROLS.md
- Example: within-layer permutation preserves per-layer counts exactly.
- Failure modes: permutation not per-layer, accidental cross-layer permutation, non-deterministic RNG.
- Acceptance: marginal preservation check passes for every layer and sequence.

### spec/08_GAIN_INTERVENTION_AND_GSTAR.md
- Example: g grid and selection criterion b_tot near one.
- Failure modes: g applied at wrong hookpoint, gstar chosen using performance, tie-breaking drift.
- Acceptance: gstar recorded and selection uses mechanistic metric only.

### spec/09_QUASI_CRITICAL_SIGNATURES.md
- Example: multi-signature evaluation pipeline and explicit falsifiers.
- Failure modes: overclaiming criticality from heavy tails, fit instability, insufficient points.
- Acceptance: signatures computed with logged ranges and uncertainty.

### spec/18_RESULTS_TABLE_SKELETONS.md
- Example: required header subset for T01_SUMMARY and canonical path under CANON.EXAMPLE.RUN_DIR_S05.
- Failure modes: schema mismatch, missing rows, provenance loss.
- Acceptance: all required columns exist and row coverage matches config.

### spec/19_FIGURES_TABLES_PLAN.md
- Example: expected presence of figs/fig_F03_BRANCHING_CURVES.pdf under CANON.EXAMPLE.RUN_DIR_S05.
- Failure modes: plotting uses model forward pass, filenames drift, missing labels.
- Acceptance: all required PDF and PNG figures exported and registered in run_record.json.

### spec/23_IMPLEMENTATION_CONTRACT.md
- Example: Phase 5 required artifact set.
- Failure modes: CANON not resolved, run overwrite, wrong hookpoint tensor.
- Acceptance: end-to-end runbook can be executed without undocumented decisions.

### tasks/PHASE_5_ANALYSIS_METRICS_FIGURES.md
- Example outputs: results/metrics.parquet, tables/table_T01_SUMMARY.csv, and required figures.
- Failure modes: missing upstream rasters, schema mismatch.
- Acceptance: table and figure validation passes.

### tasks/PHASE_6_GENERALIZATION_TASK_EVAL.md
- Example outputs: T02_GENERALIZATION and T03_ARC plus F06 and F07.
- Failure modes: ARC prompt mismatch, dataset slice drift.
- Acceptance: paired g1 versus gstar rows exist and validate.

### tasks/PHASE_7_PAPER_CAMERA_READY.md
- Example outputs: updated paper markdown with claim-to-evidence mapping.
- Failure modes: missing artifact paths, bib key mismatch.
- Acceptance: citation lint and artifact lint pass.

## Known limitations
- The bibliography included in this pack is intentionally minimal and fully verified; expanding it requires verifying additional entries via primary sources and recording them in bib/citations_verified.md.

