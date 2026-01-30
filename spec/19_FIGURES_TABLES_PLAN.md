# 19 FIGURES TABLES PLAN

## Purpose and scope
This document defines the exact publishable figures that MUST be exported, including filenames, provenance requirements, and acceptance checks. The goal is that Codex can generate camera-ready PDF figures directly from run artifacts without guessing.

Scope:
- The required main-paper figures (F01 through F08).
- The required appendix figures (F09 through F11).
- The exact filenames and required formats.
- The data sources for each figure and the producing stage.
- Minimum visual requirements and validation gates.

## Normative requirements
- MUST export each figure as both PDF and PNG using the canonical filenames in CANON.OUTPUT.FIG_FILE_PDF and CANON.OUTPUT.FIG_FILE_PNG.
- MUST place figures inside the producing run directory under the figs subdirectory.
- MUST embed provenance metadata in the PDF and PNG where feasible:
  - run_id
  - model_id
  - dataset_role
  - spike_def_id
  - target_rate
  - g or g_condition
  - config_hash
- MUST generate figures from immutable parquet and CSV artifacts written by earlier phases. Do not re-run the model during figure export.
- MUST keep all axes labeled and units explicit.
- SHOULD use consistent styling across all figures.
- MUST export appendix figures listed in CANON.OUTPUT.FIG_FILE_* beyond F08 (currently F09 through F11), placing them under figs/appendix and listing them in run_record.json artifact_index.
- MAY include additional appendix figures beyond F11, but they MUST also be placed under figs/appendix and listed in run_record.json artifact_index.

## Definitions
Figure IDs are defined in spec/00_CANONICAL.md under CANON.ID.FIG.

Canonical filenames inside a run directory:
- CANON.OUTPUT.FIG_FILE_PDF.F01_RASTER_EXAMPLE
- CANON.OUTPUT.FIG_FILE_PDF.F02_RATE_MATCH_CHECK
- CANON.OUTPUT.FIG_FILE_PDF.F03_BRANCHING_CURVES
- CANON.OUTPUT.FIG_FILE_PDF.F04_NULL_DELTAB
- CANON.OUTPUT.FIG_FILE_PDF.F05_GSTAR_SELECTION
- CANON.OUTPUT.FIG_FILE_PDF.F06_GENERALIZATION_B
- CANON.OUTPUT.FIG_FILE_PDF.F07_ARC_MCQ
- CANON.OUTPUT.FIG_FILE_PDF.F08_SPIKEDEF_ROBUST

Data sources:
- metrics.parquet and avalanches.parquet in the producing run results directory.
- tables written by spec/18.

Producing stages:
- Phase 5 produces F01 to F05 and F08.
- Phase 5 produces appendix figures F09 to F11.
- Phase 6 produces F06 and F07.

## Procedure
For each figure, implement a deterministic plotting function that reads from the declared sources and writes to the declared output paths.

General steps:
1. Load run_record.json for the producing run.
2. Load required parquet and table CSV or Parquet files.
3. Validate that required columns exist.
4. Create the figure with labeled axes and legends.
5. Write PDF and PNG files using the canonical filenames.
6. Validate output existence and minimum file size.
7. Register the figure artifacts in run_record.json with sha256 and file size.

### Figure F01_RASTER_EXAMPLE
Goal: Show a token by layer event raster for one representative sequence, with avalanche connected components highlighted.

Sources:
- rasters_token_layer.npz from Phase 3 or Phase 4 output
- avalanche component labeling output from Phase 5 analysis

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F01_RASTER_EXAMPLE
- CANON.OUTPUT.FIG_FILE_PNG.F01_RASTER_EXAMPLE

Acceptance checks:
- Raster axis labels include token index and layer index.
- At least one connected component is visible.

### Figure F02_RATE_MATCH_CHECK
Goal: Demonstrate that achieved per-layer rates match the target across gains due to tau_l(g) calibration.

Sources:
- Phase 1 calibration stats
- Phase 4 rate-matching verification logs
- Table T01_SUMMARY columns achieved_rate_mean and achieved_rate_max_abs_err

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F02_RATE_MATCH_CHECK
- CANON.OUTPUT.FIG_FILE_PNG.F02_RATE_MATCH_CHECK

Acceptance checks:
- Plot shows achieved_rate_max_abs_err below CANON.CONST.RATE_MATCH_TOL_ABS for all gains.

### Figure F03_BRANCHING_CURVES
Goal: Show branching measures b_time, b_depth, and b_tot as a function of gain g under rate-matching, for both spike definitions.

Sources:
- metrics.parquet for real data
- Table T01_SUMMARY columns g, b_time, b_depth, b_tot, spike_def_id, target_rate

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F03_BRANCHING_CURVES
- CANON.OUTPUT.FIG_FILE_PNG.F03_BRANCHING_CURVES

Acceptance checks:
- At least two lines per spike_def_id if both spike defs enabled.
- Axis includes gain g on x-axis and branching value on y-axis.

### Figure F04_NULL_DELTAB
Goal: Show that delta_b remains non-zero relative to the within-layer time permutation null, despite rate-matching.

Sources:
- Table T01_SUMMARY columns delta_b_time, delta_b_depth, delta_b_tot

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F04_NULL_DELTAB
- CANON.OUTPUT.FIG_FILE_PNG.F04_NULL_DELTAB

Acceptance checks:
- Includes error bars or bootstrap CI if computed.
- Clearly labels that the null preserves marginals.

### Figure F05_GSTAR_SELECTION
Goal: Visualize gstar selection on Dataset A using b_tot crossing near one, without using performance signals.

Sources:
- Table T01_SUMMARY b_tot versus g
- gstar.json written by Phase 2 selection

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F05_GSTAR_SELECTION
- CANON.OUTPUT.FIG_FILE_PNG.F05_GSTAR_SELECTION

Acceptance checks:
- gstar annotated on the curve.
- Plot caption or embedded note states selection criterion b_tot near one.

### Figure F06_GENERALIZATION_B
Goal: Compare Dataset B metrics at g=1 versus gstar, including mechanistic and task proxies.

Sources:
- Table T02_GENERALIZATION columns ppl, b_tot, delta_b_tot, chi

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F06_GENERALIZATION_B
- CANON.OUTPUT.FIG_FILE_PNG.F06_GENERALIZATION_B

Acceptance checks:
- Includes both conditions g1 and gstar.
- Includes a paired comparison visualization.

### Figure F07_ARC_MCQ
Goal: Compare ARC multiple-choice accuracy at g=1 and gstar with uncertainty.

Sources:
- Table T03_ARC

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F07_ARC_MCQ
- CANON.OUTPUT.FIG_FILE_PNG.F07_ARC_MCQ

Acceptance checks:
- Accuracy bounded within 0 and 1.
- Confidence interval or bootstrap interval shown.

### Figure F08_SPIKEDEF_ROBUST
Goal: Show robustness across spike definitions and one-sided versus two-sided events.

Sources:
- Table T01_SUMMARY filtered to a target_rate and using both spike defs
- metrics.parquet for additional breakdowns if needed

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F08_SPIKEDEF_ROBUST
- CANON.OUTPUT.FIG_FILE_PNG.F08_SPIKEDEF_ROBUST

Acceptance checks:
- Figure explicitly labels spike defs.
- Demonstrates qualitative consistency of the main signatures across defs.

### Figure F09_CHI_CURVES (appendix)
Goal: Show susceptibility proxy chi as a function of gain g under rate matching, with uncertainty.

Sources:
- Table T01_SUMMARY columns chi and opt_chi_ci_low/high.

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F09_CHI_CURVES
- CANON.OUTPUT.FIG_FILE_PNG.F09_CHI_CURVES

Acceptance checks:
- Axis includes gain g on x-axis and chi on y-axis.
- Error bars or CI bands shown.

### Figure F10_NULL_COMPARE (appendix)
Goal: Compare delta_b_tot under the within-layer permutation null versus the within-layer circular shift null, and optionally contrast token-shuffled-input null on the g1/gstar subset.

Sources:
- Table T01_SUMMARY columns delta_b_tot, opt_delta_b_tot_ci_low/high.
- Table T01_SUMMARY columns opt_delta_b_tot_shift, opt_delta_b_tot_shift_ci_low/high.
- Phase 4 rasters_nulls.npz and metrics.parquet when token-shuffle subset is included.

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F10_NULL_COMPARE
- CANON.OUTPUT.FIG_FILE_PNG.F10_NULL_COMPARE

Acceptance checks:
- Legend labels clearly identify each null.
- Includes uncertainty (error bars) at least for the primary curves.

### Figure F11_ABLATIONS (appendix)
Goal: Compare mechanistic signatures under the primary MLP gain intervention versus negative controls (attention gain and layer-local gains).

Sources:
- Table T06_ABLATIONS.

Output:
- CANON.OUTPUT.FIG_FILE_PDF.F11_ABLATIONS
- CANON.OUTPUT.FIG_FILE_PNG.F11_ABLATIONS

Acceptance checks:
- Figure explicitly labels intervention_id and gain_target.
- Includes a clear comparison between MLP_GLOBAL and ATTN_GLOBAL at gstar.

## Worked example
Example run directory for Phase 5 analysis: CANON.EXAMPLE.RUN_DIR_S05.

Inside that run directory, the branching curve figure PDF MUST exist at:
- figs/fig_F03_BRANCHING_CURVES.pdf

## Failure modes
1. Figure uses raw activations instead of derived artifacts.
   Detect: figure export code calls the model forward pass.
   Fix: enforce artifact-only plotting, and read metrics.parquet and tables only.
2. Filenames drift.
   Detect: output file does not match CANON.OUTPUT.FIG_FILE_PDF and CANON.OUTPUT.FIG_FILE_PNG mappings.
   Fix: write a single figure exporter that uses CANON.OUTPUT keys.
3. Missing labels or ambiguous axes.
   Detect: visual inspection or lint rule checks for missing axis labels.
   Fix: enforce an axis labeling checklist in the exporter.
4. Non-deterministic plots.
   Detect: repeated export yields different raster subsample.
   Fix: select representative examples by deterministic index and log selection in run_record.json.

## Acceptance criteria
- All required figure PDFs and PNGs exist under the producing run figs directory.
- Each figure is registered in run_record.json artifact_index with sha256 and file size.
- File naming exactly matches CANON.OUTPUT.FIG_FILE_PDF and CANON.OUTPUT.FIG_FILE_PNG mappings.

## Cross references
- spec/18_RESULTS_TABLE_SKELETONS.md for table sources
- spec/06_BRANCHING_METRICS_AND_DELTAB.md for definitions
- spec/09_QUASI_CRITICAL_SIGNATURES.md for chi and crackling
- tasks/PHASE_5_ANALYSIS_METRICS_FIGURES.md and tasks/PHASE_6_GENERALIZATION_TASK_EVAL.md
