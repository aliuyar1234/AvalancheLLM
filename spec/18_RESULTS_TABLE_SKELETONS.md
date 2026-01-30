# 18 RESULTS TABLE SKELETONS

## Purpose and scope
This document defines the exact table artifacts that MUST be produced for an ICLR-ready submission and the exact schemas Codex MUST implement. The intent is that every paper claim is backed by a table row that can be regenerated from immutable run outputs.

Scope includes:
- The summary table across gains on Dataset A.
- The cross-dataset generalization table on Dataset B at g=1 and gstar.
- The ARC multiple-choice evaluation table at g=1 and gstar.
- Schema validation rules and acceptance gates.

## Normative requirements
- MUST write each table in both CSV and Parquet formats using the canonical filenames in CANON.OUTPUT.TABLE_FILE_CSV and CANON.OUTPUT.TABLE_FILE_PARQUET.
- MUST place tables inside the run directory for the stage that produced them, under the tables subdirectory.
- MUST include a run_id column for provenance and a config_hash column for reproducibility.
- MUST ensure that column names and meanings match this spec exactly. Do not rename columns without updating this spec and CANON.
- MUST validate that required rows exist for all conditions listed in the resolved config for that run.
- SHOULD keep tables denormalized enough that paper writing does not require joining multiple files.
- MAY include additional optional columns, but optional columns MUST be prefixed with opt_ to avoid accidental dependency.

## Definitions
All identifiers are defined in spec/00_CANONICAL.md.

Table IDs:
- CANON.ID.TABLE.T01_SUMMARY
- CANON.ID.TABLE.T02_GENERALIZATION
- CANON.ID.TABLE.T03_ARC

Canonical filenames inside a run directory:
- CANON.OUTPUT.TABLE_FILE_CSV.T01_SUMMARY
- CANON.OUTPUT.TABLE_FILE_CSV.T02_GENERALIZATION
- CANON.OUTPUT.TABLE_FILE_CSV.T03_ARC

Type conventions:
- string: UTF-8 string
- int: signed 64-bit integer
- float: 64-bit float
- bool: boolean

Required provenance columns in every table:
- run_id (string)
- stage_id (string)
- model_id (string)
- dataset_role (string, one of CANON.ENUM.DATASET_ROLE)
- config_hash (string, sha256 of resolved config file bytes)
- code_version (string, git commit hash if available, otherwise the literal string no_git_metadata)

## Procedure
Generation is deterministic once upstream run artifacts exist.

1. Load the producing stage run_record.json.
2. Load the producing stage parquet artifacts:
   - CANON.OUTPUT.METRICS_PARQUET_BASENAME
   - CANON.OUTPUT.AVALANCHES_PARQUET_BASENAME
3. Compute per-condition aggregates exactly as specified per table below.
4. Write CSV and Parquet outputs into the run tables directory using CANON.OUTPUT.TABLE_FILE_CSV and CANON.OUTPUT.TABLE_FILE_PARQUET mappings.
5. Validate schema and required row coverage.
6. Register the written table artifacts in run_record.json under artifact_index with sha256 hashes and file sizes.

### Table T01_SUMMARY schema and required rows
Purpose: Provide the core mechanistic summary across the gain grid on Dataset A, including rate-matched branching and marginals-controlled connectivity residuals.

File:
- CSV: CANON.OUTPUT.TABLE_FILE_CSV.T01_SUMMARY
- Parquet: CANON.OUTPUT.TABLE_FILE_PARQUET.T01_SUMMARY

Required columns:
- run_id (string)
- stage_id (string)
- model_id (string)
- model_role (string, one of CANON.ENUM.MODEL_ROLE)
- dataset_role (string, fixed to A for this table)
- spike_def_id (string, one of CANON.ENUM.SPIKE_DEF_ID)
- target_rate (float, one of CANON.CONST.TARGET_LAYER_RATES)
- g (float)
- tau0_baseline (float, fixed to CANON.CONST.TAU0_BASELINE)
- rate_match_tol_abs (float, fixed to CANON.CONST.RATE_MATCH_TOL_ABS)
- achieved_rate_mean (float, mean achieved per-layer firing rate across layers)
- achieved_rate_max_abs_err (float, max absolute deviation from target across layers)
- b_time (float)
- b_depth (float)
- b_tot (float)
- b_time_perm (float, within-layer permutation null)
- b_depth_perm (float)
- b_tot_perm (float)
- delta_b_time (float, b_time minus b_time_perm)
- delta_b_depth (float)
- delta_b_tot (float)
- chi (float, susceptibility proxy as in spec/09)
- crackling_gamma (float, slope estimate as in spec/09)
- crackling_ci_low (float)
- crackling_ci_high (float)
- n_sequences (int, number of sequences analyzed)
- n_avalanches (int)
- avalanche_size_mean (float, mean S)
- avalanche_size_median (float)
- avalanche_span_tokens_mean (float, mean D)
- avalanche_span_layers_mean (float, mean H)
- config_hash (string)
- code_version (string)

Required opt_ columns (still prefixed opt_ for forward-compatibility):
- opt_b_time_ci_low (float)
- opt_b_time_ci_high (float)
- opt_b_depth_ci_low (float)
- opt_b_depth_ci_high (float)
- opt_b_tot_ci_low (float)
- opt_b_tot_ci_high (float)
- opt_delta_b_time_ci_low (float)
- opt_delta_b_time_ci_high (float)
- opt_delta_b_depth_ci_low (float)
- opt_delta_b_depth_ci_high (float)
- opt_delta_b_tot_ci_low (float)
- opt_delta_b_tot_ci_high (float)
- opt_chi_ci_low (float)
- opt_chi_ci_high (float)
- opt_b_tot_shift (float, circular-shift null b_tot)
- opt_delta_b_tot_shift (float, b_tot minus circular-shift null b_tot)
- opt_delta_b_tot_shift_ci_low (float)
- opt_delta_b_tot_shift_ci_high (float)
- opt_phase4_perm_null_marginals_exact (bool)
- opt_phase4_shift_null_marginals_exact (bool)

Required rows:
- For each combination of spike_def_id and target_rate in the resolved config.
- For each g in the gain grid for Dataset A in the resolved config.
- Exactly one row per condition.

Acceptance checks:
- Row count equals (num_spike_defs times num_target_rates times num_gains).
- achieved_rate_max_abs_err is less than or equal to CANON.CONST.RATE_MATCH_TOL_ABS for all rows.
- No NaN values in b_tot, delta_b_tot, chi.
- Required opt_ columns listed above are present and finite.

### Table T02_GENERALIZATION schema and required rows
Purpose: Evaluate whether the mechanistically calibrated gstar from Dataset A transfers to Dataset B in mechanistic metrics and does not collapse task proxy metrics.

File:
- CSV: CANON.OUTPUT.TABLE_FILE_CSV.T02_GENERALIZATION
- Parquet: CANON.OUTPUT.TABLE_FILE_PARQUET.T02_GENERALIZATION

Required columns:
- run_id (string)
- stage_id (string)
- model_id (string)
- model_role (string)
- dataset_role (string, fixed to B)
- spike_def_id (string)
- target_rate (float)
- g_condition (string, one of g1 or gstar)
- g (float)
- b_time (float)
- b_depth (float)
- b_tot (float)
- delta_b_tot (float)
- chi (float)
- n_sequences (int)
- ppl (float, perplexity on Dataset B slice)
- nll_mean (float, mean negative log likelihood)
- config_hash (string)
- code_version (string)

Required rows:
- For each spike_def_id and target_rate.
- For each g_condition in {g1, gstar}.
- Exactly one row per condition.

Acceptance checks:
- Row count equals (num_spike_defs times num_target_rates times 2).
- No NaN in ppl and b_tot.
- g_condition values are exactly g1 and gstar.

### Table T03_ARC schema and required rows
Purpose: Evaluate ARC multiple-choice accuracy at g=1 and gstar, and record uncertainty.

File:
- CSV: CANON.OUTPUT.TABLE_FILE_CSV.T03_ARC
- Parquet: CANON.OUTPUT.TABLE_FILE_PARQUET.T03_ARC

Required columns:
- run_id (string)
- stage_id (string)
- model_id (string)
- model_role (string)
- dataset_role (string, fixed to ARC_MCQ)
- g_condition (string, one of g1 or gstar)
- g (float)
- n_questions (int)
- accuracy (float)
- accuracy_ci_low (float)
- accuracy_ci_high (float)
- mean_logprob_correct_minus_incorrect (float)
- config_hash (string)
- code_version (string)

Required rows:
- One row for g1 and one row for gstar per spike_def_id and target_rate, unless ARC evaluation is configured to run only one spike_def. If ARC evaluation is limited, that limitation MUST be recorded in run_record.json and in an opt_note column.

Acceptance checks:
- accuracy is between 0 and 1 inclusive.
- n_questions is greater than 0.
- CI bounds are within 0 and 1 inclusive.

### Table T04_TAIL_FITS schema and required rows
Purpose: Descriptive heavy-tail model comparisons for avalanche size S on Dataset A. These fits are descriptive only and are included to preempt reviewer concerns; they are not used as sole evidence of criticality.

File:
- CSV: CANON.OUTPUT.TABLE_FILE_CSV.T04_TAIL_FITS
- Parquet: CANON.OUTPUT.TABLE_FILE_PARQUET.T04_TAIL_FITS

Required columns:
- run_id (string)
- stage_id (string)
- model_id (string)
- model_role (string)
- dataset_role (string, fixed to A)
- spike_def_id (string)
- target_rate (float)
- g (float)
- xmin (float, tail threshold used)
- n_tail (int, number of samples with size >= xmin)
- alpha_powerlaw (float)
- lambda_exponential (float)
- lognorm_mu (float)
- lognorm_sigma (float)
- ll_powerlaw (float, log-likelihood under power law)
- ll_exponential (float)
- ll_lognormal (float)
- llr_powerlaw_vs_lognormal (float, ll_powerlaw - ll_lognormal)
- llr_powerlaw_vs_exponential (float, ll_powerlaw - ll_exponential)
- config_hash (string)
- code_version (string)

Required rows:
- One row per T01 condition (spike_def_id, target_rate, g).

Acceptance checks:
- Row count equals T01 row count.
- n_tail >= CANON.CONST.TAIL_FIT_MIN_TAIL_SAMPLES for all rows (otherwise hard-fail).

### Table T05_CRACKLING_DIAGNOSTICS schema and required rows
Purpose: Crackling fit quality diagnostics per condition to support quasi-critical signature S3 under explicit quality gates.

File:
- CSV: CANON.OUTPUT.TABLE_FILE_CSV.T05_CRACKLING_DIAGNOSTICS
- Parquet: CANON.OUTPUT.TABLE_FILE_PARQUET.T05_CRACKLING_DIAGNOSTICS

Required columns:
- run_id (string)
- stage_id (string)
- model_id (string)
- model_role (string)
- dataset_role (string, fixed to A)
- spike_def_id (string)
- target_rate (float)
- g (float)
- crackling_gamma (float)
- crackling_ci_low (float)
- crackling_ci_high (float)
- crackling_ci_width (float, crackling_ci_high - crackling_ci_low)
- n_duration_points (int)
- n_avalanches_used (int)
- r2 (float, fit r^2 in log-log space)
- pass (bool, quality gate)
- config_hash (string)
- code_version (string)

Required rows:
- One row per T01 condition.

Acceptance checks:
- Row count equals T01 row count.
- pass is true for all rows (otherwise hard-fail).

### Table T06_ABLATIONS schema and required rows
Purpose: Mechanistic ablations and negative controls to test whether the signature suite is specific to the MLP gain intervention.

File:
- CSV: CANON.OUTPUT.TABLE_FILE_CSV.T06_ABLATIONS
- Parquet: CANON.OUTPUT.TABLE_FILE_PARQUET.T06_ABLATIONS

Required columns:
- run_id (string)
- stage_id (string)
- model_id (string)
- model_role (string)
- dataset_role (string, fixed to A)
- spike_def_id (string)
- target_rate (float)
- g_condition (string, one of g1 or gstar)
- g (float)
- intervention_id (string, one of CANON.ENUM.INTERVENTION_ID)
- gain_target (string, one of CANON.ENUM.GAIN_TARGET)
- b_tot (float)
- delta_b_tot (float, vs within-layer permutation null)
- chi (float)
- achieved_rate_max_abs_err (float)
- n_sequences (int)
- config_hash (string)
- code_version (string)

Required rows:
- For each spike_def_id and target_rate, include at minimum:
  - one g1 row for intervention_id=MLP_GLOBAL
  - one gstar row for each intervention_id in CANON.ENUM.INTERVENTION_ID

Acceptance checks:
- achieved_rate_max_abs_err <= CANON.CONST.RATE_MATCH_TOL_ABS for all rows.
- No NaN in b_tot, delta_b_tot, chi.

### Table T07_REPLICATION_SUMMARY schema and required rows
Purpose: Cross-model replication summary comparing the main mechanistic effects between at least two model roles (BASE and INSTRUCT).

File:
- CSV: CANON.OUTPUT.TABLE_FILE_CSV.T07_REPLICATION_SUMMARY
- Parquet: CANON.OUTPUT.TABLE_FILE_PARQUET.T07_REPLICATION_SUMMARY

Required columns:
- run_id (string, producing Phase7 run)
- stage_id (string)
- model_id_base (string)
- model_id_instruct (string)
- run_id_s05_base (string)
- run_id_s05_instruct (string)
- spike_def_id (string)
- target_rate (float)
- gstar_base (float)
- gstar_instruct (float)
- delta_b_tot_at_gstar_base (float)
- delta_b_tot_at_gstar_instruct (float)
- chi_at_gstar_base (float)
- chi_at_gstar_instruct (float)
- config_hash (string)
- code_version (string)

Required rows:
- One row per (spike_def_id, target_rate).

Acceptance checks:
- No NaN values in the numeric comparison columns.

## Worked example
Example run directory: CANON.EXAMPLE.RUN_DIR_S05.

Inside that directory, the summary table CSV MUST exist at:
- tables/table_T01_SUMMARY.csv

The header row MUST contain, at minimum:
run_id,stage_id,model_id,dataset_role,spike_def_id,target_rate,g,b_tot,delta_b_tot,chi,config_hash

## Failure modes
1. Missing required columns.
   Detect: schema validation reports missing column names.
   Fix: update table writer to emit the required columns and re-run Phase 5.
2. Wrong row coverage.
   Detect: row count does not match the required condition coverage.
   Fix: ensure the analysis enumerates the full gain grid and both spike defs and target rates.
3. Provenance loss.
   Detect: run_id or config_hash missing or empty.
   Fix: enforce that every table row copies provenance fields from run_record.json and resolved config.
4. Rate matching drift.
   Detect: achieved_rate_max_abs_err exceeds CANON.CONST.RATE_MATCH_TOL_ABS.
   Fix: increase quantile resolution or calibration sample size, then re-run Phase 1 and Phase 4.

## Acceptance criteria
- All required tables are written in both CSV and Parquet formats.
- Schema validation passes for each table.
- Required row coverage passes for each table.
- Each table is registered in the producing run_record.json with sha256 and file size.

## Cross references
- spec/00_CANONICAL.md for IDs and filenames
- spec/04_RATE_MATCHING_PROTOCOL.md for achieved rate checks
- spec/06_BRANCHING_METRICS_AND_DELTAB.md for branching and delta b definitions
- spec/09_QUASI_CRITICAL_SIGNATURES.md for chi and crackling
- tasks/PHASE_5_ANALYSIS_METRICS_FIGURES.md and tasks/PHASE_6_GENERALIZATION_TASK_EVAL.md
