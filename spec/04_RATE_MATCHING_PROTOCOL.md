# 04 RATE MATCHING PROTOCOL

## Purpose and scope
- Define per-layer thresholds tau_l(g) so that the marginal spike rate in each layer is matched across gain values g.
- Provide the reviewer-critical control: connectivity and branching changes must persist under rate matching.
- Apply to both spike definitions in spec/03.

## Normative requirements
- MUST compute tau_l(g) for every layer l and each gain g in the configured gain grid.
- MUST compute tau_l(g) separately for each spike_def_id:
  - CANON.ENUM.SPIKE_DEF_ID.SPIKE_ONE_SIDED_POS
  - CANON.ENUM.SPIKE_DEF_ID.SPIKE_TWO_SIDED_ABS
- MUST use only Dataset A calibration slices as specified by spec/11 to compute tau_l(g).
- MUST verify that achieved per-layer rates match the target rates within CANON.CONST.RATE_MATCH_TOL_ABS.
- MUST write the tau_l(g) table to the producing run under:
  - results/CANON.OUTPUT.TAU_RATE_MATCHED_PARQUET_BASENAME
- SHOULD implement the deterministic histogram quantile estimator described below.

## Definitions
- z_{t,l,i}(g): standardized activation for token t, layer l, unit i at gain g (spec/03).
- v_{t,l,i}(g): the thresholded quantity:
  - For SPIKE_ONE_SIDED_POS: v = z
  - For SPIKE_TWO_SIDED_ABS: v = absolute value of z
- s(t,l,i; tau): spike indicator equals 1 if v exceeds tau, else 0.
- Target rate per layer r_star_l:
  - computed at baseline gain g = CANON.CONST.GAIN_BASELINE and baseline threshold tau0 = CANON.CONST.TAU0_BASELINE.
  - stored by Phase 1 in results/CANON.OUTPUT.RATE_TARGETS_JSON_BASENAME.
- Achieved rate:
  r_l(g, tau) = mean over calibration tokens and units of s(t,l,i; tau).

## Procedure (deterministic histogram quantile)
Inputs:
- v samples for a fixed (gain g, layer l, spike_def_id) collected from the deterministic calibration slice.
- r_star_l from Phase 1.
- Constants:
  - CANON.CONST.RATE_MATCH_HIST_BINS
  - CANON.CONST.RATE_MATCH_EDGE_ROUND_ABS

Algorithm:
1. Collect v samples in deterministic order:
   - sequence indices in ascending order,
   - token indices in ascending order,
   - unit indices in ascending order.
2. Compute v_min and v_max over collected samples.
3. Round to stable histogram edges:
   - v_min_round is v_min rounded down to a multiple of CANON.CONST.RATE_MATCH_EDGE_ROUND_ABS.
   - v_max_round is v_max rounded up to a multiple of CANON.CONST.RATE_MATCH_EDGE_ROUND_ABS.
4. Build an equal-width histogram with CANON.CONST.RATE_MATCH_HIST_BINS bins spanning [v_min_round, v_max_round].
5. Compute the empirical CDF.
6. Select tau_l(g) as the smallest threshold such that the empirical tail probability equals r_star_l.
7. Verification pass (hard-fail on violation):
   - recompute r_l(g, tau_l(g)) on the same calibration slice.
   - require absolute error is less than or equal to CANON.CONST.RATE_MATCH_TOL_ABS.

## Worked example (real IDs and paths)
Example run path:
- A Phase 1 run id is CANON.EXAMPLE.RUN_ID_S05 and the corresponding directory is CANON.EXAMPLE.RUN_DIR_S05.

Files produced by Phase 1 that Phase 4 or Phase 5 will consume:
- runs/RUN_S05_9b8f2e0a1c3d/results/CANON.OUTPUT.CALIBRATION_STATS_NPZ_BASENAME
- runs/RUN_S05_9b8f2e0a1c3d/results/CANON.OUTPUT.RATE_TARGETS_JSON_BASENAME

In Phase 4, for a specific gain g and layer l, the program computes tau_l(g) and appends a row into:
- runs/RUN_S05_9b8f2e0a1c3d/results/CANON.OUTPUT.TAU_RATE_MATCHED_PARQUET_BASENAME

A passing run will also record into run_record.json a metrics block containing:
- max_abs_rate_error is less than or equal to CANON.CONST.RATE_MATCH_TOL_ABS

## Failure modes (detect and fix)
1. Insufficient calibration samples.
   Detect: sample count is less than CANON.CONST.RATE_MATCH_MIN_CAL_SAMPLES.
   Fix: increase calibration slice size in configs/pipeline.yaml and rerun Phase 1.
2. Rate mismatch after tau selection.
   Detect: abs(r_l - r_star_l) exceeds CANON.CONST.RATE_MATCH_TOL_ABS for any (l,g).
   Fix: increase CANON.CONST.RATE_MATCH_HIST_BINS or broaden the calibration slice; rerun the producing phase.
3. Non-deterministic tau values.
   Detect: tau_l(g) differs across two identical runs with identical inputs and seeds.
   Fix: ensure deterministic sample order and deterministic histogram edges; log edges and bin count.
4. Degenerate histogram range.
   Detect: v_min_round equals v_max_round.
   Fix: validate activation hook correctness (spec/12) and check for NaNs or constant activations.

## Acceptance criteria
- For all layers l and gains g, the achieved rate error is within CANON.CONST.RATE_MATCH_TOL_ABS.
- The tau table exists and includes rows for every configured gain, layer, and spike_def_id.
- The within-layer time permutation null (spec/07) preserves per-layer counts and rate matching remains satisfied.

## Cross references
- spec/03_SPIKE_DEFS_AND_STANDARDIZATION.md
- spec/07_NULL_MODELS_AND_CONTROLS.md
- spec/11_DATASETS_SAMPLING_HASHING.md
- spec/12_MODELS_HOOKS_TENSORS.md
- spec/16_TEST_PLAN.md
