# 03 SPIKE DEFINITIONS AND STANDARDIZATION

## Purpose and scope
- Define the standardized activation z and the two spike event definitions used throughout the project.
- Specify exactly which calibration artifacts are produced in Phase 1 and how downstream phases must consume them.
- Ensure determinism by pinning:
  - the calibration slice,
  - the activation hookpoint tensor u,
  - the standardization statistics (mu, sigma),
  - and the spike definition identifiers.

## Normative requirements
- MUST compute calibration statistics mu[l,i] and sigma[l,i] on Dataset A at gain g = CANON.CONST.GAIN_BASELINE.
- MUST use the activation tensor u defined in spec/12 (post-gate, pre-down-projection for SwiGLU-style MLP).
- MUST standardize with epsilon: z = (u - mu) / (sigma + CANON.CONST.EPS).
- MUST implement both spike definitions and label them with CANON.ENUM.SPIKE_DEF_ID:
  - CANON.ENUM.SPIKE_DEF_ID.SPIKE_ONE_SIDED_POS
  - CANON.ENUM.SPIKE_DEF_ID.SPIKE_TWO_SIDED_ABS
- MUST write Phase 1 calibration artifacts to the producing run directory under results:
  - results/CANON.OUTPUT.CALIBRATION_STATS_NPZ_BASENAME
  - results/CANON.OUTPUT.RATE_TARGETS_JSON_BASENAME
- MUST register these artifacts in run_record.json with sha256 and file size.
- MUST treat these artifacts as immutable inputs for later phases.
- SHOULD store mu and sigma in float32 for compactness; dtype MUST be logged.

## Definitions
- u[t,l,i]: activation tensor for token t, layer l, unit i (spec/12).
- mu[l,i], sigma[l,i]: calibration mean and standard deviation (Phase 1).
- z[t,l,i] = (u[t,l,i] - mu[l,i]) / (sigma[l,i] + CANON.CONST.EPS).
- Spike indicators for a given threshold tau_l(g):
  - s_pos(t,l,i) equals 1 if z[t,l,i] is strictly greater than tau_l(g), else 0.
  - s_abs(t,l,i) equals 1 if absolute value of z[t,l,i] is strictly greater than tau_l(g), else 0.

## Procedure
1. Load the deterministic calibration slice from Dataset A (spec/11).
2. Run the model with gain g = CANON.CONST.GAIN_BASELINE and capture u (spec/12).
3. Compute mu and sigma per (l,i) using streaming sums in deterministic iteration order:
   - sequences in ascending index,
   - tokens in ascending index,
   - units in ascending index.
4. Compute the baseline target rates r_star_l using threshold tau0 = CANON.CONST.TAU0_BASELINE and each spike definition.
5. Write:
   - CALIBRATION_STATS_NPZ containing arrays mu and sigma
   - RATE_TARGETS_JSON containing r_star_l for both spike definitions
6. Downstream phases compute tau_l(g) by rate matching (spec/04) and build event rasters (spec/02).

## Worked example
If a single unit has z value 3.0 at some site and tau is 2.5:
- s_pos equals 1 and s_abs equals 1.
If z value is -3.0:
- s_pos equals 0 and s_abs equals 1.

## Failure modes (detect and fix)
1. Zero or near-zero sigma.
   Detect: sigma[l,i] less than or equal to CANON.CONST.EPS for any unit.
   Fix: clamp sigma to sigma + CANON.CONST.EPS and record the count of affected units; treat them as low-variance units in analysis.
2. Calibration slice drift.
   Detect: dataset_slice_hash in run_record.json differs from previous runs under identical config.
   Fix: enforce deterministic sampling in spec/11 and pin dataset revision and tokenizer version.
3. Wrong hookpoint tensor.
   Detect: u shape mismatches expected MLP intermediate dimensionality for the model; or u values fail the hook sanity tolerance in spec/12.
   Fix: update hook mapping in spec/12 and rerun Phase 0 validation and Phase 1 calibration.

## Acceptance criteria
- The calibration stats file exists and loads with required arrays mu and sigma with shapes [L, d_ff].
- The rate targets file exists and includes r_star_l for both spike definitions.
- For a fixed input and identical run config, the computed z tensor matches within floating-point tolerance.

## Cross references
- spec/02_FORMAL_OBJECTS_EVENT_LATTICE.md
- spec/04_RATE_MATCHING_PROTOCOL.md
- spec/11_DATASETS_SAMPLING_HASHING.md
- spec/12_MODELS_HOOKS_TENSORS.md
- spec/16_TEST_PLAN.md
