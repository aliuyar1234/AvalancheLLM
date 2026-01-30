# PHASE 4 NULLS AND RATE MATCHING CONTROLS

## Purpose
Phase 4 generates the required null controls and ensures rate matching holds under each gain:
- compute and persist tau_l(g) for each gain, layer, spike definition, and target rate
- generate within-layer time permutation null rasters (marginals-preserving)
- generate within-layer time circular-shift null rasters (marginals-preserving, autocorrelation-preserving)
- generate input token shuffle null rasters (model-level)

This phase is critical for reviewer-proof controls.

## Prerequisites
- Phase 3 produced real rasters in results/CANON.OUTPUT.RASTER_NPZ_BASENAME.
- Phase 1 produced baseline rate targets and calibration stats.

## Dependency graph
- Inputs:
  - Phase 1 calibration outputs
  - Phase 3 raster NPZ
- Produces:
  - null NPZ consumed by Phase 5 analysis

## Commands
Run both commands (INSTRUCT then BASE):
- CANON.CLI.CMD.PHASE4_RUN_NULLS
- CANON.CLI.CMD.PHASE4_RUN_NULLS_BASE

## Expected outputs
A new run directory is created. It MUST contain:

Run metadata:
- CANON.OUTPUT.RUN_RECORD_JSON
- CANON.OUTPUT.CONFIG_RESOLVED_YAML

Results under results:
- results/CANON.OUTPUT.TAU_RATE_MATCHED_PARQUET_BASENAME
- results/CANON.OUTPUT.NULL_NPZ_BASENAME

The null NPZ MUST follow spec/02 schema with shapes [C,N,L,T] and required keys.

## Definition of Done
- tau table exists and passes the rate-matching tolerance check for every condition.
- within-layer permutation null passes the exact marginals preservation check.
- within-layer circular-shift null passes the exact marginals preservation check.
- input token shuffle null is generated and recorded as a distinct null_id condition.

## Acceptance tests
1. Rate matching:
   - compute max_abs_rate_error across all tau rows and confirm it is less than or equal to CANON.CONST.RATE_MATCH_TOL_ABS
2. Within-layer marginals preservation:
   - for every (cond, seq, layer), sorted A over tokens is identical before and after permutation
3. Within-layer circular-shift marginals preservation:
   - for every (cond, seq, layer), sorted A over tokens is identical before and after circular shift
4. Null NPZ schema:
   - required keys exist and shapes follow [C,N,L,T]
5. Provenance:
   - run_record.json artifact_index includes tau parquet and null NPZ with sha256

## Common failures and fixes
1. Marginals not preserved.
   Detect: mismatch in sorted per-layer token counts.
   Fix: ensure permutation is applied independently within each layer row and only on token axis.
2. Rate matching fails.
   Detect: rate error exceeds tolerance.
   Fix: recompute tau_l(g) per spec/04 and verify calibration slice order.
3. Token shuffle breaks special tokens.
   Detect: tokenizer errors or extremely short sequences.
   Fix: keep special tokens fixed and permute only interior tokens.

## Cross references
- spec/04_RATE_MATCHING_PROTOCOL.md
- spec/07_NULL_MODELS_AND_CONTROLS.md
- spec/02_FORMAL_OBJECTS_EVENT_LATTICE.md
- spec/16_TEST_PLAN.md
