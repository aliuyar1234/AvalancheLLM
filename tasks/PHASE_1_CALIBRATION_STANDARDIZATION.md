# PHASE 1 CALIBRATION STANDARDIZATION

## Purpose
Phase 1 computes and freezes the calibration statistics and baseline rate targets used everywhere else:
- mu and sigma for standardization (spec/03)
- baseline per-layer target rates r_star_l for rate matching (spec/04)

This phase is inference-only and MUST NOT train or update weights.

## Prerequisites
- Phase 0 environment validation passed.
- Model weights and tokenizer load successfully for the chosen model role.
- Dataset A is accessible and deterministic sampling rules are configured (spec/11).

## Dependency graph
- Inputs:
  - configs/models.yaml
  - configs/datasets.yaml
  - configs/pipeline.yaml
- Produces artifacts consumed by:
  - Phase 2 (gain grid)
  - Phase 3 (raster extraction)
  - Phase 4 (nulls and rate matching)
  - Phase 5 (analysis and figure export)

## Commands
Run both commands (INSTRUCT then BASE) to enable model-replication results later:
- CANON.CLI.CMD.PHASE1_CALIBRATE
- CANON.CLI.CMD.PHASE1_CALIBRATE_BASE

## Expected outputs (paths are relative to the new run directory)
A new immutable run directory under CANON.PATH.RUNS_DIR is created. It MUST contain:

Run metadata:
- CANON.OUTPUT.RUN_RECORD_JSON
- CANON.OUTPUT.CONFIG_RESOLVED_YAML

Results artifacts under CANON.OUTPUT.RUN_SUBDIR.RESULTS:
- results/CANON.OUTPUT.CALIBRATION_STATS_NPZ_BASENAME
- results/CANON.OUTPUT.RATE_TARGETS_JSON_BASENAME

Logs under CANON.OUTPUT.RUN_SUBDIR.LOGS:
- logs/phase1_calibrate.jsonl

## Definition of Done
- Calibration stats and baseline rate targets exist and are registered in run_record.json.
- Dataset slice hash is recorded and stable.
- sigma clamp count and any NaN checks are logged.

## Acceptance tests
1. File existence:
   - results/CANON.OUTPUT.CALIBRATION_STATS_NPZ_BASENAME exists
   - results/CANON.OUTPUT.RATE_TARGETS_JSON_BASENAME exists
2. Schema sanity:
   - calibration_stats contains arrays mu and sigma with shape [L, d_ff]
   - rate_targets contains r_star_l for both spike defs
3. Determinism spot check:
   - re-run Phase 1 with identical config and confirm identical sha256 for both results files

## Common failures and fixes
1. Missing calibration outputs.
   Detect: expected files do not exist.
   Fix: implement Phase 1 writer using the basenames in CANON.OUTPUT and register artifacts in run_record.json.
2. sigma contains zeros or NaNs.
   Detect: sigma below EPS or NaN count nonzero.
   Fix: clamp sigma with CANON.CONST.EPS and verify hook correctness (spec/12).
3. Dataset slice hash changes unexpectedly.
   Detect: slice hash differs across runs with identical config.
   Fix: implement deterministic sampling exactly as spec/11.

## Cross references
- spec/03_SPIKE_DEFS_AND_STANDARDIZATION.md
- spec/04_RATE_MATCHING_PROTOCOL.md
- spec/11_DATASETS_SAMPLING_HASHING.md
- spec/12_MODELS_HOOKS_TENSORS.md
- spec/16_TEST_PLAN.md
