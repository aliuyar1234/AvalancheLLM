# PHASE 3 EXTRACTION RASTERS

## Purpose
Phase 3 runs inference and extracts the token by layer event rasters for Dataset A under the selected conditions:
- real rasters at g = CANON.CONST.GAIN_BASELINE and g = gstar
- for both spike definitions and target rates configured

This phase writes the primary raster NPZ used by later analysis and null generation.

## Prerequisites
- Phase 1 produced calibration stats and baseline rate targets.
- Phase 2 produced gstar in results/CANON.OUTPUT.GSTAR_JSON_BASENAME.

## Dependency graph
- Inputs:
  - Phase 1 calibration run artifacts
  - Phase 2 gstar artifact
- Produces:
  - rasters consumed by Phase 4 and Phase 5

## Commands
Run both commands (INSTRUCT then BASE):
- CANON.CLI.CMD.PHASE3_EXTRACT_RASTERS
- CANON.CLI.CMD.PHASE3_EXTRACT_RASTERS_BASE

## Expected outputs
A new run directory is created. It MUST contain:

Run metadata:
- CANON.OUTPUT.RUN_RECORD_JSON
- CANON.OUTPUT.CONFIG_RESOLVED_YAML

Results under results:
- results/CANON.OUTPUT.RASTER_NPZ_BASENAME

NPZ schema MUST match spec/02:
- required keys: X, A, cond_id, seq_id using CANON.ENUM.NPZ_KEY
- X and A shapes are [C, N, L, T]
- cond_id length C, seq_id length N

## Definition of Done
- Raster NPZ exists, schema valid, and registered in run_record.json.
- Condition mapping for each cond_id is recorded into results/metrics.parquet or equivalent per spec/02.

## Acceptance tests
1. NPZ exists and loads.
2. Required NPZ keys exist:
   - CANON.ENUM.NPZ_KEY.X_OCCUPANCY
   - CANON.ENUM.NPZ_KEY.A_COUNT
   - CANON.ENUM.NPZ_KEY.COND_ID
   - CANON.ENUM.NPZ_KEY.SEQ_ID
3. Shapes match:
   - X.ndim equals 4 and axis order is [C,N,L,T]
   - A.ndim equals 4 and matches X shape
4. Hashes are logged:
   - run_record.json artifact_index includes the NPZ with sha256

## Common failures and fixes
1. Missing gstar dependency.
   Detect: Phase 3 cannot find gstar.json.
   Fix: run Phase 2 selection, then rerun Phase 3.
2. Wrong NPZ axis order.
   Detect: shapes do not match [C,N,L,T].
   Fix: enforce spec/02 schema and write a unit test on load.
3. Missing keys in NPZ.
   Detect: downstream phases cannot load X or A.
   Fix: always write both X and A plus cond_id and seq_id.

## Cross references
- spec/02_FORMAL_OBJECTS_EVENT_LATTICE.md
- spec/03_SPIKE_DEFS_AND_STANDARDIZATION.md
- spec/08_GAIN_INTERVENTION_AND_GSTAR.md
- spec/12_MODELS_HOOKS_TENSORS.md
