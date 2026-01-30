# 14 RUN RECORD SCHEMA

## Purpose and scope
- Define run_record.json minimal schema and required fields for reproducibility and audit.

## Normative requirements
- MUST write run_record.json into each run directory.
- MUST validate against spec/run_record.schema.json.
- MUST include hashes for config, dataset slice, model revision, and code version.

## Definitions
- run_record.json is the authoritative provenance artifact.
- conditions is list of evaluated condition ids.
- artifacts is mapping from logical name to relative paths.

## Procedure
Procedure:
1. At CLI start, generate run_id and create run directory.
2. Resolve config and compute config_sha256.
3. Record python, torch, and transformers versions (transformers should match CANON.CONST.TRANSFORMERS_VERSION_PIN for reproducibility).
4. Load model and record HF id and revision string.
5. Load dataset slice and compute dataset_slice_sha256.
6. Write run_record.json at end including artifact paths and metric summaries.
7. Validate schema and hard fail if invalid.

## Worked example
Example: runs/RUN_20260128_0002/run_record.json includes phase_id PHASE2_GAIN_GRID and artifacts.metrics_parquet set to results/RUN_20260128_0002/metrics.parquet.

## Failure modes
1. Missing hash.
   Detect: schema validation fails.
   Fix: compute missing hash and rerun.
2. Non deterministic timestamps.
   Detect: run_id includes local time.
   Fix: use UTC only.
3. Overwriting run_record.
   Detect: file exists.
   Fix: new run_id and never overwrite.

## Acceptance criteria
- run_record validates against JSON schema.
- All required hashes present.

## Cross references
- spec/run_record.schema.json
- spec/15 logging
