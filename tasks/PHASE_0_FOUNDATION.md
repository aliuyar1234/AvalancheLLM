# Phase 0 Foundation

## Prerequisites
Python env, GPU drivers, huggingface auth if needed.
- transformers pinned to CANON.CONST.TRANSFORMERS_VERSION_PIN.
- Ensure model weights remain fully on GPU when --device cuda is used (no CPU offload).

## Commands
Run CANON.CLI.CMD.PHASE0_VALIDATE_ENV

## Expected outputs
runs/RUN_ID/run_record.json and logs/RUN_ID/events.jsonl

## Definition of Done
Env validated, versions logged, determinism flags set.

## Acceptance tests
CUDA visible, torch deterministic flags set.
- transformers version matches CANON.CONST.TRANSFORMERS_VERSION_PIN.

## Common failures
- See spec/16 for failure modes.
- If a stage fails, do not overwrite; create a new run_id.
