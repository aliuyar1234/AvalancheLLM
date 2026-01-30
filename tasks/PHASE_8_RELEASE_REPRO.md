# Phase 8 Release and Repro

## Prerequisites
All tests pass.

## Commands
Run CANON.CLI.CMD.PHASE8_RELEASE

## Expected outputs
MANIFEST.sha256 and release zip

## Definition of Done
Manifest regenerated and zip created.

## Acceptance tests
Manifest hashes match recomputation.

## Common failures
- See spec/16 for failure modes.
- If a stage fails, do not overwrite; create a new run_id.
