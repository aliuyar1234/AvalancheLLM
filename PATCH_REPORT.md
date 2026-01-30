# PATCH REPORT

This report describes changes from v1.0.1 to v1.0.3.

## High-level changes
- Converted the pack from a partially templated scaffold into a Codex-proof SSOT for end-to-end implementation.
- Added hard artifact contracts for tables and figures and tied them to acceptance tests.
- Added a concrete implementation contract defining repo structure, CLI arguments, run id hashing, and artifact writing semantics.
- Replaced placeholder paper sections with conservative prose grounded in the normative specs.

## Files changed
### Canonical SDR
- spec/00_CANONICAL.md
  - Updated PACK_VERSION to 1.0.3.
  - Added canonical figure and table filename mappings under CANON.OUTPUT.FIG_FILE_PDF, CANON.OUTPUT.FIG_FILE_PNG, CANON.OUTPUT.TABLE_FILE_CSV, and CANON.OUTPUT.TABLE_FILE_PARQUET.
  - Added CANON.EXAMPLE keys used by worked examples.
  - Updated CLI command literals to include config path, device, dtype, and run id mode flags.

### Artifact contracts
- spec/18_RESULTS_TABLE_SKELETONS.md
  - Added exact schemas for T01_SUMMARY, T02_GENERALIZATION, T03_ARC.
  - Added required row coverage rules and acceptance checks.
- spec/19_FIGURES_TABLES_PLAN.md
  - Added per-figure data sources, filenames, producing phases, and validation gates.

### Implementation contract
- spec/23_IMPLEMENTATION_CONTRACT.md
  - Added required repo tree and module responsibilities.
  - Added CLI interface contract and exit code semantics.
  - Added config resolver, hashing rules, and run id strategy.

### Task runbook
- tasks/PHASE_5_ANALYSIS_METRICS_FIGURES.md
- tasks/PHASE_6_GENERALIZATION_TASK_EVAL.md
- tasks/PHASE_7_PAPER_CAMERA_READY.md
  - Expanded with prerequisites, outputs, DoD, acceptance tests, and failure fixes.

### Paper prose
- paper/00 through paper/08
  - Replaced placeholder text with conservative prose referencing stable figure and table IDs.

### Root docs
- README.md, SPEC.md, AGENTS.md, progress.md, QUALITY_SELF_AUDIT.md
  - Expanded to guide an autonomous agent through end-to-end implementation and release gating.

## Rationale
The changes eliminate the main drift vectors: undefined artifact schemas, unclear figure provenance, ambiguous CLI interfaces, and placeholder paper text. This enables implementation from zero without undocumented decisions.
