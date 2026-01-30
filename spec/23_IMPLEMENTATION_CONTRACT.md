# 23 IMPLEMENTATION CONTRACT

## Purpose and scope
This document is the engineering contract for implementing the full project as a deterministic, reproducible, end-to-end pipeline. It is written so Codex can build the repository from zero without guessing.

Scope:
- The required Python package structure and module responsibilities.
- CLI interfaces for phases 0 through 8, matching CANON.CLI.CMD.
- Configuration resolution for CANON references and resolved-config hashing.
- Run directory layout, immutability rules, and artifact registration.
- Minimum public interfaces that other modules may rely on.

## Normative requirements
- MUST implement a Python package named avalanche_llm.
- MUST implement all CLI commands listed in CANON.CLI.CMD exactly.
- MUST ensure that every CLI execution produces a new immutable run directory unless it is a pure validation command.
- MUST write run_record.json in every producing run directory using the schema in spec/14_RUN_RECORD_SCHEMA.md.
- MUST never overwrite an existing run directory. If a computed run_id already exists, the program MUST abort with a clear error unless the user passes an explicit allow_resume flag.
- MUST resolve CANON references in YAML configs deterministically and write the resolved config to CANON.OUTPUT.CONFIG_RESOLVED_YAML in the run directory.
- MUST treat model identifiers in config (e.g., models.*.hf_id) as either a Hugging Face repo id or a local model directory path.
- MUST not silently offload model weights to CPU when --device is cuda; if the model cannot be placed fully on GPU, abort (fail-closed) with a clear error.
- SHOULD pin key dependencies for reproducibility (notably transformers to CANON.CONST.TRANSFORMERS_VERSION_PIN).
- MUST record the resolved versions of python, torch, and transformers in run_record.json for every producing run.
- MUST log sufficient information to reproduce all tables and figures from run artifacts without re-running the model.
- SHOULD keep GPU usage within the budget in CANON.CONST and configs/budgets.yaml by using inference-only passes and small slices.
- MAY support additional models and datasets, but defaults MUST remain those in CANON.

## Definitions
Canonical registry:
- spec/00_CANONICAL.md is the authoritative CANON registry.

Run directory:
- A run directory is a folder under CANON.PATH.RUNS_DIR whose name begins with RUN_.
- Each run directory contains:
  - run_record.json
  - config_resolved.yaml
  - results subdirectory with parquet outputs
  - figs subdirectory with PDF and PNG exports
  - tables subdirectory with CSV and Parquet tables
  - logs subdirectory with JSONL event logs

Run id strategy:
- content_hash mode: compute a deterministic run_id from the resolved config and the stage id.
- The run_id string format MUST be:
  - RUN_{STAGE_TAG}_{HASH12}
Where:
- STAGE_TAG is a short token such as S01, S05, S06B, S06ARC, S07, S08.
- HASH12 is the first 12 hex characters of sha256(resolved_config_bytes concatenated with stage_tag bytes).

Model hooks:
- Model tensor semantics are defined in spec/12_MODELS_HOOKS_TENSORS.md.
- Gain intervention semantics are defined in spec/08_GAIN_INTERVENTION_AND_GSTAR.md.

## Required repository tree
Codex MUST create a repository that includes at least the following paths:

- avalanche_llm
  - __init__.py
  - __main__.py
  - canon.py
  - cli.py
  - config.py
  - run_id.py
  - io
    - __init__.py
    - hashing.py
    - artifacts.py
    - schema_validate.py
    - jsonl.py
  - model
    - __init__.py
    - loader.py
    - hooks.py
    - gain.py
    - forward.py
  - events
    - __init__.py
    - standardize.py
    - spike.py
    - rate_match.py
  - raster
    - __init__.py
    - build.py
    - cc.py
    - nulls.py
  - metrics
    - __init__.py
    - branching.py
    - avalanches.py
    - signatures.py
    - stats.py
  - analysis
    - __init__.py
    - summarize.py
  - plotting
    - __init__.py
    - figures.py
    - tables.py
  - paper
    - __init__.py
    - export.py
  - tests
    - __init__.py
    - test_toy_raster.py
    - test_rate_match.py
    - test_null_marginals.py
    - test_run_record_schema.py

## CLI interface contract
The CLI entrypoint is CANON.CLI.ENTRYPOINT.

Each command under CANON.CLI.CMD MUST be implemented as a subcommand. Each subcommand MUST accept at minimum:

Common arguments:
- --config: path to the pipeline YAML config (default CANON.PATH.CONFIG_PIPELINE)
- --device: device string such as cuda or cpu (default set in CANON.CLI.CMD strings)
- --dtype: bf16 or fp16 or fp32 (default set in CANON.CLI.CMD strings)
- --run_id_mode: one of content_hash or timestamp_counter (default set in CANON.CLI.CMD strings)
- --allow_resume: boolean; if true, allow resuming an existing run directory only if run_record.json indicates incomplete stage
- --dry_run: boolean; if true, perform resolution and validation but do not run model passes

Phase-specific arguments:
- Phase 2 selection MUST allow --gstar_method set to btot_near_one and MUST default to that.
- Phase 6 ARC MUST allow --arc_split and --arc_config, but defaults MUST come from CANON.DATASET.ARC_MCQ.

Exit codes:
- Return code 0 on success.
- Return code 2 on schema or determinism failures.
- Return code 3 on missing dependency artifacts.
- Return code 4 on existing run directory when allow_resume is false.

## Configuration resolution and hashing
The config resolver MUST:
1. Load YAML from the path passed to --config.
2. Replace any scalar string that begins with CANON. with the value found in the parsed CANON registry in spec/00.
3. For lists, recursively resolve each element.
4. For dictionaries, recursively resolve values.
5. After resolution, serialize the resolved config in canonical form:
   - UTF-8
   - YAML with sorted keys disabled
   - newlines normalized to LF
6. Compute config_hash as sha256 of the resolved config bytes.
7. Write resolved config bytes to CANON.OUTPUT.CONFIG_RESOLVED_YAML in the run directory.

Acceptance test:
- Resolving the same config twice yields identical resolved bytes and identical config_hash.

## Artifact writer and run_record.json
Every producing phase MUST:
1. Create a new run directory according to run id strategy.
2. Write run_record.json early with stage metadata and a stage_status field set to started.
3. Write artifacts into subdirectories:
   - results for parquet
   - figs for figures
   - tables for tables
   - logs for JSONL logs
4. After each artifact is written, compute sha256 and update artifact_index in run_record.json.
5. At the end, set stage_status to complete and write run_record.json again.

The artifact_index entries MUST include:
- relative_path
- sha256
- bytes
- created_utc

## Determinism contract
The implementation MUST support a determinism smoke test described in spec/16_TEST_PLAN.md.

Minimum requirements:
- Set PYTHONHASHSEED.
- Set random, numpy, and torch seeds.
- For torch, set deterministic algorithms where supported and record the flags in run_record.json.
- When deterministic kernels are unavailable, record that fact in run_record.json and continue if the config allows.

## Worked example
Goal: Phase 5 analysis produces the summary table and branching curve figure.

Using the example run directory in CANON.EXAMPLE.RUN_DIR_S05, the following artifacts MUST be present:
- results/metrics.parquet
- tables/table_T01_SUMMARY.csv
- figs/fig_F03_BRANCHING_CURVES.pdf

The run_record.json artifact_index MUST contain entries for each of the above with sha256 hashes.

## Failure modes
1. CANON reference not resolved.
   Detect: config resolver leaves a CANON. prefix string in resolved config.
   Fix: implement recursive resolution and add a hard-fail check that no scalar string begins with CANON. after resolution.
2. Wrong model hook tensor.
   Detect: u tensor does not match expected shape (tokens, hidden) or per-unit stats explode.
   Fix: follow spec/12 and validate hook names on a small forward pass, logging activation statistics.
3. Run overwrite.
   Detect: attempting to create a run directory that already exists.
   Fix: abort unless allow_resume is true, and ensure allow_resume only resumes incomplete stages.
4. Non-deterministic sampling.
   Detect: dataset slice hash changes across runs with same config.
   Fix: enforce deterministic index generation and record indices hash in run_record.json.

## Acceptance criteria
- Running Phase 0 validates environment and produces a success message without creating a run directory.
- Running Phases 1 through 7 each creates a new run directory and writes run_record.json that validates against the schema.
- Tables and figures match the filenames in CANON.OUTPUT mappings and validate against spec/18 and spec/19.
- The pack can be executed end-to-end following tasks/TASK_INDEX.md without requiring any undocumented decisions.

## Cross references
- spec/00_CANONICAL.md for canonical IDs and CLI strings
- spec/12_MODELS_HOOKS_TENSORS.md for hook semantics
- spec/18_RESULTS_TABLE_SKELETONS.md for table schemas
- spec/19_FIGURES_TABLES_PLAN.md for figure outputs
- spec/14_RUN_RECORD_SCHEMA.md for schema details
