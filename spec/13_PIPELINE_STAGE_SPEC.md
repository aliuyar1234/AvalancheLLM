# 13 PIPELINE STAGE SPEC

## Purpose and scope
- Define the end-to-end pipeline stages, their responsibilities, and their artifact contracts.
- Eliminate implementation guessing by specifying which stage produces which canonical files.
- Ensure run immutability and provenance so every table and figure can be regenerated from run outputs.

## Normative requirements
- MUST implement phases 0 through 8 as CLI commands listed in CANON.CLI.CMD.
- MUST create a new immutable run directory for every producing phase (all except pure validation).
- MUST write:
  - CANON.OUTPUT.RUN_RECORD_JSON
  - CANON.OUTPUT.CONFIG_RESOLVED_YAML
  in every producing run directory.
- MUST never overwrite prior runs. If a run_id collision occurs, the program MUST abort unless allow_resume is explicitly enabled and the run is marked incomplete.
- MUST store stage outputs under the canonical subdirectories in CANON.OUTPUT.RUN_SUBDIR.
- MUST register every produced artifact in run_record.json with sha256 and file size.

## Definitions
- A stage is one CLI subcommand, e.g. phase3_extract_rasters.
- A producing stage writes at least one artifact under results, figs, or tables.
- A dependency is a prior run_id whose artifacts are consumed by the current stage.

## Stage contracts (required outputs)
### S00 Phase 0 Validate Environment
Purpose: verify model load, tokenizer, deterministic flags, and basic hook sanity.
Outputs:
- This stage MAY be non-producing; if it produces a run directory, it MUST still write run_record.json.

### S01 Phase 1 Calibrate
Purpose: compute calibration stats and baseline rate targets on Dataset A at baseline gain.
Required results:
- results/CANON.OUTPUT.CALIBRATION_STATS_NPZ_BASENAME
- results/CANON.OUTPUT.RATE_TARGETS_JSON_BASENAME

### S02 Phase 2 Gain Grid
Purpose: compute branching metrics across gains on Dataset A (rate matched).
Required results:
- results/CANON.OUTPUT.METRICS_PARQUET_BASENAME

### S02b Phase 2 Select gstar
Purpose: select gstar solely from branching metrics on Dataset A.
Required results:
- results/CANON.OUTPUT.GSTAR_JSON_BASENAME

### S03 Phase 3 Extract Rasters
Purpose: extract real rasters on Dataset A for baseline gain and gstar.
Required results:
- results/CANON.OUTPUT.RASTER_NPZ_BASENAME
This NPZ MUST follow the schema in spec/02 with shapes [C,N,L,T].

### S04 Phase 4 Run Nulls
Purpose: compute tau_l(g) tables and generate null rasters.
Required results:
- results/CANON.OUTPUT.TAU_RATE_MATCHED_PARQUET_BASENAME
- results/CANON.OUTPUT.NULL_NPZ_BASENAME

### S05 Phase 5 Analyze and Export
Purpose: compute avalanche statistics, branching curves, signatures, and export paper tables and figures.
Required results:
- results/CANON.OUTPUT.METRICS_PARQUET_BASENAME
- results/CANON.OUTPUT.AVALANCHES_PARQUET_BASENAME
Required tables:
- CANON.OUTPUT.TABLE_FILE_CSV.T01_SUMMARY and CANON.OUTPUT.TABLE_FILE_PARQUET.T01_SUMMARY
 - CANON.OUTPUT.TABLE_FILE_CSV.T04_TAIL_FITS and CANON.OUTPUT.TABLE_FILE_PARQUET.T04_TAIL_FITS
 - CANON.OUTPUT.TABLE_FILE_CSV.T05_CRACKLING_DIAGNOSTICS and CANON.OUTPUT.TABLE_FILE_PARQUET.T05_CRACKLING_DIAGNOSTICS
 - CANON.OUTPUT.TABLE_FILE_CSV.T06_ABLATIONS and CANON.OUTPUT.TABLE_FILE_PARQUET.T06_ABLATIONS
Required figures:
- F01, F02, F03, F04, F05, F08 per CANON.OUTPUT.FIG_FILE_PDF and FIG_FILE_PNG
 - Appendix figures F09, F10, F11 per CANON.OUTPUT.FIG_FILE_PDF and FIG_FILE_PNG

### S06 Phase 6 Generalization and ARC
Purpose: evaluate g = baseline and g = gstar on Dataset B and ARC MCQ.
Required tables:
- CANON.OUTPUT.TABLE_FILE_CSV.T02_GENERALIZATION and CANON.OUTPUT.TABLE_FILE_PARQUET.T02_GENERALIZATION
- CANON.OUTPUT.TABLE_FILE_CSV.T03_ARC and CANON.OUTPUT.TABLE_FILE_PARQUET.T03_ARC
Required figures:
- F06_GENERALIZATION_B and F07_ARC_MCQ.

### S07 Phase 7 Paper Export
Purpose: materialize paper markdown with claim to evidence mapping using actual run ids and artifact hashes.
Required outputs:
- paper directory files updated under the producing run directory or exported into CANON.PATH.PAPER_DIR as configured.
 - tables/appendix/table_T07_REPLICATION_SUMMARY.csv and tables/appendix/table_T07_REPLICATION_SUMMARY.parquet (when both BASE and INSTRUCT model-role runs are available)

### S08 Phase 8 Release
Purpose: regenerate MANIFEST.sha256 and create the distributable zip.
Required outputs at pack root:
- MANIFEST.sha256 updated to match all files.

## Worked example
Example run ordering:
1. Run CANON.CLI.CMD.PHASE1_CALIBRATE to produce a run directory containing calibration_stats and rate_targets.
2. Run CANON.CLI.CMD.PHASE2_GAIN_GRID and then PHASE2_SELECT_GSTAR to produce gstar.
3. Run PHASE3_EXTRACT_RASTERS to produce rasters_token_layer.npz.
4. Run PHASE4_RUN_NULLS to produce rasters_nulls.npz and tau_rate_matched.parquet.
5. Run PHASE5_ANALYZE_AND_EXPORT to export T01 and figures F01-F05 and F08.

## Failure modes (detect and fix)
1. Missing stage outputs.
   Detect: expected canonical basenames not present in results or tables or figs.
   Fix: ensure stage writes required files using CANON.OUTPUT keys and registers them in run_record.json.
2. Run overwrite.
   Detect: attempting to reuse an existing run directory.
   Fix: use content_hash run_id strategy and abort on collision.
3. Dependency mismatch.
   Detect: downstream stage consumes artifacts whose config_hash does not match recorded dependency.
   Fix: enforce dependency recording and hash checking in run loader.

## Acceptance criteria
- Running stages in order yields all required tables and figures with canonical filenames.
- Every produced artifact listed in this spec is present in run_record.json artifact_index with sha256.

## Cross references
- tasks/TASK_INDEX.md
- spec/02_FORMAL_OBJECTS_EVENT_LATTICE.md
- spec/14_RUN_RECORD_SCHEMA.md
- spec/18_RESULTS_TABLE_SKELETONS.md
- spec/19_FIGURES_TABLES_PLAN.md
