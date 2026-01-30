# PHASE 6 GENERALIZATION TASK EVAL

## Purpose
Phase 6 evaluates whether the gain value gstar calibrated mechanistically on Dataset A transfers to Dataset B and to ARC multiple-choice evaluation, without retuning.

This phase MUST:
- Compare g=1 versus gstar on Dataset B mechanistic metrics and task proxy metrics.
- Compare g=1 versus gstar on ARC multiple-choice accuracy.
- Record uncertainty using bootstrap confidence intervals.

## Prerequisites
- Phase 2 gstar selection completed and gstar.json is available.
- Phase 5 analysis completed and produced Dataset A summary table and figures.
- Calibration statistics (mu and sigma) and rate-matched thresholds are available for the chosen spike definitions.

## Commands
Execute all commands (INSTRUCT then BASE):

- CANON.CLI.CMD.PHASE6_GENERALIZE_B_METRICS
- CANON.CLI.CMD.PHASE6_ARC_MCQ_EVAL
- CANON.CLI.CMD.PHASE6_GENERALIZE_B_METRICS_BASE
- CANON.CLI.CMD.PHASE6_ARC_MCQ_EVAL_BASE

Each command MUST create its own run directory under runs and MUST write a run_record.json.

## Expected outputs

### Dataset B mechanistic and proxy evaluation run
A new run directory MUST be created for Dataset B evaluation. Inside it:

- run_record.json
- config_resolved.yaml
- results/metrics.parquet
- tables/table_T02_GENERALIZATION.csv
- tables/table_T02_GENERALIZATION.parquet
- figs/fig_F06_GENERALIZATION_B.pdf and figs/fig_F06_GENERALIZATION_B.png

### ARC multiple-choice evaluation run
A new run directory MUST be created for ARC evaluation. Inside it:

- run_record.json
- config_resolved.yaml
- tables/table_T03_ARC.csv
- tables/table_T03_ARC.parquet
- figs/fig_F07_ARC_MCQ.pdf and figs/fig_F07_ARC_MCQ.png

## Definition of Done
All of the following are true:
- T02_GENERALIZATION schema and row coverage pass spec/18.
- T03_ARC schema and row coverage pass spec/18.
- Figures F06 and F07 pass spec/19.
- run_record.json artifact_index contains all required artifacts with sha256 hashes.

## Acceptance tests
1. Check that each run directory exists and contains run_record.json and config_resolved.yaml.
2. Validate the required tables against spec/18.
3. Validate the required figures against spec/19.
4. Check the generalization comparison:
   - There is one row for g1 and one row for gstar per configured spike_def_id and target_rate in T02.
   - There is one row for g1 and one row for gstar in T03 for the configured evaluation mode.

## Common failures and fixes
1. ARC prompt formatting mismatch.
   Detect: accuracy near random and mean logprob margin near zero across all questions.
   Fix: ensure the ARC prompt template referenced by CANON.DATASET.ARC_MCQ.PROMPT_TEMPLATE_ID is used exactly and logged.
2. Dataset B slice drift.
   Detect: dataset slice hash changes across runs.
   Fix: enforce deterministic sampling and hashing in spec/11 and record the slice hash.
3. Missing gstar.
   Detect: Phase 6 cannot find gstar.json.
   Fix: run Phase 2 selection or point the config dependency to the correct run.

## Cross references
- spec/08_GAIN_INTERVENTION_AND_GSTAR.md
- spec/11_DATASETS_SAMPLING_HASHING.md
- spec/18_RESULTS_TABLE_SKELETONS.md
- spec/19_FIGURES_TABLES_PLAN.md
