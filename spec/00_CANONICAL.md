# 00 CANONICAL

This file is the Single Definition Rule authority. Every identifier, constant, enum value, canonical path, CLI command literal, and canonical output filename used anywhere else in the pack MUST be defined here.

## CANON registry

```yaml
CANON:
  PROJECT:
    PROJECT_ID: avalanche_llm_iclr
    PACK_NAME: avalanche_llm_iclr_pack_v1_0_4
    PACK_VERSION: 1.0.4
    BUILD_DATE: '2026-01-29'
    LICENSE_NOTE: No weights included; users must accept model and dataset licenses.
  PATH:
    ROOT: avalanche_llm_iclr
    RUNS_DIR: runs
    LOGS_DIR: logs
    FIGS_DIR: figs
    TABLES_DIR: tables
    RESULTS_DIR: results
    CACHE_DIR: cache
    PAPER_DIR: paper
    CONFIGS_DIR: configs
    SPEC_DIR: spec
    TASKS_DIR: tasks
    BIB_DIR: bib
    CHECKLISTS_DIR: checklists
    CONFIG_PIPELINE: configs/pipeline.yaml
  CLI:
    ENTRYPOINT: python -m avalanche_llm
    CMD:
      PHASE0_VALIDATE_ENV: python -m avalanche_llm phase0_validate_env --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash
      PHASE1_CALIBRATE: python -m avalanche_llm phase1_calibrate --config configs/pipeline.yaml --device
        cuda --dtype bf16 --run_id_mode content_hash
      PHASE2_GAIN_GRID: python -m avalanche_llm phase2_gain_grid --config configs/pipeline.yaml --device
        cuda --dtype bf16 --run_id_mode content_hash
      PHASE2_SELECT_GSTAR: python -m avalanche_llm phase2_select_gstar --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash
      PHASE3_EXTRACT_RASTERS: python -m avalanche_llm phase3_extract_rasters --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash
      PHASE4_RUN_NULLS: python -m avalanche_llm phase4_run_nulls --config configs/pipeline.yaml --device
        cuda --dtype bf16 --run_id_mode content_hash
      PHASE5_ANALYZE_AND_EXPORT: python -m avalanche_llm phase5_analyze_and_export --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash
      PHASE6_GENERALIZE_B_METRICS: python -m avalanche_llm phase6_generalize_b_metrics --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash
      PHASE6_ARC_MCQ_EVAL: python -m avalanche_llm phase6_arc_mcq_eval --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash
      PHASE7_PAPER_EXPORT: python -m avalanche_llm phase7_paper_export --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash
      PHASE8_RELEASE: python -m avalanche_llm phase8_release --config configs/pipeline.yaml --device cuda
        --dtype bf16 --run_id_mode content_hash
      PHASE1_CALIBRATE_BASE: python -m avalanche_llm phase1_calibrate --config configs/pipeline.yaml --device
        cuda --dtype bf16 --run_id_mode content_hash --model_role BASE
      PHASE2_GAIN_GRID_BASE: python -m avalanche_llm phase2_gain_grid --config configs/pipeline.yaml --device
        cuda --dtype bf16 --run_id_mode content_hash --model_role BASE
      PHASE2_SELECT_GSTAR_BASE: python -m avalanche_llm phase2_select_gstar --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash --model_role BASE
      PHASE3_EXTRACT_RASTERS_BASE: python -m avalanche_llm phase3_extract_rasters --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash --model_role BASE
      PHASE4_RUN_NULLS_BASE: python -m avalanche_llm phase4_run_nulls --config configs/pipeline.yaml --device
        cuda --dtype bf16 --run_id_mode content_hash --model_role BASE
      PHASE5_ANALYZE_AND_EXPORT_BASE: python -m avalanche_llm phase5_analyze_and_export --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash --model_role BASE
      PHASE6_GENERALIZE_B_METRICS_BASE: python -m avalanche_llm phase6_generalize_b_metrics --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash --model_role BASE
      PHASE6_ARC_MCQ_EVAL_BASE: python -m avalanche_llm phase6_arc_mcq_eval --config configs/pipeline.yaml
        --device cuda --dtype bf16 --run_id_mode content_hash --model_role BASE
  ENUM:
    DATASET_ROLE:
      A: A
      B: B
      ARC_MCQ: ARC_MCQ
    MODEL_ROLE:
      BASE: BASE
      INSTRUCT: INSTRUCT
    SPIKE_DEF_ID:
      SPIKE_ONE_SIDED_POS: SPIKE_ONE_SIDED_POS
      SPIKE_TWO_SIDED_ABS: SPIKE_TWO_SIDED_ABS
    NULL_ID:
      NULL_TOKEN_SHUFFLE_INPUT: NULL_TOKEN_SHUFFLE_INPUT
      NULL_RASTER_WITHIN_LAYER_TIME_PERM: NULL_RASTER_WITHIN_LAYER_TIME_PERM
      NULL_RASTER_WITHIN_LAYER_TIME_CIRC_SHIFT: NULL_RASTER_WITHIN_LAYER_TIME_CIRC_SHIFT
      NULL_NONE: NULL_NONE
    SIGNATURE_ID:
      S1_BRANCHING_CROSSING: S1_BRANCHING_CROSSING
      S2_SUSCEPTIBILITY_PEAK: S2_SUSCEPTIBILITY_PEAK
      S3_CRACKLING_RELATION: S3_CRACKLING_RELATION
    ADJACENCY_ID:
      ADJ_4N: ADJ_4N
      ADJ_8N: ADJ_8N
    GAIN_TARGET:
      MLP: MLP
      ATTN: ATTN
    INTERVENTION_ID:
      MLP_GLOBAL: MLP_GLOBAL
      ATTN_GLOBAL: ATTN_GLOBAL
      MLP_BAND_EARLY: MLP_BAND_EARLY
      MLP_BAND_MID: MLP_BAND_MID
      MLP_BAND_LATE: MLP_BAND_LATE
    NPZ_KEY:
      X_OCCUPANCY: X
      A_COUNT: A
      COND_ID: cond_id
      SEQ_ID: seq_id
  CONST:
    SEQ_LEN_TOKENS: 256
    EPS: 1.0e-06
    TAU0_BASELINE: 2.5
    TARGET_LAYER_RATES:
    - 1.0e-05
    - 2.0e-05
    - 4.0e-05
    - 8.0e-05
    GAIN_GRID_DEFAULT:
    - 0.7
    - 0.8
    - 0.85
    - 0.9
    - 0.95
    - 1.0
    - 1.05
    - 1.1
    - 1.15
    - 1.2
    - 1.3
    N_WINDOWS_A_CALIBRATION: 64
    N_WINDOWS_A_RASTERS: 128
    N_WINDOWS_B_EVAL: 96
    WINDOW_POOL_FACTOR: 10
    BOOTSTRAP_REPS: 500
    BOOTSTRAP_SEED: 1337
    RATE_MATCH_TOL_ABS: 0.0005
    CC_MAX_COMPONENTS_PER_SEQ_HARDFAIL: 20000
    CHI_EPS: 1.0e-08
    CRACKLING_MIN_POINTS: 8
    CRACKLING_D_RANGE_MIN: 3
    CRACKLING_D_RANGE_MAX: 64
    GAIN_BASELINE: 1.0
    GPU_HOURS_MAX: 24
    RATE_MATCH_HIST_BINS: 4096
    RATE_MATCH_EDGE_ROUND_ABS: 0.001
    RATE_MATCH_MIN_CAL_SAMPLES: 20000
    SIGNATURE_ALIGNMENT_G_DIFF_MAX: 0.2
    BOOTSTRAP_CI_WIDTH_MAX: 0.5
    TAIL_FIT_XMIN_PERCENTILE: 0.8
    TAIL_FIT_MIN_TAIL_SAMPLES: 100
    U_HOOK_MATCH_TOL_ABS: 1.0e-05
    TRANSFORMERS_VERSION_PIN: '5.0.0'
    CUBLAS_WORKSPACE_CONFIG: ':4096:8'
  DATASET:
    A:
      HF_ID: Salesforce/wikitext
      HF_CONFIG: wikitext-103-v1
      SPLIT: validation
    B:
      HF_ID: allenai/c4
      HF_CONFIG: en
      SPLIT: validation
    ARC_MCQ:
      HF_ID: allenai/ai2_arc
      HF_CONFIG: ARC-Challenge
      SPLIT: test
      PROMPT_TEMPLATE_ID: ARC_MCQ_V1
      PROMPT_TEMPLATES:
        ARC_MCQ_V1: |
          Question: {question}
          Choices:
          {choices}
          Answer:
  MODEL:
    QWEN25_7B_INSTRUCT_LOCAL:
      HF_ID: 'D:\models\Qwen2.5-7B-Instruct'
      ORIGIN_HF_ID: Qwen/Qwen2.5-7B-Instruct
      REVISION_PIN_POLICY: local_path_immutable
    QWEN25_7B_BASE:
      HF_ID: Qwen/Qwen2.5-7B
      REVISION_PIN_POLICY: use_explicit_commit_if_available
    LLAMA31_8B_BASE:
      HF_ID: meta-llama/Llama-3.1-8B
      REVISION_PIN_POLICY: use_explicit_commit_if_available
    LLAMA31_8B_INSTRUCT:
      HF_ID: meta-llama/Llama-3.1-8B-Instruct
      REVISION_PIN_POLICY: use_explicit_commit_if_available
  ID:
    FIG:
      F01_RASTER_EXAMPLE: F01_RASTER_EXAMPLE
      F02_RATE_MATCH_CHECK: F02_RATE_MATCH_CHECK
      F03_BRANCHING_CURVES: F03_BRANCHING_CURVES
      F04_NULL_DELTAB: F04_NULL_DELTAB
      F05_GSTAR_SELECTION: F05_GSTAR_SELECTION
      F06_GENERALIZATION_B: F06_GENERALIZATION_B
      F07_ARC_MCQ: F07_ARC_MCQ
      F08_SPIKEDEF_ROBUST: F08_SPIKEDEF_ROBUST
      F09_CHI_CURVES: F09_CHI_CURVES
      F10_NULL_COMPARE: F10_NULL_COMPARE
      F11_ABLATIONS: F11_ABLATIONS
    TABLE:
      T01_SUMMARY: T01_SUMMARY
      T02_GENERALIZATION: T02_GENERALIZATION
      T03_ARC: T03_ARC
      T04_TAIL_FITS: T04_TAIL_FITS
      T05_CRACKLING_DIAGNOSTICS: T05_CRACKLING_DIAGNOSTICS
      T06_ABLATIONS: T06_ABLATIONS
      T07_REPLICATION_SUMMARY: T07_REPLICATION_SUMMARY
  OUTPUT:
    RUN_RECORD_JSON: run_record.json
    CONFIG_RESOLVED_YAML: config_resolved.yaml
    MANIFEST_SHA256_BASENAME: MANIFEST.sha256
    BIB_REFERENCES_BASENAME: references.bib
    PAPER_SNAPSHOT_MD_BASENAME: 00_PAPER_SNAPSHOT.md
    RASTER_NPZ_BASENAME: rasters_token_layer.npz
    NULL_NPZ_BASENAME: rasters_nulls.npz
    METRICS_PARQUET_BASENAME: metrics.parquet
    AVALANCHES_PARQUET_BASENAME: avalanches.parquet
    FIG_PREFIX: fig_
    TABLE_PREFIX: table_
    RUN_SUBDIR:
      RESULTS: results
      FIGS: figs
      TABLES: tables
      LOGS: logs
      CACHE: cache
    FIG_FILE_PDF:
      F01_RASTER_EXAMPLE: figs/fig_F01_RASTER_EXAMPLE.pdf
      F02_RATE_MATCH_CHECK: figs/fig_F02_RATE_MATCH_CHECK.pdf
      F03_BRANCHING_CURVES: figs/fig_F03_BRANCHING_CURVES.pdf
      F04_NULL_DELTAB: figs/fig_F04_NULL_DELTAB.pdf
      F05_GSTAR_SELECTION: figs/fig_F05_GSTAR_SELECTION.pdf
      F06_GENERALIZATION_B: figs/fig_F06_GENERALIZATION_B.pdf
      F07_ARC_MCQ: figs/fig_F07_ARC_MCQ.pdf
      F08_SPIKEDEF_ROBUST: figs/fig_F08_SPIKEDEF_ROBUST.pdf
      F09_CHI_CURVES: figs/appendix/fig_F09_CHI_CURVES.pdf
      F10_NULL_COMPARE: figs/appendix/fig_F10_NULL_COMPARE.pdf
      F11_ABLATIONS: figs/appendix/fig_F11_ABLATIONS.pdf
    FIG_FILE_PNG:
      F01_RASTER_EXAMPLE: figs/fig_F01_RASTER_EXAMPLE.png
      F02_RATE_MATCH_CHECK: figs/fig_F02_RATE_MATCH_CHECK.png
      F03_BRANCHING_CURVES: figs/fig_F03_BRANCHING_CURVES.png
      F04_NULL_DELTAB: figs/fig_F04_NULL_DELTAB.png
      F05_GSTAR_SELECTION: figs/fig_F05_GSTAR_SELECTION.png
      F06_GENERALIZATION_B: figs/fig_F06_GENERALIZATION_B.png
      F07_ARC_MCQ: figs/fig_F07_ARC_MCQ.png
      F08_SPIKEDEF_ROBUST: figs/fig_F08_SPIKEDEF_ROBUST.png
      F09_CHI_CURVES: figs/appendix/fig_F09_CHI_CURVES.png
      F10_NULL_COMPARE: figs/appendix/fig_F10_NULL_COMPARE.png
      F11_ABLATIONS: figs/appendix/fig_F11_ABLATIONS.png
    TABLE_FILE_CSV:
      T01_SUMMARY: tables/table_T01_SUMMARY.csv
      T02_GENERALIZATION: tables/table_T02_GENERALIZATION.csv
      T03_ARC: tables/table_T03_ARC.csv
      T04_TAIL_FITS: tables/appendix/table_T04_TAIL_FITS.csv
      T05_CRACKLING_DIAGNOSTICS: tables/appendix/table_T05_CRACKLING_DIAGNOSTICS.csv
      T06_ABLATIONS: tables/appendix/table_T06_ABLATIONS.csv
      T07_REPLICATION_SUMMARY: tables/appendix/table_T07_REPLICATION_SUMMARY.csv
    TABLE_FILE_PARQUET:
      T01_SUMMARY: tables/table_T01_SUMMARY.parquet
      T02_GENERALIZATION: tables/table_T02_GENERALIZATION.parquet
      T03_ARC: tables/table_T03_ARC.parquet
      T04_TAIL_FITS: tables/appendix/table_T04_TAIL_FITS.parquet
      T05_CRACKLING_DIAGNOSTICS: tables/appendix/table_T05_CRACKLING_DIAGNOSTICS.parquet
      T06_ABLATIONS: tables/appendix/table_T06_ABLATIONS.parquet
      T07_REPLICATION_SUMMARY: tables/appendix/table_T07_REPLICATION_SUMMARY.parquet
    CALIBRATION_STATS_NPZ_BASENAME: calibration_stats.npz
    RATE_TARGETS_JSON_BASENAME: rate_targets.json
    TAU_RATE_MATCHED_PARQUET_BASENAME: tau_rate_matched.parquet
    GSTAR_JSON_BASENAME: gstar.json
  EXAMPLE:
    RUN_ID_S05: RUN_S05_9b8f2e0a1c3d
    RUN_ID_S06B: RUN_S06B_4d2c9a77e531
    RUN_ID_S06ARC: RUN_S06ARC_c0ffee12ab34
    RUN_ID_S07: RUN_S07_7e11cafe55aa
    RUN_DIR_S05: runs/RUN_S05_9b8f2e0a1c3d
```

## Worked example

- Example run directory: CANON.EXAMPLE.RUN_DIR_S05
- Example figures output inside that run directory:
  - CANON.OUTPUT.FIG_FILE_PDF.F01_RASTER_EXAMPLE
  - CANON.OUTPUT.FIG_FILE_PDF.F03_BRANCHING_CURVES
- Example tables output inside that run directory:
  - CANON.OUTPUT.TABLE_FILE_CSV.T01_SUMMARY

## Failure modes

1. A file hardcodes a literal that differs from CANON.
   Detect: run the SDR scan in spec/16_TEST_PLAN.md.
   Fix: replace literal with CANON key reference and update CANON if a key is missing.
2. A CANON key referenced elsewhere is missing here.
   Detect: config resolver fails to resolve a CANON reference.
   Fix: add the key here and rerun the resolver.
3. Output filenames differ from CANON.
   Detect: exporter writes a different filename than CANON.OUTPUT values or manifest check fails.
   Fix: align exporter and update CANON.OUTPUT mapping.

## Acceptance criteria

- CANON registry parses as valid YAML.
- Tasks and specs refer to command identifiers under CANON.CLI.CMD and do not repeat literals elsewhere.
- Figure and table output filenames match CANON.OUTPUT mappings.
