# Avalanche LLM: Token–Layer Activation Event Cascades in LLMs

This repository is a single-source-of-truth (SSOT) reproducibility pack for analyzing thresholded gated-MLP activation events as connected cascades on a token-by-layer lattice, and for testing a simple gain intervention under strong controls.

The core scientific goal is mechanistic: measure how local connectivity signatures change with MLP gain scaling while explicitly controlling for marginal event rates and layerwise marginals. The pack is designed to be inference-heavy (no training) and to run within a practical single-GPU budget via fixed, deterministic dataset slices.

Author: Ali Uyar (Independent Researcher).

## Paper and artifacts
- Compiled paper PDF: `paper.pdf`
- LaTeX sources (Overleaf-compatible): `paper/latex/`
- Citation metadata: `CITATION.cff`

Note: ZIP bundles are intentionally not tracked by git. If you want single-file distribution (Overleaf upload or external review), generate a ZIP from `paper/latex/` and attach it to a GitHub Release.

## What is implemented
The pipeline is specified to be fail-closed and reviewer-proof by construction:
- Event lattice: standardize a fixed internal tensor and threshold into sparse binary events (two spike definitions).
- Avalanches: connected components on the token-by-layer lattice with well-defined adjacency.
- Mechanistic signatures: directional branching metrics, a susceptibility proxy, and descriptive tail-fit diagnostics.
- Controls: per-layer rate-matched thresholds plus strong marginals-preserving nulls (including a structure-preserving null).
- Gain calibration: select a mechanistic gain g-star on Dataset A and evaluate unchanged on Dataset B and ARC multiple-choice.
- Replication: compare base vs instruction-tuned checkpoints within the same model family.
- Provenance: every producing stage writes immutable run artifacts with schema-checked metadata and content hashes.

## Reproducibility contract (start here)
This pack is intentionally spec-first. If you are new, read in this order:
1. `spec/00_CANONICAL.md` (Single Definition Rule authority for IDs, paths, commands, and filenames)
2. `spec/23_IMPLEMENTATION_CONTRACT.md` (required repository tree, CLI, run layout, determinism rules)
3. `tasks/TASK_INDEX.md` (execution order and definition-of-done checklist)
4. `spec/18_RESULTS_TABLE_SKELETONS.md` and `spec/19_FIGURES_TABLES_PLAN.md` (paper-facing artifact contracts)

## How to run
All CLI command strings are defined under `CANON.CLI.CMD` in `spec/00_CANONICAL.md`. The recommended workflow is to execute phases in the exact order listed in `tasks/TASK_INDEX.md` and to treat each phase as an audited producing step.

If you want a guided narrative of what was recently implemented and how to regenerate paper-facing LaTeX artifacts, see `progress.md`.

Automation helpers:
- `tools/ssot_pipeline.py` orchestrates the end-to-end runbook into a fresh output workspace (keeps this directory clean).
- `tools/run_paperready.ps1` is a PowerShell wrapper for the same workflow.

## Repository layout
- `avalanche_llm/`: Python package (CLI entrypoint, phases, metrics, plotting, tests)
- `spec/`: normative method + artifact specifications (the SSOT)
- `tasks/`: phase-by-phase runbook (what to run, what must exist afterward)
- `configs/`: resolved configuration inputs referenced by the CLI
- `paper/latex/`: camera-ready LaTeX project that mirrors exported figures and tables
- `checklists/`: reproducibility and artifact release gates

## License and model/data note
No model weights are included. Users must ensure they comply with upstream model and dataset licenses and any access terms. The pack is intended for research and reproducibility auditing, not deployment.
