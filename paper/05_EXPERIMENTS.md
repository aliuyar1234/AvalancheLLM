# EXPERIMENTS

We describe the models, datasets, conditions, metrics, and compute budget used to generate the figures and tables.

## Models
We use open-weight transformer models that fit on a single GPU for inference. The default configuration in the pack includes:
- An instruction-tuned model at the ~7B parameter scale.
- Optionally, a base model at similar scale for comparison.

Model identifiers and revision pinning policies are defined in CANON under CANON.MODEL. The implementation MUST record the exact revision or commit hash used in run_record.json.

## Datasets
We use three dataset roles:
- Dataset A: used for calibration, rate-matching thresholds, and gstar selection.
- Dataset B: used only for evaluation of generalization at fixed gstar.
- ARC multiple-choice: used for a task-oriented generalization test.

The exact dataset identifiers, splits, and configs are in CANON under CANON.DATASET. The deterministic sampling and hashing protocol is in spec/11.

## Conditions and controls
We run conditions defined by:
- gain g values on a grid (Dataset A gain grid)
- spike definition (one-sided positive and two-sided absolute)
- target per-layer rate r_star
- null condition (real raster, within-layer permutation null, and input token shuffle null)

Reviewer-proof controls:
- Rate matching is performed per layer and per gain, so that marginal event rates match across g.
- A within-layer permutation null preserves per-layer marginals exactly and destroys temporal structure.
- Dual spike definitions test robustness.

## Primary metrics
Mechanistic metrics:
- b_time, b_depth, b_tot, and delta-b variants
- avalanche component statistics S, D, H computed from connected components (size uses event counts A)
- quasi-critical signatures chi and the S3 crackling fit when quality gates allow (otherwise recorded as not available)

Task proxy metrics:
- perplexity on Dataset B slice
- ARC multiple-choice accuracy with uncertainty

## gstar protocol
We select gstar on Dataset A only using the mechanistic criterion that minimizes the absolute deviation abs(b_tot minus 1). We then evaluate g=1 and gstar on Dataset B and ARC without retuning. This addresses the concern that g=1 is a trivial optimum and ensures that calibration is not based on performance.

## Compute budget and implementation constraints
The pack is designed to stay within a single GPU budget by:
- using inference-only passes with small deterministic slices
- storing intermediate artifacts so later stages do not re-run the model
- generating nulls post hoc on CPU

The budgets are specified in configs/budgets.yaml. The implementation MUST log GPU time per stage in run_record.json.

## Baselines
We compare against:
- g=1 as the default unmodified model behavior
- the within-layer permutation null for connectivity measures
- the input token shuffle null for semantic disruption

We do not train any learned probes or adapters in the default pipeline.
