# RESULTS

This pack is designed so that results are produced as immutable artifacts and referenced by stable figure and table identifiers. The figures and tables listed below are the authoritative outputs for the main claims.

## Rate matching and controls
Figure F02_RATE_MATCH_CHECK and Table T01_SUMMARY report achieved firing rates under the per-layer rate-matching protocol. The acceptance criterion is that achieved_rate_max_abs_err remains within the configured tolerance for all gains and spike definitions. This control is required to interpret any gain-dependent changes in branching.

In the default run (see provenance footnotes), rate matching succeeds across all 20 Dataset A conditions with achieved_rate_max_abs_err at most 4.83e-7 under a tolerance of 5.0e-4 (Table T01_SUMMARY).

## Branching versus gain under rate matching
Figure F03_BRANCHING_CURVES plots b_time, b_depth, and b_tot as a function of gain for each spike definition. The key question is whether branching measures vary with gain after controlling marginal rates. Figure F04_NULL_DELTAB complements this by reporting delta-b relative to the within-layer permutation null, which preserves marginals exactly.

In the default run, b_tot varies with gain under rate matching (min 0.795, max 1.104; Table T01_SUMMARY). The null-controlled residual delta_b_tot is non-zero across conditions (min 0.228, max 0.323; Figure F04_NULL_DELTAB and Table T01_SUMMARY).

## Mechanistic gstar selection
Figure F05_GSTAR_SELECTION shows the gstar selection criterion on Dataset A as the gain that minimizes abs(b_tot minus 1) under rate matching. This selection is mechanistic and does not use task performance signals.

In the default run, the gstar criterion selects different gains across the four (spike_def_id, target_rate) conditions (gstar takes values 0.7, 1.0, and 1.3; gstar.json).

## Cross-dataset evaluation
Table T02_GENERALIZATION and Figure F06_GENERALIZATION_B compare g=1 and gstar on Dataset B. We report both mechanistic metrics (branching and delta-b) and task proxy metrics (perplexity and mean negative log likelihood) to evaluate whether the mechanistic calibration transfers.

Table T03_ARC and Figure F07_ARC_MCQ report ARC multiple-choice accuracy at g=1 and gstar with uncertainty.

In the default run, gstar does not improve Dataset B perplexity relative to g=1; perplexity increases by between 0.000 and 2.413 depending on condition (Table T02_GENERALIZATION). On ARC multiple-choice, accuracy decreases by up to 0.028 at gstar relative to g=1, with the worst-case gstar accuracy 0.857 (95 percent CI 0.838 to 0.875) compared to the g=1 baseline 0.885 (95 percent CI 0.866 to 0.902) (Table T03_ARC).

## Robustness across spike definitions
Figure F08_SPIKEDEF_ROBUST summarizes whether qualitative patterns in branching and delta-b persist under both spike definitions. This reduces the risk that results depend on a particular thresholding convention.

## Claim boundaries
We do not infer criticality from tail shapes alone. Tail fits are reported as descriptive diagnostics, and the signature suite is treated as a set of falsifiable probes that can yield negative results under the same controls.

In the default run, avalanche connected components are not lattice-spanning: across conditions, the mean token span is 1.575 to 2.065 and the mean layer span is 1.947 to 2.790 (Table T01_SUMMARY). The crackling fit S3 is available for all conditions, with crackling_gamma ranging from 1.464 to 1.862 (Table T01_SUMMARY).
