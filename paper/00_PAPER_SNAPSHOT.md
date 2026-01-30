# PAPER SNAPSHOT

## Title
Token by Layer Activation Event Cascades in LLMs: Rate Matched Avalanche Connectivity under Gain Scaling

## One paragraph summary
We study activation events in large language models as a two-dimensional lattice indexed by token position and layer depth. By thresholding standardized gated-MLP pre-down-projection activations into sparse binary events, we can define connected components on the token by layer lattice that resemble avalanche-like cascades. We introduce a set of mechanistic connectivity metrics based on directional branching, and we test a gain intervention that scales each layer's MLP residual contribution. Crucially, we add reviewer-proof controls: per-layer rate-matched thresholds that equalize marginal firing rates across gains, and a post-hoc marginals-preserving raster shuffle that destroys temporal structure within each layer while preserving per-layer event counts exactly. We use multiple signatures to probe a quasi-critical regime without equating heavy tails with criticality, and we calibrate a gain value gstar mechanistically on Dataset A and evaluate it unchanged on Dataset B and ARC multiple-choice.
In the default run provided in this pack, b_tot varies with gain under rate matching and the gstar criterion selects different gains across conditions, but this mechanistic gstar does not improve downstream proxy metrics on Dataset B or ARC, illustrating how the same pipeline reports negative results under the same controls.

## Contributions
1. Event lattice construct: a token by layer binary event raster derived from standardized gated-MLP activations, enabling avalanche-style connected component analysis in transformers.
2. Directional branching decomposition: b_time, b_depth, and b_tot metrics that quantify local propagation along token and depth directions and support a marginals-controlled delta-b residual.
3. Rate-matched mechanistic control: for each gain g we choose per-layer thresholds tau_l(g) to match marginal firing rates, showing that connectivity changes are not explained by rate changes.
4. Strong null model: a within-layer time permutation that preserves per-layer event counts and destroys token order, enabling delta-b connectivity residuals that control for marginals.
5. Cross-dataset mechanistic calibration: select gstar on Dataset A by minimizing abs(b_tot minus 1), then evaluate the same gstar on Dataset B and ARC multiple-choice, comparing against g=1.

## Claim to evidence map
Main claims and the primary artifacts that support them:

- Claim C1: Rate matching succeeds across gains and spike definitions.
  Evidence: Figure F02_RATE_MATCH_CHECK and Table T01_SUMMARY columns achieved_rate_max_abs_err.

- Claim C2: Branching metrics vary with gain even under rate matching, and the effect is not removed by marginals-preserving shuffles.
  Evidence: Figure F03_BRANCHING_CURVES, Figure F04_NULL_DELTAB, and Table T01_SUMMARY delta_b_tot.

- Claim C3: gstar is calibrated mechanically on Dataset A without using task performance and evaluated unchanged on Dataset B and ARC multiple-choice.
  Evidence: Figure F05_GSTAR_SELECTION, Table T02_GENERALIZATION, Figure F06_GENERALIZATION_B, Table T03_ARC, Figure F07_ARC_MCQ.

- Claim C4: The qualitative signatures are robust to the spike definition choice.
  Evidence: Figure F08_SPIKEDEF_ROBUST and Table T01_SUMMARY filtered by spike_def_id.

## Explicit falsifiers
The following outcomes would falsify key claims:

- If achieved_rate_max_abs_err exceeds the tolerance for many conditions, then rate-matching control fails and conclusions about connectivity must be suspended.
- If delta_b_tot is near zero for all gains, then the observed branching changes are explained by marginals and not by event connectivity.
- If gstar selected on Dataset A does not transfer in mechanistic metrics and consistently harms proxy task metrics relative to g=1, then the mechanistic calibration claim is weakened.

## Reproducibility anchors
- Specs: spec/00_CANONICAL.md and spec/23_IMPLEMENTATION_CONTRACT.md
- Table schemas: spec/18_RESULTS_TABLE_SKELETONS.md
- Figure plan: spec/19_FIGURES_TABLES_PLAN.md
- Run provenance: run_record.json schema in spec/14

PROVENANCE_BEGIN

## Provenance

- F03_BRANCHING_CURVES: run_id=RUN_S05_d185f0bd5af0 config_hash=d14a80cac58367877c03eab7dd71ee0df526395171ee96eba5232412b3629e87
- F06_GENERALIZATION_B: run_id=RUN_S06B_e3f75727e3c8 config_hash=d14a80cac58367877c03eab7dd71ee0df526395171ee96eba5232412b3629e87
- F07_ARC_MCQ: run_id=RUN_S06ARC_83b46f42502d config_hash=d14a80cac58367877c03eab7dd71ee0df526395171ee96eba5232412b3629e87
- T01_SUMMARY: run_id=RUN_S05_d185f0bd5af0 config_hash=d14a80cac58367877c03eab7dd71ee0df526395171ee96eba5232412b3629e87
- T02_GENERALIZATION: run_id=RUN_S06B_e3f75727e3c8 config_hash=d14a80cac58367877c03eab7dd71ee0df526395171ee96eba5232412b3629e87
- T03_ARC: run_id=RUN_S06ARC_83b46f42502d config_hash=d14a80cac58367877c03eab7dd71ee0df526395171ee96eba5232412b3629e87

PROVENANCE_END
