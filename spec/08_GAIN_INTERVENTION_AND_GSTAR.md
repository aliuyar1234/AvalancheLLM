# 08 GAIN INTERVENTION AND GSTAR

## Purpose and scope
- Define how gain g modifies forward pass and how gstar is selected on Dataset A.
- Define generalization evaluation on Dataset B and ARC MCQ.

## Normative requirements
- MUST apply gain on MLP residual contribution only: h_next=h+attn+g*mlp.
- MUST implement gain via forward hook or module wrapper, no model weight edits.
- MUST select gstar using only b_tot near one on Dataset A with rate matching.
- MUST evaluate gstar on Dataset B and ARC MCQ vs g=1 without retuning.

## Definitions
- Condition is tuple: model_role, dataset_role, spike_def_id, target_rate, gain, null_id.
- gstar defined per model_role, spike_def_id, target_rate.
- Tie break: pick gain closest to 1.0 if multiple minima.

## Procedure
Procedure:
1. Run gain grid on Dataset A for gains in CANON.CONST.GAIN_GRID_DEFAULT.
2. For each condition compute b_tot under rate matching and delta b.
3. Select gstar by minimizing abs(b_tot-1).
4. Freeze gstar in a results/CANON.OUTPUT.GSTAR_JSON_BASENAME artifact.
5. Run Dataset B metrics and ARC MCQ at g=1 and gstar.
6. Export Table T02 and T03.

## Worked example
Example: If b_tot at gains [0.7,0.85,1.0,1.15,1.3] is [0.6,0.9,1.02,1.3,1.5], gstar=1.0. If b_tot is [0.6,0.98,1.1,1.35,1.6], gstar=0.85.

## Failure modes
1. gstar uses task metric.
   Detect: run_record shows accuracy used.
   Fix: enforce selection uses only b_tot.
2. Gain applied in wrong place.
   Detect: activation magnitudes change but b curves flat.
   Fix: apply gain on mlp output before residual add.
3. gstar not generalizing.
   Detect: Dataset B b_tot far from one and task degrades.
   Fix: report as falsifier, do not claim robust regime.

## Acceptance criteria
- gstar selection artifact exists and includes hashes.
- Dataset B and ARC eval completed and exported.

## Cross references
- spec/12 hooks
- spec/06 branching
- tasks/PHASE_2 and PHASE_6

