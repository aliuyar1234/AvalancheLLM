# 09 QUASI CRITICAL SIGNATURES

## Purpose and scope
- Define the multi signature evidence suite S1 through S3 and explicit falsifiers.
- Ensure language avoids power law implies criticality.

## Normative requirements
- MUST treat tail fits as descriptive only.
- MUST require agreement of S1 and S2 for quasi critical regime claim.
- SHOULD report S3 crackling relation if fit passes quality gates.
- MUST report falsifiers and negative results.

## Definitions
- S1: branching crossing, b_tot approaches one and delta b shows structure.
- S2: susceptibility proxy chi peaks near S1.
- S3: crackling relation E[S|D] proportional to D^gamma over a stable D range.

## Procedure
S1 computation: compute b_tot(g) on Dataset A grid, identify crossing interval.
S2: per sequence total activity Y=sum A. chi=Var(Y)/(Mean(Y)+CHI_EPS).
S3: group avalanches by duration D, compute mean S, fit log mean S vs log D using OLS on durations within [CRACKLING_D_RANGE_MIN,CRACKLING_D_RANGE_MAX] with at least CRACKLING_MIN_POINTS points. Bootstrap CI using BOOTSTRAP_REPS.
Failure handling: if not enough points for S3, record S3 as not available and do not claim it.

## Worked example
Example: If chi(g) is maximal at g=0.85 and b_tot is near one at g=0.85, we label that as quasi critical region for that model and spike def.

## Failure modes
1. chi peak is far from b crossing.
   Detect: g argmax chi differs from gstar by more than CANON.CONST.SIGNATURE_ALIGNMENT_G_DIFF_MAX.
   Fix: interpret as decoupling and weaken claim.
2. S3 fit unstable.
   Detect: CI width above CANON.CONST.BOOTSTRAP_CI_WIDTH_MAX.
   Fix: increase sample count or report not available.
3. Overclaim.
   Detect: paper states criticality proven.
   Fix: use quasi critical wording and reference falsifiers section.

## Acceptance criteria
- Report includes S1 and S2 with plots.
- Paper includes explicit falsifiers.
- Tail fits labeled descriptive only.

## Cross references
- spec/10 stats
- paper/07 discussion

