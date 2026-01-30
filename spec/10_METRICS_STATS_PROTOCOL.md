# 10 METRICS AND STATS PROTOCOL

## Purpose and scope
- Define statistical comparisons, bootstrap confidence intervals, and reporting conventions.
- Ensure reproducible and conservative inference.

## Normative requirements
- MUST compute paired differences between g=1 and g=gstar for Dataset B and ARC.
- MUST use bootstrap with fixed seed and replicate count.
- MUST report effect sizes and CI rather than p values.
- SHOULD store intermediate metrics in parquet for recomputation.

## Definitions
- Metric groups: branching metrics, delta metrics, chi, task metrics.
- Bootstrap: resample sequences with replacement, compute metric, repeat BOOTSTRAP_REPS.

## Procedure
Procedure:
1. For each run create metrics table at per sequence granularity.
2. For each comparison compute bootstrap CI of mean difference.
3. Export summary tables T01 to T03.
4. For plots, include error bars from bootstrap.

## Worked example
Example: For delta b_tot difference between gstar and g1, report mean and 95 percent CI derived from percentile bootstrap.

## Failure modes
1. Non deterministic bootstrap.
   Detect: CI changes between reruns.
   Fix: set bootstrap seed and fixed resampling order.
2. Mixing per sequence and pooled metrics.
   Detect: mismatch between figure and table.
   Fix: adhere to definitions in spec/06 and spec/18.
3. CI computed on already averaged data.
   Detect: too narrow CI.
   Fix: bootstrap on per sequence samples.

## Acceptance criteria
- Bootstrap CIs reproduce across reruns.
- Tables include required columns and CI bounds.

## Cross references
- spec/18 tables
- spec/19 figures

