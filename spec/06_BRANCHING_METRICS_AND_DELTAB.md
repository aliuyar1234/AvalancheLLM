# 06 BRANCHING METRICS AND DELTA B

## Purpose and scope
- Define directional branching metrics b_time, b_depth, b_tot and delta b using the marginals preserving raster null.

## Normative requirements
- MUST compute b metrics on occupancy X and optionally on counts A.
- MUST handle boundaries by excluding invalid edges.
- MUST compute b_perm on permuted raster from spec/07.
- MUST report delta b as b-b_perm.

## Definitions
- b_time = sum X[t,l]*X[t+1,l] over valid t,l divided by sum X[t,l] over valid t,l.
- b_depth = sum X[t,l]*X[t,l+1] over valid t,l divided by sum X[t,l] over valid t,l.
- b_tot = b_time + b_depth.
- delta metrics computed against within layer permutation null.

## Procedure
Procedure:
1. For each sequence compute numerator and denominator for time and depth.
2. Aggregate across sequences by summing numerators and denominators then dividing.
3. Generate permuted raster X_perm and compute same metrics.
4. delta_b = b - b_perm.
Toy acceptance test uses a handcrafted raster in spec/16.

## Worked example
Example: If two occupied nodes always followed in time, b_time=1. If permutation destroys alignment, b_time_perm near marginal rate.

## Failure modes
1. Denominator zero.
   Detect: sum X is zero.
   Fix: skip sequence and record skip count.
2. Wrong aggregation.
   Detect: per sequence average differs from pooled.
   Fix: use pooled estimator defined above.
3. Permutation reuse.
   Detect: permuted metrics identical across sequences.
   Fix: seed permutation per sequence id.

## Acceptance criteria
- b metrics reproduce known toy values.
- delta b is non negative for structured raster in toy case.

## Cross references
- spec/07 nulls
- spec/10 stats protocol

