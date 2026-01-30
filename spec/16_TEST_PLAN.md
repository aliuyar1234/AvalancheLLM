# 16 TEST PLAN

## Purpose and scope
- Define deterministic and correctness tests required before release.

## Normative requirements
- MUST run forbidden token scan: disallow to do marker, to be determined marker, three dots, angle brackets.
- MUST run SDR scan: no literal duplication for keys listed in CANON.
- MUST run toy raster tests for CC and branching.
- MUST run rate match tolerance test.
- MUST run null marginal preservation test.

## Definitions
- Toy raster is a small fixed X grid defined in this spec.
- Determinism test runs same stage twice and compares hashes.

## Procedure
Procedure:
1. Forbidden scan: grep repository for forbidden substrings.
2. SDR scan: assert literals from CANON do not appear outside spec/00.
3. Toy CC test: run CC BFS and compare component stats.
4. Toy branching test: compare b_time and b_depth.
5. Rate match test: on calibration slice compute achieved rates.
6. Null test: verify marginal preservation and delta b behaves as expected.

## Worked example
Worked toy raster: T=4,L=3 occupancy nodes at (2,2),(3,2),(3,3) yields one component with D=2,H=2. Branching time edge count 1, denominator 3 gives b_time 0.333 for occupancy.

## Failure modes
1. Forbidden token present.
   Detect: scan finds match.
   Fix: rewrite file and rerun scan.
2. SDR violation.
   Detect: literal appears outside spec/00.
   Fix: replace with CANON key.
3. Rate match fails.
   Detect: abs diff above tol.
   Fix: adjust histogram bins and recompute.

## Acceptance criteria
- All tests pass.
- Release gate requires green test plan.

## Cross references
- tasks/PHASE_8

