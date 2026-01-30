# APPENDIX

## Additional robustness checks
The default pipeline includes the following robustness checks as optional appendix artifacts:
- Adjacency robustness: repeat connected component and branching computation with 8-neighborhood adjacency (CANON.ENUM.ADJACENCY_ID).
- Threshold baseline sensitivity: report how tau0_baseline influences the rate-matching quantiles and achieved rates.
- Null comparison: compare within-layer permutation null against input token shuffle null for selected conditions.

Appendix figures and tables, if produced, MUST be placed under figs/appendix and tables/appendix within the producing run directory and MUST be registered in run_record.json.

## Implementation details
### Connected components
Connected components are computed on a binary grid A_{t,l} using a BFS traversal with an explicit queue. The implementation must handle boundaries and must avoid recursion depth issues.

### Branching metrics
Branching metrics are computed by counting active forward neighbors and normalizing by the number of active sites eligible for that neighbor relation. Boundary handling and normalization are specified in spec/06.

### Crackling relation fit
The crackling relation fit uses a declared range of spans and a minimum number of points. If the fit fails due to insufficient points, the implementation must record the failure reason and write NaN for the slope with an opt_note column in the relevant table.

## Artifact index
The appendix artifacts are optional, but if created they must be reproducible:
- registered in run_record.json with sha256
- listed in the claim-to-evidence map if referenced in the paper
