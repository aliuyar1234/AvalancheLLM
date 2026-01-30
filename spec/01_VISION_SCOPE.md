# 01 VISION AND SCOPE

## Purpose and scope
- Build an ICLR ready, reproducible analysis of token by layer activation events in LLMs.
- Core claim is avalanche like event cascades and gain tuned quasi critical regime with strong controls.
- No training runs; inference and analysis only within 24 GPU hours.

## Normative requirements
- MUST use conservative language: activation events, avalanche like, quasi critical.
- MUST include rate matched thresholds and marginals preserving raster null.
- MUST support cross dataset g star calibration: choose g star on dataset A by b tot near one, evaluate on dataset B and ARC MCQ.
- MUST produce deterministic runs and immutable artifacts.

## Definitions
- Event lattice is X[t,l] occupancy built from A[t,l] event counts as in spec/03 and spec/05.
- b time, b depth, b tot and delta b as in spec/06.
- g is global MLP residual gain as in spec/08.

## Procedure
1. Implement pipeline stages S00 through S08 in spec/13.
2. Ensure each stage writes run_record.json and resolves configs.
3. Generate figures and tables as in spec/19 and spec/18.
4. Write paper text using paper folder files and cite only bib keys in bib/references.bib.

## Worked example
Example run set: run RUN_20260128_0001 calibrate, RUN_20260128_0002 gain grid, compute gstar, then RUN_20260128_0003 extract rasters for g one and gstar.

## Failure modes
1. Overclaim criticality from tail fits.
   Detect: paper text says power law implies criticality.
   Fix: update to multi signature wording in spec/09.
2. Exceed GPU budget.
   Detect: budgets.yaml totals above 24.
   Fix: reduce sample counts.
3. Drift in definitions.
   Detect: acceptance tests in spec/16 fail.
   Fix: align to CANON.

## Acceptance criteria
- All required figures and tables exported with correct IDs.
- All stages produce artifacts and pass tests.
- Paper claims match evidence and falsifiers are reported.

## Cross references
- spec/03 for event definitions
- spec/04 rate matching
- spec/07 nulls
- spec/08 gain and gstar
- tasks/PHASE files

