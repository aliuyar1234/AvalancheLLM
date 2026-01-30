# 02 FORMAL OBJECTS EVENT LATTICE

## Purpose and scope
- Define the token by layer event lattice and the concrete raster artifacts saved to disk.
- Specify the exact raster tensor semantics used by connected-components (avalanches) and branching metrics.
- Define the NPZ on-disk schema so later phases do not guess array keys or axis order.

## Normative requirements
- MUST index tokens t from 1 to CANON.CONST.SEQ_LEN_TOKENS in documentation; implementation uses zero-based indices.
- MUST index transformer layers l from 1 to L in documentation; implementation uses zero-based indices.
- MUST compute events from the standardized activation z defined in spec/03 and thresholded by tau_l(g) from spec/04.
- MUST store rasters in NPZ files using the canonical basenames:
  - CANON.OUTPUT.RASTER_NPZ_BASENAME
  - CANON.OUTPUT.NULL_NPZ_BASENAME
- MUST use the NPZ key names defined in CANON.ENUM.NPZ_KEY.
- MUST define a deterministic axis order for all raster arrays:
  - condition axis first, then sequence axis, then layer axis, then token axis
  - shape is [C, N, L, T]
- MUST include a condition id list cond_id of length C so downstream phases can select the correct condition slice.
- MUST include a sequence id list seq_id of length N so downstream phases can align rasters with dataset samples.

## Definitions
Activation and standardization:
- u[t,l,i] is the MLP intermediate activation tensor defined in spec/12.
- z[t,l,i] is standardized u using mu and sigma and eps from spec/03.

Unit-level spikes:
- For a given spike_def_id and layer-wise threshold tau_l(g), define a unit-level spike indicator s(t,l,i).

Layer aggregation:
- A[t,l] is the event count at lattice site (t,l), defined as the sum of unit spikes over i.
- X[t,l] is the binary occupancy at (t,l), defined as 1 if A[t,l] is strictly positive, else 0.

Lattice:
- A lattice site is the ordered pair (t,l).
- Adjacency is defined in spec/05 via CANON.ENUM.ADJACENCY_ID.

## On-disk NPZ schema (mandatory)
Primary raster file CANON.OUTPUT.RASTER_NPZ_BASENAME MUST contain:
- CANON.ENUM.NPZ_KEY.X_OCCUPANCY : uint8 array X with shape [C,N,L,T]
- CANON.ENUM.NPZ_KEY.A_COUNT     : uint16 array A with shape [C,N,L,T]
- CANON.ENUM.NPZ_KEY.COND_ID     : int32 array of length C, condition identifiers
- CANON.ENUM.NPZ_KEY.SEQ_ID      : int32 array of length N, sequence identifiers within the sampled slice

Null raster file CANON.OUTPUT.NULL_NPZ_BASENAME MUST contain the same keys, where X and A correspond to the null-generated rasters for each condition and null_id.

Condition identifiers:
- A condition id is an integer stable within a run that indexes into the resolved condition grid.
- The mapping from cond_id to the semantic condition tuple (model_role, dataset_role, spike_def_id, target_rate, gain, null_id)
  MUST be written into results/metrics.parquet for the run that produced the rasters.

## Procedure
1. For each condition in the resolved grid, run the model inference and compute z and spikes.
2. Aggregate unit spikes into A and X per sequence.
3. Stack rasters across conditions to form arrays with shape [C,N,L,T].
4. Write the NPZ file and register it in run_record.json.
5. Downstream phases:
   - compute connected components on X (spec/05),
   - compute branching metrics on X and A (spec/06),
   - generate nulls by permuting rasters (spec/07).

## Worked example
Toy example for a single condition:
- C=1, N=1, L=3, T=4.
- Suppose X has ones at lattice sites (t=2,l=2), (t=3,l=2), (t=3,l=3).
These sites are adjacent under CANON.ENUM.ADJACENCY_ID.ADJ_4N because they differ by one token or one layer.

The NPZ arrays have shapes:
- X shape [1,1,3,4]
- A shape [1,1,3,4]
- cond_id length 1
- seq_id length 1

## Failure modes (detect and fix)
1. Axis order mismatch.
   Detect: raster plots appear transposed or rotated; avalanche spans disagree with spec/16 toy tests.
   Fix: enforce axis order [C,N,L,T] and write a unit test to check shapes on load.
2. Missing keys in NPZ.
   Detect: downstream phase raises KeyError for X or A.
   Fix: implement the NPZ writer to always include CANON.ENUM.NPZ_KEY keys.
3. Condition mapping drift.
   Detect: cond_id list length does not match the number of conditions in metrics.parquet.
   Fix: generate cond_id list from the resolved condition grid and write both from the same object.

## Acceptance criteria
- NPZ file loads successfully and contains all required keys.
- X and A arrays have rank 4 and match the expected [C,N,L,T] shapes.
- Toy raster tests in spec/16 pass for connected components and branching metrics.

## Cross references
- spec/03_SPIKE_DEFS_AND_STANDARDIZATION.md
- spec/04_RATE_MATCHING_PROTOCOL.md
- spec/05_AVALANCHE_DEFINITION_AND_STATS.md
- spec/06_BRANCHING_METRICS_AND_DELTAB.md
- spec/07_NULL_MODELS_AND_CONTROLS.md
- spec/16_TEST_PLAN.md
