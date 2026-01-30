# 07 NULL MODELS AND CONTROLS

## Purpose and scope
- Specify the two required null controls:
  - input token shuffle null (model-level perturbation)
  - post-hoc within-layer time permutation null (raster-level, marginals-preserving)
- Define invariants and acceptance checks so nulls are auditable and deterministic.

## Normative requirements
- MUST represent the real, unpermuted condition with null_id = CANON.ENUM.NULL_ID.NULL_NONE.
- MUST represent null-generated conditions with null_id equal to:
  - CANON.ENUM.NULL_ID.NULL_TOKEN_SHUFFLE_INPUT
  - CANON.ENUM.NULL_ID.NULL_RASTER_WITHIN_LAYER_TIME_PERM
- MUST write null rasters into CANON.OUTPUT.NULL_NPZ_BASENAME using the NPZ schema in spec/02.
- MUST verify the within-layer time permutation null preserves per-layer marginals exactly for every sequence:
  - For each (condition, seq, layer), the multiset of A counts over tokens is unchanged.
- SHOULD record the null RNG seed derivation inputs in run_record.json:
  - run_id, condition id, sequence id, layer id.

## Definitions
Input token shuffle null:
- Acts on token ids before the forward pass.
- For each input sequence, permute token positions except keep special tokens fixed as defined by the tokenizer.

Within-layer time permutation null (marginals-preserving):
- Acts on extracted raster tensors after extraction.
- For each condition c, sequence n, layer l, sample a permutation pi_{c,n,l} over token indices.
- Apply to the token axis:
  - A_perm[c,n,l,t] = A[c,n,l,pi_{c,n,l}(t)]
  - X_perm is recomputed as occupancy from A_perm (strictly positive counts).

Delta connectivity:
- Compute branching metrics b on real rasters and b_perm on permuted rasters.
- Define delta_b = b - b_perm (spec/06).

## Procedure
### Procedure for within-layer time permutation null
1. Load real rasters from CANON.OUTPUT.RASTER_NPZ_BASENAME.
2. For each (c,n,l), derive a deterministic seed from:
   - global seed (CANON.CONST.BOOTSTRAP_SEED),
   - run_id string,
   - cond_id,
   - seq_id,
   - layer index.
3. Sample a permutation pi over token indices 1..T.
4. Apply the permutation to A along token axis and recompute X from permuted A.
5. Invariant check (hard-fail on violation):
   - sort the vector A[c,n,l,:] and A_perm[c,n,l,:] and require exact equality.
6. Write null rasters to CANON.OUTPUT.NULL_NPZ_BASENAME and register artifact in run_record.json.

### Procedure for input token shuffle null
1. For each input sequence, permute token ids at positions 2..T-1, keeping BOS and EOS fixed when present.
2. Run model forward pass with the permuted input and extract rasters as in Phase 3.
3. Store resulting rasters as a condition with null_id = CANON.ENUM.NULL_ID.NULL_TOKEN_SHUFFLE_INPUT.

## Worked example
Example for a single layer row:
- A over time is [0,5,0,1].
- A_perm could be [0,0,1,5].
The sorted vectors are identical, so the per-layer marginal multiset is preserved.

## Failure modes (detect and fix)
1. Marginals not preserved for within-layer permutation.
   Detect: sorted mismatch occurs for any (c,n,l).
   Fix: bug in indexing or using inconsistent permutation per axis; ensure permutation is applied only on token axis.
2. Shuffling breaks special tokens.
   Detect: tokenizer errors or degenerate outputs.
   Fix: keep tokenizer-defined special tokens fixed and permute only the interior positions.
3. Non-deterministic null generation.
   Detect: null rasters differ across identical runs with identical seeds and inputs.
   Fix: use a deterministic seed derivation rule and log all inputs used to derive the seed.

## Acceptance criteria
- Within-layer permutation null passes the marginal preservation check for every (c,n,l).
- Null rasters NPZ exists and follows the spec/02 schema.
- Delta branching metrics computed from nulls are finite and stable across repeated runs.

## Cross references
- spec/02_FORMAL_OBJECTS_EVENT_LATTICE.md
- spec/06_BRANCHING_METRICS_AND_DELTAB.md
- spec/11_DATASETS_SAMPLING_HASHING.md
- spec/16_TEST_PLAN.md
