# 11 DATASETS SAMPLING AND HASHING

## Purpose and scope
- Define dataset selection, deterministic sampling, chunking, tokenization, and hashing.
- Ensure run records are sufficient to reproduce slices and prompts.

## Normative requirements
- MUST use datasets specified in CANON.DATASET with HF IDs.
- MUST record dataset revision string if provided by HF.
- MUST deterministically select sequence indices using PRNG seeded from CANON.CONST.BOOTSTRAP_SEED and run_id.
- MUST hash selected sample ids and token ids and store in run_record.json.

## Definitions
- Dataset A is Wikitext 103 validation. Dataset B is C4 en validation. ARC is ai2_arc ARC Challenge test.
- Chunking: concatenate text, tokenize with model tokenizer, split into windows of SEQ_LEN_TOKENS with stride equal to SEQ_LEN_TOKENS.
- Sample id is tuple dataset_role, chunk_index.

## Procedure
Procedure:
1. Load dataset split.
2. Build token windows with exact length.
3. Deterministically pick N windows by seeded PRNG.
4. For ARC, build MCQ prompts using template id ARC_MCQ_V1 and score options via log prob.
5. Compute sha256 hash over ordered list of selected sample ids and token ids.

## Worked example
Example: With SEQ_LEN_TOKENS 256, choose N=300 windows for Dataset A gain grid and N=300 for Dataset B evaluation. Hash list written to run_record.json as dataset_slice_sha256.

## Failure modes
1. HF dataset revision changes.
   Detect: revision string differs.
   Fix: pin dataset revision or record commit.
2. Tokenizer drift.
   Detect: tokenizer hash differs.
   Fix: pin tokenizer revision.
3. Non deterministic concatenation.
   Detect: sample hashes differ.
   Fix: enforce stable ordering and no multiprocessing in dataset map.

## Acceptance criteria
- Recomputing dataset slice hash yields same value.
- ARC prompt construction deterministic and hashed.

## Cross references
- spec/14 run_record schema
- tasks/PHASE_1 and PHASE_6

