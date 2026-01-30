# PHASE 7 PAPER CAMERA READY

## Purpose
Phase 7 turns the already-exported tables and figures into a camera-ready paper draft. It MUST not run model inference. It only formats, inserts provenance, and validates references.

This phase produces:
- A complete set of paper markdown sections.
- A claim-to-evidence map tying each claim to figure and table IDs and the producing run ids.
- A reproducibility statement and artifact checklist completion.

## Prerequisites
- Phase 5 completed and exported all Dataset A tables and figures.
- Phase 6 completed and exported Dataset B and ARC tables and figures.
- The bib file bib/references.bib contains every citation key used in the paper markdown.
- MANIFEST.sha256 exists and matches current repository state, or Phase 7 will request Phase 8 release rebuild.

## Commands
Execute:
- CANON.CLI.CMD.PHASE7_PAPER_EXPORT

The exporter MUST:
- Read run_record.json from the producing runs and extract run_id, config_hash, and artifact paths.
- Update the paper markdown files under paper to include:
  - figure references by canonical ID
  - table references by canonical ID
  - provenance footnotes with run ids and config hashes
- Validate that every cite key used exists in bib/references.bib.

## Expected outputs
Within the repository paper directory:
- paper/00_PAPER_SNAPSHOT.md updated with claim-to-evidence mapping that includes run ids.
- paper/01_ABSTRACT.md updated to match conservative claim language.
- paper/02_INTRO_NOVELTY.md updated to reference the exported artifacts.
- paper/03_RELATED_WORK.md updated to cite only keys present in bib/references.bib.
- paper/04_METHOD.md updated to match spec definitions.
- paper/05_EXPERIMENTS.md updated to match the config and budgets.
- paper/06_RESULTS.md updated to include the latest table and figure references and provenance.
- paper/07_DISCUSSION_LIMITATIONS_ETHICS.md updated to include limitations and ethics.
- paper/08_APPENDIX.md updated for appendix figures and extra robustness checks.
- tables/appendix/table_T07_REPLICATION_SUMMARY.csv and .parquet written into the Phase 7 run directory when both BASE and INSTRUCT Phase 5 runs are available.

Optionally, if a PDF build tool is installed, the phase MAY also produce:
- paper/build/paper_draft.pdf
This optional artifact MUST be recorded in run_record.json if produced.

## Definition of Done
All of the following are true:
- Every figure and table referenced in the paper exists in at least one producing run directory.
- Every citation key referenced in paper files exists in bib/references.bib.
- The paper snapshot claim-to-evidence map includes run ids and config hashes for each main artifact.
- No forbidden markers are present in any paper file.

## Acceptance tests
1. Citation lint:
- Scan paper for cite keys and verify each exists in bib/references.bib.
2. Artifact lint:
- Scan paper for figure and table IDs and verify the corresponding files exist under the producing run directories.
3. Provenance lint:
- Verify that paper snapshot lists run ids for T01, T02, and T03 and for F03, F06, and F07.

## Common failures and fixes
1. Missing artifact paths.
   Detect: exporter cannot find a figure file referenced by CANON.OUTPUT.
   Fix: re-run Phase 5 or Phase 6 exporters and ensure filenames match CANON.
2. Bib key mismatch.
   Detect: cite key used in paper does not exist in references.bib.
   Fix: add a verified BibTeX entry or remove the cite from paper.
3. Drift between paper and spec.
   Detect: method section wording contradicts spec definitions.
   Fix: treat spec files as normative and update paper to match.

## Cross references
- spec/20_PAPER_WRITING_PLAN.md
- spec/21_SUBMISSION_CHECKLIST.md
- checklists/artifact_checklist_iclr.md
