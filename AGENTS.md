# AGENTS

This pack is designed as a single source of truth handoff to an autonomous coding agent. The implementation MUST be fail-closed: if a required artifact is missing or a schema check fails, the program MUST stop and return a non-zero exit code.

## Non-negotiable rules
- Do not invent citations. Only use bib/references.bib keys.
- Single Definition Rule: literals for IDs, paths, CLI commands, and constants are defined only in spec/00_CANONICAL.md.
- Never overwrite runs. Runs are immutable and append-only.
- Every producing CLI command writes run_record.json and config_resolved.yaml.
- Every exported table and figure must match the filenames in CANON.OUTPUT.
- Do not add new outputs without updating spec/18, spec/19, and CANON.

## Codex workflow
1. Read spec/00_CANONICAL.md.
2. Read spec/23_IMPLEMENTATION_CONTRACT.md and implement the required repository tree and CLI.
3. Implement phases in order, using tasks/TASK_INDEX.md as the runbook.
4. Implement the acceptance tests in spec/16_TEST_PLAN.md and make the pipeline fail if they do not pass.
5. Ensure Phase 7 paper export reads run_record.json and inserts run ids and config hashes into paper/00_PAPER_SNAPSHOT.md.

## Output expectations
At the end of Phase 8 release, the repository MUST contain:
- a populated runs directory with immutable run artifacts
- paper markdown updated with claim-to-evidence mapping
- MANIFEST.sha256 consistent with all files in the pack and produced artifacts
