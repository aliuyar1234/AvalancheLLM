# 22 RISK REGISTER

## Purpose and scope
- Provide the operational plan for this part of the project.

## Normative requirements
- MUST follow CANON keys and never hardcode values.
- MUST be executable by Codex without guessing.

## Definitions
- All identifiers referenced are defined in spec/00.

## Procedure
Step by step instructions are defined in tasks folder and referenced here.
This spec defines the required outputs and checks for this topic.

## Worked example
Example: See tasks/PHASE_7_PAPER_CAMERA_READY.md for concrete commands and expected outputs.

## Failure modes
1. Missing required output.
   Detect: phase acceptance fails.
   Fix: rerun phase.
2. Drift in IDs.
   Detect: mismatch with CANON.
   Fix: update references.
3. Unclear responsibility.
   Detect: Codex deviates.
   Fix: strengthen contract in this spec.

## Acceptance criteria
- All acceptance checks for this spec topic pass.

## Cross references
- tasks/TASK_INDEX.md
- spec/00_CANONICAL.md

