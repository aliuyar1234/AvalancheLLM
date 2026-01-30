# 15 LOGGING SCHEMA

## Purpose and scope
- Define log event schema and required events for observability and postmortems.

## Normative requirements
- MUST write JSONL log file per run at logs/RUN_ID/events.jsonl.
- MUST include events: start, config_resolved, model_loaded, dataset_loaded, stage_complete, artifacts_written, end.
- SHOULD include timing per step.

## Definitions
- Each log event is JSON with fields: ts_utc, run_id, event_type, payload.
- event_type is one of the required set in this spec.

## Procedure
Procedure:
1. Open log file.
2. Emit start event.
3. Emit events at each major step.
4. Flush and close.
5. Validate each JSON line is parseable.

## Worked example
Example event: event_type model_loaded, payload contains hf_id and revision.

## Failure modes
1. Non JSON lines.
   Detect: parse fails.
   Fix: enforce json.dumps.
2. Missing end event.
   Detect: log ends unexpectedly.
   Fix: use try finally.
3. Event ordering inconsistent.
   Detect: stage_complete before start.
   Fix: central logger helper.

## Acceptance criteria
- Every run has events.jsonl with required event types.
- Logs are parseable.

## Cross references
- spec/14 run_record

