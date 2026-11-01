# CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT — The connector serves its packaged content

Owner: audio-transcriber. Cross-repo IDs: `SCION-CONNECTOR-CONTROL-R071`, `SCION-CONNECTOR-CONTROL-R073`.

## Purpose and user stories

The `audio-transcriber-mcp` server serves its packaged skills, prompts and ontology to any client, with no wiring of its own.

## Why a new spec

The specs of this repository were read; none owns this capability. This spec states it for this connector only.

## Scope and interfaces

In scope: the rows of [requirements.md](requirements.md). Out of scope: the SDK modules and the engine, which own the shared capabilities.

## User Scenarios & Testing

### User Story 1 - The connector serves its packaged content (Priority: P1)

The `audio-transcriber-mcp` server serves its packaged skills, prompts and ontology to any client, with no wiring of its own.

**Acceptance Scenarios**:

1. **Given** the connector is configured, **When** the behaviour of row `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R002` (transcript list/get operation by recording id and date) is exercised, **Then** transcript list/get operation by recording id and date.
2. **Given** the connector is configured, **When** the behaviour of row `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R003` (expose avg_logprob (confidence) per transcript segment) is exercised, **Then** expose avg_logprob (confidence) per transcript segment.
3. **Given** the connector is configured, **When** the behaviour of row `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R001` (default content) is exercised, **Then** `resources/list` includes `skill://` and `ontology://`; `prompts/list` includes each packaged prompt. Planned test: `tests/test_default_content_surfaces.py::test_default_content_lists_skill_and_ontology_resources_and_prompts`.

### Edge Cases

- If `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R002` is not met, its proving test fails: transcript list/get operation by recording id and date.
- If `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R003` is not met, its proving test fails: expose avg_logprob (confidence) per transcript segment.
- If `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R001` is not met, its proving test fails: default content.

## Requirements

- **FR-001**: Transcript list/get operation by recording id and date. Rows: `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R002`.
- **FR-002**: Expose avg_logprob (confidence) per transcript segment. Rows: `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R003`.
- **FR-003**: Default content. Rows: `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R001`.

## Success Criteria

- **SC-001**: Transcript list/get operation by recording id and date. Rows: `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R002`.
- **SC-002**: Expose avg_logprob (confidence) per transcript segment. Rows: `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R003`.
- **SC-003**: `resources/list` includes `skill://` and `ontology://`; `prompts/list` includes each packaged prompt. Planned test: `tests/test_default_content_surfaces.py::test_default_content_lists_skill_and_ontology_resources_and_prompts`. Rows: `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R001`.

## Assumptions

- The external system is reachable with operator-supplied credentials.
- The SDK and engine contracts the rows cite are at the floors declared in `pyproject.toml`.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R002` | [plan.md](plan.md) | T-001 | [test-spec.md](test-spec.md) |
| FR-002 | `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R003` | [plan.md](plan.md) | T-002 | [test-spec.md](test-spec.md) |
| FR-003 | `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R001` | [plan.md](plan.md) | T-003 | [test-spec.md](test-spec.md) |
