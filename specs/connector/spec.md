# CONN-AUDIO-TRANSCRIBER — Read-only operations

Owner: audio-transcriber. Cross-repo IDs: none.

## Purpose and user stories

A fleet operator must check the transcription engine without a transcription run. The connector must expose one read-only operation for that check.

## Why a new spec

The `specs/` tree held only the template and the index.
No local spec owned this behavior, so this spec is the first capability spec.

## Scope and interfaces

The connector owns its operations and their MCP tool annotations.
The connector SDK owns tool registration, transport and the shared client contracts.
Code: ../../audio_transcriber/mcp_server.py ../../tests/test_read_only_health.py

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| FR-001 | plan.md#architecture | T-001 | `tests/test_read_only_health.py` |

## Open questions

- None.

## User Scenarios & Testing

### User Story 1 - Read-only operations (Priority: P1)

A fleet operator must check the transcription engine without a transcription run. The connector must expose one read-only operation for that check.

**Acceptance Scenarios**:

1. **Given** the connector is configured, **When** the behaviour of row `CONN-AUDIO-TRANSCRIBER-R001` (read-only engine health) is exercised, **Then** `tests/test_read_only_health.py` must show: the tool is listed read-only, reports the engine fields without a model load, and reports DEGRADED when no backend is importable. Test: `tests/test_read_only_health.py::test_health_check_is_listed_read_only`.
2. **Given** the connector is configured, **When** the behaviour of row `CONN-AUDIO-TRANSCRIBER-R002` (fleet name) is exercised, **Then** every connector name in the package is `audio-transcriber-mcp`.
3. **Given** the connector is configured, **When** the behaviour of row `CONN-AUDIO-TRANSCRIBER-R003` (microphone extra) is exercised, **Then** the `microphone` extra lists `pyaudio`; the PR workflow passes `sync-args` that omit it. Test: `tests/test_microphone_extra.py::test_microphone_extra_is_declared`.

### Edge Cases

- If `CONN-AUDIO-TRANSCRIBER-R001` is not met, its proving test fails: read-only engine health.
- If `CONN-AUDIO-TRANSCRIBER-R002` is not met, its proving test fails: fleet name.
- If `CONN-AUDIO-TRANSCRIBER-R003` is not met, its proving test fails: microphone extra.

## Requirements

- **FR-001**: Read-only engine health. Rows: `CONN-AUDIO-TRANSCRIBER-R001`.
- **FR-002**: Fleet name. Rows: `CONN-AUDIO-TRANSCRIBER-R002`.
- **FR-003**: Microphone extra. Rows: `CONN-AUDIO-TRANSCRIBER-R003`.
- **FR-004**: Meeting segment tools. Rows: `CONN-AUDIO-TRANSCRIBER-R004`.

## Success Criteria

- **SC-001**: `tests/test_read_only_health.py` must show: the tool is listed read-only, reports the engine fields without a model load, and reports DEGRADED when no backend is importable. Rows: `CONN-AUDIO-TRANSCRIBER-R001`.
- **SC-002**: Every connector name in the package is `audio-transcriber-mcp`. Rows: `CONN-AUDIO-TRANSCRIBER-R002`.
- **SC-003**: The `microphone` extra lists `pyaudio`; the PR workflow passes `sync-args` that omit it. Rows: `CONN-AUDIO-TRANSCRIBER-R003`.
- **SC-004**: Meeting segment tools. Rows: `CONN-AUDIO-TRANSCRIBER-R004`.

## Assumptions

- The external system is reachable with operator-supplied credentials.
- The SDK and engine contracts the rows cite are at the floors declared in `pyproject.toml`.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-AUDIO-TRANSCRIBER-R001` | [plan.md](plan.md) | T-001 | [test-spec.md](test-spec.md) |
| FR-002 | `CONN-AUDIO-TRANSCRIBER-R002` | [plan.md](plan.md) | T-002 | [test-spec.md](test-spec.md) |
| FR-003 | `CONN-AUDIO-TRANSCRIBER-R003` | [plan.md](plan.md) | T-003 | [test-spec.md](test-spec.md) |
| FR-004 | `CONN-AUDIO-TRANSCRIBER-R004` | [plan.md](plan.md) | T-004 | [test-spec.md](test-spec.md) |
