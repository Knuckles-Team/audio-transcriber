# TRANSCRIBE — Local-model speech transcription with egress controls

Owner: audio-transcriber. Cross-repo IDs: agent-connector-sdk, medulla.
Owner: audio-transcriber. Cross-repo IDs: scion-sdk, medulla.
Public provenance: EG:49 (spec audit 2026-10-15).

## Purpose and user stories

Speech input routes locally to inference without external transmission. Raw audio egress is prohibited. SDK governs integration with medulla transcript consumers.

## Why a new spec

No existing spec owns local transcription execution contracts or egress refusal policy. agent-connector-sdk owns connector transport. medulla owns transcript consumption. This spec bridges them.

## Scope and interfaces
audio-transcriber owns: local model execution, raw audio handling, egress checks, SDK transport policy links. agent-connector-sdk owns: connector transport contracts. medulla owns: transcript consumer integration. Non-goals: model selection, training, or tuning.
No existing spec owns local transcription execution contracts or egress refusal policy. scion-sdk owns connector transport. medulla owns transcript consumption. This spec bridges them.
audio-transcriber owns: local model execution, raw audio handling, egress checks, SDK transport policy links. scion-sdk owns: connector transport contracts. medulla owns: transcript consumer integration. Non-goals: model selection, training, or tuning.

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| FR-001 | plan.md Local model authority | T-001 |  |
| FR-002 | plan.md Refusal execution | T-002 |  |
| SC-001 | test-spec.md Negative cases | T-003 |  |

## Open questions

None at entry.

## User Scenarios & Testing

### User Story 1 - Local-model speech transcription with egress controls (Priority: P1)

Speech input routes locally to inference without external transmission. Raw audio egress is prohibited. SDK governs integration with medulla transcript consumers.

**Acceptance Scenarios**:

1. **Given** the connector is configured, **When** the behaviour of row `TRANSCRIBE-R001` (local-model speech transcription with egress controls) is exercised, **Then** integration test proves local execution, zero external transmission, and refusal gates.

### Edge Cases

- If `TRANSCRIBE-R001` is not met, its proving test fails: local-model speech transcription with egress controls.

## Requirements

- **FR-001**: Local-model speech transcription with egress controls. Rows: `TRANSCRIBE-R001`.

## Success Criteria

- **SC-001**: Integration test proves local execution, zero external transmission, and refusal gates. Rows: `TRANSCRIBE-R001`.

## Assumptions

- The external system is reachable with operator-supplied credentials.
- The SDK and engine contracts the rows cite are at the floors declared in `pyproject.toml`.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `TRANSCRIBE-R001` | [plan.md](plan.md) | T-001 | [test-spec.md](test-spec.md) |
