# TRANSCRIBE — Local-model speech transcription with egress controls

Owner: audio-transcriber. Cross-repo IDs: agent-connector-sdk, epistemic-graph.
Public provenance: EG:49 (spec audit 2026-10-15).

## Purpose and user stories

Speech input routes locally to inference without external transmission. Raw audio egress is prohibited. SDK governs integration with epistemic-graph transcript consumers.

## Requirements and acceptance

- FR-001: Local inference transcribes speech; raw audio never leaves the host.
- FR-002: Misconfiguration refusals prevent egress before transmission occurs.
- SC-001: Integration tests prove local model execution and zero external transmission.

## Why a new spec

No existing spec owns local transcription execution contracts or egress refusal policy. agent-connector-sdk owns connector transport. epistemic-graph owns transcript consumption. This spec bridges them.

## Scope and interfaces

audio-transcriber owns: local model execution, raw audio handling, egress checks, SDK transport policy links. agent-connector-sdk owns: connector transport contracts. epistemic-graph owns: transcript consumer integration. Non-goals: model selection, training, or tuning.

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| FR-001 | plan.md Local model authority | T-001 |  |
| FR-002 | plan.md Refusal execution | T-002 |  |
| SC-001 | test-spec.md Negative cases | T-003 |  |

## Open questions

None at entry.

