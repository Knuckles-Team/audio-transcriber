# CONN-AUDIO-TRANSCRIBER — Read-only operations

Owner: audio-transcriber. Cross-repo IDs: none.

## Purpose and user stories

A fleet operator must check the transcription engine without a transcription run. The connector must expose one read-only operation for that check.

## Requirements and acceptance

- FR-001: The connector must expose `health_check` with the read-only annotation in `audio_transcriber/mcp_server.py`. The tool must report the importable backends, ffmpeg presence, the default model and the transcription directory without a model load.
- SC-001: the tool is listed read-only, reports the engine fields without a model load, and reports DEGRADED when no backend is importable.

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
