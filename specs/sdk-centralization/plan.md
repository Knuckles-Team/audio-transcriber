# CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

The SDK owns the capability rows cited in [requirements.md](requirements.md). The connector adds no transport code of its own. The connector builds on the SDK. HTTP goes through the SDK governed client, and TLS comes from `resolve_tls_profile` for the connector's settings prefix.

## Technical Context

- **Language/Version**: Python 3.14 (floor 3.12)
- **Primary Dependencies**: scion-sdk, fastmcp
- **Storage**: N/A; the connector stores nothing
- **Testing**: pytest; the proof level each row names
- **Target Platform**: Linux containers
- **Project Type**: connector (MCP server)
- **Performance Goals**: N/A
- **Constraints**: one tool per API operation; shared capabilities come from the SDK
- **Scale/Scope**: 3 rows

## Constitution Check

- One owner per concept: pass. The connector owns one external system's calls, mappings and tools only.
- Connectors depend on the SDK and are called through arbora: pass.
- No ratchets: pass. The rows add no suppression.

## Project Structure

```text
audio_transcriber/personaplex_client.py  existing
tests/test_sdk_transport.py  existing
```

## Phase 0: Research

No unknowns. Each row names the code it touches.

## Phase 1: Design

### Interfaces and data model

No tool, argument or result changes.

### Live integration path

Every tool reaches the PersonaPlex streaming server through the connector client, which sends each request through the SDK.

### Quality and release gates

The focused tests in [test-spec.md](test-spec.md) must pass. The SDK floor must be the latest published release.

### Risks, alternatives, and decisions

A private certificate authority must be supplied through the TLS profile. A flag that disables verification must not return.

## Phase 2: Task approach

`tasks.md` holds one task line per row, producer first.

## Complexity Tracking

None. The constitution check passed.

## Build contract

The repository pre-commit hooks apply: `ruff-format`, `connector-standard`, `secrets`, `tests-fast`. Each row names its proving test. No suppression and no duplicate implementation.

## Production caller chain

1. `CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION-R001.1`: The `audio-transcriber` console script runs `audio_transcriber/audio_transcriber.py::audio_transcriber` with `--interact --server wss://...`, which calls `AudioTranscriber.interact` (line 528) and then `audio_transcriber/personaplex_client.py::PersonaPlexClient.connect` (line 20), which calls `resolve_tls_profile("personaplex")`.
2. `CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION-R001.2`: Installing the package resolves the `dependencies` list in pyproject.toml; the `websockets` driver imported by `audio_transcriber/personaplex_client.py` is then supplied by the `stream` extra of `scion-sdk[mcp,stream]` that the `audio-transcriber` console script path uses.

## Cross-repo seams

none. The cited owner rows are library capabilities the connector calls in process; spec.md names them.
