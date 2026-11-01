# CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

`pyproject.toml` declares `audio-transcriber-mcp` as the connector identity. `create_mcp_server` resolves the packaged content of `audio_transcriber` from it.

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
audio_transcriber/mcp_server.py  existing
audio_transcriber/connector_sources.py  existing
pyproject.toml  existing
audio_transcriber/__init__.py  existing
```

## Phase 0: Research

No unknowns. Each row names the code it touches.

## Phase 1: Design

The rows are delivered in place; the design is the caller chain below.

## Phase 2: Task approach

`tasks.md` holds one task line per row, producer first.

## Complexity Tracking

None. The constitution check passed.

## Build contract

The repository pre-commit hooks apply: `ruff-format`, `connector-standard`, `secrets`, `tests-fast`. Each row names its proving test. No suppression and no duplicate implementation.

## Production caller chain

1. `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R002`: (planned) The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve`, loads `audio_transcriber/__init__.py::connector`, and arbora will call the new tool through `fleet.call`; no list or get tool exists, only `audio_transcriber/mcp_server.py::transcribe_audio`, `transcribe_media` and `audio_ingest_transcription`.
2. `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R003`: (planned) The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve`; the tools `transcribe_audio` and `audio_ingest_transcription` in `audio_transcriber/mcp_server.py` call `AudioTranscriber.transcribe`, whose `FasterWhisperBackend.transcribe` at `audio_transcriber/audio_transcriber.py:126` already builds each segment with `avg_logprob`.
3. `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R001`: The `audio-transcriber-mcp` console script in pyproject.toml runs `scion_sdk.cli:serve`, which loads the `scion_sdk.connector` entry point `audio-transcriber-mcp = audio_transcriber:connector` and builds the server from `audio_transcriber/__init__.py::connector`.

## Cross-repo seams

none. The cited owner rows are library capabilities the connector calls in process; spec.md names them.
