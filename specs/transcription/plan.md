# TRANSCRIBE — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

audio-transcriber owns transcription via local models. agent-connector-sdk owns transport contracts. medulla owns transcript consumers. No duplicated transcription authority. audio-transcriber owns transcription via local models. scion-sdk owns transport contracts. medulla owns transcript consumers. No duplicated transcription authority. Speech input enters audio-transcriber transcribe endpoint. Local model inference produces text output. SDK contracts govern egress checks: raw audio blocked, transcripts validated, origin traced to medulla. Refusal gates block misconfigured remote endpoints before transmission.

## Technical Context

- **Language/Version**: Python 3.14 (floor 3.12)
- **Primary Dependencies**: scion-sdk, fastmcp
- **Storage**: N/A; the connector stores nothing
- **Testing**: pytest; the proof level each row names
- **Target Platform**: Linux containers
- **Project Type**: connector (MCP server)
- **Performance Goals**: N/A
- **Constraints**: one tool per API operation; shared capabilities come from the SDK
- **Scale/Scope**: 7 rows

## Constitution Check

- One owner per concept: pass. The connector owns one external system's calls, mappings and tools only.
- Connectors depend on the SDK and are called through arbora: pass.
- No ratchets: pass. The rows add no suppression.

## Project Structure

```text
audio_transcriber/mcp_server.py  existing
audio_transcriber/audio_transcriber.py  existing
```

## Phase 0: Research

No unknowns. Each row names the code it touches.

## Phase 1: Design

### Interfaces and data model

Transcribe endpoint accepts raw audio input via ConnectorRequest. Response schema carries transcribed text only, not audio. SDK transport policy enforces audio payload exclusion. EG consumer validates transcript schema and origin.

### Live integration path

SDK calls audio-transcriber transcribe. audio-transcriber invokes local model. Output text flows to medulla transcript consumer. Refusal checks intercept before transmission. No external inference APIs called.

### Quality and release gates

Contract gate: SDK transport policy verified. Integration test: test_transcription_local_egress_zero proves zero external calls. Language-native: pytest test suite. KISS: no redundant inference authority.

### Risks, alternatives, and decisions

Risk: Model size impacts latency; local inference chosen over remote for security. Decision: Refusal gates prevent misconfiguration egress; tests verify. Authority: audio-transcriber owns execution; SDK owns transport.

## Phase 2: Task approach

`tasks.md` holds one task line per row, producer first.

## Complexity Tracking

None. The constitution check passed.

## Build contract

The repository pre-commit hooks apply: `ruff-format`, `connector-standard`, `secrets`, `tests-fast`. Each row names its proving test. No suppression and no duplicate implementation.

## Production caller chain

1. `TRANSCRIBE-R001.1`: The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve` and loads `audio_transcriber/__init__.py::connector`; the served tool `transcribe_audio` in `audio_transcriber/mcp_server.py` (line 108) calls `audio_transcriber/audio_transcriber.py::AudioTranscriber.transcribe` (line 363), which runs `FasterWhisperBackend.transcribe` or `OpenAIWhisperBackend.transcribe` locally.
2. `TRANSCRIBE-R001.2`: The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve` and loads `audio_transcriber/__init__.py::connector`; the served tool `transcribe_audio` in `audio_transcriber/mcp_server.py` (line 108) calls `audio_transcriber/audio_transcriber.py::AudioTranscriber.transcribe` (line 363), which runs `FasterWhisperBackend.transcribe` or `OpenAIWhisperBackend.transcribe` locally.
3. `TRANSCRIBE-R001.3`: The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve` and loads `audio_transcriber/__init__.py::connector`; the served tool `transcribe_audio` in `audio_transcriber/mcp_server.py` (line 108) calls `audio_transcriber/audio_transcriber.py::AudioTranscriber.transcribe` (line 363), which runs `FasterWhisperBackend.transcribe` or `OpenAIWhisperBackend.transcribe` locally.
4. `TRANSCRIBE-R001.4`: The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve` and loads `audio_transcriber/__init__.py::connector`; the served tool `transcribe_audio` in `audio_transcriber/mcp_server.py` (line 108) calls `audio_transcriber/audio_transcriber.py::AudioTranscriber.transcribe` (line 363), which runs `FasterWhisperBackend.transcribe` or `OpenAIWhisperBackend.transcribe` locally.
5. `TRANSCRIBE-R001.5`: The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve` and loads `audio_transcriber/__init__.py::connector`; the served tool `transcribe_audio` in `audio_transcriber/mcp_server.py` (line 108) calls `audio_transcriber/audio_transcriber.py::AudioTranscriber.transcribe` (line 363), which runs `FasterWhisperBackend.transcribe` or `OpenAIWhisperBackend.transcribe` locally.
6. `TRANSCRIBE-R001.6`: The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve` and loads `audio_transcriber/__init__.py::connector`; the served tool `transcribe_audio` in `audio_transcriber/mcp_server.py` (line 108) calls `audio_transcriber/audio_transcriber.py::AudioTranscriber.transcribe` (line 363), which runs `FasterWhisperBackend.transcribe` or `OpenAIWhisperBackend.transcribe` locally.

## Cross-repo seams

none
