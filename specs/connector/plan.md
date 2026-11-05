# CONN-AUDIO-TRANSCRIBER — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

The `health_check` tool existed and returned a fixed value without an annotation. This row makes it read the engine state. The tool reads the installed backends, the ffmpeg path, the default model and the transcription directory. It loads no model and writes nothing.

## Technical Context

- **Language/Version**: Python 3.14 (floor 3.12)
- **Primary Dependencies**: scion-sdk, fastmcp
- **Storage**: N/A; the connector stores nothing
- **Testing**: pytest; the proof level each row names
- **Target Platform**: Linux containers
- **Project Type**: connector (MCP server)
- **Performance Goals**: N/A
- **Constraints**: one tool per API operation; shared capabilities come from the SDK
- **Scale/Scope**: 4 rows

## Constitution Check

- One owner per concept: pass. The connector owns one external system's calls, mappings and tools only.
- Connectors depend on the SDK and are called through arbora: pass.
- No ratchets: pass. The rows add no suppression.

## Project Structure

```text
audio_transcriber/mcp_server.py  existing
pyproject.toml  existing
tests/test_microphone_extra.py  existing
audio_transcriber/__init__.py  existing
```

## Phase 0: Research

No unknowns. Each row names the code it touches.

## Phase 1: Design

### Interfaces and data model

The change adds no wire field. The tool returns the API response document.

### Live integration path

The fleet gateway lists the tool, reads its annotation and calls it as the gateway identity.

### Quality and release gates

The focused tests run with `pytest -q`. A merge to the default branch publishes the package and the image.

### Risks, alternatives, and decisions

No new risk. The operation reads state and changes nothing.

## Phase 2: Task approach

`tasks.md` holds one task line per row, producer first.

## Complexity Tracking

None. The constitution check passed.

## Build contract

The repository pre-commit hooks apply: `ruff-format`, `connector-standard`, `secrets`, `tests-fast`. Each row names its proving test. No suppression and no duplicate implementation.

## Production caller chain

1. `CONN-AUDIO-TRANSCRIBER-R001`: The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve`, which builds the server from `audio_transcriber/__init__.py::connector` (tools=`audio_transcriber.mcp_server`) and serves `audio_transcriber/mcp_server.py::register_misc_tools.health_check`, which returns `audio_transcriber/mcp_server.py::engine_health`.
2. `CONN-AUDIO-TRANSCRIBER-R002`: The `audio-transcriber-mcp` console script in pyproject.toml runs `scion_sdk.cli:serve`, which loads the single `scion_sdk.connector` entry point `audio-transcriber-mcp = audio_transcriber:connector` defined in `audio_transcriber/__init__.py::connector`.
3. `CONN-AUDIO-TRANSCRIBER-R003`: A user runs `pip install audio-transcriber[microphone]`, which resolves the `microphone` extra in pyproject.toml; the `audio-transcriber` console script (`audio_transcriber.audio_transcriber:main` entry `audio_transcriber()`) then records through `audio_transcriber/audio_transcriber.py::AudioTranscriber.record`.
4. `CONN-AUDIO-TRANSCRIBER-R004`: (planned) The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve` and loads `audio_transcriber/__init__.py::connector`; the four meeting tools will be registered from `audio_transcriber/mcp_server.py` through `extra_registrations` like the existing `register_prompts` hook, and no meeting code exists in the package today.
5. `CONN-AUDIO-TRANSCRIBER-R005`: (planned) The `audio-transcriber-mcp` console script runs `scion_sdk.cli:serve` and loads `audio_transcriber/__init__.py::connector`; a second `extra_registrations` hook calls `register_transcript_tools` in `audio_transcriber/mcp_server.py`, which registers `list_transcripts` and `get_transcript`; each reads the `<id>.json` exports under `DEFAULT_TRANSCRIBE_DIRECTORY`.

## Cross-repo seams

none
