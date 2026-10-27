---
name: audio-transcriber-development
skill_type: skill
description: >-
  Develop and maintain the Audio Transcriber connector while preserving its
  Whisper backend boundary, MCP tool surface, typed transcript ingestion,
  optional hardware behavior, package resources, and public-first evidence.
license: MIT
tags: [audio, transcription, whisper, connector, mcp, knowledge-graph]
metadata:
  author: Genius
  version: '0.1.0'
---
# Audio Transcriber Development

This is the repository development skill for `audio-transcriber`. The owning
central delivery row is `PIPE-CONNSPEC-R107.4`; use this skill with the
repository `AGENTS.md`, the connector-content specifications, and focused tests.

## Repository and runtime map

- `audio_transcriber/mcp_server.py` registers transcription, media-sidecar,
  health, and KG-ingestion tools plus the packaged prompts. Preserve the public
  tool names, progress reporting, and Connector registration boundary.
- `audio_transcriber/audio_transcriber.py` owns the `TranscriberBackend`
  interface, faster-whisper/openai-whisper selection, file and microphone paths,
  bounded exports, and best-effort KG handoff. Keep microphone dependencies
  optional so file transcription remains importable without PortAudio.
- `audio_transcriber/connector_sources.py` is the live transcript source and
  schema declaration. Change source IDs, record mappings, and resource names
  there rather than duplicating them in tool code.
- `audio_transcriber/kg_ingest.py` and `kg_media.py` map transcript documents,
  segments, and audio assets through the SDK/epistemic-graph ingestion boundary.
  Preserve stable IDs and no-op behavior when no engine is available.
- `audio_transcriber/skills/**`, `prompts/**`, `ontology/**`, and
  `connectors/**` are package data. A skill change needs a package-content check
  before release.

## Runtime and security contract

`WHISPER_MODEL` selects the default model and `TRANSCRIBE_DIRECTORY` selects
connector-owned scratch storage, with `XDG_DATA_HOME` supplying the safe default
root. Keep paths explicit and bounded; do not persist audio, transcripts, model
credentials, or provider data in source, skills, fixtures, or logs.

Preserve the optional `pyaudio` import boundary, the supported faster-whisper
and openai-whisper backends, `MCP_TOOL_MODE` surface selection, SDK transport
configuration, and the SDK credential/TLS settings inherited by the connector.
Model downloads and microphone access remain operator-controlled; never add a
network or hardware bypass.

## Rapid delivery: the 5-minute contract

1. Read the owning connector-content specs, this skill, `AGENTS.md`, and the
   directly bound tests before changing a surface.
2. Inspect the SDK `Connector`, MCP registrars and prompts, backend interface,
   file/microphone paths, source declarations, KG/media mappers, and package-data
   rules.
3. Make the smallest connector-owned change. Preserve public tool names and
   payloads, backend selection and optional dependencies, safe file/path
   handling, progress behavior, transcript source IDs, stable KG IDs, and
   canonical/legacy server routes.
4. Run the smallest directly bound tests with
   `uv run --locked --extra test pytest`, then run repository hooks. Use the
   approved narrow wrapper when the lane requires it; never bypass a hook or a
   failing gate.
5. If package content changes, advance the version with the repository's normal
   release metadata, refresh `uv.lock`, and inspect the built wheel and sdist
   for the intended `audio_transcriber/skills/**` bytes before publication.
6. Treat Pages and containers as separate evidence. After the exact main commit
   publishes, verify the Pages revision and `/spec-delivery/` artifacts. Keep
   package publication ahead of any image pin or generated connector build.

## Focused validation

For backend or MCP changes, run the smallest relevant test file and keep file,
microphone, optional-backend, and unavailable-KG paths distinct. For this
skill/package-data change, verify frontmatter, the required rapid-delivery
heading, Audio Transcriber-specific source references, package-data inclusion,
and absence of token-like material without loading a model or using hardware.

Use repository commands from `AGENTS.md`:

```bash
uv sync --locked
uv run --locked --extra test pytest -q tests/test_audio_transcriber_development_skill.py
uvx pre-commit run --all-files
```

## Publication and Pages

`.github/workflows/pipeline.yml` publishes the package through
`Knuckles-Team/pipelines/.github/workflows/python_pipeline.yml@main` with
`connector: true`, then invokes the container pipeline with
`render_connector: true`. `.github/workflows/pages.yml` delegates README/docs
publication to `pages_pipeline.yml@main`. Keep the public wheel/source guard
and package-before-image ordering intact.

A successful local test or workflow run is not package, Pages, or image proof.
Record exact merged source, public wheel/sdist hashes, skill bytes, Pages
revision, and image digest separately. Do not hand-edit central Pipelines
status, promote a rollup from this skill change, or put credentials, provider
data, generated release artifacts, or bypass instructions in this skill.
