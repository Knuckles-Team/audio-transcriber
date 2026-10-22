# CONN-AUDIO-TRANSCRIBER — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Existing system and reuse

The `health_check` tool existed and returned a fixed value without an annotation. This row makes it read the engine state.

## Architecture

The tool reads the installed backends, the ffmpeg path, the default model and the transcription directory. It loads no model and writes nothing.

## Interfaces and data model

The change adds no wire field. The tool returns the API response document.

## Live integration path

The fleet gateway lists the tool, reads its annotation and calls it as the gateway identity.

## Quality and release gates

The focused tests run with `pytest -q`. A merge to the default branch publishes the package and the image.

## Risks, alternatives, and decisions

No new risk. The operation reads state and changes nothing.
