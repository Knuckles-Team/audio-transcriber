# audio-transcriber

Transcribe `.wav`, `.mp4`, `.mp3`, and `.flac` files to text — or record your own
audio — through a CLI, a Python API, an **MCP server**, and an A2A agent, built on
the agent-utilities ecosystem.

!!! info "Official documentation"
    This site is the canonical reference for `audio-transcriber`, maintained
    alongside every release.

[![PyPI](https://img.shields.io/pypi/v/audio-transcriber)](https://pypi.org/project/audio-transcriber/)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
[![License](https://img.shields.io/pypi/l/audio-transcriber)](https://github.com/Knuckles-Team/audio-transcriber/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/Knuckles-Team/audio-transcriber)

## Overview

`audio-transcriber` wraps OpenAI Whisper — via the fast
[`faster-whisper`](https://pypi.org/project/faster-whisper/) (CTranslate2) backend
with an `openai-whisper` fallback — behind a typed, deterministic tool surface. It
provides:

- **`AudioTranscriber`** — a Python class that records microphone audio, transcribes
  local media files, and exports `txt` / `srt` / `vtt` / `json` results.
- **An MCP server** (`audio-transcriber-mcp`) exposing the `transcribe_audio` tool
  for agents and IDE assistants.
- **An A2A agent** (`audio-transcriber-agent`) that drives the MCP tools over the
  Agent Control Protocol with an optional web interface.

Transcription runs entirely in process — the Whisper model is loaded locally, so no
external transcription service is required.

## Explore the documentation

<div class="grid cards" markdown>

- :material-rocket-launch: **[Installation](installation.md)** — pip, source, extras, and the prebuilt Docker image.
- :material-server-network: **[Deployment](deployment.md)** — run the MCP server and the agent, Docker Compose, Caddy + Technitium.
- :material-console: **[Usage](usage.md)** — the MCP tool surface, the `AudioTranscriber` API, and the CLI.
- :material-sitemap: **[Overview](overview.md)** — capability summary and ecosystem role.
- :material-tag-multiple: **[Concepts](concepts.md)** — the `CONCEPT:AUDIO-*` registry.

</div>

## Quick start

```bash
pip install "audio-transcriber[mcp]"
audio-transcriber-mcp            # stdio MCP server (default transport)
```

Transcribe a file directly from the command line:

```bash
audio-transcriber --file '~/Downloads/meeting.mp4' --model base --export
```

See **[Installation](installation.md)** and **[Deployment](deployment.md)** for the
full matrix (PyPI extras, Docker image, all transports, reverse proxy, DNS).
