# CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Architecture

`pyproject.toml` declares `audio-transcriber-mcp` as the connector identity. `create_mcp_server` resolves the packaged content of `audio_transcriber` from it.
