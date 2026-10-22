# CONN-AUDIO-TRANSCRIBER — Requirements

Governing spec: [spec.md](spec.md).

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-AUDIO-TRANSCRIBER-R001` | **Read-only engine health.** The connector must expose `health_check` with the read-only annotation in `audio_transcriber/mcp_server.py`. The tool must report the importable backends, ffmpeg presence, the default model and the transcription directory without a model load. | `tests/test_read_only_health.py` must show: the tool is listed read-only, reports the engine fields without a model load, and reports DEGRADED when no backend is importable. |
