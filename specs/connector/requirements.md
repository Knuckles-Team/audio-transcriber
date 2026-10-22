# CONN-AUDIO-TRANSCRIBER — Requirements

Governing spec: [spec.md](spec.md).

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-AUDIO-TRANSCRIBER-R001` | **Read-only engine health.** The connector must expose `health_check` with the read-only annotation in `audio_transcriber/mcp_server.py`. The tool must report the importable backends, ffmpeg presence, the default model and the transcription directory without a model load. | `tests/test_read_only_health.py` must show: the tool is listed read-only, reports the engine fields without a model load, and reports DEGRADED when no backend is importable. |
| `CONN-AUDIO-TRANSCRIBER-R002` | **Fleet name.** `pyproject.toml` declares `audio-transcriber-mcp` as the single `agent_connector_sdk.connector` entry point. The content pack, the manifest `connector`, the fingerprints and the ingest binding all carry `audio-transcriber-mcp`. Proving test: `tests/test_fleet_name.py::test_connector_name_is_the_fleet_name`. | Every connector name in the package is `audio-transcriber-mcp`. |
