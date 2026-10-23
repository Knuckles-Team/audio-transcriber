# CONN-AUDIO-TRANSCRIBER — Requirements

Governing spec: [spec.md](spec.md).

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-AUDIO-TRANSCRIBER-R001` | **Read-only engine health.** The connector must expose `health_check` with the read-only annotation in `audio_transcriber/mcp_server.py`. The tool must report the importable backends, ffmpeg presence, the default model and the transcription directory without a model load. | `tests/test_read_only_health.py` must show: the tool is listed read-only, reports the engine fields without a model load, and reports DEGRADED when no backend is importable. |
| `CONN-AUDIO-TRANSCRIBER-R002` | **Fleet name.** `pyproject.toml` declares `audio-transcriber-mcp` as the single `agent_connector_sdk.connector` entry point. The content pack, the manifest `connector`, the fingerprints and the ingest binding all carry `audio-transcriber-mcp`. Proving test: `tests/test_fleet_name.py::test_connector_name_is_the_fleet_name`. | Every connector name in the package is `audio-transcriber-mcp`. |
| `CONN-AUDIO-TRANSCRIBER-R003` | **Microphone extra.** `pyproject.toml` declares the `microphone` optional extra with `pyaudio`, so `pip install audio-transcriber[microphone]` installs microphone recording. The PR pipeline installs the extras without `microphone` (the `sync-args` input of the shared pipeline), because `pyaudio` needs PortAudio headers that CI runners lack. Proving test: `tests/test_microphone_extra.py`. | The `microphone` extra lists `pyaudio`; the PR workflow passes `sync-args` that omit it. |
