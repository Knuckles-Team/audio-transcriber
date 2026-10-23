# CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | `CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION-R001` | Unit | Every child row. | See each child. |  |
| T-002 | `CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION-R001.1` | Unit | `tests/test_sdk_transport.py::test_secure_stream_uses_the_tls_profile` | A `wss://` connect passes the profile context, and a `ws://` connect passes none. |  |
| T-003 | `CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION-R001.2` | Unit | `tests/test_sdk_transport.py::test_websockets_comes_from_the_sdk_stream_extra` | `pyproject.toml` pins no `websockets` and requests `agent-connector-sdk[mcp,stream]`. |  |
