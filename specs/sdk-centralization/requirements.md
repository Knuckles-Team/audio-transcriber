# CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A child row `.n` is one five-minute unit of work. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION-R001` | **Capabilities come from the SDK (rollup).** The connector must use the SDK for HTTP, retry, TLS and optional dependencies. Children `.1` to `.2` deliver it. | See each child. |
| `CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION-R001.1` | **TLS profile for secure streams.** A `wss://` connection to the PersonaPlex server must take its SSL context from `resolve_tls_profile` for the `personaplex` prefix (`SCION-CENTRALIZED-CAPABILITIES-R008`). The client must not build an SSL context or disable verification. A `ws://` connection uses no TLS. Proving test: `tests/test_sdk_transport.py::test_secure_stream_uses_the_tls_profile`. | A `wss://` connect passes the profile context, and a `ws://` connect passes none. |
| `CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION-R001.2` | **Stream dependency from the SDK.** The `mcp` extra must take the websockets driver from the SDK `stream` extra and must not pin `websockets` itself (`SCION-CENTRALIZED-CAPABILITIES-R040`). The SDK floor must be `0.3.11` or later. Proving test: `tests/test_sdk_transport.py::test_websockets_comes_from_the_sdk_stream_extra`. | `pyproject.toml` pins no `websockets` and requests `scion-sdk[mcp,stream]`. |
