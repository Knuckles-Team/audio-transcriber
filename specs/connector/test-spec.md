# CONN-AUDIO-TRANSCRIBER — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | FR-001 | Unit | Mocked HTTP | the tool is listed read-only, reports the engine fields without a model load, and reports DEGRADED when no backend is importable. | `tests/test_read_only_health.py` |

## Negative and boundary cases

The tests cover the error path that each row names.

## Quality and release proof

Run `pytest -q`. The served proof is the fleet acceptance check through the gateway.
| T-002 | `CONN-AUDIO-TRANSCRIBER-R002` | `tests/test_fleet_name.py::test_connector_name_is_the_fleet_name` | Every connector name in the package is `audio-transcriber-mcp`. |
| T-003 | `CONN-AUDIO-TRANSCRIBER-R003` | `tests/test_microphone_extra.py::test_microphone_extra_is_declared` | The `microphone` extra declares `pyaudio`. |
