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

## Rows awaiting bound tests
- `CONN-AUDIO-TRANSCRIBER-R004`: planned proof `tests/test_meeting_tools.py::test_tools_listed_with_hints_and_tenant_from_context`. Expected: The served tool list contains meeting_segment_append, meeting_segment_correct, meeting_finalize and meeting_get_final with annotations.. Refusal: A call that passes a tenant argument is processed with the tenant from the authenticated context and the argument is ignored..
- `CONN-AUDIO-TRANSCRIBER-R005`: planned proof `tests/test_transcript_listing.py::test_list_and_get_by_recording_id_and_date` and `tests/test_transcript_listing.py::test_refusals_write_nothing`. Expected: Both tools are listed read-only; the list filters by recording id and date range in date-descending order within `limit`; `get_transcript` returns segments with `start`, `end` and `no_speech_prob`; each refusal code leaves the directory unchanged.
