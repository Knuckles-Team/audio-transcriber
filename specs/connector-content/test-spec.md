# CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R001` | `tests/test_default_content_surfaces.py` | The listings include the packaged skills, ontology and prompts. |

## Rows awaiting bound tests
- `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R002`: planned proof `tests/test_mcp_server.py::test_transcript_list_filters_by_recording_id_and_date`. Expected: A fixture transcript is returned for its recording id and date with the documented fields.. Refusal: An unknown recording id or a date outside the range returns an empty list, and a malformed date is refused..
- `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R003`: planned proof `tests/test_kg_ingest.py::test_ingest_transcription_carries_avg_logprob_per_segment`. Expected: Every returned segment carries avg_logprob equal to the backend value.. Refusal: A backend without avg_logprob yields a null value for the segment and the result is still returned..
