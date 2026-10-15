# TRANSCRIBE — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | TRANSCRIBE-R001.1 | Contract/integration | Raw audio payload to transcribe endpoint | Local model inference returns text transcript; network audit shows zero egress. |  |
| T-002 | TRANSCRIBE-R001.2 | Contract/integration | Speech input to local inference | API response contains only text; audio payload absent. |  |
| T-003 | TRANSCRIBE-R001.3 | Contract/integration | SDK calls transcribe; EG consumer validates output | Transcript schema valid; origin traced to audio-transcriber; traceability chain intact. |  |
| T-004 | TRANSCRIBE-R001.4 | Negative | Misconfigured remote endpoint setting | Refusal logged; transmission blocked; no network call made. |  |
| T-005 | TRANSCRIBE-R001.5 | Contract/integration | End-to-end local inference | Positive case passes; negative case blocks external transmission. |  |
| T-006 | TRANSCRIBE-R001.6 | Integration | pytest test_transcription_local_egress_zero | Test passes; local inference completes; zero external calls observed. |  |

## Negative and boundary cases

Misconfigured remote endpoint: refusal gate blocks transmission. Invalid audio: graceful error. Network interruption during refusal check: transcript not sent. Idempotency: same audio input produces same transcript.

## Quality and release proof

Command: pytest tests/integration/test_transcription_sdk.py::test_transcription_local_egress_zero -v. Environment: local model inference, network audit enabled. Proof: test passes; audit shows zero external calls. Evidence: CI log and test result.

