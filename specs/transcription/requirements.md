# TRANSCRIBE — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A child row `.n` is one five-minute unit of work. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `TRANSCRIBE-R001` | **Local-model speech transcription with egress controls.** audio-transcriber routes speech to local inference, prohibits raw audio egress, and integrates transcripts with epistemic-graph consumers via SDK contracts. | Integration test proves local execution, zero external transmission, and refusal gates. |
| `TRANSCRIBE-R001.1` | **Model.** audio-transcriber executes inference locally using packaged models; no remote API calls for transcription. | Test T-001: Local model inference produces transcripts; network audit shows zero egress during inference. |
| `TRANSCRIBE-R001.2` | **Behavior.** Transcription routes raw audio to local inference; API returns text transcripts only. | Test T-002: Raw audio input produces text-only output; audio payload absent from SDK transport. |
| `TRANSCRIBE-R001.3` | **Wiring.** audio-transcriber exposes transcription endpoints; SDK transport policy governs output format; epistemic-graph consumer validates transcript origin. | Test T-003: SDK contract satisfied; EG validation passes; traceability chain end-to-end. |
| `TRANSCRIBE-R001.4` | **Refusal.** Misconfiguration preventing local inference triggers refusal before transmission; misconfigured remote endpoints are rejected. | Test T-004: Misconfiguration test rejects misconfigured remote endpoint; refusal logged. |
| `TRANSCRIBE-R001.5` | **Evidence.** Positive test exercises local inference end-to-end; negative test observes zero network egress under misconfiguration. | Test T-005: Positive case: local transcription succeeds. Negative case: external transmission blocked. |
| `TRANSCRIBE-R001.6` | **Hot-path proof.** audio-transcriber transcribe endpoint integration test calls local inference, validates SDK output contract, observes zero external calls in test_transcription_local_egress_zero. | Test T-006: pytest test_transcription_local_egress_zero in tests/integration/test_transcription_sdk.py passes. |

