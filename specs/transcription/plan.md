# TRANSCRIBE — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Existing system and reuse

audio-transcriber owns transcription via local models. agent-connector-sdk owns transport contracts. epistemic-graph owns transcript consumers. No duplicated transcription authority.

## Architecture

Speech input enters audio-transcriber transcribe endpoint. Local model inference produces text output. SDK contracts govern egress checks: raw audio blocked, transcripts validated, origin traced to epistemic-graph. Refusal gates block misconfigured remote endpoints before transmission.

## Interfaces and data model

Transcribe endpoint accepts raw audio input via ConnectorRequest. Response schema carries transcribed text only, not audio. SDK transport policy enforces audio payload exclusion. EG consumer validates transcript schema and origin.

## Live integration path

SDK calls audio-transcriber transcribe. audio-transcriber invokes local model. Output text flows to epistemic-graph transcript consumer. Refusal checks intercept before transmission. No external inference APIs called.

## Quality and release gates

Contract gate: SDK transport policy verified. Integration test: test_transcription_local_egress_zero proves zero external calls. Language-native: pytest test suite. KISS: no redundant inference authority.

## Risks, alternatives, and decisions

Risk: Model size impacts latency; local inference chosen over remote for security. Decision: Refusal gates prevent misconfiguration egress; tests verify. Authority: audio-transcriber owns execution; SDK owns transport.

