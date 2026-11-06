# TRANSCRIBE — Implementation tasks

Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md). Rows: [requirements.md](requirements.md).

Each task line reads `- [ ] T001 [P] [US1] <ID>-R012.4 description`. Order is producer first.

- [ ] T001 [US1] TRANSCRIBE-R001 Local-model speech transcription with egress controls.
- [ ] T002 [US1] TRANSCRIBE-R001.1 Model.
- [ ] T003 [US1] TRANSCRIBE-R001.2 Behavior.
- [ ] T004 [US1] TRANSCRIBE-R001.3 Wiring.
- [ ] T005 [US1] TRANSCRIBE-R001.4 Refusal.
- [ ] T006 [US1] TRANSCRIBE-R001.5 Evidence.
- [ ] T007 [US1] TRANSCRIBE-R001.6 Hot-path proof.
- [ ] T008 [US2] TRANSCRIBE-R002 Typed speech-to-text contract for front-door audio prompts (rollup).
- [ ] T009 [US2] TRANSCRIBE-R002.1 Typed tool and result.
- [ ] T010 [US2] TRANSCRIBE-R002.2 Language detection, timestamps and diarisation report.
- [ ] T011 [US2] TRANSCRIBE-R002.3 ffmpeg format conversion in the image.
- [ ] T012 [US2] TRANSCRIBE-R002.4 The faster-whisper backend loads and decodes on a torch-free, current-PyAV install.
