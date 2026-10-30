# CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R001` | **Default content.** `pyproject.toml` declares `audio-transcriber-mcp` as the single `agent_connector_sdk.connector` entry point. The server lists the packaged `skill://` and `ontology://` resources and the packaged prompts with no `content=` argument. Proving test: `tests/test_default_content_surfaces.py`. | `resources/list` includes `skill://` and `ontology://`; `prompts/list` includes each packaged prompt. |
| `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R002` | **Transcript list/get operation by recording id and date.** Add an operation returning transcripts filtered by recording id and date. It unblocks the audio-transcriber competency question, which Medulla answers with a typed abstention naming this operation until it lands. | Test: the operation returns the documented fields. |
| `CONN-AUDIO-TRANSCRIBER-CONNECTOR-CONTENT-R003` | **Expose avg_logprob (confidence) per transcript segment.** Expose the per-segment confidence so the audio-transcriber competency question can be answered. Medulla abstains naming this operation until it lands. | Test: the operation returns avg_logprob for each segment. |
