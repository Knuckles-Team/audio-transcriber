---
name: audio-transcriber-knowledge-graph
skill_type: skill
description: >-
  Native knowledge-graph ingestion of Whisper transcripts on the audio-transcriber
  MCP server — transcribe an audio/video file and push it into the epistemic-graph in
  one call: the raw audio as a shared :MediaAsset blob, the transcript text as a
  :Document, and each Whisper segment as a :TranscriptSegment node linked :segmentOf
  the transcript and :transcribedFrom the audio. Use when a transcript must be durable
  and semantically searchable in the KG. Do NOT use for a one-off transcription with no
  KG persistence (use audio-transcriber-transcription).
license: MIT
tags: [audio, transcription, knowledge-graph, ingestion, whisper, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Audio Transcript → Knowledge Graph

Wire-First native ingestion: the **`audio-transcriber`** MCP server transcribes a file
and natively writes the result into the ONE epistemic-graph engine across every modality
(CONCEPT:AU-KG.ingest.enterprise-source-extractor). Ordinary `transcribe_audio` calls also
ingest by default; this skill covers the explicit ingest tool and the graph shape.

## When to use
- Transcribe **and persist** an audio/video file into the knowledge graph in one step.
- Make a transcript semantically searchable (the `:Document` is chunked/embedded hub-side).
- Keep the raw audio durable + deduped as a content-addressed blob tied to its transcript.

## When NOT to use
- A throwaway transcription or caption export with no KG → `audio-transcriber-transcription`.
- Querying/searching transcripts already in the KG → the graph query/search tools
  (`graph_search` / `graph_query`), not this ingest tool.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`audio-transcriber`** MCP server. A live
epistemic-graph engine must be reachable for anything to be written — otherwise ingestion
**no-ops** cleanly (the tool still returns the transcript, with `ingested: null`).

| Variable | Required | Notes |
|----------|----------|-------|
| `WHISPER_MODEL` | optional | Default model when `model` is omitted |

## Tools & actions
| Tool | Purpose |
|------|---------|
| `audio_ingest_transcription` | Transcribe a file and ingest the result (blob + document + segments) |

### Key parameters
- `audio_file` — required path to the audio/video file.
- `model` — Whisper model (default `WHISPER_MODEL`/`base`).
- `language` — pin the language, else auto-detect.
- `task` — `transcribe` (default) or `translate` (English).

## Graph shape
| Node / link | Meaning |
|-------------|---------|
| `:MediaAsset` (`audio:` blob) | the raw audio bytes, content-addressed |
| `:Document` `audio:transcript:<slug>` | the full transcript text + `language`/`duration`/`whisperModel` |
| `:TranscriptSegment` `audio:segment:<slug>:<id>` | one timestamped span (`startTime`/`endTime`/`noSpeechProb`) |
| `:transcribedFrom` | `:Transcript` → source `:MediaAsset` |
| `:segmentOf` | `:TranscriptSegment` → its `:Transcript` |

## Recipes
Transcribe and ingest a lecture recording:
```json
{"audio_file":"/data/lecture.mp3","model":"small"}
```
Translate + ingest foreign audio (transcript stored in English):
```json
{"audio_file":"/data/interview_fr.wav","task":"translate"}
```

## Gotchas
- Ingestion is **best-effort and default-on**: with no reachable engine the tool returns
  the transcript with `ingested: null` — check that key to confirm a write happened.
- Node ids are stable per file name (`audio:transcript:<slugified-stem>`); re-transcribing
  the same file MERGEs onto the same transcript node rather than duplicating it.
- Segment ingestion is capped (default 500 segments) to bound very long recordings.
- The `:Document` carries the text so it is embedded hub-side; the `:MediaAsset` is only
  written when the audio file path is available on disk.

## Related
- **`audio-transcriber-transcription`** — transcription + caption export without the KG focus.
- The in-repo `connectors/mcp_source_presets.json` maps this tool as a Tier-1 KG source.
