---
name: audio-transcriber-transcription
description: >-
  Speech-to-text on the audio-transcriber MCP server — run Whisper (faster-whisper,
  falling back to openai-whisper) over a local audio/video file or a microphone
  recording, and export txt/srt/vtt/json captions. Use when the agent must
  transcribe or translate spoken audio, generate subtitle/caption files, or pick a
  Whisper model for an accuracy/latency trade-off. Do NOT use to push a transcript
  into the knowledge graph (use audio-transcriber-knowledge-graph) or to download
  the media first (use media-downloader).
license: MIT
tags: [audio, transcription, whisper, speech-to-text, captions, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Audio Transcription

Whisper transcription of audio/video via the **`audio-transcriber`** MCP server. The
server prefers **faster-whisper** (CTranslate2) and falls back to **openai-whisper**;
GPU is used automatically when available (`float16`), else CPU (`int8`).

## When to use
- Transcribe a local audio/video file (`.wav`, `.mp3`, `.mp4`, `.flac`, …) to text.
- Record from the microphone for N seconds and transcribe.
- Translate foreign-language audio to English (`task='translate'`).
- Emit subtitle/caption files (`txt`, `srt`, `vtt`, `json`).

## When NOT to use
- Persisting the transcript / audio into the knowledge graph → use
  `audio-transcriber-knowledge-graph`.
- Fetching the media from a URL first → use the `media-downloader` package, then
  transcribe the downloaded file.
- Real-time audio-to-audio interaction (PersonaPlex) — that is a separate CLI mode,
  not part of this tool surface.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`audio-transcriber`** MCP server.
`ffmpeg` should be installed for broad input-format support.

| Variable | Required | Notes |
|----------|----------|-------|
| `WHISPER_MODEL` | optional | Default model when the caller omits `model` (default `base`) |
| `TRANSCRIBE_DIRECTORY` | optional | Default directory for recordings/exports |

`MCP_TOOL_MODE` (`condensed`|`verbose`|`both`) selects the condensed surface vs. the
1:1 verbose tools.

## Tools & actions
| Tool | Purpose |
|------|---------|
| `transcribe_audio` | Transcribe a file or a microphone recording; optionally export captions |

### Key parameters
- `audio_file` — path to transcribe. Provide this **or** a positive `record_seconds`.
- `record_seconds` — seconds to capture from the mic (only when no `audio_file`).
- `model` — `tiny`/`base`/`small`/`medium`/`large`/`large-v3`/`distil-large-v3`/`turbo`.
- `language` — pin (e.g. `en`) to skip auto-detection; omit to auto-detect.
- `task` — `transcribe` (default) or `translate` (to English).
- `word_timestamps` — set for word-level timing.
- `export_formats` — subset of `["txt","srt","vtt","json"]`.
- `backend` — force `faster-whisper` or `openai-whisper` (default: auto).

## Recipes
Transcribe a file with SRT + VTT captions:
```json
{"audio_file":"/data/talk.mp3","model":"small","export_formats":["srt","vtt"]}
```
Translate Japanese audio to English text:
```json
{"audio_file":"/data/jp_interview.wav","task":"translate","language":"ja"}
```
Record 30 seconds from the mic and transcribe:
```json
{"record_seconds":30,"model":"base"}
```

## Gotchas
- Provide **either** `audio_file` **or** a positive `record_seconds` — supplying
  neither raises "Either audio_file must be provided or record_seconds must be positive."
- Bigger models (`large-v3`) are far more accurate but much slower; on CPU prefer
  `base`/`small` or `distil-large-v3`.
- `task='translate'` always targets **English**; it does not translate to other languages.
- Recording needs a real input device; on a headless server mic capture will fail —
  transcribe a file instead.
- Exports are written next to the source file, named `<stem>.<fmt>`.

## Related
- **`audio-transcriber-knowledge-graph`** — ingest the transcript + audio into the KG.
- **`media-downloader`** — fetch the media before transcribing.
