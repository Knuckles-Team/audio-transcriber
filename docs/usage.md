# Usage — API / CLI / MCP

`audio-transcriber` exposes the same capability three ways: as an **MCP tool** an
agent calls, as a **Python API** (`AudioTranscriber`) you import, and as a **CLI**.

## As an MCP server

Once [deployed](deployment.md), the server registers the audio-processing tool set.
Tool registration is gated by `AUDIO_PROCESSINGTOOL` (default `True`).

| Tool | Description |
|---|---|
| `transcribe_audio` | Transcribe a provided audio file, or record from the microphone for a number of seconds and transcribe the recording. |

Example agent prompts that map onto this tool:

- *"Transcribe `~/Downloads/meeting.mp4` with the `base` model"* → `transcribe_audio`
- *"Record 30 seconds from the microphone and transcribe it"* → `transcribe_audio`
- *"Transcribe and translate this French clip to English"* → `transcribe_audio` (`task="translate"`)

## As a Python API

`AudioTranscriber` records microphone audio, transcribes local media files, and
exports the result to `txt` / `srt` / `vtt` / `json`. The Whisper model is loaded
locally — the fast `faster-whisper` backend is used by default, with an
`openai-whisper` fallback.

```python
from audio_transcriber.audio_transcriber import AudioTranscriber

# Transcribe an existing file
transcriber = AudioTranscriber(model="base", file="~/Downloads/meeting.mp4")
result = transcriber.transcribe(language="en")
print(result["text"])

# Export to multiple formats next to the source file
transcriber.export(result, formats=["txt", "srt", "vtt", "json"])
```

Record from the microphone, then transcribe:

```python
transcriber = AudioTranscriber(
    model="tiny",
    file_name="my_recording.wav",
    directory="~/Downloads",
)
transcriber.initiate_stream()
transcriber.record(seconds=30)        # capture 30 seconds
transcriber.stop_stream()
transcriber.save_stream()
result = transcriber.transcribe()
print(result["text"])
```

## As a CLI

The `audio-transcriber` console script transcribes files and records audio directly:

| Short | Long | Description |
|---|---|---|
| `-f` | `--file` | File to transcribe |
| `-m` | `--model` | Model: `tiny`, `base`, `small`, `medium`, `large` |
| `-l` | `--language` | Language to transcribe |
| `-e` | `--export` | Export `txt`, `srt`, and `vtt` files |
| `-r` | `--record` | Seconds to record from the microphone |
| `-d` | `--directory` | Directory to save the recording |
| `-n` | `--name` | Name of the recording |
| `-b` | `--bitrate` | Bitrate to use during recording |
| `-c` | `--channels` | Number of channels to use during recording |

Transcribe a file with the large model:

```bash
audio-transcriber --file '~/Downloads/Federal_Reserve.mp4' --model large --export
```

Record 60 seconds and transcribe with the tiny model:

```bash
audio-transcriber --record 60 --directory '~/Downloads/' --name 'my_recording.wav' --model tiny
```

### Model reference

Whisper model sizes (courtesy of [OpenAI Whisper](https://github.com/openai/whisper)):

| Size | Parameters | English-only | Multilingual | Required VRAM | Relative speed |
|:---:|:---:|:---:|:---:|:---:|:---:|
| tiny | 39 M | `tiny.en` | `tiny` | ~1 GB | ~32x |
| base | 74 M | `base.en` | `base` | ~1 GB | ~16x |
| small | 244 M | `small.en` | `small` | ~2 GB | ~6x |
| medium | 769 M | `medium.en` | `medium` | ~5 GB | ~2x |
| large | 1550 M | N/A | `large` | ~10 GB | 1x |
