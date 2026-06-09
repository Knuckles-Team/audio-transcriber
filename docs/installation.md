# Installation

`audio-transcriber` is a standard Python package and a prebuilt container image.
Pick the path that matches how you want to run it.

## Requirements

- **Python 3.11 – 3.14**.
- **`ffmpeg`** for broad audio-format support, and the PortAudio system libraries
  for microphone recording. On Debian / Ubuntu:

  ```bash
  sudo apt-get update
  sudo apt-get install -y libasound-dev portaudio19-dev libportaudio2 \
    libportaudiocpp0 ffmpeg gcc
  ```

The Whisper model is loaded locally; no external transcription service is required.

## From PyPI (recommended)

```bash
pip install audio-transcriber
```

### Optional extras

The base install ships the CLI and the `faster-whisper` backend. Install the extra
for the interface you need:

| Extra | Install | Pulls in |
|---|---|---|
| `mcp` | `pip install "audio-transcriber[mcp]"` | FastMCP MCP-server runtime (`agent-utilities[mcp]`) + `websockets` |
| `agent` | `pip install "audio-transcriber[agent]"` | Pydantic-AI agent + Logfire tracing |
| `local` | `pip install "audio-transcriber[local]"` | `openai-whisper` reference backend (fallback to `faster-whisper`) |
| `all` | `pip install "audio-transcriber[all]"` | The `mcp` and `agent` extras together |

```bash
# Typical: run the MCP server and the A2A agent
pip install "audio-transcriber[all]"
```

## From source

```bash
git clone https://github.com/Knuckles-Team/audio-transcriber.git
cd audio-transcriber
pip install -e ".[all]"          # editable install with every extra
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install -e ".[all]"
uv run audio-transcriber-mcp
```

## Prebuilt Docker image

A slim image is published on every release (entrypoint `audio-transcriber-mcp`):

```bash
docker pull knucklessg1/audio-transcriber:latest

docker run --rm -i \
  -e WHISPER_MODEL=base \
  knucklessg1/audio-transcriber:latest        # stdio transport (default)
```

For an HTTP server with a published port, see [Deployment](deployment.md).

## Verify the install

```bash
audio-transcriber --help
audio-transcriber-mcp --help
python -c "import audio_transcriber; print(audio_transcriber.__version__)"
```

## Next steps

- **[Deployment](deployment.md)** — run it as a long-lived MCP server and agent behind Caddy + DNS.
- **[Usage](usage.md)** — call the tool, the API, and the CLI.
- **[Configuration](deployment.md#configuration-environment)** — every environment variable.
