# Audio Transcriber
## CLI or API | MCP

![PyPI - Version](https://img.shields.io/pypi/v/audio-transcriber)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
![PyPI - Downloads](https://img.shields.io/pypi/dd/audio-transcriber)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/audio-transcriber)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/audio-transcriber)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/audio-transcriber)
![PyPI - License](https://img.shields.io/pypi/l/audio-transcriber)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/audio-transcriber)
![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/audio-transcriber)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/audio-transcriber)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/audio-transcriber)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/audio-transcriber)
![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/audio-transcriber)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/audio-transcriber)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/audio-transcriber)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/audio-transcriber)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/audio-transcriber)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/audio-transcriber)

*Version: 2.2.1*

> **Documentation** — Installation, deployment, and usage across the CLI, Python API,
> [official documentation](https://knuckles-team.github.io/audio-transcriber/).

---

## Overview

**Audio Transcriber** is a production-grade Agent and Model Context Protocol (MCP) server designed to interface directly with Transcribe the operator's .wav .mp4 .mp3 .flac files to text or record the operator's own audio!.

---

## Key Features

- **Consolidated Action-Routed MCP Tools:** Minimizes token overhead and eliminates tool bloat in LLM contexts by grouping methods into optimized, togglable tool modules.
- **Enterprise-Grade Security:** Complete support for Eunomia policies, OIDC token delegation, and granular execution context tracking.
- **Native Telemetry & Tracing:** Out-of-the-box OpenTelemetry exports and native Langfuse tracing.

---

## CLI or API

This agent wraps the Transcribe the operator's .wav .mp4 .mp3 .flac files to text or record the operator's own audio! API. The operator can interact with it programmatically or via its integrated execution entrypoints.

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](docs/index.md).

---

## MCP

This server use dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools

The table below is auto-generated from the live server — do not edit by hand.

<!-- MCP-TOOLS-TABLE:START -->

#### Condensed action-routed tools (`MCP_TOOL_MODE=condensed`)

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `audio_ingest_transcription` | `MISCTOOL` | Transcribe an audio file and natively ingest it into the knowledge graph. |
| `health_check` | `MISCTOOL` |  |
| `transcribe_audio` | `AUDIO_PROCESSINGTOOL` | Transcribes audio from a provided file or by recording from the microphone. |
| `transcribe_media` | `MEDIA_SIDECARTOOL` | Verify the digest, transcribe via Whisper, and return AudioSegment-shaped |

#### Verbose 1:1 API-mapped tools (`MCP_TOOL_MODE=verbose` or `both`)

<details>
<summary>7 per-operation tools — one per public API method (click to expand)</summary>

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `audio_transcriber_export` | `AUDIO_TRANSCRIBERTOOL` | Export transcription to specified formats. |
| `audio_transcriber_initiate_stream` | `AUDIO_TRANSCRIBERTOOL` | Initiate the audio input stream. |
| `audio_transcriber_interact` | `AUDIO_TRANSCRIBERTOOL` | Interact with PersonaPlex server via WebSocket. |
| `audio_transcriber_record` | `AUDIO_TRANSCRIBERTOOL` | Record audio for a specified duration or until stopped. |
| `audio_transcriber_save_stream` | `AUDIO_TRANSCRIBERTOOL` | Save the recorded frames to a WAV file. |
| `audio_transcriber_stop_stream` | `AUDIO_TRANSCRIBERTOOL` | Stop and close the audio stream. |
| `audio_transcriber_transcribe` | `AUDIO_TRANSCRIBERTOOL` | Transcribe the audio file using the initialized backend. |

</details>

_4 action-routed tool(s) · 7 verbose 1:1 tool(s). Each is enabled unless its `<DOMAIN>TOOL` toggle is set false; `MCP_TOOL_MODE` selects the surface (**`intent` default** — the six verb-tools, granular set loaded on demand · `condensed` action-routed · `verbose` 1:1 · `both`). Auto-generated — do not edit._
<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/usage.md](docs/usage.md).

### Dynamic Tool Selection & Visibility

This MCP server supports dynamic toolset selection and visibility filtering at runtime. This allows the operator to restrict the set of exposed tools to prevent blowing up the LLM's context window.

The operator can configure tool filtering via multiple input channels:

- **CLI Arguments:** Pass `--tools` or `--toolsets` (or their disabled counterparts `--disabled-tools` and `--disabled-toolsets`) during startup.
- **Environment Variables:** Define standard environment variables:
  - `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS`
  - `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS`
- **HTTP SSE Request Headers:** Pass custom headers during transport initialization:
  - `x-mcp-enabled-tools` / `x-mcp-disabled-tools`
  - `x-mcp-enabled-tags` / `x-mcp-disabled-tags`
- **HTTP SSE Request Query Parameters:** Append query parameters directly to the operator's transport connection URL:
  - `?tools=tool1,tool2`
  - `?tags=tag1`

When query strings or parameters are provided, an LLM-free **Knowledge Graph resolution layer** (using `DynamicToolOrchestrator`) matches query intents against known tool tags, names, or descriptions, with safe fallback and automated 24-hour background cache refreshing.

---

### MCP Configuration Examples

<!-- MCP-CONFIG-EXAMPLES:START -->

> **Install the connector-focused `[mcp]` extra.** Examples use `audio-transcriber[mcp]` to add
> FastMCP / FastAPI through `agent-connector-sdk`; the required Agent Connector SDK core
> still carries `epistemic-graph[full]`. The `[agent-runtime]` extra additionally
> enables model orchestration.

#### stdio Transport (local IDEs — Cursor, Claude Desktop, VS Code)

```json
{
  "mcpServers": {
    "audio-transcriber-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "audio-transcriber[mcp]",
        "audio-transcriber-mcp"
      ],
      "env": {
        "MCP_TOOL_MODE": "intent",
        "AUDIO_PROCESSINGTOOL": "True",
        "MEDIA_SIDECARTOOL": "True",
        "MISCTOOL": "True",
        "TRANSCRIBE_DIRECTORY": "/path/to/transcribe_directory",
        "WHISPER_MODEL": "base"
      }
    }
  }
}
```

Runtime references require an alias-aware launcher such as Arbora. Other
launchers must omit those entries and inject the resolved values through their
own runtime secret boundary.

#### Streamable-HTTP Transport (networked / production)

```json
{
  "mcpServers": {
    "audio-transcriber-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "audio-transcriber[mcp]",
        "audio-transcriber-mcp",
        "--transport",
        "streamable-http",
        "--port",
        "8000"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "127.0.0.1",
        "PORT": "8000",
        "MCP_TOOL_MODE": "intent",
        "AUDIO_PROCESSINGTOOL": "True",
        "MEDIA_SIDECARTOOL": "True",
        "MISCTOOL": "True",
        "TRANSCRIBE_DIRECTORY": "/path/to/transcribe_directory",
        "WHISPER_MODEL": "base"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed Streamable-HTTP instance by `url`:

```json
{
  "mcpServers": {
    "audio-transcriber-mcp": {
      "url": "http://localhost:8000/audio-transcriber-mcp/mcp"
    }
  }
}
```

Run a reviewed container image as a least-privilege stdio child (no
listener or published port):

```bash
docker run -i --rm \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --pids-limit=256 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  -e TRANSPORT=stdio \
  -e MCP_TOOL_MODE=intent \
  -e AUDIO_PROCESSINGTOOL=True \
  -e MEDIA_SIDECARTOOL=True \
  -e MISCTOOL=True \
  -e TRANSCRIBE_DIRECTORY=/path/to/transcribe_directory \
  -e WHISPER_MODEL=base \
  registry.example.invalid/audio-transcriber@sha256:<digest> audio-transcriber-mcp
```

For containerized network HTTP, supply an authenticated TLS ingress (or
direct server TLS), exact `MCP_ALLOWED_HOSTS`, and an exact trusted-proxy
CIDR policy through the operator-owned deployment profile. The generator
does not emit an unauthenticated non-loopback listener.

_Auto-generated from the code-read env surface (`MCP_TOOL_MODE` + package vars) — do not edit._
<!-- MCP-CONFIG-EXAMPLES:END -->

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`audio-transcriber` can run as a local stdio process or container, or behind a remote
network boundary. The
[Deployment guide](https://knuckles-team.github.io/audio-transcriber/deployment/) carries
the detailed transport contract.

- **Local container** — launch a reviewed immutable image as a least-privilege
  stdio child with no listener or published port.
- **Remote URL** — connect through an operator-supplied authenticated HTTPS
  ingress. Keep its URL, outbound identity references, trust profile, and exact
  `MCP_ALLOWED_HOSTS` in `AgentConfig`.
<!-- END GENERATED: additional-deployment-options -->

## Agent

# Configure transcription (optional)
export WHISPER_MODEL="base"
export TRANSCRIBE_DIRECTORY="/path/to/transcribe_directory"

# Connector-focused MCP server (includes the shared graph engine)
uv pip install "audio-transcriber[mcp]"

# Everything (development)
uv pip install "audio-transcriber[all]"      # or: python -m pip install "audio-transcriber[all]"
```

### Container images (`:mcp`)

One multi-stage `docker/Dockerfile` builds the MCP-only image (`--target mcp`):

| Image tag | Build target | Contents | Entrypoint |
|-----------|--------------|----------|------------|
| `example/audio-transcriber:mcp` | `--target mcp` | `audio-transcriber[mcp]` — **connector-focused**, includes `epistemic-graph[full]`; no model-orchestration stack | `audio-transcriber-mcp` |

```bash
docker build --target mcp   -t example/audio-transcriber:mcp    docker/   # connector-focused MCP server
```

agent (`immutable agent digest`) with a co-located `:mcp` sidecar.

### Knowledge-graph database (`epistemic-graph`)

Agent Connector SDK core dependency (`epistemic-graph[full]`). The `[mcp]` extra keeps
deployments can use the bundled engine. For production or shared state, run
**epistemic-graph as a dedicated database service** and configure the runtime to use it.
Deployment recipes (single-node + Raft HA), connection configuration, and architecture
diagrams are documented in the
[epistemic-graph deployment guide](https://knuckles-team.github.io/epistemic-graph/deployment/).

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=example&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/example)
![GitHub User's stars](https://img.shields.io/github/stars/example)

---

## Documentation

The complete documentation is published as the
[official documentation site](https://knuckles-team.github.io/audio-transcriber/) and
is the recommended reference for installation, deployment, and day-to-day operation.

| Page | Contents |
|---|---|
| [Installation](https://knuckles-team.github.io/audio-transcriber/installation/) | pip, source, extras, prebuilt Docker image |
| [Deployment](https://knuckles-team.github.io/audio-transcriber/deployment/) | run the MCP server and agent, Compose, Caddy + Technitium, env config |
| [Usage](https://knuckles-team.github.io/audio-transcriber/usage/) | the MCP tool, the `AudioTranscriber` API, the CLI |
| [Overview](https://knuckles-team.github.io/audio-transcriber/overview/) | capability summary and ecosystem role |
| [Concepts](https://knuckles-team.github.io/audio-transcriber/concepts/) | concept registry (`CONCEPT:AUDIO-*`) |

---

## Contribute

Contributions are welcome! Please ensure code quality by running local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Ran test suites using `pytest`

<!-- BEGIN agent-connector-sdk-deployment (generated; do not edit between markers) -->

## Deploy with `agent-connector-sdk-deployment`

Provision this package with the consolidated **`agent-connector-sdk-deployment`**
workflow. It selects an installed-package, editable-source, or immutable-container
path; records only runtime secret and TLS-profile references in `AgentConfig`; and
runs doctor, registration, policy, observability, and rollback gates. Ask your agent
to **"deploy `audio-transcriber` with agent-connector-sdk-deployment"**.

| Install mode | Command |
|------|---------|
| Installed package | `uv tool install "audio-transcriber[mcp]"`, then run `audio-transcriber-mcp` |
| Immutable container | deploy `registry.example.invalid/audio-transcriber@sha256:<digest>` through the operator-selected orchestrator |

The repository embeds no deployment profile, credential value, certificate path, or
environment-specific endpoint. Supply those at runtime through `AgentConfig` and the
configured secret provider.

<!-- END agent-connector-sdk-deployment -->

<!-- GOVERNED-CAPABILITY:START -->
## Governed capability contract

This package ships a compact canonical skill surface with specialist procedures
kept as referenced workflows. The current MCP tools, skill metadata,
`connector_manifest.yml`, ontology, mappings, shapes, fixtures, migrations,
tool-schema fingerprints, and certification metadata form one versioned
capability contract. Validate them together; do not rely on stale tool names or
historical per-task skill wrappers.

Runtime endpoints, credentials, certificate trust, tenant identity, retention,
and observability policy are deployment inputs and are never packaged values.
See [Configuration, trust, and privacy](docs/configuration.md) before enabling a
network transport, connector ingestion, Arbora delegation, or trace export.
<!-- GOVERNED-CAPABILITY:END -->
