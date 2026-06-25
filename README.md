# Audio Transcriber
## CLI or API | MCP | Agent

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

*Version: 0.35.0*

> **Documentation** — Installation, deployment, and usage across the CLI, Python API,
> MCP server, and A2A agent are maintained in the
> [official documentation](https://knuckles-team.github.io/audio-transcriber/).

---

## Overview

**Audio Transcriber** is a production-grade Agent and Model Context Protocol (MCP) server designed to interface directly with Transcribe your .wav .mp4 .mp3 .flac files to text or record your own audio!.

---

## Key Features

- **Consolidated Action-Routed MCP Tools:** Minimizes token overhead and eliminates tool bloat in LLM contexts by grouping methods into optimized, togglable tool modules.
- **Enterprise-Grade Security:** Comprehensive support for Eunomia policies, OIDC token delegation, and granular execution context tracking.
- **Integrated Graph Agent:** Built-in Pydantic AI agent supporting the Agent Control Protocol (ACP) and standard Web interfaces (AG-UI).
- **Native Telemetry & Tracing:** Out-of-the-box OpenTelemetry exports and native Langfuse tracing.

---

## CLI or API

This agent wraps the Transcribe your .wav .mp4 .mp3 .flac files to text or record your own audio! API. You can interact with it programmatically or via its integrated execution entrypoints.

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](docs/index.md).

---

## MCP

This server utilizes dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools

The table below is auto-generated from the live server — do not edit by hand.

<!-- MCP-TOOLS-TABLE:START -->

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `health_check` | `MISCTOOL` |  |
| `transcribe_audio` | `AUDIO_PROCESSINGTOOL` | Transcribes audio from a provided file or by recording from the microphone. |

_2 action-routed tools (default `MCP_TOOL_MODE=condensed`). Each is enabled unless its toggle is set false; set `MCP_TOOL_MODE=verbose` (or `both`) for the 1:1 per-operation surface. Auto-generated — do not edit._
<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/mcp.md](docs/mcp.md).

### Dynamic Tool Selection & Visibility

This MCP server supports dynamic toolset selection and visibility filtering at runtime. This allows you to restrict the set of exposed tools in order to prevent blowing up the LLM's context window.

You can configure tool filtering via multiple input channels:

- **CLI Arguments:** Pass `--tools` or `--toolsets` (or their disabled counterparts `--disabled-tools` and `--disabled-toolsets`) during startup.
- **Environment Variables:** Define standard environment variables:
  - `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS`
  - `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS`
- **HTTP SSE Request Headers:** Pass custom headers during transport initialization:
  - `x-mcp-enabled-tools` / `x-mcp-disabled-tools`
  - `x-mcp-enabled-tags` / `x-mcp-disabled-tags`
- **HTTP SSE Request Query Parameters:** Append query parameters directly to your transport connection URL:
  - `?tools=tool1,tool2`
  - `?tags=tag1`

When query strings or parameters are supplied, an LLM-free **Knowledge Graph resolution layer** (using `DynamicToolOrchestrator`) matches query intents against known tool tags, names, or descriptions, with safe fallback and automated 24-hour background cache refreshing.

---

### MCP Configuration Examples

> **Install the slim `[mcp]` extra.** All examples below install
> `audio-transcriber[mcp]` — the MCP-server extra that pulls only the FastMCP /
> FastAPI tooling (`agent-utilities[mcp]`). It deliberately **excludes** the heavy
> agent runtime (the epistemic-graph engine, `pydantic-ai`, `dspy`, `llama-index`,
> `tree-sitter`), so `uvx`/container installs are dramatically smaller and faster.
> Use the full `[agent]` extra only when you need the integrated Pydantic AI agent
> (see [Installation](#installation)).

#### stdio Transport (Recommended for local IDEs e.g., Cursor, Claude Desktop)
Configure your IDE's `mcp.json` to launch the MCP server via `uvx`:

```json
{
  "mcpServers": {
    "audio-transcriber": {
      "command": "uvx",
      "args": [
        "--from",
        "audio-transcriber[mcp]",
        "audio-transcriber-mcp"
      ],
      "env": {
        "AUDIO_TRANSCRIPTOR_API_KEY": "your_audio_transcriptor_api_key_here",
        "LANGSMITH_DEFAULT_SYSTEM_PROMPT": "your_langsmith_default_system_prompt_here",
        "OPENROUTER_API_KEY": "your_openrouter_api_key_here"
      }
    }
  }
}
```

#### Streamable-HTTP Transport (Recommended for production deployments)
Configure your client's `mcp.json` to launch the Streamable-HTTP server via `uvx` with explicit host and port definition:

```json
{
  "mcpServers": {
    "audio-transcriber": {
      "command": "uvx",
      "args": [
        "--from",
        "audio-transcriber[mcp]",
        "audio-transcriber-mcp"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "AUDIO_TRANSCRIPTOR_API_KEY": "your_audio_transcriptor_api_key_here",
        "LANGSMITH_DEFAULT_SYSTEM_PROMPT": "your_langsmith_default_system_prompt_here",
        "OPENROUTER_API_KEY": "your_openrouter_api_key_here"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed remote or local Streamable-HTTP instance:

```json
{
  "mcpServers": {
    "audio-transcriber": {
      "url": "http://localhost:8000/audio-transcriber/mcp"
    }
  }
}
```

Deploying the Streamable-HTTP server via Docker:

```bash
docker run -d \
  --name audio-transcriber-mcp \
  -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  -e AUDIO_TRANSCRIPTOR_API_KEY="your_value" \
  -e LANGSMITH_DEFAULT_SYSTEM_PROMPT="your_value" \
  -e OPENROUTER_API_KEY="your_value" \
  knucklessg1/audio-transcriber:mcp
```

> The `:mcp` tag is the **slim MCP-server image** (built from
> `docker/Dockerfile --target mcp`, installing `audio-transcriber[mcp]`). The default
> `:latest` tag is the **full agent image** (`--target agent`, `audio-transcriber[agent]`)
> which also bundles the Pydantic AI agent and the epistemic-graph engine — use it
> when you run `audio-transcriber-agent` (the agent), not just the MCP server. See
> [Container images](#container-images-mcp-vs-agent).

---

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`audio-transcriber` can also run as a **local container** (Docker / Podman / `uv`) or be
consumed from a **remote deployment**. The
[Deployment guide](https://knuckles-team.github.io/audio-transcriber/deployment/) has full, copy-paste
`mcp_config.json` for all four transports — **stdio**, **streamable-http**,
**local container / uv**, and **remote URL**:

- **Local container / uv** — launch the server from `mcp_config.json` via `uvx`,
  `docker run`, or `podman run`, or point at a local streamable-http container by `url`.
- **Remote URL** — connect to a server deployed behind Caddy at
  `http://audio-transcriber-mcp.arpa/mcp` using the `"url"` key.
<!-- END GENERATED: additional-deployment-options -->

## Agent

This repository features a fully integrated Pydantic AI Graph Agent. It communicates over the **Agent Control Protocol (ACP)** and interacts seamlessly with the **Agent Web UI (AG-UI)** and Terminal interface.

### Running the Agent CLI
To start the interactive command-line agent:

```bash
# Set credentials
export AUDIO_TRANSCRIPTOR_API_KEY="your_value"
export LANGSMITH_DEFAULT_SYSTEM_PROMPT="your_value"
export OPENROUTER_API_KEY="your_value"

# Run the agent server
audio-transcriber-agent --provider openai --model-id gpt-4o
```

### Docker Compose Orchestration
The following `docker/agent.compose.yml` configures the Agent, Web UI, and Terminal Interface together:

```yaml
version: '3.8'

services:
  audio-transcriber-mcp:
    image: knucklessg1/audio-transcriber:mcp
    container_name: audio-transcriber-mcp
    hostname: audio-transcriber-mcp
    restart: always
    env_file:
      - ../.env
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  audio-transcriber-agent:
    image: knucklessg1/audio-transcriber:latest
    container_name: audio-transcriber-agent
    hostname: audio-transcriber-agent
    restart: always
    depends_on:
      - audio-transcriber-mcp
    env_file:
      - ../.env
    command: [ "audio-transcriber-agent" ]
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=9014
      - MCP_URL=http://audio-transcriber-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
      - ENABLE_OTEL=True
    ports:
      - "9014:9014"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9014/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

```

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/agent.md](docs/agent.md).

---

## Security & Governance

Built directly upon the enterprise-ready [`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities) core, standard security parameters are fully supported:

### Access Control & Policy Enforcement
- **Eunomia Policies:** Fine-grained, policy-driven tool authorization. Supports `none`, local `embedded` (`mcp_policies.json`), or centralized `remote` modes.
- **OIDC Token Delegation:** Compliant with RFC 8693 token exchange for flowing authenticating user credentials from Web UI / ACP → Agent → MCP.
- **Scoped Credentials:** Execution context runs restricted to the specific caller identity.

### Runtime Security Grid
| Feature | Functionality | Enablement |
|---------|---------------|------------|
| **Tool Guard** | Sensitivity inspection with human-in-the-loop validation | Enabled by default |
| **Prompt Injection Defense** | Input scanning, repetition monitoring, and recursive loop blocks | Enabled by default |
| **Context Safety Guard** | Stuck-loop detectors and contextual overflow preemptive alerts | Enabled by default |

---

## Environment Variables

<!-- ENV-VARS-TABLE:START -->

#### Package environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` |  |
| `PORT` | `8000` |  |
| `TRANSPORT` | `stdio` | options: stdio, streamable-http, sse |
| `ENABLE_OTEL` | `True` |  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:8080/api/public/otel` |  |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` | `pk-...` |  |
| `OTEL_EXPORTER_OTLP_SECRET_KEY` | `sk-...` |  |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |  |
| `EUNOMIA_TYPE` | `none` | options: none, embedded, remote |
| `EUNOMIA_POLICY_FILE` | `mcp_policies.json` |  |
| `EUNOMIA_REMOTE_URL` | `http://eunomia-server:8000` |  |
| `AUDIO_TRANSCRIPTOR_API_KEY` | `your_api_key_here` |  |
| `LANGSMITH_DEFAULT_SYSTEM_PROMPT` | `""` |  |
| `OPENROUTER_API_KEY` | `your_openrouter_api_key_here` |  |
| `MISCTOOL` | `True` |  |
| `AUDIO_PROCESSINGTOOL` | `True` |  |
| `WHISPER_MODEL` | `base` | Standard OpenAI Whisper model to use for local transcription (e.g., base, tiny, small) |

#### Inherited agent-utilities variables (apply to every connector)

| Variable | Example | Description |
|----------|---------|-------------|
| `MCP_TOOL_MODE` | `condensed` | Tool surface: `condensed` | `verbose` | `both` |
| `MCP_ENABLED_TOOLS` | — | Comma-separated tool allow-list |
| `MCP_DISABLED_TOOLS` | — | Comma-separated tool deny-list |
| `MCP_ENABLED_TAGS` | — | Comma-separated tag allow-list |
| `MCP_DISABLED_TAGS` | — | Comma-separated tag deny-list |
| `MCP_CLIENT_AUTH` | — | Outbound MCP auth (`oidc-client-credentials` for fleet calls) |
| `OIDC_CLIENT_ID` | — | OIDC client id (service-account auth) |
| `OIDC_CLIENT_SECRET` | — | OIDC client secret (service-account auth) |
| `DEBUG` | `False` | Verbose logging |
| `PYTHONUNBUFFERED` | `1` | Unbuffered stdout (recommended in containers) |
| `MCP_URL` | `http://localhost:8000/mcp` | URL of the MCP server the agent connects to |
| `PROVIDER` | `openai` | LLM provider for the agent |
| `MODEL_ID` | `gpt-4o` | Model id for the agent |
| `ENABLE_WEB_UI` | `True` | Serve the AG-UI web interface |

_17 package + 14 inherited variable(s). Auto-generated from `.env.example` + the shared agent-utilities set — do not edit._
<!-- ENV-VARS-TABLE:END -->


Every variable the server reads, grouped by purpose.

### Transcription / Credentials
| Variable | Description | Default |
|----------|-------------|---------|
| `AUDIO_TRANSCRIPTOR_API_KEY` | API key for the transcription backend | — |
| `OPENROUTER_API_KEY` | OpenRouter API key (LLM provider) | — |
| `LANGSMITH_DEFAULT_SYSTEM_PROMPT` | Default system prompt for LangSmith tracing | — |
| `WHISPER_MODEL` | Local OpenAI Whisper model (e.g. `base`, `tiny`, `small`) | `base` |

### MCP server / transport
| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSPORT` | `stdio`, `streamable-http`, or `sse` | `stdio` |
| `HOST` | Bind host (HTTP transports) | `0.0.0.0` |
| `PORT` | Bind port (HTTP transports) | `8000` |
| `MCP_TOOL_MODE` | Tool surface: `condensed`, `verbose`, or `both` | `condensed` |
| `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS` | Comma-separated tool allow/deny list | — |
| `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS` | Comma-separated tag allow/deny list | — |
| `DEBUG` | Verbose logging | `False` |
| `PYTHONUNBUFFERED` | Unbuffered stdout (recommended in containers) | `1` |

### Tool toggles
Each action-routed tool can be disabled individually via its toggle env var (set to `false`).
See the [Available MCP Tools](#available-mcp-tools) table above for the authoritative names.

| Variable | Description | Default |
|----------|-------------|---------|
| `MISCTOOL` | Toggle the miscellaneous / health-check tool | `True` |
| `AUDIO_PROCESSINGTOOL` | Toggle the audio-processing (transcription) tool | `True` |

### Telemetry & governance
| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_OTEL` | Enable OpenTelemetry export | `True` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | — |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` / `OTEL_EXPORTER_OTLP_SECRET_KEY` | OTLP auth keys | — |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol (e.g. `http/protobuf`) | — |
| `EUNOMIA_TYPE` | Authorization mode: `none`, `embedded`, `remote` | `none` |
| `EUNOMIA_POLICY_FILE` | Embedded policy file | `mcp_policies.json` |
| `EUNOMIA_REMOTE_URL` | Remote Eunomia server URL | — |

### Agent CLI (full `[agent]` runtime only)
| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_URL` | URL of the MCP server the agent connects to | `http://localhost:8000/mcp` |
| `PROVIDER` | LLM provider (e.g. `openai`) | `openai` |
| `MODEL_ID` | Model id (e.g. `gpt-4o`) | `gpt-4o` |
| `ENABLE_WEB_UI` | Serve the AG-UI web interface | `True` |

See [`.env.example`](.env.example) for a copy-paste starting point.

---

## Installation

Pick the extra that matches what you want to run:

| Extra | Installs | Use when |
|-------|----------|----------|
| `audio-transcriber[mcp]` | Slim MCP server only (`agent-utilities[mcp]` — FastMCP/FastAPI) | You only run the **MCP server** (smallest install / image) |
| `audio-transcriber[agent]` | Full agent runtime (`agent-utilities[agent,logfire]` — Pydantic AI + the epistemic-graph engine) | You run the **integrated agent** |
| `audio-transcriber[all]` | Everything (`mcp` + `agent`) | Development / both surfaces |

```bash
# MCP server only (recommended for tool hosting — slim deps)
uv pip install "audio-transcriber[mcp]"

# Full agent runtime (Pydantic AI + epistemic-graph engine)
uv pip install "audio-transcriber[agent]"

# Everything (development)
uv pip install "audio-transcriber[all]"      # or: python -m pip install "audio-transcriber[all]"
```

### Container images (`:mcp` vs `:agent`)

One multi-stage `docker/Dockerfile` builds two right-sized images, selected by `--target`:

| Image tag | Build target | Contents | Entrypoint |
|-----------|--------------|----------|------------|
| `knucklessg1/audio-transcriber:mcp` | `--target mcp` | `audio-transcriber[mcp]` — **slim**, no engine/`pydantic-ai`/`dspy`/`llama-index`/`tree-sitter` | `audio-transcriber-mcp` |
| `knucklessg1/audio-transcriber:latest` | `--target agent` (default) | `audio-transcriber[agent]` — **full** agent runtime + epistemic-graph engine | `audio-transcriber-agent` |

```bash
docker build --target mcp   -t knucklessg1/audio-transcriber:mcp    docker/   # slim MCP server
docker build --target agent -t knucklessg1/audio-transcriber:latest docker/   # full agent
```

`docker/mcp.compose.yml` runs the slim `:mcp` server; `docker/agent.compose.yml` runs the
agent (`:latest`) with a co-located `:mcp` sidecar.

### Knowledge-graph database (`epistemic-graph`)

The **full agent** (`[agent]` / `:latest`) embeds the **epistemic-graph** engine (pulled in
transitively via `agent-utilities[agent]`). For production — or to share one knowledge graph
across multiple agents — run **epistemic-graph as its own database container** and point the
agent at it instead of embedding it. Deployment recipes (single-node + Raft HA), connection
config, and the full database architecture (with diagrams) are documented in the
[epistemic-graph deployment guide](https://knuckles-team.github.io/epistemic-graph/deployment/).
The slim `[mcp]` server does **not** require the database.

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)

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

Contributions are welcome! Please ensure code quality by executing local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Execute test suites using `pytest`


<!-- BEGIN agent-os-genesis-deploy (generated; do not edit between markers) -->

## Deploy with `agent-os-genesis`

This package can be provisioned for you — skill-guided — by the **`agent-os-genesis`**
universal skill (its *single-package deploy mode*): it picks your install method, seeds
secrets to OpenBao/Vault (or `.env`), trusts your enterprise CA, registers the MCP
server, and verifies it — the same machinery that stands up the whole Agent OS, narrowed
to just this package. Ask your agent to **"deploy `audio-transcriber` with agent-os-genesis"**.

| Install mode | Command |
|------|---------|
| Bare-metal, prod (PyPI) | `uvx audio-transcriber-mcp` · or `uv tool install audio-transcriber` |
| Bare-metal, dev (editable) | `uv pip install -e ".[all]"` · or `pip install -e ".[all]"` |
| Container, prod | deploy `knucklessg1/audio-transcriber:latest` via docker-compose / swarm / podman / podman-compose / kubernetes |
| Container, dev (editable) | deploy `docker/compose.dev.yml` (source-mounted at `/src`; edits live on restart) |

Secrets are read-existing + seeded via `vault_sync` — you are only prompted for what's missing.

<!-- END agent-os-genesis-deploy -->
