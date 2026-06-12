# Deployment

<!-- BEGIN GENERATED: deployment-options -->
## Deployment Options

`audio-transcriber` exposes its MCP server (console script `audio-transcriber-mcp`) four ways. Pick the row that
matches where the server runs relative to your MCP client, then copy the matching
`mcp_config.json` below. Replace the `<your-…>` placeholders with the values from the **Configuration / Environment Variables** section.

| # | Option | Transport | Where it runs | `mcp_config.json` key |
|---|--------|-----------|---------------|------------------------|
| 1 | stdio | `stdio` | client launches a subprocess | `command` |
| 2 | Streamable-HTTP (local) | `streamable-http` | a local network port | `command` or `url` |
| 3 | Local container / uv | `stdio` or `streamable-http` | Docker / Podman / uv on this host | `command` or `url` |
| 4 | Remote URL | `streamable-http` | a remote host behind Caddy | `url` |

### 1. stdio (local subprocess)

The client launches the server over stdio via `uvx` — best for local IDEs
(Cursor, Claude Desktop, VS Code):

```json
{
  "mcpServers": {
    "audio-transcriber-mcp": {
      "command": "uvx",
      "args": ["--from", "audio-transcriber", "audio-transcriber-mcp"],
      "env": {
        "AUDIO_TRANSCRIPTOR_API_KEY": "<your-audio_transcriptor_api_key>"
      }
    }
  }
}
```

### 2. Streamable-HTTP (local process)

Run the server as a long-lived HTTP process:

```bash
uvx --from audio-transcriber audio-transcriber-mcp --transport streamable-http --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/health        # {"status":"OK"}
```

Then either let the client launch it:

```json
{
  "mcpServers": {
    "audio-transcriber-mcp": {
      "command": "uvx",
      "args": ["--from", "audio-transcriber", "audio-transcriber-mcp", "--transport", "streamable-http", "--port", "8000"],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "AUDIO_TRANSCRIPTOR_API_KEY": "<your-audio_transcriptor_api_key>"
      }
    }
  }
}
```

…or connect to the already-running process by URL:

```json
{
  "mcpServers": {
    "audio-transcriber-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

### 3. Local container / uv

**(a) Launch a container directly from `mcp_config.json`** (stdio over the container —
no ports to manage). Swap `docker` for `podman` for a daemonless runtime:

```json
{
  "mcpServers": {
    "audio-transcriber-mcp": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "TRANSPORT=stdio",
        "-e", "AUDIO_TRANSCRIPTOR_API_KEY=<your-audio_transcriptor_api_key>",
        "knucklessg1/audio-transcriber:latest"
      ]
    }
  }
}
```

**(b) Run a local streamable-http container, then connect by URL:**

```bash
docker run -d --name audio-transcriber-mcp -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  -e AUDIO_TRANSCRIPTOR_API_KEY="<your-audio_transcriptor_api_key>" \
  knucklessg1/audio-transcriber:latest
# or, from a clone of this repo:
docker compose -f docker/mcp.compose.yml up -d
```

```json
{
  "mcpServers": {
    "audio-transcriber-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

**(c) From a local checkout with `uv`:**

```bash
uv run audio-transcriber-mcp --transport streamable-http --port 8000
```

### 4. Remote URL (deployed behind Caddy)

When the server is deployed remotely (e.g. as a Docker service) and published through
Caddy on the internal `*.arpa` zone, connect with the `"url"` key — no local process or
image required:

```json
{
  "mcpServers": {
    "audio-transcriber-mcp": { "url": "http://audio-transcriber-mcp.arpa/mcp" }
  }
}
```

Caddy reverse-proxies `http://audio-transcriber-mcp.arpa` to the container's `:8000`
streamable-http listener; `http://audio-transcriber-mcp.arpa/health` returns
`{"status":"OK"}` when the service is live.
<!-- END GENERATED: deployment-options -->

This page covers running `audio-transcriber` as a long-lived server: the transports,
the optional A2A agent, a Docker Compose stack, putting it behind a Caddy reverse
proxy, and giving it a DNS name with Technitium.

> `audio-transcriber` ships an **MCP server** (console script `audio-transcriber-mcp`)
> and an **A2A agent server** (console script `audio-transcriber-agent`). The MCP
> server is the typed, deterministic tool surface; the agent drives those tools over
> the Agent Control Protocol.

## Run the MCP server

The transport is selected with `--transport` (or the `TRANSPORT` env var):

=== "stdio (default)"

    ```bash
    audio-transcriber-mcp
    ```
    For IDE / desktop MCP clients that launch the server as a subprocess.

=== "streamable-http"

    ```bash
    audio-transcriber-mcp --transport streamable-http --host 0.0.0.0 --port 8000
    ```
    A network server with a `/health` endpoint and `/mcp` route.

=== "sse"

    ```bash
    audio-transcriber-mcp --transport sse --host 0.0.0.0 --port 8000
    ```

Health check (HTTP transports):

```bash
curl -s http://localhost:8000/health        # {"status":"OK"}
```

## Configuration (environment)

`audio-transcriber` is configured from the environment. The commonly used set:

| Var | Default | Meaning |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address for HTTP transports |
| `PORT` | `8000` | Bind port for HTTP transports |
| `TRANSPORT` | `stdio` | `stdio`, `streamable-http`, or `sse` |
| `WHISPER_MODEL` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `TRANSCRIBE_DIRECTORY` | data dir | Default directory for recordings and exports |
| `AUDIO_PROCESSINGTOOL` | `True` | Register the audio-processing tool set |
| `MISC_TOOL` | `True` | Register the miscellaneous (health) tool set |
| `ENABLE_OTEL` | `True` | Export OpenTelemetry traces |
| `EUNOMIA_TYPE` | `none` | Authorization mode: `none`, `embedded`, `remote` |

Every variable, grouped by concern, is documented in
[`.env.example`](https://github.com/Knuckles-Team/audio-transcriber/blob/main/.env.example).
Copy it to `.env` and populate only what you use.

## Docker Compose

The repo ships [`docker/mcp.compose.yml`](https://github.com/Knuckles-Team/audio-transcriber/blob/main/docker/mcp.compose.yml).
It reads a sibling `.env` and publishes the HTTP server on `:8000`:

```yaml
services:
  audio-transcriber-mcp:
    image: knucklessg1/audio-transcriber:latest
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
```

```bash
cp .env.example .env          # then edit WHISPER_MODEL and any other values
docker compose -f docker/mcp.compose.yml up -d
docker compose -f docker/mcp.compose.yml logs -f
```

## Run the A2A agent

The agent connects to the MCP server and exposes an Agent Control Protocol endpoint
(and an optional web interface). The console script is `audio-transcriber-agent`:

```bash
export MCP_URL=http://localhost:8000/mcp
audio-transcriber-agent --provider openai --model-id gpt-4o
```

The repo ships [`docker/agent.compose.yml`](https://github.com/Knuckles-Team/audio-transcriber/blob/main/docker/agent.compose.yml),
which deploys the MCP server and the agent together. The agent listens on `:9014`
and is wired to the MCP server by container name through `MCP_URL`:

```yaml
services:
  audio-transcriber-mcp:
    image: knucklessg1/audio-transcriber:latest
    hostname: audio-transcriber-mcp
    environment:
      - TRANSPORT=streamable-http
      - HOST=0.0.0.0
      - PORT=8000
    ports:
      - "8000:8000"

  audio-transcriber-agent:
    image: knucklessg1/audio-transcriber:latest
    depends_on:
      - audio-transcriber-mcp
    command: ["audio-transcriber-agent"]
    environment:
      - HOST=0.0.0.0
      - PORT=9014
      - MCP_URL=http://audio-transcriber-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
    ports:
      - "9014:9014"
```

```bash
docker compose -f docker/agent.compose.yml up -d
```

The agent endpoints are then available at `http://localhost:9014/a2a` (discovery at
`/a2a/.well-known/agent.json`) and, when enabled, the web interface at
`http://localhost:9014/`.

## Behind a Caddy reverse proxy

Expose the HTTP server on a hostname with automatic TLS. Add to your `Caddyfile`:

```caddy
# Internal (self-signed) — homelab .arpa zone
audio-transcriber.arpa {
    tls internal
    reverse_proxy audio-transcriber-mcp:8000
}
```

```caddy
# Public — automatic Let's Encrypt
audio-transcriber.example.com {
    reverse_proxy audio-transcriber-mcp:8000
}
```

Reload Caddy:

```bash
docker compose -f services/caddy/compose.yml exec caddy caddy reload --config /etc/caddy/Caddyfile
```

## DNS with Technitium

Point the hostname at the host running Caddy. Via the Technitium API:

```bash
curl -s "http://technitium.arpa:5380/api/zones/records/add" \
  --data-urlencode "token=$TECHNITIUM_DNS_TOKEN" \
  --data-urlencode "domain=audio-transcriber.arpa" \
  --data-urlencode "zone=arpa" \
  --data-urlencode "type=A" \
  --data-urlencode "ipAddress=10.0.0.10" \
  --data-urlencode "ttl=3600"
```

…or add an **A record** `audio-transcriber.arpa → <caddy-host-ip>` in the Technitium
web console (`http://technitium.arpa:5380`). The ecosystem
[`technitium-dns-mcp`](https://knuckles-team.github.io/technitium-dns-mcp/) automates
this as a tool.

## Register with an MCP client

Add to your client's `mcp_config.json`:

```json
{
  "mcpServers": {
    "audio-transcriber": {
      "command": "uv",
      "args": ["run", "audio-transcriber-mcp"],
      "env": {
        "WHISPER_MODEL": "base",
        "TRANSCRIBE_DIRECTORY": "~/Downloads"
      }
    }
  }
}
```

For a remote HTTP server, point the client at
`http://audio-transcriber.arpa/mcp` instead.
