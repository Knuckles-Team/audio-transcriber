# Deployment

<!-- BEGIN GENERATED: deployment-options -->
## Deployment Options

`audio-transcriber` supports local stdio, a loopback-only development listener, a
least-privilege stdio container, and a remote authenticated HTTPS boundary.
Provider endpoint, credential, selector, identity, and trust material are supplied
at runtime through `AgentConfig`; none is stored in this repository.

### Installed stdio process

```json
{
  "mcpServers": {
    "audio-transcriber": {
      "command": "audio-transcriber-mcp",
      "args": [],
      "env": {"MCP_TOOL_MODE": "intent"}
    }
  }
}
```

### Loopback development listener

```bash
audio-transcriber-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

Do not expose this listener beyond loopback. Network deployments require direct TLS
or an explicitly trusted TLS-terminating ingress, configured authentication, exact
`MCP_ALLOWED_HOSTS`, and an exact trusted-proxy CIDR policy.

### Least-privilege local container

```bash
docker run -i --rm \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --pids-limit=256 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  -e TRANSPORT=stdio \
  registry.example.invalid/audio-transcriber@sha256:<digest> audio-transcriber-mcp
```

The operator projects the selected AgentConfig profile into the process at runtime;
the image remains immutable and contains no environment connection profile.

### Remote authenticated HTTPS endpoint

```json
{
  "mcpServers": {
    "audio-transcriber": {"url": "https://service.example.invalid/mcp"}
  }
}
```

Store the real remote URL, outbound identity reference, and TLS-profile reference in
`AgentConfig`, not in MCP client JSON or documentation.
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
    image: example/audio-transcriber@sha256:<digest>
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
    image: example/audio-transcriber@sha256:<digest>
    hostname: audio-transcriber-mcp
    environment:
      - TRANSPORT=streamable-http
      - HOST=0.0.0.0
      - PORT=8000
    ports:
      - "8000:8000"

  audio-transcriber-agent:
    image: example/audio-transcriber@sha256:<digest>
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
# Internal (self-signed) — homelab .example.invalid zone
audio-transcriber.example.invalid {
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
curl -s "http://technitium.example.invalid:5380/api/zones/records/add" \
  --data-urlencode "token=$TECHNITIUM_DNS_TOKEN" \
  --data-urlencode "domain=audio-transcriber.example.invalid" \
  --data-urlencode "zone=arpa" \
  --data-urlencode "type=A" \
  --data-urlencode "ipAddress=192.0.2.10" \
  --data-urlencode "ttl=3600"
```

…or add an **A record** `audio-transcriber.example.invalid → <caddy-host-ip>` in the Technitium
web console (`http://technitium.example.invalid:5380`). The ecosystem
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
`http://audio-transcriber.example.invalid/mcp` instead.
