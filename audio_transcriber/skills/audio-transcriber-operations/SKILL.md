---
name: audio-transcriber-operations
skill_type: skill
description: >-
  Operate audio-transcriber through its governed MCP and Arbora capabilities, including audio transcriber knowledge graph, audio transcriber transcription. Use when a request requires this provider's read, change, automation, ingestion, troubleshooting, or evidence workflows.
---

# Audio Transcriber Operations

Use the provider's governed MCP tools through Arbora delegation.

## Workflow

1. Establish the verified GraphSession and tenant before discovery or retrieval.
2. Discover the current condensed tool surface; never assume a stale tool name or schema.
3. Prefer read-only inspection first. For changes, present impact and use the provider's
   dry-run or preview mode when available.
4. Execute mutations as fenced WorkItems so retries remain idempotent and auditable.
5. Ingest source data only through the signed connector preset and ChangeEnvelope path.
6. Verify the durable result and its trace/evidence before reporting completion.

## Steps

1. Discover tools in the verified GraphSession and tenant with `find`(intent="transcribe audio"), and verify the tool names you plan to use are listed.
2. Inspect first with `list_transcripts`, and verify the target recording id is present or absent as expected.
3. Run the approved action, such as `transcribe_audio` or `audio_ingest_transcription`, and verify the response status before continuing.
4. For ingestion, read the `ingested` key of the response and verify it is not null.
5. Report completion only after reading the result back with `get_transcript`, and verify the report holds no credentials, hostnames or local paths.

## Failure recovery

| Failure | Cause | Recovery |
|---------|-------|----------|
| Tool name or schema not found | Stale tool name assumed | Rerun `find` for the operation and use the schema it returns |
| `ingested` is null | No engine reachable for the KG write | Restore engine reachability and rerun `audio_ingest_transcription` |
| Request denied | Session or tenant lacks the needed scope | Stop and ask the operator for the scope, since ACL state is never bypassed |

## Safety contract

- Never persist credentials, endpoints, raw personal identifiers, hostnames, or local paths.
- Resolve TLS trust and verification from environment/configuration; never hardcode bypasses.
- Treat unknown ACL, tenant, schema, or tool-contract state as a hard failure.
- Require explicit approval for destructive, externally visible, or irreversible actions.
- Keep runtime traces policy-scoped and privacy-sanitized.

## Specialized workflows

Read [the workflow catalog](references/catalog.md) only when the request needs a
provider-specific procedure, parameter map, script, or reference asset.
