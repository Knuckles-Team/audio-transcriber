"""Native epistemic-graph ingestion for audio transcriptions.

CONCEPT:AU-KG.ingest.enterprise-source-extractor. audio-transcriber is a *producer*:
after Whisper transcribes an audio/video file it natively pushes the result into the ONE
epistemic-graph engine across every modality that applies (the "maximum ingestion" bar):

* **blob**  — the raw audio bytes → shared ``:MediaAsset``/``:Blob`` (``audio_transcriber.kg_media``)
* **document** — the transcript text → shared ``:Document`` (``ingest_documents``); the hub
  chunks/embeds it for semantic search
* **typed nodes** — the Whisper segments → ``:TranscriptSegment`` nodes (``ingest_entities``),
  linked ``:segmentOf`` the transcript and ``:transcribedFrom`` the audio asset

All three ride the lightweight engine client via the shared
``agent_utilities.knowledge_graph.memory.native_ingest`` primitive; when that primitive is
not yet installed we fall back to a self-contained txn writer with the same shape. Everything
is dependency-/engine-guarded: with no KG stack or no reachable engine every entry point
**no-ops** (returns ``None``), so the transcriber runs with zero KG infrastructure. Node ids
follow ``audio:<class>:<externalId>`` and ``type`` matches the classes federated by
``audio_transcriber.ontology`` (``audio.ttl``).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger("AudioTranscriber.kg")

_SOURCE = "audio-transcriber"
_DOMAIN = "audio"
_DEFAULT_GRAPH = "__commons__"


# --------------------------------------------------------------------------- #
# Low-level write path: prefer the shared primitive, else a self-contained txn.
# --------------------------------------------------------------------------- #
def _native_client() -> tuple[Any | None, str]:
    """Return ``(engine_client, graph_name)`` or ``(None, "")`` when unavailable."""
    try:
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )
    except Exception as e:  # noqa: BLE001 — KG stack absent
        logger.debug("KG ingest unavailable (import): %s", e)
        return None, ""
    try:
        engine = GraphComputeEngine()
        client = getattr(engine, "_client", None)
        if client is None:
            return None, ""
        return client, (getattr(engine, "graph_name", None) or _DEFAULT_GRAPH)
    except Exception as e:  # noqa: BLE001 — engine unreachable
        logger.debug("KG ingest: engine unreachable: %s", e)
        return None, ""


def _fallback_write_nodes(
    client: Any,
    graph: str,
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None,
    *,
    source: str,
    domain: str,
) -> dict[str, int] | None:
    """Self-contained txn writer used when the shared primitive is not installed."""
    nodes = [n for n in nodes if n.get("id")]
    if not nodes:
        return None
    try:
        txn = client.txn.begin(graph=graph)
        for node in nodes:
            props = {k: v for k, v in node.items() if k != "id" and v is not None}
            props.setdefault("source", source)
            props.setdefault("domain", domain)
            client.txn.add_node(txn, node["id"], props)
        committed = client.txn.commit(txn)
    except Exception as e:  # noqa: BLE001 — engine/txn failure is non-fatal
        logger.warning("KG ingest: txn failed: %s", e)
        return None
    if not committed:
        logger.warning("KG ingest: txn not committed (conflict)")
        return None

    edges = 0
    for rel in relationships or []:
        try:
            client.edges.add(
                rel["source"], rel["target"], {"type": rel.get("type", "RELATED")}
            )
            edges += 1
        except Exception as e:  # noqa: BLE001 — pure edge link, best-effort
            logger.debug("KG ingest: edge skipped: %s", e)
    logger.info("KG ingest[%s]: wrote %d nodes, %d edges", domain, len(nodes), edges)
    return {"nodes": len(nodes), "edges": edges}


def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    source: str = _SOURCE,
    domain: str = _DOMAIN,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Write typed OWL nodes (+ edges) into the engine (``:TranscriptSegment`` …).

    ``entities``: ``[{"id":..., "type":<owl:Class>, ...props}]``.
    ``relationships``: ``[{"source":id, "target":id, "type":<link>}]``.
    Returns ``{"nodes":n, "edges":m}`` or ``None``. ``client``/``graph`` may be injected.
    """
    if not entities:
        return None
    # Prefer the shared primitive when both it and (if injected) no test client apply.
    if client is None:
        try:
            from agent_utilities.knowledge_graph.memory import native_ingest

            return native_ingest.ingest_entities(
                entities, relationships, source=source, domain=domain, graph=graph
            )
        except Exception as e:  # noqa: BLE001 — primitive absent; self-contained path
            logger.debug("native_ingest.ingest_entities unavailable: %s", e)
        client, graph = _native_client()
    if client is None:
        return None
    return _fallback_write_nodes(
        client,
        graph or _DEFAULT_GRAPH,
        entities,
        relationships,
        source=source,
        domain=domain,
    )


def ingest_documents(
    documents: list[dict[str, Any]],
    *,
    source: str = _SOURCE,
    domain: str = _DOMAIN,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Write transcript text records as shared ``:Document`` nodes (search fodder).

    Each doc: ``{"id":..., "text":..., "title"?:..., "source_uri"?:..., ...props}``.
    Returns ``{"nodes":n, "edges":0}`` or ``None``. ``client``/``graph`` may be injected.
    """
    if not documents:
        return None
    if client is None:
        try:
            from agent_utilities.knowledge_graph.memory import native_ingest

            return native_ingest.ingest_documents(
                documents, source=source, domain=domain, graph=graph
            )
        except Exception as e:  # noqa: BLE001 — primitive absent; self-contained path
            logger.debug("native_ingest.ingest_documents unavailable: %s", e)
        client, graph = _native_client()
    if client is None:
        return None

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nodes: list[dict[str, Any]] = []
    for doc in documents:
        did = doc.get("id")
        text = doc.get("text") or doc.get("content")
        if not did or not text:
            continue
        node = {k: v for k, v in doc.items() if k != "content" and v is not None}
        node["id"] = did
        node["type"] = "Document"
        node["text"] = text
        node.setdefault("created_at", now)
        nodes.append(node)
    return _fallback_write_nodes(
        client, graph or _DEFAULT_GRAPH, nodes, None, source=source, domain=domain
    )


def media_store() -> Any | None:
    """Return a ``MediaStore`` over a live engine for raw-blob ingestion, or ``None``."""
    from audio_transcriber.kg_media import _media_store

    return _media_store()


# --------------------------------------------------------------------------- #
# Domain mapper: a Whisper result -> blob + transcript document + segment nodes.
# --------------------------------------------------------------------------- #
def _ext_id(name: str) -> str:
    """Stable, id-safe external id from a transcript name/stem."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower()
    return slug or "transcript"


def ingest_transcription(
    result: dict[str, Any],
    *,
    audio_path: str | None = None,
    name: str | None = None,
    model: str | None = None,
    task: str = "transcribe",
    max_segments: int = 500,
    source: str = _SOURCE,
    media_store: Any | None = None,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, Any] | None:
    """Ingest a Whisper ``result`` across all modalities and link them.

    1. store the audio bytes as a shared ``:MediaAsset`` blob (best-effort),
    2. write the transcript text as a shared ``:Document`` (``audio:transcript:<ext>``),
    3. write each Whisper segment as a ``:TranscriptSegment`` typed node linked
       ``:segmentOf`` the transcript and the transcript ``:transcribedFrom`` the asset.

    Returns a summary ``{transcript_id, asset, documents, entities}`` or ``None`` when
    there is nothing to write / no engine (never raises).
    """
    if not result:
        return None
    text = (result.get("text") or "").strip()
    if not text:
        return None

    import os

    stem = name or (os.path.basename(audio_path) if audio_path else "transcript")
    ext = _ext_id(stem)
    transcript_id = f"audio:transcript:{ext}"

    language = result.get("language")
    info = {
        "name": stem,
        "language": language,
        "language_probability": result.get("language_probability"),
        "duration": result.get("duration"),
        "whisper_model": model,
        "task": task,
        "source_uri": audio_path,
    }

    # 1) blob — the raw audio bytes as a :MediaAsset.
    asset: dict[str, Any] | None = None
    if audio_path:
        from audio_transcriber.kg_media import ingest_audio_file

        asset = ingest_audio_file(
            audio_path, info=info, source=source, media_store=media_store
        )

    # 2) document — the transcript text as a :Document.
    doc = {
        "id": transcript_id,
        "title": stem,
        "text": text,
        "source_uri": audio_path,
        "language": language,
        "language_probability": result.get("language_probability"),
        "duration": result.get("duration"),
        "whisper_model": model,
        "task": task,
    }
    if asset and asset.get("asset_id"):
        doc["transcribedFrom"] = asset["asset_id"]
    documents_result = ingest_documents(
        [doc], source=source, client=client, graph=graph
    )

    # 3) typed nodes — the Whisper segments as :TranscriptSegment.
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for seg in (result.get("segments") or [])[:max_segments]:
        sid = seg.get("id")
        if sid is None:
            continue
        seg_id = f"audio:segment:{ext}:{sid}"
        entities.append(
            {
                "id": seg_id,
                "type": "TranscriptSegment",
                "text": (seg.get("text") or "").strip(),
                "startTime": seg.get("start"),
                "endTime": seg.get("end"),
                "noSpeechProb": seg.get("no_speech_prob"),
            }
        )
        relationships.append(
            {"source": seg_id, "target": transcript_id, "type": "segmentOf"}
        )
    entities_result = ingest_entities(
        entities, relationships, source=source, client=client, graph=graph
    )

    if documents_result is None and entities_result is None and asset is None:
        return None
    logger.info(
        "KG ingest: transcript %s (asset=%s, doc=%s, segments=%s)",
        transcript_id,
        bool(asset),
        documents_result,
        entities_result,
    )
    return {
        "transcript_id": transcript_id,
        "asset": asset,
        "documents": documents_result,
        "entities": entities_result,
    }
