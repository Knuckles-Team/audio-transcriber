"""Native epistemic-graph ingestion for audio transcriptions.

CONCEPT:AU-KG.ingest.enterprise-source-extractor. audio-transcriber is a *producer*:
after Whisper transcribes an audio/video file it natively pushes the result into the ONE
epistemic-graph engine across every modality that applies (the "maximum ingestion" bar):

* **blob**  — the raw audio bytes → a shared media-asset record (``audio_transcriber.kg_media``)
* **document** — the transcript text → shared ``:Document`` (``ingest_documents``); the hub
  chunks/embeds it for semantic search
* **typed nodes** — the Whisper segments → ``:TranscriptSegment`` nodes (``ingest_entities``),
  linked ``:segmentOf`` the transcript and ``:transcribedFrom`` the audio asset

All three ride the ``agent_connector_sdk.ingest`` knowledge-ingest facade
(:class:`~agent_connector_sdk.ingest.KnowledgeIngest`), which is async — every public
ingest function here is an ``async def`` and its callers must ``await`` it. Engine
failures are explicit (``IngestError``) and no partial write is acknowledged. Node ids
follow ``audio:<class>:<externalId>`` and ``node_type`` matches the classes federated by
``audio_transcriber.ontology`` (``audio.ttl``).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from agent_connector_sdk.ingest import (
    ChangeSet,
    Document,
    Entity,
    IngestBinding,
    IngestError,
    IngestUnavailableError,
    KnowledgeIngest,
    Relationship,
    current_ingest,
)

logger = logging.getLogger("AudioTranscriber.kg")

_SOURCE = "audio-transcriber"
_DOMAIN = "audio"
_BINDING = IngestBinding(connector="audio-transcriber", stream=_DOMAIN)


def _to_entity(record: dict[str, Any]) -> Entity:
    return Entity(
        id=record.get("id"),
        node_type=record.get("node_type"),
        properties={k: v for k, v in record.items() if k not in ("id", "node_type")},
    )


def _to_relationship(record: dict[str, Any]) -> Relationship:
    props = {
        k: v
        for k, v in record.items()
        if k not in ("source", "target", "relationship")
    }
    return Relationship(
        source=record["source"],
        target=record["target"],
        relationship=record["relationship"],
        properties=props or None,
    )


def _to_document(record: dict[str, Any]) -> Document:
    return Document(
        id=record["id"],
        text=record["text"],
        title=record.get("title"),
        source_uri=record.get("source_uri"),
        properties={
            k: v
            for k, v in record.items()
            if k not in ("id", "text", "title", "source_uri")
        },
    )


async def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    ingest: KnowledgeIngest | None = None,
) -> dict[str, int]:
    """Write typed OWL nodes (+ edges) into the engine (``:TranscriptSegment`` …).

    ``entities`` use ``node_type`` and relationships use ``relationship``.
    ``ingest`` may be injected (tests); otherwise the process-installed
    :class:`~agent_connector_sdk.ingest.KnowledgeIngest` is used.
    """
    if not entities:
        raise IngestError("ingest_entities needs at least one entity")
    change_set = ChangeSet(
        entities=tuple(_to_entity(e) for e in entities),
        relationships=tuple(_to_relationship(r) for r in relationships or ()),
    )
    service = ingest or current_ingest()
    receipt = await service.submit(_BINDING, change_set)
    return {"nodes": receipt.affected_count, "edges": receipt.relationship_count}


async def ingest_documents(
    documents: list[dict[str, Any]],
    *,
    ingest: KnowledgeIngest | None = None,
) -> dict[str, int]:
    """Write transcript text records as shared ``:Document`` nodes (search fodder).

    Each doc: ``{"id":..., "text":..., "title"?:..., "source_uri"?:..., ...props}``.
    ``ingest`` may be injected (tests); otherwise the process-installed service.
    """
    if not documents:
        raise IngestError("ingest_documents needs at least one document")
    change_set = ChangeSet(documents=tuple(_to_document(d) for d in documents))
    service = ingest or current_ingest()
    receipt = await service.submit(_BINDING, change_set)
    return {"nodes": receipt.affected_count, "edges": receipt.relationship_count}


# --------------------------------------------------------------------------- #
# Domain mapper: a Whisper result -> blob + transcript document + segment nodes.
# --------------------------------------------------------------------------- #
def _ext_id(name: str) -> str:
    """Stable, id-safe external id from a transcript name/stem."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower()
    return slug or "transcript"


async def ingest_transcription(
    result: dict[str, Any],
    *,
    audio_path: str | None = None,
    name: str | None = None,
    model: str | None = None,
    task: str = "transcribe",
    max_segments: int = 500,
    source: str = _SOURCE,
    ingest: KnowledgeIngest | None = None,
) -> dict[str, Any] | None:
    """Ingest a Whisper ``result`` across all modalities and link them.

    1. store the audio bytes as a shared media-asset blob (best-effort),
    2. write the transcript text as a shared ``:Document`` (``audio:transcript:<ext>``),
    3. write each Whisper segment as a ``:TranscriptSegment`` typed node linked
       ``:segmentOf`` the transcript and the transcript ``:transcribedFrom`` the asset.

    Returns a summary ``{transcript_id, asset, documents, entities}`` or ``None`` when
    there is nothing to write / no engine (never raises). ``ingest`` may be injected
    (tests); otherwise the process-installed service is used for all three steps.
    """
    if not result:
        return None
    text = (result.get("text") or "").strip()
    if not text:
        return None

    service = ingest
    if service is None:
        try:
            service = current_ingest()
        except IngestUnavailableError:
            return None

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

    # 1) blob — the raw audio bytes as a media-asset record.
    asset: dict[str, Any] | None = None
    if audio_path:
        from audio_transcriber.kg_media import ingest_audio_file

        asset = await ingest_audio_file(
            audio_path, info=info, source=source, ingest=service
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
    documents_result = await ingest_documents([doc], ingest=service)

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
                "node_type": "TranscriptSegment",
                "text": (seg.get("text") or "").strip(),
                "startTime": seg.get("start"),
                "endTime": seg.get("end"),
                "noSpeechProb": seg.get("no_speech_prob"),
            }
        )
        relationships.append(
            {"source": seg_id, "target": transcript_id, "relationship": "segmentOf"}
        )
    entities_result: dict[str, int] | None = None
    if entities:
        entities_result = await ingest_entities(entities, relationships, ingest=service)

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
