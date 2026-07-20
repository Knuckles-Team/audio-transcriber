"""Native epistemic-graph ingestion for audio transcriptions.

CONCEPT:AU-KG.ingest.enterprise-source-extractor. audio-transcriber is a *producer*:
after Whisper transcribes an audio/video file it natively pushes the result into the ONE
epistemic-graph engine across every modality that applies (the "maximum ingestion" bar):

* **blob**  — the raw audio bytes → shared ``:AssetOccurrence``/``:Blob`` (``audio_transcriber.kg_media``)
* **document** — the transcript text → shared ``:Document`` (``ingest_documents``); the hub
  chunks/embeds it for semantic search
* **typed nodes** — the Whisper segments → ``:TranscriptSegment`` nodes (``ingest_entities``),
  linked ``:segmentOf`` the transcript and ``:transcribedFrom`` the audio asset

All three ride the required
``agent_utilities.knowledge_graph.memory.native_ingest`` transaction primitive. Engine
failures are explicit and no partial write is acknowledged. Node ids follow
``audio:<class>:<externalId>`` and ``node_type`` matches the classes federated by
``audio_transcriber.ontology`` (``audio.ttl``).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent_utilities.knowledge_graph.memory.native_ingest import (
    ingest_documents as _native_ingest_documents,
)
from agent_utilities.knowledge_graph.memory.native_ingest import (
    ingest_entities as _native_ingest_entities,
)

logger = logging.getLogger("AudioTranscriber.kg")

_SOURCE = "audio-transcriber"
_DOMAIN = "audio"
def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    source: str = _SOURCE,
    domain: str = _DOMAIN,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Write typed OWL nodes (+ edges) into the engine (``:TranscriptSegment`` …).

    ``entities`` use ``node_type`` and relationships use ``relationship``.
    ``client``/``graph`` may be injected for isolated validation.
    """
    return _native_ingest_entities(
        entities,
        relationships,
        source=source,
        domain=domain,
        client=client,
        graph=graph,
    )


def ingest_documents(
    documents: list[dict[str, Any]],
    *,
    source: str = _SOURCE,
    domain: str = _DOMAIN,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Write transcript text records as shared ``:Document`` nodes (search fodder).

    Each doc: ``{"id":..., "text":..., "title"?:..., "source_uri"?:..., ...props}``.
    ``client``/``graph`` may be injected for isolated validation.
    """
    return _native_ingest_documents(
        documents,
        source=source,
        domain=domain,
        client=client,
        graph=graph,
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

    1. store the audio bytes as a shared ``:AssetOccurrence`` blob (best-effort),
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

    # 1) blob — the raw audio bytes as a :AssetOccurrence.
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
