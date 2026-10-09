"""Native epistemic-graph blob ingestion for transcribed audio.

CONCEPT:AU-KG.ingest.list-durable-media. When a live epistemic-graph engine is
reachable, the audio file that was transcribed is stored as a content-addressed
**blob** with a shared media-asset graph record (carrying its Whisper metadata),
via the agent-connector-sdk knowledge-ingest facade
(:mod:`agent_connector_sdk.ingest`). This makes the raw audio bytes — not just a
filesystem path — durable, deduped, and queryable inside the knowledge graph, so
a ``:Transcript`` can point back at it via ``:transcribedFrom``.

The ingest facade fails closed when the authoritative engine is unavailable or
unconfigured; this module never reports an uncommitted blob as ingested.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
from typing import Any

from agent_connector_sdk.ingest import (
    ChangeSet,
    IngestBinding,
    IngestUnavailableError,
    KnowledgeIngest,
    MediaAsset,
    current_ingest,
)

logger = logging.getLogger("AudioTranscriber.kg")

_SOURCE = "audio-transcriber"
_BINDING = IngestBinding(connector="audio-transcriber", stream="audio")

# Whisper/transcription info keys worth carrying onto the media-asset record.
_INFO_FIELDS = (
    "language",
    "language_probability",
    "duration",
    "whisper_model",
    "task",
    "source_uri",
)


async def ingest_audio_file(
    file_path: str | None,
    *,
    info: dict[str, Any] | None = None,
    source: str = _SOURCE,
    ingest: KnowledgeIngest | None = None,
) -> dict[str, Any] | None:
    """Store a transcribed audio file as a blob + media record in the graph.

    Returns ``{asset_id, digest, size_bytes, media_type}`` on success, or
    ``None`` when there is no file, no engine is reachable, or the store
    failed (never raises). ``ingest`` may be injected (tests); otherwise the
    process-installed :class:`~agent_connector_sdk.ingest.KnowledgeIngest` is
    used.
    """
    if not file_path or not os.path.exists(file_path):
        return None

    service = ingest
    if service is None:
        try:
            service = current_ingest()
        except IngestUnavailableError:
            return None

    info = info or {}
    mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    media_type = "video" if mime.startswith("video") else "audio"

    try:
        with open(file_path, "rb") as fh:
            data = fh.read()
    except OSError as e:
        logger.warning("Operation failed: error_type=%s", type(e).__name__)
        return None

    extra = {k: info[k] for k in _INFO_FIELDS if info.get(k) is not None}
    name = info.get("name") or os.path.basename(file_path)
    asset = MediaAsset(data=data, mime_type=mime, name=name, properties=extra)

    try:
        receipt = await service.submit(_BINDING, ChangeSet(media=(asset,)))
    except Exception as e:  # noqa: BLE001 — engine/store failure is non-fatal
        logger.warning("Operation failed: error_type=%s", type(e).__name__)
        return None

    # The SDK resolves the stored blob's own content digest server-side and
    # folds it into a generated media-asset record; the identical sha256 of
    # the bytes we just sent is the closest honest local analogue to surface
    # here (same digest, computed without a round trip).
    digest = hashlib.sha256(data).hexdigest()
    asset_id = f"blob:{digest}"
    logger.info(
        "KG media ingest: stored %s (%s bytes) as asset %s digest %s (affected=%s)",
        name,
        len(data),
        asset_id,
        digest[:16],
        receipt.affected_count,
    )
    return {
        "asset_id": asset_id,
        "digest": digest,
        "size_bytes": len(data),
        "media_type": media_type,
    }
