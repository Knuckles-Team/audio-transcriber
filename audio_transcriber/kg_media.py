"""Native epistemic-graph blob ingestion for transcribed audio.

CONCEPT:AU-KG.ingest.list-durable-media. When a live epistemic-graph engine is
reachable, the audio file that was transcribed is stored as a content-addressed
**blob** with a shared ``:AssetOccurrence`` graph node (carrying its Whisper metadata) in
ONE cross-modal ACID commit, via the agent-utilities ``MediaStore``. This makes the raw
audio bytes — not just a filesystem path — durable, deduped, and queryable inside the
knowledge graph, so a ``:Transcript`` can point back at it via ``:transcribedFrom``.

The required native media store fails closed when the authoritative engine is unavailable;
the connector never reports an uncommitted blob as ingested.
"""

from __future__ import annotations

import logging
import mimetypes
import os
from typing import Any

from agent_utilities.knowledge_graph.memory.native_ingest import media_store

logger = logging.getLogger("AudioTranscriber.kg")

_SOURCE = "audio-transcriber"

# Whisper/transcription info keys worth carrying onto the :AssetOccurrence node.
_INFO_FIELDS = (
    "language",
    "language_probability",
    "duration",
    "whisper_model",
    "task",
    "source_uri",
)


def _media_store() -> Any:
    """Return the authoritative native media store."""
    return media_store()


def ingest_audio_file(
    file_path: str | None,
    *,
    info: dict[str, Any] | None = None,
    source: str = _SOURCE,
    media_store: Any | None = None,
) -> dict[str, Any] | None:
    """Store a transcribed audio file as a blob + ``:AssetOccurrence`` in the graph.

    Returns ``{asset_id, digest, size_bytes, media_type}`` on success, or ``None``
    when there is no engine, no file, or the store failed (never raises).
    ``media_store`` may be injected (tests); otherwise one is built on demand.
    """
    if not file_path or not os.path.exists(file_path):
        return None
    store = media_store if media_store is not None else _media_store()
    if store is None:
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

    try:
        stored = store.store_media(
            data,
            media_type=media_type,
            mime_type=mime,
            source=source,
            name=name,
            extra=extra,
        )
    except Exception as e:  # noqa: BLE001 — engine/store failure is non-fatal
        logger.warning("Operation failed: error_type=%s", type(e).__name__)
        return None
    if stored is None:
        return None

    asset_id = getattr(stored, "asset_id", None)
    digest = getattr(stored, "digest", "") or ""
    logger.info(
        "KG media ingest: stored %s (%s bytes) as asset %s digest %s",
        name,
        len(data),
        asset_id,
        digest[:16],
    )
    return {
        "asset_id": asset_id,
        "digest": digest,
        "size_bytes": len(data),
        "media_type": media_type,
    }
