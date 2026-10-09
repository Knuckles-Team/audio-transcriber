"""Native epistemic-graph transcription ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` / ``ingest_documents`` / ``ingest_transcription``
seam with a fake epistemic-graph ingest transport (no engine required), asserting the
generated ``SourceIngestionRequest`` records/relationships the SDK's own request
builder produces, and the Whisper-result -> blob/document/segment mapping.
CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_connector_sdk.ingest import IngestError, KnowledgeIngest

from audio_transcriber.kg_ingest import (
    ingest_documents,
    ingest_entities,
    ingest_transcription,
)


class _FakeTransport:
    """Fakes the transport boundary; the SDK's real request builder runs on top."""

    def __init__(self):
        self.requests = []

    async def source_status(self, connector, stream):
        return SimpleNamespace(accepted_checkpoint=None)

    async def submit(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            affected_count=len(request.records),
            relationship_count=len(request.relationships),
        )

    async def store_blob(self, data):
        digest = "aabb"
        return digest


@pytest.fixture
def ingest():
    transport = _FakeTransport()
    return KnowledgeIngest(transport, loop=None), transport


_RESULT = {
    "text": " hello world ",
    "language": "en",
    "language_probability": 0.99,
    "duration": 3.2,
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.5, "text": " hello", "no_speech_prob": 0.01},
        {"id": 1, "start": 1.5, "end": 3.2, "text": " world", "no_speech_prob": 0.02},
    ],
}


@pytest.mark.asyncio
async def test_ingest_entities_writes_nodes_and_edges(ingest):
    service, transport = ingest
    res = await ingest_entities(
        [
            {"id": "audio:segment:x:0", "node_type": "TranscriptSegment", "text": "hi"},
        ],
        [
            {
                "source": "audio:segment:x:0",
                "target": "audio:transcript:x",
                "relationship": "segmentOf",
            }
        ],
        ingest=service,
    )
    assert res == {"nodes": 1, "edges": 1}
    assert len(transport.requests) == 1
    record = transport.requests[0].records[0]
    assert record.record_id == "audio:segment:x:0"
    assert record.payload["text"] == "hi"
    relationship = transport.requests[0].relationships[0]
    assert relationship.source.record_id == "audio:segment:x:0"
    assert relationship.target.record_id == "audio:transcript:x"


@pytest.mark.asyncio
async def test_ingest_documents_writes_document_node(ingest):
    service, transport = ingest
    res = await ingest_documents(
        [{"id": "audio:transcript:x", "title": "x", "text": "hello world"}],
        ingest=service,
    )
    assert res == {"nodes": 1, "edges": 0}
    record = transport.requests[0].records[0]
    assert record.record_id == "audio:transcript:x"
    assert record.payload["text"] == "hello world"


@pytest.mark.asyncio
async def test_ingest_transcription_maps_all_modalities(ingest, tmp_path):
    service, transport = ingest
    audio = tmp_path / "My Talk.mp3"
    audio.write_bytes(b"audio-bytes")
    res = await ingest_transcription(
        _RESULT,
        audio_path=str(audio),
        name="My Talk",
        model="base",
        ingest=service,
    )
    assert res is not None
    assert res["transcript_id"] == "audio:transcript:my-talk"
    # blob stored (media change set submitted first)
    assert res["asset"]["asset_id"].startswith("blob:")
    # document + 2 segment nodes written, each its own submission
    assert res["documents"] == {"nodes": 1, "edges": 0}
    assert res["entities"] == {"nodes": 2, "edges": 2}
    # three submissions: media, document, entities+relationships
    assert len(transport.requests) == 3
    doc_request = transport.requests[1]
    doc_record = doc_request.records[0]
    assert doc_record.record_id == "audio:transcript:my-talk"
    assert doc_record.payload["text"] == "hello world"
    assert doc_record.payload["transcribedFrom"].startswith("blob:")
    assert doc_record.payload["whisper_model"] == "base"
    entities_request = transport.requests[2]
    record_ids = {r.record_id for r in entities_request.records}
    assert record_ids == {"audio:segment:my-talk:0", "audio:segment:my-talk:1"}
    relationship_pairs = {
        (r.source.record_id, r.target.record_id) for r in entities_request.relationships
    }
    assert ("audio:segment:my-talk:1", "audio:transcript:my-talk") in relationship_pairs


@pytest.mark.asyncio
async def test_ingest_transcription_noops_on_empty_text(ingest):
    service, _ = ingest
    assert await ingest_transcription({"text": "   "}, ingest=service) is None
    assert await ingest_transcription({}, ingest=service) is None


@pytest.mark.asyncio
async def test_ingest_entities_rejects_missing_node_type(ingest):
    service, _ = ingest
    with pytest.raises(IngestError, match="id and a node_type"):
        await ingest_entities([{"id": "a"}], ingest=service)


@pytest.mark.asyncio
async def test_empty_entities_is_rejected(ingest):
    service, _ = ingest
    with pytest.raises(IngestError, match="at least one entity"):
        await ingest_entities([], ingest=service)
