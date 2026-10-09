"""Native epistemic-graph audio-blob ingestion — Wire-First live-path coverage.

Exercises the real ``ingest_audio_file`` seam with a fake epistemic-graph ingest
transport (no engine required). CONCEPT:AU-KG.ingest.list-durable-media.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
from agent_connector_sdk.ingest import KnowledgeIngest

from audio_transcriber.kg_media import ingest_audio_file


class _FakeTransport:
    """Captures the media submission the way the real transport is invoked."""

    def __init__(self):
        self.requests = []
        self.stored_blobs = []

    async def source_status(self, connector, stream):
        return SimpleNamespace(accepted_checkpoint=None)

    async def submit(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            affected_count=len(request.records),
            relationship_count=len(request.relationships),
        )

    async def store_blob(self, data):
        self.stored_blobs.append(data)
        return hashlib.sha256(data).hexdigest()


@pytest.fixture
def ingest():
    transport = _FakeTransport()
    return KnowledgeIngest(transport, loop=None), transport


@pytest.mark.asyncio
async def test_ingest_audio_file_stores_bytes_and_metadata(ingest, tmp_path):
    service, transport = ingest
    f = tmp_path / "talk.mp3"
    f.write_bytes(b"\x00ID3-audio-bytes\x01")

    res = await ingest_audio_file(
        str(f),
        info={"language": "en", "duration": 12.3, "whisper_model": "base"},
        ingest=service,
    )

    assert res is not None
    expected_digest = hashlib.sha256(f.read_bytes()).hexdigest()
    assert res["digest"] == expected_digest
    assert res["asset_id"] == f"blob:{expected_digest}"
    assert res["media_type"] == "audio"
    assert res["size_bytes"] == f.stat().st_size

    assert len(transport.stored_blobs) == 1
    assert transport.stored_blobs[0] == f.read_bytes()
    record = transport.requests[0].records[0]
    assert record.payload["mime_type"] == "audio/mpeg"
    assert record.payload["name"] == "talk.mp3"
    assert record.payload["language"] == "en"
    assert record.payload["whisper_model"] == "base"


@pytest.mark.asyncio
async def test_ingest_audio_file_detects_video(ingest, tmp_path):
    service, _ = ingest
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"video")
    res = await ingest_audio_file(str(f), ingest=service)
    assert res is not None
    assert res["media_type"] == "video"


@pytest.mark.asyncio
async def test_ingest_audio_file_noops_without_engine(tmp_path):
    f = tmp_path / "talk.wav"
    f.write_bytes(b"x")
    # No injected service + no reachable/configured engine -> clean no-op.
    assert await ingest_audio_file(str(f)) is None


@pytest.mark.asyncio
async def test_ingest_audio_file_noops_on_missing_file(ingest):
    service, _ = ingest
    assert await ingest_audio_file("/no/such/file.mp3", ingest=service) is None
