"""Native epistemic-graph audio-blob ingestion — Wire-First live-path coverage.

Exercises the real ``ingest_audio_file`` seam with a fake ``MediaStore`` (no engine
required). CONCEPT:AU-KG.ingest.list-durable-media.
"""

from __future__ import annotations

from dataclasses import dataclass

from audio_transcriber.kg_media import ingest_audio_file


@dataclass
class _Stored:
    asset_id: str
    digest: str


class _FakeMediaStore:
    """Captures the store_media call the way the real MediaStore is invoked."""

    def __init__(self):
        self.calls = []

    def store_media(self, data, **kw):
        self.calls.append((data, kw))
        return _Stored(asset_id="media:cafe", digest="cafebabe")


def test_ingest_audio_file_stores_bytes_and_metadata(tmp_path):
    f = tmp_path / "talk.mp3"
    f.write_bytes(b"\x00ID3-audio-bytes\x01")
    store = _FakeMediaStore()

    res = ingest_audio_file(
        str(f),
        info={"language": "en", "duration": 12.3, "whisper_model": "base"},
        media_store=store,
    )

    assert res is not None
    assert res["asset_id"] == "media:cafe"
    assert res["digest"] == "cafebabe"
    assert res["media_type"] == "audio"
    assert res["size_bytes"] == f.stat().st_size

    assert len(store.calls) == 1
    data, kw = store.calls[0]
    assert data == f.read_bytes()
    assert kw["source"] == "audio-transcriber"
    assert kw["mime_type"] == "audio/mpeg"
    assert kw["name"] == "talk.mp3"
    assert kw["extra"]["language"] == "en"
    assert kw["extra"]["whisper_model"] == "base"


def test_ingest_audio_file_detects_video(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"video")
    store = _FakeMediaStore()
    res = ingest_audio_file(str(f), media_store=store)
    assert res is not None
    assert res["media_type"] == "video"


def test_ingest_audio_file_noops_without_engine(tmp_path):
    f = tmp_path / "talk.wav"
    f.write_bytes(b"x")
    # No injected store + no reachable engine -> clean no-op (never raises).
    assert ingest_audio_file(str(f)) is None


def test_ingest_audio_file_noops_on_missing_file():
    assert ingest_audio_file("/no/such/file.mp3", media_store=_FakeMediaStore()) is None
