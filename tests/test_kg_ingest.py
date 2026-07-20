"""Native epistemic-graph transcription ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` / ``ingest_documents`` / ``ingest_transcription``
seam with a fake engine client (no engine required), asserting the txn add_node/commit +
edge calls and the Whisper-result -> blob/document/segment mapping.
CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from agent_utilities.knowledge_graph.memory.native_ingest import NativeIngestError

from audio_transcriber.kg_ingest import (
    ingest_documents,
    ingest_entities,
    ingest_transcription,
)


class _FakeTxn:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.committed = False

    def begin(self, graph=None):
        self.graph = graph
        return "txn-1"

    def add_node(self, txn, node_id, props):
        self.nodes[node_id] = props

    def add_edge(self, txn, source, target, props):
        self.edges.append((source, target, props))

    def commit(self, txn):
        self.committed = True
        return True


class _FakeClient:
    def __init__(self):
        self.txn = _FakeTxn()


@dataclass
class _Stored:
    asset_id: str
    digest: str


class _FakeMediaStore:
    def __init__(self):
        self.calls = []

    def store_media(self, data, **kw):
        self.calls.append((data, kw))
        return _Stored(asset_id="media:aa", digest="aabb")


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


def test_ingest_entities_writes_nodes_and_edges():
    c = _FakeClient()
    res = ingest_entities(
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
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 1, "edges": 1}
    assert c.txn.committed is True
    node = c.txn.nodes["audio:segment:x:0"]
    assert node["node_type"] == "TranscriptSegment"
    assert node["source"] == "audio-transcriber"
    assert node["domain"] == "audio"
    assert c.txn.edges == [
        ("audio:segment:x:0", "audio:transcript:x", {"relationship": "segmentOf"})
    ]


def test_ingest_documents_writes_document_node():
    c = _FakeClient()
    res = ingest_documents(
        [{"id": "audio:transcript:x", "title": "x", "text": "hello world"}],
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 1, "edges": 0}
    node = c.txn.nodes["audio:transcript:x"]
    assert node["node_type"] == "Document"
    assert node["text"] == "hello world"
    assert "created_at" in node


def test_ingest_transcription_maps_all_modalities(tmp_path):
    c = _FakeClient()
    store = _FakeMediaStore()
    audio = tmp_path / "My Talk.mp3"
    audio.write_bytes(b"audio-bytes")
    res = ingest_transcription(
        _RESULT,
        audio_path=str(audio),
        name="My Talk",
        model="base",
        media_store=store,
        client=c,
        graph="__commons__",
    )
    assert res is not None
    assert res["transcript_id"] == "audio:transcript:my-talk"
    # blob stored
    assert res["asset"]["asset_id"] == "media:aa"
    assert len(store.calls) == 1
    # document + 2 segment nodes written
    assert res["documents"] == {"nodes": 1, "edges": 0}
    assert res["entities"] == {"nodes": 2, "edges": 2}
    # transcript document carries provenance + link to the asset
    doc = c.txn.nodes["audio:transcript:my-talk"]
    assert doc["node_type"] == "Document"
    assert doc["text"] == "hello world"
    assert doc["transcribedFrom"] == "media:aa"
    assert doc["whisper_model"] == "base"
    # segments typed + linked
    assert c.txn.nodes["audio:segment:my-talk:0"]["node_type"] == "TranscriptSegment"
    assert (
        "audio:segment:my-talk:1",
        "audio:transcript:my-talk",
        {"relationship": "segmentOf"},
    ) in c.txn.edges


def test_ingest_transcription_noops_on_empty_text():
    assert ingest_transcription({"text": "   "}, client=_FakeClient()) is None
    assert ingest_transcription({}, client=_FakeClient()) is None


def test_retired_structural_alias_is_rejected():
    with pytest.raises(NativeIngestError, match="canonical node_type"):
        ingest_entities([{"id": "a", "type": "TranscriptSegment"}], client=_FakeClient())


def test_empty_native_ingest_is_rejected():
    with pytest.raises(NativeIngestError, match="at least one entity"):
        ingest_entities([], client=_FakeClient())
