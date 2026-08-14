"""Native epistemic-graph transcription ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` / ``ingest_documents`` / ``ingest_transcription``
seam with a fake engine client (no engine required), asserting the txn add_node/commit +
edge calls and the Whisper-result -> blob/document/segment mapping.
CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass

import msgpack
import pytest
from agent_utilities.knowledge_graph.memory.native_ingest import NativeIngestError
from agent_utilities.security.brain_context import ActorContext, use_actor
from agent_utilities.models.company_brain import ActorType
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session

from audio_transcriber.kg_ingest import (
    ingest_documents,
    ingest_entities,
    ingest_transcription,
)


@pytest.fixture(autouse=True)
def _governed_session():
    actor = ActorContext(
        actor_id="subject:opaque:synthetic",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=(),
        tenant_id="tenant:opaque:synthetic",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:write"}),
        graph="graph:opaque:synthetic",
        policy_version="policy:opaque:synthetic",
        audience="epistemic-graph",
    )
    with use_actor(actor), use_session(session):
        yield


class _FakeNodes:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {}

    def properties(self, node_id: str) -> dict[str, Any] | None:
        return self.values.get(node_id)

    def list(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self.values.items())


class _FakeChanges:
    def __init__(self, nodes: _FakeNodes) -> None:
        self.nodes = nodes
        self.edges: list[tuple[str, str, dict[str, Any]]] = []
        self.applied: list[dict[str, Any]] = []
        self.records: dict[str, dict[str, Any]] = {}
        self.versions: dict[str, dict[str, Any]] = {}

    def get(self, envelope_id: str) -> dict[str, Any] | None:
        return self.records.get(envelope_id)

    def content_version(self, object_id: str) -> dict[str, Any] | None:
        return self.versions.get(object_id)

    def cursor(self, _source: str, _partition: str = "") -> None:
        return None

    def apply(self, envelope: dict[str, Any]) -> dict[str, Any]:
        self.applied.append(envelope)
        mutation = envelope["mutation"]
        for operation in mutation["operations"]:
            method = operation["method"]
            params = method["params"]
            properties = msgpack.unpackb(params["properties_msgpack"], raw=False)
            if method["method"] == "AddNode":
                self.nodes.values[params["node_id"]] = properties
            elif method["method"] == "AddEdge":
                self.edges.append(
                    (params["source_id"], params["target_id"], properties)
                )
        version = envelope["content_version"]
        self.versions[version["object_id"]] = version
        self.records[envelope["envelope_id"]] = envelope
        return {
            "batch_id": mutation["batch_id"],
            "replayed": False,
            "projection_pending": False,
        }


class _FakeRdf:
    def validate_shacl(self, _shapes: str, _data_graph: str) -> dict[str, Any]:
        return {"conforms": True, "results": []}


class _FakeClient:
    def __init__(self) -> None:
        self.nodes = _FakeNodes()
        self.changes = _FakeChanges(self.nodes)
        self.rdf = _FakeRdf()

    @staticmethod
    def supports(operation: str) -> bool:
        return operation == "ApplyChangeEnvelope"


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
    )
    assert res == {"nodes": 1, "edges": 1}
    assert len(c.changes.applied) == 1
    node = c.nodes.values["audio:segment:x:0"]
    assert node["node_type"] == "TranscriptSegment"
    assert node["source"] == "audio-transcriber"
    assert node["domain"] == "audio"
    assert c.changes.edges == [
        ("audio:segment:x:0", "audio:transcript:x", {"relationship": "segmentOf"})
    ]


def test_ingest_documents_writes_document_node():
    c = _FakeClient()
    res = ingest_documents(
        [{"id": "audio:transcript:x", "title": "x", "text": "hello world"}],
        client=c,
    )
    assert res == {"nodes": 1, "edges": 0}
    node = c.nodes.values["audio:transcript:x"]
    assert node["node_type"] == "Document"
    assert node["text"] == "hello world"
    assert node["needs_enrichment"] is True


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
    doc = c.nodes.values["audio:transcript:my-talk"]
    assert doc["node_type"] == "Document"
    assert doc["text"] == "hello world"
    assert doc["transcribedFrom"] == "media:aa"
    assert doc["whisper_model"] == "base"
    # segments typed + linked
    assert c.nodes.values["audio:segment:my-talk:0"]["node_type"] == "TranscriptSegment"
    assert (
        "audio:segment:my-talk:1",
        "audio:transcript:my-talk",
        {"relationship": "segmentOf"},
    ) in c.changes.edges


def test_ingest_transcription_noops_on_empty_text():
    assert ingest_transcription({"text": "   "}, client=_FakeClient()) is None
    assert ingest_transcription({}, client=_FakeClient()) is None


def test_retired_structural_alias_is_rejected():
    with pytest.raises(NativeIngestError, match="canonical node_type"):
        ingest_entities([{"id": "a", "type": "TranscriptSegment"}], client=_FakeClient())


def test_empty_native_ingest_is_rejected():
    with pytest.raises(NativeIngestError, match="at least one entity"):
        ingest_entities([], client=_FakeClient())
