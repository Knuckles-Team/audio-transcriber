"""Transcript list and get tools (CONN-AUDIO-TRANSCRIBER-R005)."""

import asyncio
import importlib
import json
import os

import pytest
from fastmcp import Client
from scion_sdk.connector import build_server

mcp_server = importlib.import_module("audio_transcriber.mcp_server")

pytestmark = pytest.mark.spec("CONN-AUDIO-TRANSCRIBER-R005")

DAY = 86400
BASE = 1_767_225_600  # 2026-01-01T00:00:00Z


def _write(directory, name, texts, days):
    path = directory / f"{name}.json"
    segments = [
        {
            "id": i,
            "start": float(i),
            "end": i + 1.0,
            "text": t,
            "no_speech_prob": 0.1 * i,
        }
        for i, t in enumerate(texts)
    ]
    path.write_text(json.dumps({"text": " ".join(texts), "segments": segments}))
    os.utime(path, (BASE + days * DAY, BASE + days * DAY))


def _server(monkeypatch, directory):
    monkeypatch.setattr(mcp_server, "DEFAULT_TRANSCRIBE_DIRECTORY", str(directory))
    monkeypatch.setattr("sys.argv", ["audio-transcriber-mcp"])
    return build_server("audio_transcriber", command_args=[])[1]


async def _call(mcp, name, args, raises=False):
    async with Client(mcp) as client:
        return await client.call_tool(name, args, raise_on_error=raises)


def test_list_and_get_by_recording_id_and_date(monkeypatch, tmp_path):
    _write(tmp_path, "b", ["two", "parts"], 1)
    _write(tmp_path, "a", ["alpha"], 1)
    _write(tmp_path, "c", ["old"], 0)
    mcp = _server(monkeypatch, tmp_path)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("list_transcripts", "get_transcript"):
        ann = tools[name].annotations
        assert ann.read_only_hint is True and ann.destructive_hint is False
        assert ann.idempotent_hint is True and ann.open_world_hint is False

    full = asyncio.run(_call(mcp, "list_transcripts", {})).data
    assert [i["id"] for i in full["items"]] == ["a", "b", "c"]
    assert full["items"][1]["text"] == "two parts"
    assert full["items"][0]["date"] == "2026-01-02"
    assert full["truncated"] is False

    by_id = asyncio.run(_call(mcp, "list_transcripts", {"recording_id": "c"})).data
    assert [i["id"] for i in by_id["items"]] == ["c"]
    ranged = asyncio.run(
        _call(
            mcp,
            "list_transcripts",
            {"date_from": "2026-01-02", "date_to": "2026-01-02"},
        )
    ).data
    assert [i["id"] for i in ranged["items"]] == ["a", "b"]
    limited = asyncio.run(_call(mcp, "list_transcripts", {"limit": 2})).data
    assert [i["id"] for i in limited["items"]] == ["a", "b"] and limited[
        "truncated"
    ] is True

    got = asyncio.run(_call(mcp, "get_transcript", {"recording_id": "b"})).data
    assert (
        got["id"] == "b" and got["date"] == "2026-01-02" and got["text"] == "two parts"
    )
    assert got["segments"][1] == {"start": 1.0, "end": 2.0, "no_speech_prob": 0.1}


def test_refusals_write_nothing(monkeypatch, tmp_path):
    _write(tmp_path, "a", ["x"], 0)
    (tmp_path / "bad.json").write_text("not json")
    before = sorted((p.name, p.read_bytes()) for p in tmp_path.iterdir())
    mcp = _server(monkeypatch, tmp_path)
    cases = [
        ("list_transcripts", {"recording_id": ""}, "INVALID_RECORDING_ID"),
        ("get_transcript", {"recording_id": "a/b"}, "INVALID_RECORDING_ID"),
        ("get_transcript", {"recording_id": ".."}, "INVALID_RECORDING_ID"),
        ("list_transcripts", {"date_from": "2026-13-01"}, "INVALID_DATE_RANGE"),
        (
            "list_transcripts",
            {"date_from": "2026-02-01", "date_to": "2026-01-01"},
            "INVALID_DATE_RANGE",
        ),
        ("list_transcripts", {"limit": 0}, "LIMIT_OUT_OF_RANGE"),
        ("list_transcripts", {"limit": 501}, "LIMIT_OUT_OF_RANGE"),
        ("get_transcript", {"recording_id": "missing"}, "TRANSCRIPT_NOT_FOUND"),
        ("get_transcript", {"recording_id": "bad"}, "TRANSCRIPT_UNREADABLE"),
    ]
    for tool, args, code in cases:
        with pytest.raises(Exception, match=code):
            asyncio.run(_call(mcp, tool, args, raises=True))
    assert sorted((p.name, p.read_bytes()) for p in tmp_path.iterdir()) == before
