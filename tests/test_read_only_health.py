"""The read-only engine health tool (CONN-AUDIO-TRANSCRIBER-R001)."""

import asyncio

import pytest

import importlib

mcp_server = importlib.import_module("audio_transcriber.mcp_server")

pytestmark = pytest.mark.spec("CONN-AUDIO-TRANSCRIBER-R001")


def test_health_check_is_listed_read_only(monkeypatch):
    monkeypatch.setattr("sys.argv", ["audio-transcriber-mcp"])
    mcp, *_ = mcp_server.get_mcp_instance()
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    annotations = tools["health_check"].annotations
    assert annotations.read_only_hint is True
    assert annotations.destructive_hint is False
    assert sum(1 for t in tools.values() if t.annotations and t.annotations.read_only_hint) >= 1


def test_health_reports_engine_state_without_loading_a_model(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_server, "DEFAULT_TRANSCRIBE_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(
        mcp_server, "AudioTranscriber", lambda *_args, **_kwargs: pytest.fail("model loaded")
    )
    health = mcp_server.engine_health()
    assert health["version"] == mcp_server.__version__
    assert health["transcribe_directory_exists"] is True
    assert set(health) >= {"status", "backends", "ffmpeg", "default_model"}


def test_health_is_degraded_without_a_backend(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    health = mcp_server.engine_health()
    assert health["status"] == "DEGRADED"
    assert health["backends"] == []
