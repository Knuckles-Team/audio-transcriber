import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from fastmcp import FastMCP, Context

from audio_transcriber.mcp_server import (
    register_audio_processing_tools,
    register_media_sidecar_tools,
    register_prompts,
    get_mcp_instance,
    mcp_server,
)


class MockMCP:
    def __init__(self):
        self.tools = {}
        self.prompts = {}
        self.middlewares = []

    def tool(self, *args, **kwargs):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator

    def prompt(self, func=None, **kwargs):
        if func is not None:
            self.prompts[func.__name__] = func
            return func

        def decorator(f):
            self.prompts[f.__name__] = f
            return f

        return decorator

    def add_middleware(self, mw):
        self.middlewares.append(mw)



def test_get_mcp_instance():
    with patch("audio_transcriber.mcp_server.create_mcp_server") as mock_create:
        mock_args = MagicMock()
        mock_mcp = MagicMock(spec=FastMCP)
        mock_mw = MagicMock()
        mock_create.return_value = (mock_args, mock_mcp, [mock_mw])

        with patch.dict(os.environ, {"AUDIO_PROCESSINGTOOL": "True"}):
            mcp, args, middlewares, tags = get_mcp_instance()
            assert mcp == mock_mcp
            mock_mcp.add_middleware.assert_any_call(mock_mw)

        with patch.dict(os.environ, {"AUDIO_PROCESSINGTOOL": "False"}):
            mcp, args, middlewares, tags = get_mcp_instance()
            assert mcp == mock_mcp


def test_prompts():
    mcp = MockMCP()
    register_prompts(mcp)

    assert "transcribe_file_prompt" in mcp.prompts
    assert "record_and_transcribe_prompt" in mcp.prompts
    assert "translate_audio_prompt" in mcp.prompts

    prompt_file = mcp.prompts["transcribe_file_prompt"]
    assert "test.wav" in prompt_file("test.wav", "en", "base")

    prompt_record = mcp.prompts["record_and_transcribe_prompt"]
    assert "10 seconds" in prompt_record(10, "en", "base")

    prompt_translate = mcp.prompts["translate_audio_prompt"]
    assert "test.wav" in prompt_translate("test.wav", "base")


@pytest.mark.asyncio
async def test_transcribe_audio_tool():
    mcp = MockMCP()
    register_audio_processing_tools(mcp)

    assert "transcribe_audio" in mcp.tools
    tool_fn = mcp.tools["transcribe_audio"]

    # Helper to call tool_fn with defaults filled in (since Pydantic defaults are not applied without FastMCP)
    async def call_tool(
        audio_file=None,
        record_seconds=0,
        directory=".",
        model="base",
        language=None,
        task="transcribe",
        fp16=True,
        word_timestamps=False,
        temperature=0.0,
        initial_prompt=None,
        export_formats=None,
        backend=None,
        ctx=None,
    ):
        return await tool_fn(
            audio_file=audio_file,
            record_seconds=record_seconds,
            directory=directory,
            model=model,
            language=language,
            task=task,
            fp16=fp16,
            word_timestamps=word_timestamps,
            temperature=temperature,
            initial_prompt=initial_prompt,
            export_formats=export_formats,
            backend=backend,
            ctx=ctx,
        )

    # 1. Invalid args (neither audio_file nor positive record_seconds)
    with pytest.raises(RuntimeError) as exc_info:
        await call_tool(audio_file=None, record_seconds=0)
    assert (
        "Either audio_file must be provided or record_seconds must be positive"
        in str(exc_info.value)
    )

    # 2. File not found
    mock_transcriber_missing = MagicMock()
    with (
        patch(
            "audio_transcriber.mcp_server.AudioTranscriber",
            return_value=mock_transcriber_missing,
        ),
        patch("pathlib.Path.exists", return_value=False),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            await call_tool(audio_file="nonexistent.wav", record_seconds=0)
        assert "Audio file not found" in str(exc_info.value)

    # 3. Successful file transcription with mock AudioTranscriber
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = {"text": "hello world"}

    with (
        patch(
            "audio_transcriber.mcp_server.AudioTranscriber",
            return_value=mock_transcriber,
        ) as mock_trans_cls,
        patch("pathlib.Path.exists", return_value=True),
    ):
        ctx = AsyncMock(spec=Context)
        res = await call_tool(
            audio_file="test.wav", record_seconds=0, export_formats=["txt"], ctx=ctx
        )

        assert res == "hello world"
        mock_transcriber.transcribe.assert_called_once()
        mock_transcriber.export.assert_called_once_with(
            {"text": "hello world"}, formats=["txt"]
        )
        assert ctx.report_progress.call_count >= 1

    # 4. Successful recording transcription with mock AudioTranscriber
    mock_transcriber_rec = MagicMock()
    mock_transcriber_rec.transcribe.return_value = {"text": "hello recording"}

    with patch(
        "audio_transcriber.mcp_server.AudioTranscriber",
        return_value=mock_transcriber_rec,
    ):
        ctx = AsyncMock(spec=Context)
        res = await call_tool(audio_file=None, record_seconds=5, ctx=ctx)

        assert res == "hello recording"
        mock_transcriber_rec.initiate_stream.assert_called_once()
        mock_transcriber_rec.record.assert_called_once_with(seconds=5)
        mock_transcriber_rec.stop_stream.assert_called_once()
        mock_transcriber_rec.save_stream.assert_called_once()


@pytest.mark.asyncio
async def test_transcribe_media_tool():
    """BUG-271: the governed sidecar contract's ``transcribe_media`` action —
    action-routed (``action`` + ``params_json``) with ``digest``/``media_type``/
    ``artifact_b64``, matching ``agent_utilities.media.sidecar_contract``'s
    ``SIDECAR_CAPABILITIES['audio']`` manifest entry exactly."""
    import base64
    import hashlib
    import json

    mcp = MockMCP()
    register_media_sidecar_tools(mcp)

    assert "transcribe_media" in mcp.tools
    tool_fn = mcp.tools["transcribe_media"]

    # Unsupported action: never fabricates, distinct error.
    res = await tool_fn(action="not_a_real_action", params_json="{}", ctx=None)
    assert res == {"available": False, "error": "unsupported action: 'not_a_real_action'"}

    # Malformed params_json: distinct failure, not a crash.
    res = await tool_fn(action="transcribe_segments", params_json="not json", ctx=None)
    assert res["available"] is False

    # Missing required fields: distinct failure.
    res = await tool_fn(action="transcribe_segments", params_json="{}", ctx=None)
    assert res == {
        "available": False,
        "error": "missing required 'digest' or 'artifact_b64'",
    }

    raw = b"not-really-audio-but-bytes"
    good_digest = hashlib.sha256(raw).hexdigest()
    b64 = base64.b64encode(raw).decode("ascii")

    # Digest mismatch: verified BEFORE transcribing, never silently transcribed anyway.
    res = await tool_fn(
        action="transcribe_segments",
        params_json=json.dumps(
            {"digest": "sha256:deadbeef", "media_type": "audio/wav", "artifact_b64": b64}
        ),
        ctx=None,
    )
    assert res == {"available": False, "error": "digest mismatch"}

    # Successful path: delegates to the SAME AudioTranscriber/Whisper backend
    # transcribe_audio uses (no second transcription implementation).
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "hello"},
            {"start": 1.5, "end": 3.0, "text": "world"},
        ],
        "language": "en",
        "duration": 3.0,
    }
    with (
        patch(
            "audio_transcriber.mcp_server.AudioTranscriber",
            return_value=mock_transcriber,
        ) as mock_cls,
        patch("pathlib.Path.write_bytes"),
    ):
        res = await tool_fn(
            action="transcribe_segments",
            params_json=json.dumps(
                {
                    "digest": f"sha256:{good_digest}",
                    "media_type": "audio/wav",
                    "artifact_b64": b64,
                }
            ),
            ctx=None,
        )
    assert res["available"] is True
    assert res["segments"] == [
        {"start_ms": 0, "end_ms": 1500, "text": "hello"},
        {"start_ms": 1500, "end_ms": 3000, "text": "world"},
    ]
    assert res["language"] == "en"
    assert res["duration"] == 3.0
    # ingest_to_kg=False: a dictation/sidecar clip never becomes durable KG evidence.
    assert mock_cls.call_args.kwargs["ingest_to_kg"] is False


def test_mcp_server_main():
    for transport in ["stdio", "streamable-http", "sse"]:
        mock_args = MagicMock()
        mock_args.transport = transport
        mock_args.host = "localhost"
        mock_args.port = 8000
        mock_args.auth_type = "none"

        mock_mcp = MagicMock(spec=FastMCP)

        with patch(
            "audio_transcriber.mcp_server.get_mcp_instance",
            return_value=(mock_mcp, mock_args, [], []),
        ):
            mcp_server()
            mock_mcp.run.assert_called_once()

    # Invalid transport
    mock_args = MagicMock()
    mock_args.transport = "invalid"
    mock_args.auth_type = "none"
    mock_mcp = MagicMock(spec=FastMCP)
    with (
        patch(
            "audio_transcriber.mcp_server.get_mcp_instance",
            return_value=(mock_mcp, mock_args, [], []),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        mcp_server()
    assert exc_info.value.code == 1
