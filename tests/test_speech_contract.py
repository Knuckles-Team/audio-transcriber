"""TRANSCRIBE-R002: the typed speech-transcription contract."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import shutil
import wave
from pathlib import Path

import pytest

from audio_transcriber import speech_contract
from audio_transcriber.speech_contract import SpeechTranscript, transcribe_speech


def _wav(rate: int = 16_000, seconds: float = 0.2, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * channels * int(rate * seconds))
    return buf.getvalue()


def _payload(raw: bytes) -> dict:
    return {
        "artifact_b64": base64.b64encode(raw).decode(),
        "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


class _Backend:
    def __init__(self, duration: float = 1.5) -> None:
        self.calls: list[tuple[Path, dict]] = []
        self.duration = duration

    def __call__(self, path: Path, **kwargs):
        self.calls.append((path, kwargs, path.read_bytes()[:4]))
        return {
            "text": " turn on the lights ",
            "language": "en",
            "language_probability": 0.97,
            "duration": self.duration,
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": " turn on the lights",
                    "words": [
                        {"word": "turn", "start": 0.0, "end": 0.3, "probability": 0.9}
                    ],
                }
            ],
        }


def test_typed_result_carries_language_timestamps_and_diarisation_report():
    backend = _Backend()
    result = transcribe_speech(
        **_payload(_wav()),
        media_type="audio/wav",
        transcribe=backend,
        word_timestamps=True,
        diarize=True,
        model="base",
    )
    assert isinstance(result, SpeechTranscript)
    assert result.status == "ok" and result.refusal_code is None
    assert result.text == "turn on the lights"
    assert result.language == "en" and result.language_probability == pytest.approx(
        0.97
    )
    assert result.segments[0].start_ms == 0 and result.segments[0].end_ms == 1200
    assert result.segments[0].words[0].end_ms == 300
    assert result.diarization.requested and not result.diarization.supported
    assert result.segments[0].speaker is None
    assert result.schema_version == "speech-transcript/1"
    assert backend.calls[0][1]["word_timestamps"] is True
    # Canonical 16 kHz mono WAV needs no conversion.
    assert result.converted is False


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda p: {**p, "digest": "sha256:" + "0" * 64}, "DIGEST_MISMATCH"),
        (lambda p: {**p, "artifact_b64": "!!notb64"}, "INVALID_ARGUMENT"),
        (lambda p: {**p, "digest": ""}, "INVALID_ARGUMENT"),
        (lambda p: {**p, "media_type": "image/png"}, "UNSUPPORTED_MEDIA"),
    ],
)
def test_caller_errors_are_typed_refusals_with_zero_backend_calls(mutate, code):
    backend = _Backend()
    args = mutate({**_payload(_wav()), "media_type": "audio/wav"})
    result = transcribe_speech(**args, transcribe=backend)
    assert result.status == "refused" and result.refusal_code == code
    assert backend.calls == []


def test_size_and_duration_limits_refuse():
    backend = _Backend(duration=900.0)
    raw = _wav(seconds=1.0)
    too_big = transcribe_speech(
        **_payload(raw), media_type="audio/wav", transcribe=backend, max_bytes=1000
    )
    assert too_big.refusal_code == "MEDIA_TOO_LARGE" and backend.calls == []
    too_long = transcribe_speech(
        **_payload(raw), media_type="audio/wav", transcribe=backend, max_duration_s=60
    )
    assert too_long.refusal_code == "MEDIA_TOO_LONG" and too_long.text == ""


def test_backend_failure_never_fabricates_text():
    def broken(path, **kwargs):
        raise RuntimeError("model missing")

    result = transcribe_speech(
        **_payload(_wav()), media_type="audio/wav", transcribe=broken
    )
    assert result.refusal_code == "TRANSCRIPTION_FAILED" and result.text == ""


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg binary not installed"
)
def test_non_canonical_audio_is_converted_with_ffmpeg():
    backend = _Backend()
    result = transcribe_speech(
        **_payload(_wav(rate=44_100, channels=2)),
        media_type="audio/wav",
        transcribe=backend,
    )
    assert result.status == "ok" and result.converted is True
    path, _, head = backend.calls[0]
    assert path.name == "normalized.wav" and head == b"RIFF"


def test_undecodable_media_without_ffmpeg_falls_through_and_with_ffmpeg_refuses(
    monkeypatch,
):
    backend = _Backend()
    garbage = b"not really audio" * 10
    # No ffmpeg: the backend's own decoder gets the original file.
    monkeypatch.setattr(speech_contract.shutil, "which", lambda name: None)
    result = transcribe_speech(
        **_payload(garbage), media_type="audio/ogg", transcribe=backend
    )
    assert result.status == "ok" and result.converted is False
    assert backend.calls[0][0].suffix in {".ogg", ".oga", ".bin"}


def test_transcribe_speech_tool_is_registered_typed_and_returns_the_contract(
    monkeypatch,
):
    from audio_transcriber import mcp_server

    tools: dict = {}

    class _MCP:
        def tool(self, *args, **kwargs):
            def deco(fn):
                tools[kwargs.get("name", fn.__name__)] = (fn, kwargs)
                return fn

            return deco

    mcp_server.register_speech_tools(_MCP())
    fn, meta = tools["transcribe_speech"]
    assert (
        "speech-to-text" in meta["tags"]
        and "Capability: speech-to-text" in meta["description"]
    )

    class _Fake:
        def __init__(self, **kwargs):
            self.backend_instance = object()

        def transcribe(self, **kwargs):
            return _Backend()(Path(__file__), **kwargs)

    monkeypatch.setattr(mcp_server, "AudioTranscriber", _Fake)
    raw = _wav()
    out = asyncio.run(
        fn(
            **_payload(raw),
            media_type="audio/wav",
            language=None,
            task="transcribe",
            word_timestamps=False,
            diarize=False,
        )
    )
    assert out["status"] == "ok" and out["text"] == "turn on the lights"
    assert out["source_digest"] == hashlib.sha256(raw).hexdigest()
    assert out["backend"] == "object"


def test_device_probe_needs_no_torch(monkeypatch):
    import builtins

    from audio_transcriber import audio_transcriber as at

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert at._ctranslate2_device() in {"cpu", "cuda"}


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg binary not installed"
)
def test_faster_whisper_gets_ffmpeg_decoded_samples(tmp_path):
    import numpy as np

    from audio_transcriber import audio_transcriber as at

    path = tmp_path / "clip.wav"
    path.write_bytes(_wav(rate=44_100, seconds=0.5, channels=2))
    samples = at._decode_with_ffmpeg(path)
    assert isinstance(samples, np.ndarray) and samples.dtype == np.float32
    assert abs(len(samples) - 8_000) <= 200
