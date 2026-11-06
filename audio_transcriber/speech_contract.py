"""The typed speech-transcription contract (TRANSCRIBE-R002).

One bounded audio payload in, one typed :class:`SpeechTranscript` out. Callers
(for example a front door that turns a voice note into a prompt) discover this
tool by its contract: the input fields ``artifact_b64``, ``digest`` and
``media_type`` plus the ``speech-to-text`` capability line in its description.

The result always carries a ``status``. ``ok`` carries the transcript with
detected language, segment timestamps and (when asked) word timestamps.
Anything else is a typed refusal with a ``refusal_code``; the tool never
fabricates a transcript and never raises for a caller error.

Format conversion: when the ``ffmpeg`` binary is present (the runtime image
installs it), any input that is not already 16 kHz mono PCM WAV is converted
to that form before inference, so container formats such as Ogg/Opus voice
notes, WebM, M4A and MP3 transcribe the same way on every backend.
Diarisation: no diarisation backend ships with this connector, so a request
for it returns ``diarization.supported = false`` with the reason, and the
segments carry no speaker labels. It is reported, never faked.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import mimetypes
import shutil
import subprocess  # nosec B404 - fixed argv, no shell, used only for ffmpeg
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "speech-transcript/1"

#: The capability marker callers match on (see the module docstring).
CAPABILITY = "speech-to-text"

RefusalCode = Literal[
    "INVALID_ARGUMENT",
    "DIGEST_MISMATCH",
    "MEDIA_TOO_LARGE",
    "MEDIA_TOO_LONG",
    "UNSUPPORTED_MEDIA",
    "TRANSCRIPTION_FAILED",
]

DEFAULT_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_DURATION_S = 600.0


class Word(BaseModel):
    word: str
    start_ms: int
    end_ms: int
    probability: float | None = None


class Segment(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)


class Diarization(BaseModel):
    requested: bool = False
    supported: bool = False
    reason: str = "no diarisation backend is installed in this connector"


class SpeechTranscript(BaseModel):
    """The one result shape of ``transcribe_speech``."""

    schema_version: str = SCHEMA_VERSION
    status: Literal["ok", "refused"]
    refusal_code: RefusalCode | None = None
    refusal_detail: str = ""
    source_digest: str = ""
    media_type: str = ""
    text: str = ""
    language: str | None = None
    language_probability: float | None = None
    duration_s: float | None = None
    segments: list[Segment] = Field(default_factory=list)
    diarization: Diarization = Field(default_factory=Diarization)
    converted: bool = False
    model: str = ""
    backend: str = ""


def _refuse(code: RefusalCode, detail: str, **fields: Any) -> SpeechTranscript:
    return SpeechTranscript(
        status="refused", refusal_code=code, refusal_detail=detail, **fields
    )


def _ms(value: Any) -> int | None:
    try:
        return int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return None


def _is_pcm16_mono_wav(raw: bytes) -> bool:
    """A canonical RIFF/WAVE PCM header at 16 kHz, mono."""
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return False
    channels = int.from_bytes(raw[22:24], "little")
    rate = int.from_bytes(raw[24:28], "little")
    return channels == 1 and rate == 16_000


def convert_to_wav(source: Path, target: Path, ffmpeg: str | None = None) -> bool:
    """Convert ``source`` to 16 kHz mono PCM WAV with ffmpeg; True when it ran.

    Returns False (no conversion) when the ffmpeg binary is absent, so the
    caller falls back to the backend's own decoder.
    """
    binary = ffmpeg or shutil.which("ffmpeg")
    if not binary:
        return False
    completed = subprocess.run(  # nosec B603 - fixed argv, no shell
        [
            binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(target),
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("ffmpeg could not decode the media")
    return True


def _segments(result: dict[str, Any]) -> list[Segment]:
    out: list[Segment] = []
    for raw in result.get("segments") or []:
        start, end = _ms(raw.get("start")), _ms(raw.get("end"))
        if start is None or end is None:
            continue
        words = []
        for w in raw.get("words") or []:
            ws, we = _ms(w.get("start")), _ms(w.get("end"))
            if ws is None or we is None:
                continue
            words.append(
                Word(
                    word=str(w.get("word", "")),
                    start_ms=ws,
                    end_ms=we,
                    probability=w.get("probability"),
                )
            )
        out.append(
            Segment(
                start_ms=start, end_ms=end, text=str(raw.get("text", "")), words=words
            )
        )
    return out


def transcribe_speech(
    *,
    artifact_b64: str,
    digest: str,
    media_type: str,
    transcribe: Callable[..., dict[str, Any]],
    language: str | None = None,
    task: str = "transcribe",
    word_timestamps: bool = False,
    diarize: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
    model: str = "",
    backend: str = "",
    ffmpeg: str | None = None,
    scratch_root: str = "/var/tmp",  # nosec B108 - mkdtemp below, private 0700 dir
) -> SpeechTranscript:
    """Validate, convert and transcribe one payload; return the typed result.

    ``transcribe(path, language=..., task=..., word_timestamps=...)`` is the
    backend call (the connector passes ``AudioTranscriber.transcribe`` bound
    to a file); it returns the backend's dict with ``text``, ``segments``,
    ``language``, ``language_probability`` and ``duration``.
    """
    media_type = (media_type or "").split(";")[0].strip().lower()
    expected = digest.split(":", 1)[-1].strip().lower() if digest else ""
    common = {"source_digest": expected, "media_type": media_type}
    diarization = Diarization(requested=diarize)
    if task not in ("transcribe", "translate"):
        return _refuse(
            "INVALID_ARGUMENT", "task must be transcribe or translate", **common
        )
    if not media_type.startswith(("audio/", "video/")):
        return _refuse(
            "UNSUPPORTED_MEDIA", "media_type must be audio/* or video/*", **common
        )
    if not expected or not artifact_b64:
        return _refuse(
            "INVALID_ARGUMENT", "digest and artifact_b64 are required", **common
        )
    # Bound the encoded size before decoding (4 b64 chars carry 3 bytes).
    if len(artifact_b64) > (max_bytes * 4) // 3 + 4:
        return _refuse(
            "MEDIA_TOO_LARGE", f"payload exceeds {max_bytes} bytes", **common
        )
    try:
        raw = base64.b64decode(artifact_b64, validate=True)
    except (binascii.Error, ValueError):
        return _refuse("INVALID_ARGUMENT", "artifact_b64 is not valid base64", **common)
    if len(raw) > max_bytes:
        return _refuse(
            "MEDIA_TOO_LARGE", f"payload exceeds {max_bytes} bytes", **common
        )
    if hashlib.sha256(raw).hexdigest() != expected:
        return _refuse(
            "DIGEST_MISMATCH", "sha256 of the payload does not match", **common
        )

    scratch = Path(
        tempfile.mkdtemp(prefix="audio-transcriber-speech-", dir=scratch_root)
    )
    ext = mimetypes.guess_extension(media_type) or ".bin"
    source = scratch / f"source{ext}"
    converted = False
    try:
        source.write_bytes(raw)
        path = source
        if not _is_pcm16_mono_wav(raw):
            target = scratch / "normalized.wav"
            try:
                converted = convert_to_wav(source, target, ffmpeg)
            except (ValueError, subprocess.TimeoutExpired, OSError):
                return _refuse(
                    "UNSUPPORTED_MEDIA", "the media could not be decoded", **common
                )
            if converted:
                path = target
        try:
            result = transcribe(
                path, language=language, task=task, word_timestamps=word_timestamps
            )
        except Exception as exc:  # noqa: BLE001 - typed refusal, never a fake transcript
            return _refuse(
                "TRANSCRIPTION_FAILED",
                f"{type(exc).__name__}: transcription failed",
                **common,
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    duration = result.get("duration")
    duration_s = float(duration) if isinstance(duration, (int, float)) else None
    if duration_s is not None and duration_s > max_duration_s:
        return _refuse(
            "MEDIA_TOO_LONG", f"audio exceeds {max_duration_s:g} seconds", **common
        )
    probability = result.get("language_probability")
    return SpeechTranscript(
        status="ok",
        text=str(result.get("text") or "").strip(),
        language=result.get("language"),
        language_probability=float(probability)
        if isinstance(probability, (int, float))
        else None,
        duration_s=duration_s,
        segments=_segments(result),
        diarization=diarization,
        converted=converted,
        model=model,
        backend=backend,
        **common,
    )
