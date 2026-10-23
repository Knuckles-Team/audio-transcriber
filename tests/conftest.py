"""Shared test fixtures for Audio Transcriber."""

import sys
import types
from typing import Any

import pytest

try:
    import pyaudio  # noqa: F401
except ModuleNotFoundError:
    # pyaudio compiles against PortAudio and is not a package dependency; the
    # tests mock every call, so a stand-in module is enough.
    _stub: Any = types.ModuleType("pyaudio")
    _stub.paInt16 = 8
    _stub.paContinue = 0
    _stub.paComplete = 1

    class _PyAudio:
        def get_default_input_device_info(self):
            return {"index": 0}

        def get_sample_size(self, fmt):
            return 2

        def open(self, *args, **kwargs):
            raise OSError("no audio device")

        def terminate(self):
            return None

    _stub.PyAudio = _PyAudio
    sys.modules["pyaudio"] = _stub


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("AUDIO_URL", "https://test.example.com")
    monkeypatch.setenv("AUDIO_TOKEN", "test-token-12345")
    monkeypatch.setenv("AUDIO_SSL_VERIFY", "False")
