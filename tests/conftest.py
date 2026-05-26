"""Shared test fixtures for Audio Transcriber."""

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("AUDIO_URL", "https://test.example.com")
    monkeypatch.setenv("AUDIO_TOKEN", "test-token-12345")
    monkeypatch.setenv("AUDIO_SSL_VERIFY", "False")
