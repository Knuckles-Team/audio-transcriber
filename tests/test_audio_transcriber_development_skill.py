"""PIPE-CONNSPEC-R107.4: Audio Transcriber rapid-delivery skill contract."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "audio_transcriber/skills/audio-transcriber-development/SKILL.md"

pytestmark = pytest.mark.spec("PIPE-CONNSPEC-R107.4")


def test_development_skill_binds_audio_contract_and_package_data() -> None:
    text = SKILL.read_text()
    assert text.startswith("---\nname: audio-transcriber-development\n")
    assert "## Rapid delivery: the 5-minute contract" in text
    assert "PIPE-CONNSPEC-R107.4" in text
    for reference in (
        "audio_transcriber/mcp_server.py",
        "audio_transcriber/audio_transcriber.py",
        "audio_transcriber/connector_sources.py",
        "WHISPER_MODEL",
        "TRANSCRIBE_DIRECTORY",
        "render_connector: true",
    ):
        assert reference in text

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "skills/**" in data["tool"]["setuptools"]["package-data"]["audio_transcriber"]
    assert SKILL.is_file()
    assert not re.search(r"(?:sk-|ghp_|Bearer\s+)[A-Za-z0-9_./-]{12,}", text)
    assert data["project"]["name"] == "audio-transcriber"
