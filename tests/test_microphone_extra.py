"""CONN-AUDIO-TRANSCRIBER-R003: the microphone extra stays declared."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_microphone_extra_is_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extra = project["optional-dependencies"]["microphone"]
    assert any(dep.startswith("pyaudio") for dep in extra)
