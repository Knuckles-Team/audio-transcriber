"""CONN-AUDIO-TRANSCRIBER-R003: the microphone extra stays declared."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_microphone_extra_is_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extra = project["optional-dependencies"]["microphone"]
    assert any(dep.startswith("pyaudio") for dep in extra)


def test_pr_workflow_skips_microphone_through_sync_args():
    workflow = (ROOT / ".github/workflows/connector-pr.yml").read_text()
    sync_line = next(line for line in workflow.splitlines() if "sync-args:" in line)
    assert "microphone" not in sync_line
