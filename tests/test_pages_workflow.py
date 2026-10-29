"""Keep the spec dashboard wired into the canonical Pages publisher."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_spec_dashboard_sources_generated_status_files() -> None:
    config = json.loads((ROOT / ".github/spec-dashboard.json").read_text())
    assert config == {
        "version": 1,
        "sources": [{"glob": "specs/*/status.json", "kind": "spec"}],
    }
