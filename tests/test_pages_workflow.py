"""Keep the spec dashboard wired into the canonical Pages publisher."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_pages_workflow_publishes_the_spec_dashboard() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/pages.yml").read_text())
    events = workflow[True]
    assert events["push"]["branches"] == ["main"]
    assert {
        "docs/**",
        "mkdocs.yml",
        "README.md",
        "specs/**",
        ".github/spec-dashboard.json",
        ".github/workflows/pages.yml",
    } <= set(events["push"]["paths"])
    assert "workflow_dispatch" in events

    publisher = workflow["jobs"]["publish-pages"]
    assert publisher["uses"] == (
        "Knuckles-Team/pipelines/.github/workflows/pages_pipeline.yml@main"
    )
    assert publisher["with"] == {
        "content_source": "docs",
        "spec_dashboard_enabled": True,
        "spec_dashboard_config": ".github/spec-dashboard.json",
    }
    assert publisher["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }


def test_spec_dashboard_sources_generated_status_files() -> None:
    config = json.loads((ROOT / ".github/spec-dashboard.json").read_text())
    assert config == {
        "version": 1,
        "sources": [{"glob": "specs/*/status.json", "kind": "spec"}],
    }
