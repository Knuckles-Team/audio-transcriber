import sys
from unittest.mock import MagicMock, patch
import pytest

from audio_transcriber.agent_server import agent_server


def test_agent_server_runs():
    """Verify that agent server starts up correctly.

    Traces the ecosystem server setup:
    - CONCEPT:OS-5.4
    - CONCEPT:OS-5.1
    - CONCEPT:OS-5.3
    - CONCEPT:ORCH-1.4
    - CONCEPT:OS-5.2
    """
    mock_parser = MagicMock()

    mock_args = MagicMock()
    mock_args.debug = True
    mock_args.mcp_url = "http://localhost:8000"
    mock_args.mcp_config = "mcp_config.json"
    mock_args.host = "localhost"
    mock_args.port = 8000
    mock_args.provider = "openai"
    mock_args.model_id = "gpt-4"
    mock_args.base_url = "http://localhost:8000"
    mock_args.api_key = "key"
    mock_args.custom_skills_directory = None
    mock_args.web = True
    mock_args.otel = False
    mock_args.otel_endpoint = None
    mock_args.otel_headers = None
    mock_args.otel_public_key = None
    mock_args.otel_secret_key = None
    mock_args.otel_protocol = None
    mock_parser.parse_args.return_value = mock_args

    mock_identity = {
        "name": "Mock Transcriber",
        "description": "Mock Description",
        "content": "mock content",
    }

    with (
        patch("agent_utilities.initialize_workspace") as mock_init_ws,
        patch(
            "agent_utilities.load_identity", return_value=mock_identity
        ) as mock_load_id,
        patch(
            "agent_utilities.create_agent_parser", return_value=mock_parser
        ) as mock_create_parser,
        patch("agent_utilities.create_agent_server") as mock_create_server,
    ):
        agent_server()

        mock_init_ws.assert_called_once()
        mock_load_id.assert_called_once()
        mock_create_parser.assert_called_once()
        mock_create_server.assert_called_once()

        call_kwargs = mock_create_server.call_args[1]
        assert call_kwargs["mcp_url"] == "http://localhost:8000"
        assert call_kwargs["model_id"] == "gpt-4"
        assert call_kwargs["debug"] is True
