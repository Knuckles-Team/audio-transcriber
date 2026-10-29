"""Capabilities through the SDK (CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION-R001)."""

from __future__ import annotations

import pathlib
import re
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from audio_transcriber.personaplex_client import PersonaPlexClient

ID = "CONN-AUDIO-TRANSCRIBER-SDK-CENTRALIZATION-R001"
ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.spec(f"{ID}.1")
async def test_secure_stream_uses_the_tls_profile() -> None:
    context = ssl.create_default_context()
    profile = MagicMock(ssl_context=context)
    with (
        patch(
            "audio_transcriber.personaplex_client.resolve_tls_profile",
            return_value=profile,
        ) as resolve,
        patch("websockets.connect", new_callable=AsyncMock) as connect,
    ):
        secure = PersonaPlexClient("wss://pp.example:8998")
        await secure.connect()
        resolve.assert_called_once_with("personaplex")
        connect.assert_called_once_with("wss://pp.example:8998", ssl=context)
        await secure.disconnect()
        profile.cleanup.assert_called_once()
        connect.reset_mock()
        plain = PersonaPlexClient("ws://localhost:8998")
        await plain.connect()
        connect.assert_called_once_with("ws://localhost:8998")
        assert resolve.call_count == 1


@pytest.mark.spec(f"{ID}.2")
def test_websockets_comes_from_the_sdk_stream_extra() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    assert "websockets" not in text
    match = re.search(r"scion-sdk\[mcp,stream\]>=(\d+)\.(\d+)\.(\d+)", text)
    assert match and tuple(map(int, match.groups())) >= (0, 3, 11)
