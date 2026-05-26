import base64
import json
import logging
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
import pyaudio

from audio_transcriber.audio_transcriber import AudioTranscriber
from audio_transcriber.personaplex_client import PersonaPlexClient


@pytest.mark.asyncio
async def test_personaplex_client_init():
    client = PersonaPlexClient("ws://localhost:8998")
    assert client.uri == "ws://localhost:8998"
    assert isinstance(client.logger, logging.Logger)

    custom_logger = logging.getLogger("custom")
    client2 = PersonaPlexClient("ws://localhost:8998", custom_logger)
    assert client2.logger == custom_logger


@pytest.mark.asyncio
async def test_personaplex_client_connect_success():
    client = PersonaPlexClient("ws://localhost:8998")
    mock_websocket = MagicMock()
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_websocket
        await client.connect()
        mock_connect.assert_called_once_with("ws://localhost:8998")
        assert client.websocket == mock_websocket


@pytest.mark.asyncio
async def test_personaplex_client_connect_failure():
    client = PersonaPlexClient("ws://localhost:8998")
    with patch("websockets.connect", side_effect=Exception("Connection refused")):
        with pytest.raises(Exception) as exc:
            await client.connect()
        assert "Connection refused" in str(exc.value)


@pytest.mark.asyncio
async def test_personaplex_client_disconnect():
    client = PersonaPlexClient("ws://localhost:8998")

    # 1. Without active websocket
    await client.disconnect()

    # 2. With active websocket
    mock_websocket = AsyncMock()
    client.websocket = mock_websocket
    await client.disconnect()
    mock_websocket.close.assert_called_once()


@pytest.mark.asyncio
async def test_personaplex_client_send_audio():
    client = PersonaPlexClient("ws://localhost:8998")

    # 1. Not connected raises RuntimeError
    with pytest.raises(RuntimeError) as exc:
        await client.send_audio(b"audio_bytes")
    assert "Not connected" in str(exc.value)

    # 2. Connected successfully sends
    mock_websocket = AsyncMock()
    client.websocket = mock_websocket
    await client.send_audio(b"audio_bytes")
    mock_websocket.send.assert_called_once_with(b"audio_bytes")

    # 3. Connection error handling
    mock_websocket.send.side_effect = Exception("Send failed")
    # Should not raise exception, but log it
    await client.send_audio(b"audio_bytes")


@pytest.mark.asyncio
async def test_personaplex_client_receive_audio():
    client = PersonaPlexClient("ws://localhost:8998")

    # 1. Not connected raises RuntimeError
    with pytest.raises(RuntimeError):
        async for _ in client.receive_audio():
            pass

    # 2. Receive message types
    mock_websocket = AsyncMock()

    async def mock_iter(*args):
        # Bytes message
        yield b"chunk1"
        # JSON message
        yield json.dumps({"audio": base64.b64encode(b"chunk2").decode("utf-8")})
        # Invalid JSON text message
        yield "invalid_json"
        # JSON without audio field
        yield json.dumps({"other": "field"})
        # Yield exception
        raise Exception("Websocket closed abnormally")

    mock_websocket.__aiter__ = mock_iter
    client.websocket = mock_websocket

    chunks = []
    async for chunk in client.receive_audio():
        chunks.append(chunk)

    assert chunks == [b"chunk1", b"chunk2"]


@pytest.mark.asyncio
async def test_personaplex_client_stream_audio():
    client = PersonaPlexClient("ws://localhost:8998")
    mock_websocket = AsyncMock()
    client.websocket = mock_websocket

    async def audio_iter():
        yield b"chunk1"
        yield b"chunk2"

    await client.stream_audio(audio_iter())
    assert mock_websocket.send.call_count == 2


@pytest.mark.asyncio
async def test_audio_transcriber_interact():
    with patch("pyaudio.PyAudio") as mock_pa:
        pa_instance = mock_pa.return_value
        pa_instance.get_default_input_device_info.return_value = {"index": 0}

        mock_input_stream = MagicMock()
        mock_output_stream = MagicMock()

        def open_side_effect(**kwargs):
            if kwargs.get("input"):
                return mock_input_stream
            return mock_output_stream

        pa_instance.open.side_effect = open_side_effect

        with patch(
            "audio_transcriber.audio_transcriber.OpenAIWhisperBackend.load_model"
        ):
            transcriber = AudioTranscriber(backend="openai-whisper")
        transcriber.device_index = 0

        mock_client_instance = AsyncMock()

        # We simulate receive_audio yielding one chunk and then stopping
        async def mock_recv():
            yield b"chunk_response"
            while not transcriber.stop:
                await asyncio.sleep(0.01)

        mock_client_instance.receive_audio = mock_recv

        with patch(
            "audio_transcriber.personaplex_client.PersonaPlexClient",
            return_value=mock_client_instance,
        ):
            # Capture callback passed to open
            with patch.object(
                transcriber.pyaudio_instance, "open", side_effect=open_side_effect
            ) as mock_open:
                # We want to feed one chunk to input_callback so send_audio_loop gets it
                async def run_interact():
                    await transcriber.interact("ws://localhost:8998")

                # Run interact task in background
                task = asyncio.create_task(run_interact())
                await asyncio.sleep(0.05)

                # Check that open was called with a callback
                callback = None
                for call in mock_open.call_args_list:
                    if call[1].get("input"):
                        callback = call[1].get("stream_callback")

                assert callback is not None

                # Trigger callback
                res_cb = callback(b"input_audio", 1024, None, None)
                assert res_cb == (None, pyaudio.paContinue)

                # Stop interaction
                transcriber.stop = True

                # Wait for interaction loop to finish
                await task

                # Callback after stop should return paComplete
                res_cb_stop = callback(b"input_audio", 1024, None, None)
                assert res_cb_stop == (None, pyaudio.paComplete)

                mock_input_stream.start_stream.assert_called_once()
                mock_output_stream.start_stream.assert_called_once()
                mock_input_stream.stop_stream.assert_called_once()
                mock_input_stream.close.assert_called_once()
                mock_output_stream.stop_stream.assert_called_once()
                mock_output_stream.close.assert_called_once()
                pa_instance.terminate.assert_called_once()
