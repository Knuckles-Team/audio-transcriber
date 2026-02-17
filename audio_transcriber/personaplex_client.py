import logging
import json
import websockets
import base64
from typing import Optional, AsyncGenerator


class PersonaPlexClient:
    """Client for interacting with the PersonaPlex/Moshi server."""

    def __init__(self, uri: str, logger: Optional[logging.Logger] = None):
        self.uri = uri
        self.logger = logger or logging.getLogger(__name__)
        self.websocket = None

    async def connect(self):
        """Connect to the PersonaPlex server."""
        try:
            self.logger.info(f"Connecting to PersonaPlex at {self.uri}...")
            self.websocket = await websockets.connect(self.uri)
            self.logger.info("Connected to PersonaPlex.")
        except Exception as e:
            self.logger.error(f"Failed to connect to PersonaPlex: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.logger.info("Disconnected from PersonaPlex.")

    async def send_audio(self, audio_data: bytes):
        """Send audio data to the server.

        This assumes the server accepts raw bytes or a JSON wrapper.
        For now, we will try sending raw bytes if it's a stream, or investigate protocol.
        """
        if not self.websocket:
            raise RuntimeError("Not connected to PersonaPlex.")

        try:
            # Placeholder: sending raw bytes. Adjust based on actual protocol.
            # Moshi/PersonaPlex might expect a specific message format.
            await self.websocket.send(audio_data)
        except Exception as e:
            self.logger.error(f"Error sending audio: {e}")

    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """Receive audio data from the server."""
        if not self.websocket:
            raise RuntimeError("Not connected to PersonaPlex.")

        try:
            async for message in self.websocket:
                # Placeholder: receiving raw bytes.
                if isinstance(message, bytes):
                    yield message
                elif isinstance(message, str):
                    # Handle text messages or JSON encoded audio
                    try:
                        data = json.loads(message)
                        if "audio" in data:
                            yield base64.b64decode(data["audio"])
                        # Handle other message types if needed
                    except json.JSONDecodeError:
                        self.logger.warning(
                            f"Received non-JSON text message: {message}"
                        )
        except Exception as e:
            self.logger.error(f"Error receiving audio: {e}")

    async def stream_audio(self, audio_iterator: AsyncGenerator[bytes, None]):
        """Stream audio from an iterator to the server."""
        async for chunk in audio_iterator:
            await self.send_audio(chunk)
