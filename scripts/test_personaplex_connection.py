import asyncio
import sys
import os

# Add parent directory to path so we can import audio_transcriber
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from audio_transcriber.audio_transcriber.personaplex_client import PersonaPlexClient

async def main():
    uri = "ws://localhost:8998"
    print(f" attempting to connect to {uri}...")
    client = PersonaPlexClient(uri)
    try:
        await client.connect()
        print("Successfully connected to PersonaPlex server!")
        await client.disconnect()
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
