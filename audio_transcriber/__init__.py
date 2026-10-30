#!/usr/bin/env python

from scion_sdk.connector import Connector
from scion_sdk.connector_hooks import RegistrationContext

from audio_transcriber.audio_transcriber import AudioTranscriber
from audio_transcriber.mcp_server import get_client, register_prompts


def _register_prompts(context: RegistrationContext) -> None:
    register_prompts(context.server)


connector = Connector(
    api=AudioTranscriber,
    get_client=get_client,
    tools="audio_transcriber.mcp_server",
    instructions="Audio Transcriber MCP Server - Run Whisper transcription on audio files or microphone input.",
    extra_registrations=(_register_prompts,),
)
