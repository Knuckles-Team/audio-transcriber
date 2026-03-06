#!/usr/bin/python
# coding: utf-8

from dotenv import load_dotenv, find_dotenv
from agent_utilities.base_utilities import to_boolean
import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastmcp import FastMCP, Context
from fastmcp.utilities.logging import get_logger
from audio_transcriber.audio_transcriber import AudioTranscriber
from agent_utilities.mcp_utilities import (
    create_mcp_server,
    config,
)

__version__ = "0.6.27"

logger = get_logger(name="TokenMiddleware")
logger.setLevel(logging.DEBUG)


DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
DEFAULT_TRANSCRIBE_DIRECTORY = os.environ.get(
    "TRANSCRIBE_DIRECTORY", str(Path.home() / "Downloads")
)


def register_misc_tools(mcp: FastMCP):
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})


def register_audio_processing_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Transcribe Audio",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"audio_processing"},
    )
    async def transcribe_audio(
        audio_file: Optional[str] = Field(
            description="Path to the audio file to transcribe. If provided, transcription is performed on this file.",
            default=None,
        ),
        record_seconds: int = Field(
            description="Number of seconds to record audio from microphone. Must be positive if no audio_file is provided. 0 or negative not supported for recording in this context.",
            default=0,
        ),
        directory: Optional[str] = Field(
            description="Directory for saving recordings or exports.",
            default=DEFAULT_TRANSCRIBE_DIRECTORY,
        ),
        model: str = Field(
            description="Whisper model to use (e.g., 'base', 'small', 'turbo').",
            default=DEFAULT_WHISPER_MODEL,
        ),
        language: Optional[str] = Field(
            description="Language code for transcription (e.g., 'en', 'fr'). Auto-detected if not specified.",
            default=None,
        ),
        task: str = Field(
            description="Task to perform: 'transcribe' or 'translate' (to English).",
            default="transcribe",
        ),
        fp16: bool = Field(description="Use FP16 for faster inference.", default=True),
        word_timestamps: bool = Field(
            description="Include word-level timestamps in the output.", default=False
        ),
        temperature: float = Field(
            description="Temperature for sampling diversity (0.0 for deterministic).",
            default=0.0,
        ),
        initial_prompt: Optional[str] = Field(
            description="Initial text prompt to guide the transcription.", default=None
        ),
        export_formats: List[str] = Field(
            description="Formats to export the transcription (e.g., ['txt', 'srt']).",
            default=None,
        ),
        backend: Optional[str] = Field(
            description="Transcription backend to use: 'faster-whisper' or 'openai-whisper'. Defaults to auto-detect (preferring faster-whisper).",
            default=None,
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting.", default=None
        ),
    ) -> str:
        """Transcribes audio from a provided file or by recording from the microphone."""
        logger.info(
            f"Starting transcription: audio_file={audio_file}, record_seconds={record_seconds}, "
            f"directory={directory}, model={model}, language={language}, task={task}, backend={backend}"
        )

        try:
            if not audio_file and record_seconds <= 0:
                raise ValueError(
                    "Either audio_file must be provided or record_seconds must be positive."
                )

            transcriber = AudioTranscriber(
                model=model,
                directory=Path(directory),
                file=audio_file if audio_file else None,
                logger=logger,
                backend=backend,
            )

            if ctx:
                await ctx.report_progress(progress=0, total=100)
                logger.debug("Reported initial progress: 0/100")

            if audio_file:
                file_path = Path(audio_file)
                if not file_path.exists():
                    raise ValueError(f"Audio file not found: {audio_file}")
            else:
                logger.info(f"Starting recording for {record_seconds} seconds.")
                transcriber.initiate_stream()

                transcriber.record(seconds=record_seconds)
                transcriber.stop_stream()
                transcriber.save_stream()

                if ctx:
                    await ctx.report_progress(progress=40, total=100)
                    logger.debug("Reported progress after recording: 40/100")

            logger.info("Starting Whisper transcription.")
            result = transcriber.transcribe(
                language=language,
                task=task,
                fp16=fp16,
                word_timestamps=word_timestamps,
                temperature=temperature,
                initial_prompt=initial_prompt,
                verbose=True,
            )

            if ctx:
                await ctx.report_progress(progress=90, total=100)
                logger.debug("Reported progress after transcription: 90/100")

            if export_formats:
                transcriber.export(result, formats=export_formats)
                logger.info(f"Exported transcription to formats: {export_formats}")

            if ctx:
                await ctx.report_progress(progress=100, total=100)
                logger.debug("Reported final progress: 100/100")

            logger.info("Transcription completed successfully.")
            return result["text"]
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}")
            raise RuntimeError(f"Failed to transcribe audio: {str(e)}")


def register_prompts(mcp: FastMCP):
    @mcp.prompt
    def transcribe_file_prompt(
        audio_file: str,
        language: str = "",
        model: str = "base",
    ) -> str:
        """
        Generates a prompt for transcribing a specific audio file.
        """
        return f"Transcribe the audio file '{audio_file}'. Language: '{language}', Model: '{model}'. Use the `transcribe_audio` tool with `audio_file`."

    @mcp.prompt
    def record_and_transcribe_prompt(
        record_seconds: int,
        language: str = "",
        model: str = "base",
    ) -> str:
        """
        Generates a prompt for recording from the microphone and transcribing.
        """
        return f"Record audio for {record_seconds} seconds and transcribe it. Language: '{language}', Model: '{model}'. Use the `transcribe_audio` tool with `record_seconds`."

    @mcp.prompt
    def translate_audio_prompt(
        audio_file: str,
        model: str = "base",
    ) -> str:
        """
        Generates a prompt for transcribing an audio file and translating it to English.
        """
        return f"Transcribe and translate the audio file '{audio_file}' to English. Model: '{model}'. Use the `transcribe_audio` tool with `task='translate'`."


def mcp_server():
    load_dotenv(find_dotenv())

    args, mcp, middlewares = create_mcp_server(
        name="Audio Transcriber",
        version=__version__,
        instructions="Audio Transcriber MCP Server - Run Whisper transcription on audio files or microphone input.",
    )

    DEFAULT_MISCTOOL = to_boolean(os.getenv("MISCTOOL", "True"))
    if DEFAULT_MISCTOOL:
        register_misc_tools(mcp)
    DEFAULT_AUDIO_PROCESSINGTOOL = to_boolean(os.getenv("AUDIO_PROCESSINGTOOL", "True"))
    if DEFAULT_AUDIO_PROCESSINGTOOL:
        register_audio_processing_tools(mcp)
    register_prompts(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)

    print(f"Audio Transcriber MCP v{__version__}")
    print("\nStarting Audio Transcriber MCP Server")
    print(f"  Transport: {args.transport.upper()}")
    print(f"  Auth: {args.auth_type}")
    print(f"  Delegation: {'ON' if config['enable_delegation'] else 'OFF'}")
    print(f"  Eunomia: {args.eunomia_type}")

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Invalid transport", extra={"transport": args.transport})
        sys.exit(1)


if __name__ == "__main__":
    mcp_server()
