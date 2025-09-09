#!/usr/bin/python
# coding: utf-8

import getopt
import os
import sys
from typing import List, Optional
from pathlib import Path
from audio_transcriber import AudioTranscriber, setup_logging
from fastmcp import FastMCP, Context
from pydantic import Field

# Initialize logging for MCP server (logs to file, verbose for details)
logger = setup_logging(verbose=True, log_file="audio_transcriber_mcp.log")

mcp = FastMCP(name="AudioTranscriberServer")

# Environment variables for defaults
environment_model = os.environ.get("WHISPER_MODEL", "base")
environment_directory = os.environ.get(
    "TRANSCRIBE_DIRECTORY", str(Path.home() / "Downloads")
)


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
        default=environment_directory,
    ),
    model: str = Field(
        description="Whisper model to use (e.g., 'base', 'small', 'turbo').",
        default=environment_model,
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
    ctx: Context = Field(
        description="MCP context for progress reporting.", default=None
    ),
) -> str:
    """Transcribes audio from a provided file or by recording from the microphone."""
    logger.info(
        f"Starting transcription: audio_file={audio_file}, record_seconds={record_seconds}, "
        f"directory={directory}, model={model}, language={language}, task={task}"
    )

    try:
        if not audio_file and record_seconds <= 0:
            raise ValueError(
                "Either audio_file must be provided or record_seconds must be positive."
            )

        # Create transcriber instance
        transcriber = AudioTranscriber(
            model=model,
            directory=Path(directory),
            file=audio_file if audio_file else None,
            logger=logger,
        )

        # Report initial progress
        if ctx:
            await ctx.report_progress(progress=0, total=100)
            logger.debug("Reported initial progress: 0/100")

        if audio_file:
            # Validate file existence
            file_path = Path(audio_file)
            if not file_path.exists():
                raise ValueError(f"Audio file not found: {audio_file}")
        else:
            # Recording mode (only fixed duration supported)
            logger.info(f"Starting recording for {record_seconds} seconds.")
            transcriber.initiate_stream()

            # Coarse progress for recording (sync call, so limited granularity)
            transcriber.record(seconds=record_seconds)
            transcriber.stop_stream()
            transcriber.save_stream()

            if ctx:
                await ctx.report_progress(
                    progress=40, total=100
                )  # Arbitrary midpoint after recording
                logger.debug("Reported progress after recording: 40/100")

        # Perform transcription
        logger.info("Starting Whisper transcription.")
        result = transcriber.transcribe(
            language=language,
            task=task,
            fp16=fp16,
            word_timestamps=word_timestamps,
            temperature=temperature,
            initial_prompt=initial_prompt,
            verbose=True,  # Enable verbose for logging details
        )

        if ctx:
            await ctx.report_progress(progress=90, total=100)
            logger.debug("Reported progress after transcription: 90/100")

        # Export if requested
        if export_formats:
            transcriber.export(result, formats=export_formats)
            logger.info(f"Exported transcription to formats: {export_formats}")

        # Report completion
        if ctx:
            await ctx.report_progress(progress=100, total=100)
            logger.debug("Reported final progress: 100/100")

        logger.info("Transcription completed successfully.")
        return result["text"]
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {str(e)}")
        raise RuntimeError(f"Failed to transcribe audio: {str(e)}")


def audio_transcriber_mcp(argv):
    transport = "stdio"
    host = "0.0.0.0"
    port = 8000
    try:
        opts, args = getopt.getopt(
            argv,
            "ht:h:p:",
            ["help", "transport=", "host=", "port="],
        )
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            sys.exit()
        elif opt in ("-t", "--transport"):
            transport = arg
        elif opt in ("-h", "--host"):
            host = arg
        elif opt in ("-p", "--port"):
            try:
                port = int(arg)
                if not (0 <= port <= 65535):
                    print(f"Error: Port {arg} is out of valid range (0-65535).")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Port {arg} is not a valid integer.")
                sys.exit(1)
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "http":
        mcp.run(transport="http", host=host, port=port)
    else:
        logger.error("Transport not supported")
        sys.exit(1)


def main():
    audio_transcriber_mcp(sys.argv[1:])


if __name__ == "__main__":
    audio_transcriber_mcp(sys.argv[1:])
