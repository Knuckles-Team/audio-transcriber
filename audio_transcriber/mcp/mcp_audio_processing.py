"""MCP tools for audio processing operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from pathlib import Path

from agent_utilities.mcp_utilities import ctx_log
from fastmcp import Context, FastMCP
from pydantic import Field

from audio_transcriber.audio_transcriber import AudioTranscriber
from audio_transcriber.mcp_server import (
    DEFAULT_TRANSCRIBE_DIRECTORY,
    DEFAULT_WHISPER_MODEL,
    logger,
)


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
        audio_file: str | None = Field(
            default=None,
            description="Path to the audio file to transcribe. If provided, transcription is performed on this file.",
        ),
        record_seconds: int = Field(
            default=0,
            description="Number of seconds to record audio from microphone. Must be positive if no audio_file is provided. 0 or negative not supported for recording in this context.",
        ),
        directory: str | None = Field(
            default=DEFAULT_TRANSCRIBE_DIRECTORY,
            description="Directory for saving recordings or exports.",
        ),
        model: str = Field(
            default=DEFAULT_WHISPER_MODEL,
            description="Whisper model to use (e.g., 'base', 'small', 'turbo').",
        ),
        language: str | None = Field(
            default=None,
            description="Language code for transcription (e.g., 'en', 'fr'). Auto-detected if not specified.",
        ),
        task: str = Field(
            default="transcribe",
            description="Task to perform: 'transcribe' or 'translate' (to English).",
        ),
        fp16: bool = Field(default=True, description="Use FP16 for faster inference."),
        word_timestamps: bool = Field(
            default=False, description="Include word-level timestamps in the output."
        ),
        temperature: float = Field(
            default=0.0,
            description="Temperature for sampling diversity (0.0 for deterministic).",
        ),
        initial_prompt: str | None = Field(
            default=None, description="Initial text prompt to guide the transcription."
        ),
        export_formats: list[str] | None = Field(
            default=None,
            description="Formats to export the transcription (e.g., ['txt', 'srt']).",
        ),
        backend: str | None = Field(
            default=None,
            description="Transcription backend to use: 'faster-whisper' or 'openai-whisper'. Defaults to auto-detect (preferring faster-whisper).",
        ),
        ctx: Context | None = Field(
            description="MCP context for progress reporting.", default=None
        ),
    ) -> str:
        """Transcribes audio from a provided file or by recording from the microphone."""
        ctx_log(
            ctx,
            logger,
            "info",
            f"Starting transcription: audio_file={audio_file}, record_seconds={record_seconds}, "
            f"directory={directory}, model={model}, language={language}, task={task}, backend={backend}",
        )

        try:
            if not audio_file and record_seconds <= 0:
                raise ValueError(
                    "Either audio_file must be provided or record_seconds must be positive."
                )

            transcriber = AudioTranscriber(
                model=model,
                directory=Path(directory) if directory else Path.cwd(),
                file=audio_file if audio_file else None,
                logger=logger,
                backend=backend,
            )

            if ctx:
                await ctx.report_progress(progress=0, total=100)
                ctx_log(ctx, logger, "debug", "Reported initial progress: 0/100")

            if audio_file:
                file_path = Path(audio_file)
                if not file_path.exists():
                    raise ValueError(f"Audio file not found: {audio_file}")
            else:
                ctx_log(
                    ctx,
                    logger,
                    "info",
                    f"Starting recording for {record_seconds} seconds.",
                )
                transcriber.initiate_stream()

                transcriber.record(seconds=record_seconds)
                transcriber.stop_stream()
                transcriber.save_stream()

                if ctx:
                    await ctx.report_progress(progress=40, total=100)
                    ctx_log(
                        ctx,
                        logger,
                        "debug",
                        "Reported progress after recording: 40/100",
                    )

            ctx_log(ctx, logger, "info", "Starting Whisper transcription.")
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
                ctx_log(
                    ctx,
                    logger,
                    "debug",
                    "Reported progress after transcription: 90/100",
                )

            if export_formats:
                transcriber.export(result, formats=export_formats)
                ctx_log(
                    ctx,
                    logger,
                    "info",
                    f"Exported transcription to formats: {export_formats}",
                )

            if ctx:
                await ctx.report_progress(progress=100, total=100)
                ctx_log(ctx, logger, "debug", "Reported final progress: 100/100")

            ctx_log(ctx, logger, "info", "Transcription completed successfully.")
            return result["text"]
        except Exception as e:
            ctx_log(ctx, logger, "error", f"Failed to transcribe audio: {str(e)}")
            raise RuntimeError(f"Failed to transcribe audio: {str(e)}") from e
