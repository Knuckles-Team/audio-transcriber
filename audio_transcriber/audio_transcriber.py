#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Iterator, List, Optional, TextIO, Union

import pyaudio
import whisper
import wave

__version__ = "0.5.74"


class AudioTranscriber:
    """A class for recording audio and transcribing it using OpenAI's Whisper model."""

    def __init__(
        self,
        model: str = "base",
        channels: int = 1,
        rate: int = 16000,  # Whisper recommends 16kHz for better accuracy
        file_name: str = "output.wav",
        directory: Union[str, Path] = Path.cwd(),
        file: Optional[Union[str, Path]] = None,
        device: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = channels
        self.rate = rate
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        self.frames: List[bytes] = []
        self.file_path = Path(file) if file else Path(directory) / file_name
        self.title = self.file_path.stem
        self.directory = self.file_path.parent
        self.stop = False
        self.model = whisper.load_model(model)
        self.device_index = device or self._get_default_device()
        self.logger = logger or logging.getLogger(__name__)
        self._check_ffmpeg()

    def _get_default_device(self) -> int:
        """Get the default input device index."""
        return self.pyaudio_instance.get_default_input_device_info()["index"]

    def _check_ffmpeg(self) -> None:
        """Check if ffmpeg is installed; log warning if not."""
        import shutil

        if not shutil.which("ffmpeg"):
            self.logger.warning(
                "ffmpeg not found. Install it for better audio format support. "
                "See https://ffmpeg.org/download.html for instructions."
            )

    def initiate_stream(self) -> None:
        """Initiate the audio input stream."""
        self.stream = self.pyaudio_instance.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device_index,
        )

    def record(self, seconds: int = 0) -> None:
        """Record audio for a specified duration or until stopped."""
        self.logger.info("Recording started...")
        self.frames = []
        self.stop = False
        if seconds > 0:
            for _ in range(0, int((self.rate / self.chunk) * seconds)):
                if self.stop:
                    break
                data = self.stream.read(self.chunk)
                self.frames.append(data)
        else:
            self.logger.info("Recording indefinitely until interrupted (Ctrl+C)...")
            threading.Thread(target=self._unlimited_record, daemon=True).start()
            try:
                while not self.stop:
                    pass
            except KeyboardInterrupt:
                self.stop = True
        self.logger.info("Recording stopped.")

    def _unlimited_record(self) -> None:
        """Thread for unlimited recording."""
        while not self.stop:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

    def stop_stream(self) -> None:
        """Stop and close the audio stream."""
        self.stop = True
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio_instance.terminate()

    def save_stream(self) -> None:
        """Save the recorded frames to a WAV file."""
        if not self.frames:
            self.logger.warning("No audio frames to save.")
            return
        with wave.open(str(self.file_path), "wb") as wave_file:
            wave_file.setnchannels(self.channels)
            wave_file.setsampwidth(self.pyaudio_instance.get_sample_size(self.format))
            wave_file.setframerate(self.rate)
            wave_file.writeframes(b"".join(self.frames))
        self.logger.info(f"Audio saved to {self.file_path}")

    def transcribe(
        self,
        language: Optional[str] = None,
        task: str = "transcribe",
        fp16: bool = True,
        word_timestamps: bool = False,
        temperature: float = 0.0,
        initial_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> dict:
        """Transcribe the audio file using Whisper."""
        if task == "translate" and self.model.name.startswith("turbo"):
            raise ValueError(
                "Turbo model does not support translation. Use a multilingual model like 'medium' or 'large'."
            )

        start_time = datetime.datetime.now()
        self.logger.info(
            f"Started transcription at {start_time} for file: {self.file_path}"
        )

        options = whisper.DecodingOptions(
            language=language,
            task=task,
            fp16=fp16,
            word_timestamps=word_timestamps,
            temperature=temperature,
            prompt=initial_prompt,
        )

        result = self.model.transcribe(
            str(self.file_path), **options.__dict__, verbose=verbose
        )

        end_time = datetime.datetime.now()
        self.logger.info(
            f"Ended transcription at {end_time}. Time elapsed: {end_time - start_time}"
        )
        if verbose:
            self.logger.info(f"Transcription result: {result['text']}")

        return result

    def export(
        self,
        result: dict,
        formats: List[str],
    ) -> None:
        """Export transcription to specified formats."""
        segments = result["segments"]
        for fmt in formats:
            export_path = self.directory / f"{self.title}.{fmt}"
            if fmt == "txt":
                with open(export_path, "w", encoding="utf-8") as f:
                    self._write_txt(segments, f)
            elif fmt == "vtt":
                with open(export_path, "w", encoding="utf-8") as f:
                    self._write_vtt(segments, f)
            elif fmt == "srt":
                with open(export_path, "w", encoding="utf-8") as f:
                    self._write_srt(segments, f)
            elif fmt == "json":
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
            else:
                self.logger.warning(f"Unsupported export format: {fmt}")
            self.logger.info(f"Exported to {export_path}")

    @staticmethod
    def _srt_format_timestamp(seconds: float) -> str:
        """Format timestamp for SRT."""
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
        seconds_int = milliseconds // 1_000
        milliseconds -= seconds_int * 1_000
        return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"

    def _write_srt(self, transcript: Iterator[dict], file: TextIO) -> None:
        """Write SRT file."""
        count = 0
        for segment in transcript:
            count += 1
            print(
                f"{count}\n"
                f"{self._srt_format_timestamp(segment['start'])} --> {self._srt_format_timestamp(segment['end'])}\n"
                f"{segment['text'].replace('-->', '->').strip()}\n",
                file=file,
                flush=True,
            )

    @staticmethod
    def _write_txt(transcript: Iterator[dict], file: TextIO) -> None:
        """Write TXT file."""
        for segment in transcript:
            print(segment["text"].strip(), file=file, flush=True)

    @staticmethod
    def _write_vtt(transcript: Iterator[dict], file: TextIO) -> None:
        """Write VTT file."""
        print("WEBVTT\n", file=file)
        for segment in transcript:
            print(
                f"{AudioTranscriber._format_timestamp(segment['start'])} --> {AudioTranscriber._format_timestamp(segment['end'])}\n"
                f"{segment['text'].strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )

    @staticmethod
    def _format_timestamp(
        seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
    ) -> str:
        """Format timestamp for VTT."""
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
        seconds_int = milliseconds // 1_000
        milliseconds -= seconds_int * 1_000
        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds_int:02d}{decimal_marker}{milliseconds:03d}"


def setup_logging(
    verbose: bool = False, log_file: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def audio_transcriber() -> None:
    parser = argparse.ArgumentParser(
        description="Audio Transcriber: Record and transcribe audio using OpenAI Whisper.",
        epilog="Examples:\n"
        "  python audio_transcriber.py --file path/to/audio.mp3 --model large --task translate --language ja\n"
        "  python audio_transcriber.py --record 60 --directory ./recordings --name my_recording.wav --verbose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "turbo",
            "tiny.en",
            "base.en",
            "small.en",
            "medium.en",
        ],
        help="Whisper model to use (default: base)",
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of audio channels (default: 1)"
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=16000,
        help="Sample rate for recording (default: 16000)",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path.cwd(),
        help="Directory to save recordings/exports (default: current dir)",
    )
    parser.add_argument(
        "--name",
        default="output.wav",
        help="Name of the output file (default: output.wav)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        nargs="*",
        help="Path(s) to audio file(s) to transcribe (skips recording)",
    )
    parser.add_argument(
        "--record",
        type=int,
        default=0,
        help="Seconds to record (0 for unlimited until Ctrl+C; default: 0)",
    )
    parser.add_argument(
        "--device", type=int, help="Input device index (default: system default)"
    )
    parser.add_argument(
        "--language", help="Language code (e.g., 'en', 'fr'; auto-detected if omitted)"
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task: transcribe or translate to English (default: transcribe)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 for faster inference (default: False)",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word-level timestamps in output (default: False)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling diversity (default: 0.0)",
    )
    parser.add_argument(
        "--initial-prompt", help="Initial text prompt to guide transcription"
    )
    parser.add_argument(
        "--export",
        nargs="*",
        choices=["txt", "vtt", "srt", "json"],
        default=[],
        help="Export formats (e.g., --export txt srt)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--log-file", help="Path to log file")

    args = parser.parse_args()

    logger = setup_logging(args.verbose, args.log_file)

    if args.file:
        # Batch transcription
        for file_path in args.file:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                sys.exit(1)
            transcriber = AudioTranscriber(
                model=args.model,
                channels=args.channels,
                rate=args.rate,
                file=file_path,
                device=args.device,
                logger=logger,
            )
            result = transcriber.transcribe(
                language=args.language,
                task=args.task,
                fp16=args.fp16,
                word_timestamps=args.word_timestamps,
                temperature=args.temperature,
                initial_prompt=args.initial_prompt,
                verbose=args.verbose,
            )
            if args.export:
                transcriber.export(result, args.export)
    else:
        # Recording mode
        transcriber = AudioTranscriber(
            model=args.model,
            channels=args.channels,
            rate=args.rate,
            file_name=args.name,
            directory=args.directory,
            device=args.device,
            logger=logger,
        )
        transcriber.initiate_stream()
        transcriber.record(seconds=args.record)
        transcriber.stop_stream()
        transcriber.save_stream()
        result = transcriber.transcribe(
            language=args.language,
            task=args.task,
            fp16=args.fp16,
            word_timestamps=args.word_timestamps,
            temperature=args.temperature,
            initial_prompt=args.initial_prompt,
            verbose=args.verbose,
        )
        if args.export:
            transcriber.export(result, args.export)


if __name__ == "__main__":
    audio_transcriber()
