#!/usr/bin/env python
# coding: utf-8

from audio_transcriber.audio_transcriber import (
    main,
    AudioTranscriber,
    setup_logging,
)
from audio_transcriber.audio_transcriber_mcp import audio_transcriber_mcp

"""
audio-transcriber

Transcribe your .wav .mp4 .mp3 .flac files to text using AI!
"""


__all__ = ["audio_transcriber_mcp", "main", "AudioTranscriber", "setup_logging"]
