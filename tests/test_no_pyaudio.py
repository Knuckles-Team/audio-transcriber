"""File transcription must work when pyaudio is not installed (container image)."""

import audio_transcriber.audio_transcriber as module


def test_transcriber_builds_without_pyaudio(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "pyaudio", None)
    transcriber = module.AudioTranscriber(file=tmp_path / "a.wav")
    assert transcriber.pyaudio_instance is None
    assert transcriber.device_index == -1
    transcriber.stop_stream()
