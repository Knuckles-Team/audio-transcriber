import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest

# Ensure modules are mock-imported
sys.modules["torch"] = MagicMock()
sys.modules["whisper"] = MagicMock()
sys.modules["faster_whisper"] = MagicMock()

import audio_transcriber
from audio_transcriber.audio_transcriber import (
    FasterWhisperBackend,
    OpenAIWhisperBackend,
    AudioTranscriber,
    setup_logging,
    audio_transcriber as cli_entrypoint,
)


def test_faster_whisper_backend():
    import sys

    mock_fw = MagicMock()
    sys.modules["faster_whisper"] = mock_fw

    logger = logging.getLogger("test_logger")
    backend = FasterWhisperBackend(logger)
    assert backend.model is None

    # load_model CUDA/float16
    with patch("torch.cuda.is_available", return_value=True):
        backend.load_model("tiny")
    mock_fw.WhisperModel.assert_called_with(
        "tiny", device="cuda", compute_type="float16"
    )

    # load_model CPU/int8
    with patch("torch.cuda.is_available", return_value=False):
        backend.load_model("tiny")
    mock_fw.WhisperModel.assert_called_with("tiny", device="cpu", compute_type="int8")

    # transcribe without model
    backend.model = None
    with pytest.raises(RuntimeError):
        backend.transcribe("dummy.wav")

    # transcribe with model
    mock_model = MagicMock()
    backend.model = mock_model

    # Mock segment and info
    mock_segment = MagicMock()
    mock_segment.id = 1
    mock_segment.seek = 0
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = " Hello world "
    mock_segment.tokens = [1, 2]
    mock_segment.temperature = 0.0
    mock_segment.avg_logprob = -0.1
    mock_segment.compression_ratio = 1.2
    mock_segment.no_speech_prob = 0.05

    # Word timestamps words
    mock_word = MagicMock()
    mock_word.word = "Hello"
    mock_word.start = 0.0
    mock_word.end = 0.5
    mock_word.probability = 0.99
    mock_segment.words = [mock_word]

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95
    mock_info.duration = 1.0

    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    res = backend.transcribe("dummy.wav", word_timestamps=True)
    assert res["text"] == " Hello world "
    assert len(res["segments"]) == 1
    assert res["segments"][0]["words"][0]["word"] == "Hello"


def test_openai_whisper_backend():
    import sys

    mock_whisper = MagicMock()
    sys.modules["whisper"] = mock_whisper

    logger = logging.getLogger("test_logger")
    backend = OpenAIWhisperBackend(logger)
    assert backend.model is None

    # Test load_model CPU
    with patch("torch.cuda.is_available", return_value=False):
        backend.load_model("tiny", device=None)
    assert backend.model_name == "tiny"
    mock_whisper.load_model.assert_called_with("tiny", device="cpu")

    # Test load_model CUDA
    with patch("torch.cuda.is_available", return_value=True):
        backend.load_model("tiny", device=None)
    mock_whisper.load_model.assert_called_with("tiny", device="cuda")

    # Test transcribe without loaded model
    backend.model = None
    with pytest.raises(RuntimeError):
        backend.transcribe("dummy.wav")

    # Test transcribe
    backend.model = MagicMock()
    mock_options = MagicMock()
    mock_whisper.DecodingOptions.return_value = mock_options
    mock_options.__dict__ = {"option": "val"}

    backend.transcribe("dummy.wav", language="en", task="translate", fp16=False)
    mock_whisper.DecodingOptions.assert_called_with(
        language="en",
        task="translate",
        fp16=False,
        word_timestamps=False,
        temperature=0.0,
        prompt=None,
    )


def test_audio_transcriber_backend_initialization():
    # 1. requested openai-whisper but raises ImportError
    with patch(
        "audio_transcriber.audio_transcriber.OpenAIWhisperBackend.load_model",
        side_effect=ImportError("no openai"),
    ):
        with pytest.raises(ImportError):
            AudioTranscriber(backend="openai-whisper")

    # 2. default fails to load faster-whisper, falls back to openai-whisper
    with patch(
        "audio_transcriber.audio_transcriber.FasterWhisperBackend.load_model",
        side_effect=ImportError("no faster"),
    ):
        with patch(
            "audio_transcriber.audio_transcriber.OpenAIWhisperBackend.load_model"
        ) as mock_load:
            transcriber = AudioTranscriber(backend=None)
            assert isinstance(transcriber.backend_instance, OpenAIWhisperBackend)
            mock_load.assert_called_once()

    # 3. requested faster-whisper specifically but it fails to load
    with patch(
        "audio_transcriber.audio_transcriber.FasterWhisperBackend.load_model",
        side_effect=ImportError("no faster"),
    ):
        with pytest.raises(ImportError):
            AudioTranscriber(backend="faster-whisper")

    # 4. Both fail to load
    with patch(
        "audio_transcriber.audio_transcriber.FasterWhisperBackend.load_model",
        side_effect=ImportError("no faster"),
    ):
        with patch(
            "audio_transcriber.audio_transcriber.OpenAIWhisperBackend.load_model",
            side_effect=ImportError("no openai"),
        ):
            with pytest.raises(ImportError) as exc:
                AudioTranscriber(backend=None)
            assert "Neither faster-whisper nor openai-whisper found" in str(exc.value)


def test_audio_transcriber_device_and_stream():
    # Mock PyAudio to raise OSError on get_default_input_device_info
    with patch("pyaudio.PyAudio") as mock_pa:
        pa_instance = mock_pa.return_value
        pa_instance.get_default_input_device_info.side_effect = OSError(
            "no default device"
        )

        trans = AudioTranscriber()
        assert trans.device_index == -1

        # initiate_stream raises RuntimeError
        with pytest.raises(RuntimeError) as exc:
            trans.initiate_stream()
        assert "No input device available" in str(exc.value)


def test_check_ffmpeg_warning():
    with patch("shutil.which", return_value=None):
        logger = MagicMock()
        _ = AudioTranscriber(logger=logger)
        logger.warning.assert_any_call(
            "ffmpeg not found. Install it for better audio format support. "
            "See https://ffmpeg.org/download.html for instructions."
        )


def test_record_limited():
    with patch("pyaudio.PyAudio") as mock_pa:
        pa_instance = mock_pa.return_value
        pa_instance.get_default_input_device_info.return_value = {"index": 0}

        mock_stream = MagicMock()
        pa_instance.open.return_value = mock_stream
        mock_stream.read.return_value = b"\x00" * 1024

        trans = AudioTranscriber(rate=16000, channels=1)
        trans.initiate_stream()

        # Record with seconds = 1
        trans.record(seconds=1)
        assert len(trans.frames) > 0

        # Record with seconds = 1 but stop is set early
        trans.stop = True
        trans.record(seconds=1)


def test_record_unlimited():
    with patch("pyaudio.PyAudio") as mock_pa:
        pa_instance = mock_pa.return_value
        pa_instance.get_default_input_device_info.return_value = {"index": 0}

        mock_stream = MagicMock()
        pa_instance.open.return_value = mock_stream

        trans = AudioTranscriber(rate=16000, channels=1)
        trans.initiate_stream()

        def mock_read(chunk):
            trans.stop = True
            return b"\x00" * 1024

        mock_stream.read.side_effect = mock_read
        trans.stop = False
        trans._unlimited_record()
        assert len(trans.frames) == 1


def test_record_keyboard_interrupt():
    with patch("pyaudio.PyAudio") as mock_pa:
        pa_instance = mock_pa.return_value
        pa_instance.get_default_input_device_info.return_value = {"index": 0}

        mock_stream = MagicMock()
        pa_instance.open.return_value = mock_stream
        mock_stream.read.return_value = b"\x00" * 1024

        trans = AudioTranscriber(rate=16000, channels=1)
        trans.initiate_stream()

        # Test KeyboardInterrupt during record()
        class StopMock:
            def __init__(self):
                self.calls = 0

            def get_stop(self, obj):
                self.calls += 1
                if self.calls > 1:
                    raise KeyboardInterrupt("Ctrl+C")
                return False

            def set_stop(self, obj, val):
                pass

        original_stop = getattr(AudioTranscriber, "stop", None)
        try:
            setattr(
                AudioTranscriber,
                "stop",
                property(StopMock().get_stop, StopMock().set_stop),
            )
            trans.record(seconds=0)
        finally:
            if original_stop is not None:
                setattr(AudioTranscriber, "stop", original_stop)
            else:
                delattr(AudioTranscriber, "stop")


def test_stop_and_save_stream():
    trans = AudioTranscriber()
    # test stop_stream
    trans.stream = MagicMock()
    trans.stop_stream()
    assert trans.stop is True
    trans.stream.stop_stream.assert_called_once()
    trans.stream.close.assert_called_once()

    # test save_stream warning
    logger = MagicMock()
    trans.logger = logger
    trans.frames = []
    trans.save_stream()
    logger.warning.assert_any_call("No audio frames to save.")

    # test save_stream success
    trans.frames = [b"\x00\x00"]
    with patch("wave.open") as mock_wave_open:
        trans.save_stream()
        mock_wave_open.assert_called_once()


def test_export_formats():
    trans = AudioTranscriber()
    mock_result = {
        "text": "Hello world",
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 1.5,
                "text": "Hello world",
            },
            {
                "id": 2,
                "start": 3600.0,
                "end": 3605.2,
                "text": "An hour later --> with arrow",
            },
        ],
    }

    m_open = mock_open()
    with patch("builtins.open", m_open):
        trans.export(mock_result, ["txt", "vtt", "srt", "json", "invalid_fmt"])

    vtt_ts1 = AudioTranscriber._format_timestamp(0.5)
    vtt_ts2 = AudioTranscriber._format_timestamp(3605.2, always_include_hours=True)
    assert vtt_ts1 == "00:00.500"
    assert vtt_ts2 == "01:00:05.200"

    srt_ts = AudioTranscriber._srt_format_timestamp(3605.2)
    assert srt_ts == "01:00:05,200"

    with pytest.raises(AssertionError):
        AudioTranscriber._srt_format_timestamp(-1.0)
    with pytest.raises(AssertionError):
        AudioTranscriber._format_timestamp(-1.0)


def test_setup_logging_with_file():
    with patch("logging.FileHandler") as mock_fh:
        mock_handler = MagicMock(spec=logging.Handler)
        mock_handler.level = logging.INFO
        mock_fh.return_value = mock_handler

        logger = setup_logging(verbose=True, log_file="test.log")
        mock_fh.assert_called_once_with("test.log")
        if mock_handler in logger.handlers:
            logger.handlers.remove(mock_handler)


def test_cli_help():
    mock_args = MagicMock()
    mock_args.help = True

    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with patch("argparse.ArgumentParser.print_help") as mock_print:
            with pytest.raises(SystemExit) as exc:
                cli_entrypoint()
            assert exc.value.code == 0
            mock_print.assert_called_once()


def test_cli_interact():
    mock_args = MagicMock()
    mock_args.interact = True
    mock_args.server = "ws://localhost"
    mock_args.help = False
    mock_args.verbose = False
    mock_args.log_file = None
    mock_args.model = "tiny"
    mock_args.channels = 1
    mock_args.rate = 16000
    mock_args.device = None

    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with patch(
            "audio_transcriber.audio_transcriber.AudioTranscriber.interact"
        ) as mock_interact:

            async def dummy_interact(server):
                return

            mock_interact.side_effect = dummy_interact

            with pytest.raises(SystemExit) as exc:
                cli_entrypoint()
            assert exc.value.code == 0
            mock_interact.assert_called_once_with("ws://localhost")


def test_cli_files():
    mock_args = MagicMock()
    mock_args.interact = False
    mock_args.help = False
    mock_args.verbose = False
    mock_args.log_file = None
    mock_args.model = "tiny"
    mock_args.channels = 1
    mock_args.rate = 16000
    mock_args.device = None
    mock_args.backend = None
    mock_args.language = "en"
    mock_args.task = "transcribe"
    mock_args.fp16 = True
    mock_args.word_timestamps = False
    mock_args.temperature = 0.0
    mock_args.initial_prompt = None
    mock_args.export = ["txt"]

    # File exists
    mock_file = MagicMock()
    mock_file.exists.return_value = True
    mock_args.file = [mock_file]

    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with patch(
            "audio_transcriber.audio_transcriber.AudioTranscriber.transcribe"
        ) as mock_transcribe:
            with patch(
                "audio_transcriber.audio_transcriber.AudioTranscriber.export"
            ) as mock_export:
                cli_entrypoint()
                mock_transcribe.assert_called_once()
                mock_export.assert_called_once()

    # File does not exist
    mock_file.exists.return_value = False
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with pytest.raises(SystemExit) as exc:
            cli_entrypoint()
        assert exc.value.code == 1


def test_cli_record():
    mock_args = MagicMock()
    mock_args.file = None
    mock_args.interact = False
    mock_args.help = False
    mock_args.verbose = False
    mock_args.log_file = None
    mock_args.model = "tiny"
    mock_args.channels = 1
    mock_args.rate = 16000
    mock_args.device = None
    mock_args.backend = None
    mock_args.name = "output.wav"
    mock_args.directory = Path.cwd()
    mock_args.record = 5
    mock_args.language = "en"
    mock_args.task = "transcribe"
    mock_args.fp16 = True
    mock_args.word_timestamps = False
    mock_args.temperature = 0.0
    mock_args.initial_prompt = None
    mock_args.export = ["txt"]

    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with patch(
            "audio_transcriber.audio_transcriber.AudioTranscriber.initiate_stream"
        ) as mock_init:
            with patch(
                "audio_transcriber.audio_transcriber.AudioTranscriber.record"
            ) as mock_record:
                with patch(
                    "audio_transcriber.audio_transcriber.AudioTranscriber.stop_stream"
                ) as mock_stop:
                    with patch(
                        "audio_transcriber.audio_transcriber.AudioTranscriber.save_stream"
                    ) as mock_save:
                        with patch(
                            "audio_transcriber.audio_transcriber.AudioTranscriber.transcribe"
                        ) as mock_transcribe:
                            with patch(
                                "audio_transcriber.audio_transcriber.AudioTranscriber.export"
                            ) as mock_export:
                                cli_entrypoint()
                                mock_init.assert_called_once()
                                mock_record.assert_called_once_with(seconds=5)
                                mock_stop.assert_called_once()
                                mock_save.assert_called_once()
                                mock_transcribe.assert_called_once()
                                mock_export.assert_called_once()


def test_openai_whisper_backend_turbo():
    import sys

    mock_whisper = MagicMock()
    sys.modules["whisper"] = mock_whisper

    logger = logging.getLogger("test_logger")
    backend = OpenAIWhisperBackend(logger)
    backend.model = MagicMock()
    backend.model_name = "turbo"
    mock_options = MagicMock()
    mock_whisper.DecodingOptions.return_value = mock_options

    backend.transcribe("dummy.wav", language="en", task="translate", fp16=False)
    mock_whisper.DecodingOptions.assert_called_with(
        language="en",
        task="translate",
        fp16=False,
        word_timestamps=False,
        temperature=0.0,
        prompt=None,
    )


def test_audio_transcriber_successful_requested_openai_whisper():
    with patch(
        "audio_transcriber.audio_transcriber.OpenAIWhisperBackend.load_model"
    ) as mock_load:
        transcriber = AudioTranscriber(backend="openai-whisper")
        assert isinstance(transcriber.backend_instance, OpenAIWhisperBackend)
        mock_load.assert_called_once()


def test_audio_transcriber_transcribe():
    trans = AudioTranscriber()
    # 1. Not initialized backend
    trans.backend_instance = None
    with pytest.raises(RuntimeError) as exc:
        trans.transcribe()
    assert "Backend not initialized" in str(exc.value)

    # 2. Successfully transcribes
    mock_backend = MagicMock()
    mock_backend.transcribe.return_value = {"text": "hello"}
    trans.backend_instance = mock_backend
    res = trans.transcribe(verbose=True)
    assert res == {"text": "hello"}
    mock_backend.transcribe.assert_called_once()


def test_cli_interact_keyboard_interrupt():
    mock_args = MagicMock()
    mock_args.interact = True
    mock_args.server = "ws://localhost"
    mock_args.help = False
    mock_args.verbose = False
    mock_args.log_file = None
    mock_args.model = "tiny"
    mock_args.channels = 1
    mock_args.rate = 16000
    mock_args.device = None

    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with patch(
            "audio_transcriber.audio_transcriber.AudioTranscriber.interact",
            side_effect=KeyboardInterrupt,
        ):
            with pytest.raises(SystemExit) as exc:
                cli_entrypoint()
            assert exc.value.code == 0


def test_main_block():
    import runpy

    with patch("sys.argv", ["audio_transcriber.py", "--help"]):
        with patch("argparse.ArgumentParser.print_help"):
            with pytest.raises(SystemExit) as exc:
                runpy.run_path(
                    "audio_transcriber/audio_transcriber.py", run_name="__main__"
                )
            assert exc.value.code == 0
