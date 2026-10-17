import importlib
from unittest.mock import patch
import pytest
import audio_transcriber


def test_expose_members_and_getattr():
    # Test getattr for normal elements
    assert hasattr(audio_transcriber, "AudioTranscriber")
    assert audio_transcriber.AudioTranscriber is not None


def test_available_flags():
    # Fetch available flags
    mcp_avail = audio_transcriber._MCP_AVAILABLE
    assert isinstance(mcp_avail, bool)


def test_import_error_getattr():
    # Test when module import raises ImportError
    with patch("importlib.import_module", side_effect=ImportError("mock error")):
        from audio_transcriber import _import_module_safely

        res = _import_module_safely("non_existent_module")
        assert res is None




def test_dynamic_attributes_retrieval():
    # Retrieve dynamic attribute from optional modules (covers line 69)
    # DEFAULT_WHISPER_MODEL is a string variable, not class/function, so it triggers __getattr__
    val = getattr(audio_transcriber, "DEFAULT_WHISPER_MODEL")
    assert val is not None


def test_attribute_error():
    # Request a non-existent attribute to trigger AttributeError
    with pytest.raises(AttributeError) as exc_info:
        _ = audio_transcriber.non_existent_attribute_xyz
    assert "has no attribute" in str(exc_info.value)


def test_dir():
    # Execute __dir__
    dir_list = dir(audio_transcriber)
    assert "AudioTranscriber" in dir_list
