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
    agent_avail = audio_transcriber._AGENT_AVAILABLE
    assert isinstance(mcp_avail, bool)
    assert isinstance(agent_avail, bool)


def test_import_error_getattr():
    # Test when module import raises ImportError
    with patch("importlib.import_module", side_effect=ImportError("mock error")):
        from audio_transcriber import _import_module_safely

        res = _import_module_safely("non_existent_module")
        assert res is None


def test_available_flags_missing():
    # Patch import_module to raise ImportError when importing optional modules
    original_import = importlib.import_module

    def mock_import(name, *args, **kwargs):
        if "mcp_server" in name or "agent_server" in name:
            raise ImportError("mock optional missing")
        return original_import(name, *args, **kwargs)

    with patch("importlib.import_module", side_effect=mock_import):
        mcp_val = audio_transcriber.__getattr__("_MCP_AVAILABLE")
        agent_val = audio_transcriber.__getattr__("_AGENT_AVAILABLE")
        assert mcp_val is False
        assert agent_val is False

    # Cover fallback return False (lines 52, 57) when OPTIONAL_MODULES keys are missing
    with patch.dict(audio_transcriber.OPTIONAL_MODULES, {}, clear=True):
        mcp_val = audio_transcriber.__getattr__("_MCP_AVAILABLE")
        agent_val = audio_transcriber.__getattr__("_AGENT_AVAILABLE")
        assert mcp_val is False
        assert agent_val is False


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
