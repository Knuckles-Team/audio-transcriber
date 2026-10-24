import importlib
import pytest
import audio_transcriber


def test_attribute_error():
    # Request a non-existent attribute to trigger AttributeError
    with pytest.raises(AttributeError) as exc_info:
        _ = audio_transcriber.non_existent_attribute_xyz
    assert "has no attribute" in str(exc_info.value)
