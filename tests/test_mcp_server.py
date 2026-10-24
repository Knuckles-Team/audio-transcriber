import sys
from pathlib import Path

from audio_transcriber.mcp_server import register_prompts


class MockMCP:
    def __init__(self):
        self.tools = {}
        self.prompts = {}
        self.middlewares = []

    def tool(self, *args, **kwargs):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator

    def prompt(self, func=None, **kwargs):
        if func is not None:
            self.prompts[func.__name__] = func
            return func

        def decorator(f):
            self.prompts[f.__name__] = f
            return f

        return decorator

    def add_middleware(self, mw):
        self.middlewares.append(mw)



def test_prompts():
    mcp = MockMCP()
    register_prompts(mcp)

    assert "transcribe_file_prompt" in mcp.prompts
    assert "record_and_transcribe_prompt" in mcp.prompts
    assert "translate_audio_prompt" in mcp.prompts

    prompt_file = mcp.prompts["transcribe_file_prompt"]
    assert "test.wav" in prompt_file("test.wav", "en", "base")

    prompt_record = mcp.prompts["record_and_transcribe_prompt"]
    assert "10 seconds" in prompt_record(10, "en", "base")

    prompt_translate = mcp.prompts["translate_audio_prompt"]
    assert "test.wav" in prompt_translate("test.wav", "base")
