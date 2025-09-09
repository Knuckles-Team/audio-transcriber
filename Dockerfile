FROM python:3-slim

ARG HOST=0.0.0.0
ARG PORT=8021
ENV HOST=${HOST}
ENV PORT=${PORT}
ENV PATH="/usr/local/bin:${PATH}"
RUN apt update \
    && apt install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg -y \
    && pip install uv \
    && uv pip install --system audio-transcriber

ENTRYPOINT exec audio-transcriber-mcp --transport "http" --host "${HOST}" --port "${PORT}"
