FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libopus-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Fetch exactly the reviewed upstream revision; branch heads are intentionally rejected.
ARG PERSONAPLEX_REV=3428dfd95309a7f3c84fd93259ded0f810d1ff91
RUN git init . \
    && git remote add origin https://github.com/NVIDIA/personaplex.git \
    && git fetch --depth 1 origin "${PERSONAPLEX_REV}" \
    && git checkout --detach FETCH_HEAD \
    && pip install --no-cache-dir ./moshi \
    && mkdir -p /tmp/ssl

FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3 AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopus0 \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --no-create-home \
        --home-dir /tmp --shell /usr/sbin/nologin app \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local /usr/local
WORKDIR /app
ENV HOME=/tmp \
    XDG_CONFIG_HOME=/tmp/.config \
    XDG_CACHE_HOME=/tmp/.cache

# Expose port
EXPOSE 8998

# Default command to run the server
# We use a wrapper script or direct command.
# The user instruction: SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
# We can simulate this in CMD
USER 10001:10001
CMD ["sh", "-c", "SSL_DIR=$(mktemp -d) && python -m moshi.server --ssl \"$SSL_DIR\" --host 0.0.0.0 --port 8998"]
