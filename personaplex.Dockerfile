FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libopus-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the repository
# Using the likely repository URL based on the issue link provided
RUN git clone https://github.com/NVIDIA/personaplex.git .

# Install dependencies
# Note: The user mentioned checking issue #2 for Blackwell GPUs, we default to standard installation first
# but include the upgrade command as an optional step or commented out if needed.
# Since we are in a docker container, we can just install '.'
RUN pip install --no-cache-dir ./moshi

# Create directory for SSL certs
RUN mkdir -p /tmp/ssl

# Expose port
EXPOSE 8998

# Environment variables
ENV HF_TOKEN=""

# Default command to run the server
# We use a wrapper script or direct command.
# The user instruction: SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
# We can simulate this in CMD
CMD ["sh", "-c", "SSL_DIR=$(mktemp -d) && python -m moshi.server --ssl \"$SSL_DIR\" --host 0.0.0.0 --port 8998"]
