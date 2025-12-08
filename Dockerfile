# Dockerfile for WARC → JSONL extractor using pyproject.toml packaging

FROM python:3.11-slim

# System dependencies for pdfplumber, pandas, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Package metadata
COPY pyproject.toml ./

# Source code
COPY src/ ./src/

# Installing the package (and its dependencies) defined in pyproject.toml
RUN pip install --no-cache-dir .

# Default command:
CMD ["python", "-m", "extractor.cli"]
