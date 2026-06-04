FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd -ms /bin/bash vscode

WORKDIR /app/ticket-data-extraction

RUN chown -R vscode:vscode /app/ticket-data-extraction

COPY --chown=vscode:vscode pyproject.toml .python-version ./
COPY --chown=vscode:vscode uv.lock* ./

USER vscode

RUN uv sync

COPY --chown=vscode:vscode . .