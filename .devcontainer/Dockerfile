FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd -ms /bin/bash vscode

WORKDIR /workspaces/ticket-data-extraction

RUN chown -R vscode:vscode /workspaces/ticket-data-extraction

COPY --chown=vscode:vscode pyproject.toml uv.lock .python-version ./

USER vscode

RUN uv sync

COPY --chown=vscode:vscode . .