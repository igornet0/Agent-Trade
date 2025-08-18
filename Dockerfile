# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry lock --no-interaction && \
    poetry install --no-interaction --no-ansi --only main --no-root

COPY . .

EXPOSE 8000

# Используем упрощенную версию для запуска
CMD ["python", "run_app_simple.py"]

FROM base AS worker
CMD ["celery", "-A", "src.backend.celery_app.create_app.celery_app", "worker", "-l", "INFO"]

