FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev curl && rm -rf /var/lib/apt/lists/*
# Base image ships the lightweight API stack only; add the 'ml' extra for the
# fine-tuned classifier and the sentence-transformer encoder (pulls torch).
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY eval /app/eval
RUN pip install --upgrade pip && pip install -e .
COPY . /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
