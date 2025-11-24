FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Crea user non-root
RUN useradd -ms /bin/bash appuser

WORKDIR /app

# Installo dipendenze di sistema (incluso make)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copio requirements e installo
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copio Makefile e codice
COPY Makefile .
COPY src/ ./src
COPY data/ ./data

USER appuser

# Default: genera dati + allena modello
CMD ["make", "train"]
