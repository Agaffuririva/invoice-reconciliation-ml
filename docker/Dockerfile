# docker/Dockerfile
FROM python:3.11-slim

# Evita buffer dello stdout e migliora logging
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Crea un user non-root (best practice)
RUN useradd -ms /bin/bash appuser

# Crea le cartelle dell'app
WORKDIR /app
RUN mkdir -p /app/src /app/data /app/notebooks

# Installa dipendenze di sistema (per numpy, pandas, ecc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia i requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia il codice
COPY src/ ./src
COPY data/ ./data

# Passa all'utente non-root
USER appuser

# Comando di default: mostrare help (poi lo cambiamo / override da docker-compose o kubernetes)
CMD ["python", "-m", "src.train_model"]
