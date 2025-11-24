SHELL := /bin/bash

.PHONY: help data train test all

help:
	@echo "Comandi disponibili:"
	@echo "  make data   - genera i dati (pairs_train.csv)"
	@echo "  make train  - esegue data_prep + addestra il modello"
	@echo "  make test   - esegue pytest"
	@echo "  make all    - data + train + test"

data:
	@echo "[DATA] Genero/aggiorno i dati..."
	PYTHONPATH=. python -m src.data_prep

train: data
	@echo "[TRAIN] Addestro il modello..."
	PYTHONPATH=. python -m src.train_model

test:
	@echo "[TEST] Eseguo pytest..."
	PYTHONPATH=. pytest -q

all: data train test
