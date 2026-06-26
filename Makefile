# Variables
RAW_DATA=data/raw/full2025.csv
LOAD_PARQUET=data/processed/dvf_load.parquet
CLEAN_PARQUET=data/processed/dvf_processed.parquet

# Commande par défaut (pipeline complet)
all: load preprocess train

# Étape 1 : Ingestion
load:
	@echo "--- [1/3] Ingestion ---"
	python src/data/load.py --input $(RAW_DATA) --output $(LOAD_PARQUET)

# Étape 2 : Prétraitement
preprocess:
	@echo "--- [2/3] Prétraitement ---"
	python src/data/preprocess.py --input $(LOAD_PARQUET) --output $(CLEAN_PARQUET)

# Étape 3 : Entraînement standard
train:
	@echo "--- [3/3] Entraînement standard ---"
	python src/models/train_model.py

# Étape 3bis : Entraînement personnalisé (Exemple : make train-custom N=500 DEPTH=8)
# Utilisation : make train-custom N=500 DEPTH=8
train-custom:
	@echo "--- [3/3] Entraînement personnalisé (N=$(N), Depth=$(DEPTH)) ---"
	python src/models/train_model.py --n_estimators $(N) --max_depth $(DEPTH)

# Nettoyage Windows complet
clean-artifacts:
	del /f /q data\processed\*.parquet
	rmdir /s /q models
	rmdir /s /q experiments

# Aide
help:
	@echo "Commandes disponibles :"
	@echo "  make load           : Exécute l'ingestion."
	@echo "  make preprocess     : Exécute le nettoyage."
	@echo "  make train          : Entraîne avec paramètres par défaut."
	@echo "  make train-custom   : Entraîne avec arguments (ex: make train-custom N=500 DEPTH=8)."
	@echo "  make all            : Pipeline complet."
	@echo "  make clean-artifacts: Supprime tous les fichiers générés."