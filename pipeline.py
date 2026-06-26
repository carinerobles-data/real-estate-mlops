import subprocess
import os

def run_pipeline():
    # Définition des chemins
    # Note : Ajuste le chemin de train.py si besoin selon son emplacement exact
    RAW_DATA = "data/raw/full2025.csv"
    LOAD_PARQUET = "data/processed/dvf_load.parquet"
    CLEAN_PARQUET = "data/processed/dvf_processed.parquet"
    TRAIN_SCRIPT = "src/data/train.py" # Ajusté selon tes indications

    print("--- Lancement du pipeline MLOps complet ---")
    
    # Étape 1 : Ingestion
    print("\n[1/3] Ingestion en cours...")
    subprocess.run([
        "python", "src/data/load.py", 
        "--input", RAW_DATA, 
        "--output", LOAD_PARQUET
    ], check=True)
    
    # Étape 2 : Prétraitement
    print("\n[2/3] Nettoyage en cours...")
    subprocess.run([
        "python", "src/data/preprocess.py", 
        "--input", LOAD_PARQUET, 
        "--output", CLEAN_PARQUET
    ], check=True)
    
    # Étape 3 : Entraînement
    print("\n[3/3] Entraînement en cours...")
    subprocess.run(["python", TRAIN_SCRIPT], check=True)
    
    print("\n--- Pipeline terminé avec succès ! ---")
    print("Les résultats de l'entraînement se trouvent dans le dossier /experiments")

if __name__ == "__main__":
    run_pipeline()