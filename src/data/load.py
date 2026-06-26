import pandas as pd
import argparse
import os
import sys

# Calcul dynamique de la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_to_parquet(input_path: str, output_path: str):
    """
    Charge le CSV DVF et convertit en Parquet.
    """
    # 1. Vérification que le fichier existe
    if not os.path.exists(input_path):
        print(f"ERREUR : Fichier introuvable à : {input_path}")
        sys.exit(1)

    cols_to_keep = [
        "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
        "code_departement", "code_commune", "nom_commune",
        "adresse_nom_voie", "type_local", 
        "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain",
        "longitude", "latitude"
    ]

    try:
        print(f"--- Chargement de : {input_path} ---")
        
        # 2. Lecture optimisée
        df = pd.read_csv(input_path, sep=',', usecols=cols_to_keep, low_memory=False)
        
        # 3. Sauvegarde
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        print(f"--- Succès ! {df.shape[0]} lignes enregistrées dans : {output_path} ---")
        
    except Exception as e:
        print(f"Erreur lors du traitement : {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertisseur DVF CSV vers Parquet")
    
    # Argument pour choisir le fichier (par défaut full2025.csv)
    parser.add_argument("--filename", default="full2025.csv", help="Nom du fichier dans data/raw/")
    args = parser.parse_args()

    # Construction des chemins basés sur le nom fourni
    in_path = os.path.join(BASE_DIR, "data", "raw", args.filename)
    out_path = os.path.join(BASE_DIR, "data", "processed", args.filename.replace(".csv", ".parquet"))

    convert_to_parquet(in_path, out_path)