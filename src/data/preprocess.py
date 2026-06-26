import pandas as pd
import numpy as np
import argparse
import os
import sys

# Calcul dynamique de la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_data(input_path: str, output_path: str):
    print(f"--- Début du nettoyage : {input_path} ---")
    
    # 1. Chargement
    if not os.path.exists(input_path):
        print(f"ERREUR : Fichier source introuvable à {input_path}")
        sys.exit(1)
        
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Erreur de lecture du fichier Parquet : {e}")
        sys.exit(1)

    # 2. Nettoyage de base
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
    df["surface_reelle_bati"] = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
    df["surface_terrain"] = df["surface_terrain"].fillna(0)
    df["nombre_pieces_principales"] = pd.to_numeric(df["nombre_pieces_principales"], errors="coerce").fillna(1)

    df = df[df["nature_mutation"].isin(["Vente", "Vente en l'état futur d'achèvement"])].copy()
    df = df[df["type_local"].isin(["Maison", "Appartement"])].copy()

    # 3. Regroupement par ID Mutation
    agg_rules = {
        "surface_reelle_bati": "sum", "nombre_pieces_principales": "sum", "surface_terrain": "sum",
        "valeur_fonciere": "max", "date_mutation": "first", "nature_mutation": "first",
        "type_local": "first", "code_departement": "first", "code_commune": "first",
        "longitude": "first", "latitude": "first"
    }

    df_clean = df.groupby("id_mutation").agg(agg_rules).reset_index()
    df_clean["prix_m2"] = df_clean["valeur_fonciere"] / df_clean["surface_reelle_bati"].replace(0, np.nan)
    df_clean = df_clean.dropna(subset=["prix_m2"])

    # 4. Nettoyage Géographique
    df_clean = df_clean.dropna(subset=["longitude", "latitude"])
    df_clean = df_clean[(df_clean["longitude"] != 0) & (df_clean["latitude"] != 0)].copy()

    # 5. Nettoyage des surfaces et pièces
    df_clean.loc[df_clean["nombre_pieces_principales"] == 0, "nombre_pieces_principales"] = 1
    df_clean["surface_par_piece"] = df_clean["surface_reelle_bati"] / df_clean["nombre_pieces_principales"]

    df_clean = df_clean[
        (df_clean["surface_reelle_bati"] >= 10) & (df_clean["surface_reelle_bati"] <= 500) &
        (df_clean["surface_par_piece"] >= 7) & (df_clean["surface_par_piece"] <= 80)
    ].copy()

    # 6. Filtrage prix
    df_clean = df_clean[
        (df_clean["valeur_fonciere"] >= 20000) & (df_clean["valeur_fonciere"] <= 2000000) &
        (df_clean["prix_m2"] >= 300) & (df_clean["prix_m2"] <= 15000)
    ].copy()

    # 7. Feature Engineering
    df_clean["annee"] = df_clean["date_mutation"].dt.year
    df_clean["mois"] = df_clean["date_mutation"].dt.month
    df_clean["is_maison"] = (df_clean["type_local"] == "Maison").astype(int)
    df_clean["is_neuf"] = (df_clean["nature_mutation"] == "Vente en l'état futur d'achèvement").astype(int)
    
    # Calcul ancienneté dynamique
    df_clean["anciennete_mois"] = (2026 - df_clean["annee"]) * 12 + (6 - df_clean["mois"])

    # Nettoyage final
    df_clean = df_clean.drop(columns=["nature_mutation", "type_local"])
    
    # Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    print(f"--- Succès ! Fichier nettoyé : {output_path} ---")

if __name__ == "__main__":
    # Définition des chemins par défaut basés sur la racine
    default_input = os.path.join(BASE_DIR, "data", "processed", "full2025.parquet")
    default_output = os.path.join(BASE_DIR, "data", "processed", "dvf_processed.parquet")

    parser = argparse.ArgumentParser(description="Script de nettoyage DVF.")
    parser.add_argument("--input", default=default_input, help="Chemin du fichier source.")
    parser.add_argument("--output", default=default_output, help="Chemin du fichier cible.")
    
    args = parser.parse_args()
    preprocess_data(args.input, args.output)