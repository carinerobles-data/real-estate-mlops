import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

BASE_DIR = "/app"
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_processed.parquet")
OUTPUT_REPORT = os.path.join(BASE_DIR, "models", "data_drift_report.html")

def run_drift_analysis():
    print("\n========== DÉTECTION DE DÉRIVE (EVIDENTLY) ==========")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Erreur : Le fichier {INPUT_PATH} n'existe pas.")
        return

    # 1. Chargement du dataset processed
    df = pd.read_parquet(INPUT_PATH)
    
    # On trie par date si la colonne existe pour faire une vraie séparation temporelle
    if "date_mutation" in df.columns:
        df = df.sort_values("date_mutation")
    
    # 2. Séparation 50/50 pour simuler le "Avant" / "Après"
    midpoint = len(df) // 2
    reference_df = df.iloc[:midpoint].copy()
    current_df = df.iloc[midpoint:].copy()
    
    # On nettoie les colonnes inutiles pour l'analyse comme dans ton script d'entraînement
    cols_to_drop = ["id_mutation", "date_mutation"]
    reference_df.drop(columns=cols_to_drop, errors="ignore", inplace=True)
    current_df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    print(font="Metrics analysis started...")
    # 3. Génération du rapport Evidently
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    
    # 4. Sauvegarde du HTML
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    report.save_html(OUTPUT_REPORT)
    
    print(f"✅ Rapport de Data Drift généré avec succès ici : {OUTPUT_REPORT}")
    print("=====================================================")

if __name__ == "__main__":
    run_drift_analysis()