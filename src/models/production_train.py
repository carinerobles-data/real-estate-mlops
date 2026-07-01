import os
import sys

# ==========================================
# CONFIGURATION DE SÉCURITÉ MLFLOW
# (Doit impérativement être avant les autres imports MLflow)
# ==========================================
os.environ["MLFLOW_TRACKING_INSECURE_DISABLE_HOST_CHECK"] = "true"
TRACKING_URI = "http://mlflow:5000"
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI

import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import dvc.api  # Import pour la traçabilité
from sklearn.metrics import mean_absolute_error, r2_score

# Forçage de l'URI de tracking
mlflow.set_tracking_uri(TRACKING_URI)

# ==========================================
# PARAMÈTRES ET CHEMINS
# ==========================================
BASE_DIR = "/app"
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_processed.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

PARAMS = {
    "n_estimators": 100, 
    "max_depth": 4, 
    "learning_rate": 0.05,
    "subsample": 0.8, 
    "colsample_bytree": 0.7, 
    "n_jobs": 2, 
    "random_state": 42
}

def train_production_model():
    print("\n========== PRODUCTION TRAIN ==========")

    # 1. Vérification des données
    if not os.path.exists(INPUT_PATH):
        print(f"ERREUR : Fichier introuvable à {INPUT_PATH}")
        sys.exit(1)

    # 2. Chargement & Feature Engineering
    df = pd.read_parquet(INPUT_PATH)
    
    k = 15
    stats = df.groupby("code_commune").agg(
        commune_prix_m2=("prix_m2", "median"),
        commune_volume=("valeur_fonciere", "count")
    ).reset_index()

    global_median = df["prix_m2"].median()
    df = df.merge(stats, on="code_commune", how="left")
    df["commune_prix_m2"] = ((df["commune_volume"].fillna(0) * df["commune_prix_m2"].fillna(global_median)) + (k * global_median)) / (df["commune_volume"].fillna(0) + k)

    # 3. Préparation Features
    X = df.drop(columns=["valeur_fonciere", "prix_m2", "id_mutation", "date_mutation"], errors="ignore")
    y = df["prix_m2"]

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    # 4. Entraînement
    model = xgb.XGBRegressor(**PARAMS, enable_categorical=True)
    model.fit(X, y)
    
    # 5. Sauvegarde locale (Le point de vérité pour l'API)
    model_path = os.path.join(MODEL_DIR, "model_production.json")
    model.save_model(model_path)
    print(f"Modèle sauvegardé localement : {model_path}")

    # 6. MLflow (Avec traçabilité DVC sécurisée)
    try:
        print("Enregistrement dans MLflow...")
        mlflow.set_experiment("DVF_Production_VF")

        # Récupération sécurisée de l'empreinte DVC
        try:
            data_url = dvc.api.get_url('data/processed/dvf_processed.parquet', repo=BASE_DIR)
        except Exception as dvc_e:
            print(f"Avertissement DVC : Impossible de récupérer l'URL ({dvc_e}). Utilisation de 'local_file'.")
            data_url = "local_file_no_dvc_url"

        with mlflow.start_run(run_name="Production_Refit_Light"):
            mlflow.log_params(PARAMS)
            mlflow.log_param("data_dvc_url", data_url) # Lien Data-Modèle
            
            # Calcul et logging des métriques
            preds = model.predict(X)
            mlflow.log_metrics({
                "mae": mean_absolute_error(y, preds), 
                "r2": r2_score(y, preds)
            })
            
            # Logging du modèle
            mlflow.xgboost.log_model(
                model, 
                artifact_path="model_production",
                registered_model_name="DVF_Production_Model"
            )
        print(f"MLflow OK. Dataset version: {data_url}")
        
    except Exception as e:
        print(f"NOTE : MLflow a échoué. Le pipeline continue. Erreur : {e}")

    print("========== TRAIN FINI ==========")

if __name__ == "__main__":
    train_production_model()