import pandas as pd
import xgboost as xgb
import os
import sys
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# CONFIGURATION
# ==========================================
TRACKING_URI = "http://mlflow:5000"
BASE_DIR = "/app" # Simplifié pour être sûr dans le conteneur
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_processed.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
os.environ["MLFLOW_TRACKING_INSECURE_DISABLE_HOST_CHECK"] = "true"

PARAMS = {
    "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.7, "n_jobs": 2, "random_state": 42
}

def train_production_model():
    print("\n========== PRODUCTION TRAIN ==========")

    # 1. Vérification
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

    # 6. MLflow (Avec gestion d'erreur pour éviter le blocage du pipeline)
    try:
        print("Enregistrement dans MLflow...")
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("DVF_Production_Final")

        with mlflow.start_run(run_name="Production_Refit_Light"):
            mlflow.log_params(PARAMS)
            mlflow.log_metrics({"mae": mean_absolute_error(y, model.predict(X)), "r2": r2_score(y, model.predict(X))})
            
            # Utilisation de artifact_path au lieu de name pour éviter des conflits
            mlflow.xgboost.log_model(
                model, 
                artifact_path="model_production",
                registered_model_name="DVF_Production_Model"
            )
        print("MLflow OK")
    except Exception as e:
        print(f"NOTE : MLflow a échoué, mais le modèle est disponible localement. Erreur : {e}")

    print("========== TRAIN FINI ==========")

if __name__ == "__main__":
    train_production_model()