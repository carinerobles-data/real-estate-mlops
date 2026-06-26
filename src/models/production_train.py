import pandas as pd
import xgboost as xgb
import os
import sys
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, r2_score

# Calcul dynamique de la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemins de production
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_processed.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "n_jobs": -1,
    "random_state": 42
}

def train_production_model():
    print(f"\n--- Lancement PRODUCTION MODEL sur {INPUT_PATH} ---\n")

    if not os.path.exists(INPUT_PATH):
        print(f"ERREUR : Dataset introuvable à {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_parquet(INPUT_PATH)
    
    # --- Feature Engineering ---
    k = 15
    stats = df.groupby("code_commune").agg(
        commune_prix_m2=("prix_m2", "median"),
        commune_volume=("valeur_fonciere", "count")
    ).reset_index()
    global_median = df["prix_m2"].median()

    df = df.merge(stats, on="code_commune", how="left")
    df["commune_prix_m2"] = (df["commune_volume"].fillna(0) * df["commune_prix_m2"].fillna(global_median) + k * global_median) / (df["commune_volume"].fillna(0) + k)

    X = df.drop(columns=["valeur_fonciere", "prix_m2", "id_mutation", "date_mutation", "target"], errors='ignore')
    y = df["target"]

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    # --- Train ---
    model = xgb.XGBRegressor(**PARAMS, enable_categorical=True)
    model.fit(X, y)

    # --- Evaluation ---
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    # --- MLflow (Run unique et propre) ---
    mlflow.set_experiment("DVF_Production_Final")
    with mlflow.start_run(run_name="Production_Refit_Light"):
        mlflow.log_params(PARAMS)
        mlflow.log_metrics({"mae": mae, "r2": r2})
        mlflow.xgboost.log_model(model, "model_production")
        
        # Sauvegarde physique locale
        model_path = os.path.join(MODEL_DIR, "model_production.json")
        model.save_model(model_path)
        
        print(f"Entraînement terminé. MAE: {mae:.2f}, R2: {r2:.4f}")
        print(f"Modèle sauvegardé dans : {model_path}")

if __name__ == "__main__":
    train_production_model()