import pandas as pd
import numpy as np
import xgboost as xgb
import os
import time
import json
import argparse
import mlflow
import mlflow.xgboost
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "dvf_processed.parquet"
)

PARAMS = {
    "n_estimators": 300,        # ↓ plus léger
    "learning_rate": 0.05,
    "max_depth": 5,            # ↓ important
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "n_jobs": -1,
    "random_state": 42
}

# ============================================================
# TRAIN
# ============================================================

def train_pipeline(args):

    if args.n_estimators:
        PARAMS["n_estimators"] = args.n_estimators

    if args.max_depth:
        PARAMS["max_depth"] = args.max_depth

    mlflow.set_experiment("DVF_XGBoost_Experiment")

    with mlflow.start_run():

        mlflow.log_params(PARAMS)

        start_time = time.time()

        # ======================
        # LOAD + SPLIT
        # ======================

        df = pd.read_parquet(INPUT_PATH)

        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        # ======================
        # FEATURE ENGINEERING
        # ======================

        k = 15
        stats = train_df.groupby("code_commune").agg(
            commune_prix_m2=("prix_m2", "median"),
            commune_volume=("valeur_fonciere", "count")
        ).reset_index()

        global_median = train_df["prix_m2"].median()

        def add_commune_features(data, stats_df, global_val):
            data = data.merge(stats_df, on="code_commune", how="left")
            data["commune_prix_m2"] = data["commune_prix_m2"].fillna(global_val)
            data["commune_volume"] = data["commune_volume"].fillna(0)
            data["commune_prix_m2"] = (data["commune_volume"] * data["commune_prix_m2"] + k * global_val) / (data["commune_volume"] + k)
            return data

        train_df = add_commune_features(train_df, stats, global_median)
        test_df = add_commune_features(test_df, stats, global_median)

        # 2. MODIFICATION : La target est directement le prix au m²
        train_df["target"] = train_df["prix_m2"]
        test_df["target"] = test_df["prix_m2"]

        # 3. On définit les colonnes à supprimer
        drop_cols = [
            "valeur_fonciere",
            "prix_m2",
            "target", # Attention : ne pas supprimer target ici, on le fait après
            "id_mutation",
            "date_mutation"
        ]

        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df["target"] # C'est maintenant le prix direct

        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df["target"]

        # ======================
        # CLEAN FEATURES
        # ======================

        for col in X_train.select_dtypes(include=["object"]).columns:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

        print("Shape dataset :", X_train.shape)

        # ======================
        # TRAIN
        # ======================

        model = xgb.XGBRegressor(
            **PARAMS,
            enable_categorical=True
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=100
        )

        # ======================
        # EVAL
        # ======================

        preds = model.predict(X_test) 

        metrics = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "mape": float(np.mean(np.abs((y_test - preds) / y_test)) * 100)
        }

        mlflow.log_metrics(metrics)

        mlflow.xgboost.log_model(model, "model")

        # ======================
        # SAVE EXPERIMENT ONLY
        # ======================

        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        exp_path = os.path.join(
            BASE_DIR,
            "experiments",
            exp_name
        )

        os.makedirs(exp_path, exist_ok=True)

        with open(
            os.path.join(exp_path, "metrics.json"),
            "w"
        ) as f:
            json.dump({**metrics, "params": PARAMS}, f, indent=4)

        print("\n--- RESULTS ---")
        print(metrics)
        print("\nSaved in:", exp_path)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--max_depth", type=int)

    args = parser.parse_args()

    train_pipeline(args)