import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# =========================
# APP
# =========================

app = FastAPI(title="API DVF Immobilier")

# =========================
# MODEL PATH
# =========================

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_production.json")

# =========================
# REQUEST
# =========================

class PredictionRequest(BaseModel):
    data: dict

# =========================
# MODEL LOADER
# =========================

class ModelLoader:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            print(f"Chargement modèle XGBoost depuis {MODEL_PATH}...")
            self.model = xgb.XGBRegressor()
            self.model.load_model(MODEL_PATH)

    def build_features(self, data: dict):
        df = pd.DataFrame([data])

        # 1. Feature Engineering
        if "nombre_pieces_principales" in df.columns and df["nombre_pieces_principales"].iloc[0] != 0:
            df["surface_par_piece"] = (df["surface_reelle_bati"] / df["nombre_pieces_principales"])
        else:
            df["surface_par_piece"] = 0.0

        # 2. Features par défaut
        defaults = {
            "commune_prix_m2": 0.0, "commune_volume": 0.0,
            "longitude": 0.0, "latitude": 0.0,
            "annee": 2024, "mois": 1,
            "is_maison": 1, "is_neuf": 0, "anciennete_mois": 0
        }
        for col, val in defaults.items():
            df[col] = val

        # 3. Gestion des colonnes catégorielles (crucial pour XGBoost)
        df["code_commune"] = str(data.get("code_commune", "00000"))
        df["code_departement"] = str(data.get("code_departement", "00"))

        # 4. Ordre strict attendu par le modèle
        expected_cols = [
            'surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain',
            'code_departement', 'code_commune', 'longitude', 'latitude',
            'surface_par_piece', 'annee', 'mois', 'is_maison', 'is_neuf',
            'anciennete_mois', 'commune_prix_m2', 'commune_volume'
        ]
        
        # Réindexation et forçage des types catégoriels
        df = df.reindex(columns=expected_cols).fillna(0)
        
        df["code_commune"] = df["code_commune"].astype("category")
        df["code_departement"] = df["code_departement"].astype("category")
        
        return df

    def predict(self, input_data: dict):
        if self.model is None:
            self.load()
        
        df = self.build_features(input_data)
        
        # Appel de la prédiction
        return float(self.model.predict(df)[0])
# =========================
# INSTANCE
# =========================

loader = ModelLoader()

@app.on_event("startup")
def startup():
    loader.load()

# =========================
# HEALTH
# =========================
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction DVF", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": loader.model is not None
    }

# =========================
# PREDICT
# =========================

@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    try:
        prediction = loader.predict(request.data)
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))