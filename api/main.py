import os
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# 1. Initialisation
app = FastAPI(title="DVF Prediction API")

# 2. Instrumentation Monitoring
Instrumentator().instrument(app).expose(app)

# 3. Chargement du modèle
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model_production.json")
model = xgb.XGBRegressor()

if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
    print(f"Modèle chargé depuis : {MODEL_PATH}")
else:
    print(f"ATTENTION : Modèle introuvable à {MODEL_PATH}")

# 4. Modèle de données pour les prédictions
class PropertyData(BaseModel):
    surface_reelle_bati: float
    nombre_pieces_principales: int
    surface_terrain: float
    longitude: float
    latitude: float
    commune_prix_m2: float
    commune_volume: int

# 5. Endpoint de prédiction
@app.post("/predict")
async def predict(data: PropertyData):
    try:
        # Conversion des données en DataFrame pour XGBoost
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok"}