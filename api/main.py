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

# 3. Chargement du modèle (Gestion globale)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model_production.json")
model = xgb.XGBRegressor()

def load_model_from_disk():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model.load_model(MODEL_PATH)
            print(f"Modèle chargé avec succès depuis : {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
            return False
    return False

# Chargement initial
load_model_from_disk()

# 4. Modèle de données pour les prédictions
class PropertyData(BaseModel):
    surface_reelle_bati: float
    nombre_pieces_principales: int
    surface_terrain: float
    longitude: float
    latitude: float
    commune_prix_m2: float
    commune_volume: int

# 5. Endpoints
@app.post("/reload_model")
async def reload_model():
    """Endpoint pour Airflow : rechargement du modèle sans redémarrer le conteneur."""
    success = load_model_from_disk()
    if not success:
        raise HTTPException(status_code=500, detail="Échec du rechargement du modèle")
    return {"status": "success", "message": "Modèle mis à jour"}

@app.post("/predict")
async def predict(data: PropertyData):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok"}

# À la toute fin de ton fichier main.py
if __name__ == "__main__":
    import uvicorn
    # Important : host="0.0.0.0" pour être accessible dans le réseau Docker
    uvicorn.run(app, host="0.0.0.0", port=8000)