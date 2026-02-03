from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

# Variables globales pour les artefacts 
MODEL = None
PREPROCESSOR = None
FEATURES_IDX = None
INPUT_COLS = None

@app.on_event("startup")
def load_artifacts():
    global MODEL, PREPROCESSOR, FEATURES_IDX, INPUT_COLS
    # Chargement depuis le dossier artifacts partagé [cite: 56, 80]
    MODEL = pickle.load(open("../artifacts/model.pickle", "rb"))
    PREPROCESSOR = pickle.load(open("../artifacts/scaler.pickle", "rb"))
    FEATURES_IDX = pickle.load(open("../artifacts/pca.pickle", "rb"))
    INPUT_COLS = pickle.load(open("../artifacts/input_columns.pickle", "rb"))

class CustomerData(BaseModel):
    features: dict 

@app.post("/predict")
def predict(data: CustomerData):
    # 1. Mise en DataFrame des données brutes
    df_raw = pd.DataFrame([data.features])
    df_raw = df_raw[INPUT_COLS]
    
    # 2. Pipeline complet : Preprocessing + Sélection [cite: 55]
    X_transformed = PREPROCESSOR.transform(df_raw)
    X_final = X_transformed[:, FEATURES_IDX]
    
    # 3. Prédiction avec votre seuil optimisé
    prob = MODEL.predict_proba(X_final)[0, 1]
    prediction = int(prob >= 0.170202)
    
    return {
        "churn_probability": float(prob),
        "prediction": prediction
    }
