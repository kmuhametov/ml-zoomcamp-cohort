import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

# Load model components
model_data = joblib.load("models/best_model.pkl")
model = model_data["model"]
dv = model_data["dv"]
categorical = model_data["categorical"]
numerical = model_data["numerical"]

app = FastAPI()


# ---------------------------
# Health check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------
# Prediction endpoint
# ---------------------------
class InputData(BaseModel):
    data: Dict[str, Any]


def preprocess(raw: Dict[str, Any]):
    """Convert raw JSON into model-ready format."""
    df = pd.DataFrame([raw])

    # Ensure numerical columns exist
    for col in numerical:
        if col not in df:
            df[col] = 0

    # Ensure categorical columns exist
    for col in categorical:
        if col not in df:
            df[col] = ""

    df = df[categorical + numerical]
    return df.to_dict(orient="records")


@app.post("/predict")
def predict(payload: InputData):
    record = preprocess(payload.data)
    X = dv.transform(record)
    pred = float(model.predict_proba(X)[0, 1])
    return {"attrition_probability": pred}