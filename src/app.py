from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("uvicorn.error")

# 1. Lifespan Manager for ML Artifacts
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = joblib.load("../models/weather_predictor.pkl")
        logger.info("Model loaded successfully into app.state.")
        yield
    except FileNotFoundError:
        logger.critical("CRITICAL: weather_predictor.pkl not found.")
        raise RuntimeError("Model artifact missing; shutting down.")
    finally:
        app.state.model = None

app = FastAPI(
    title="Weather Predictor AI API",
    version="1.0.0",
    lifespan=lifespan
)

# 2. CORS Configuration for Local Dev & Production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "*"
    ],
    allow_credentials=False,  # Must be False when "*" is included in allow_origins
    allow_methods=["*"],      # Allows POST, GET, and browser OPTIONS preflight requests
    allow_headers=["*"],
)

# 3. Pydantic Schemas
class ModelMetadataResponse(BaseModel):
    features: List[str]
    feature_count: int
    model_type: str

class WeatherPayload(BaseModel):
    # Enforces exactly 15 numerical inputs matching model.feature_names_in_
    features: List[float] = Field(..., min_length=15, max_length=15)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    raw_class: int

# 4. Metadata Discovery Endpoint (Decouples Frontend from Hardcoded Arrays)
@app.get("/metadata", response_model=ModelMetadataResponse)
def get_model_metadata(request: Request):
    try:
        model = request.app.state.model
        return {
            "features": list(model.feature_names_in_),
            "feature_count": len(model.feature_names_in_),
            "model_type": type(model).__name__
        }
    except AttributeError:
        raise HTTPException(
            status_code=500, 
            detail="Loaded model artifact lacks .feature_names_in_ attribute."
        )

# 5. DataFrame-Aware Inference Endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_weather(payload: WeatherPayload, request: Request):
    try:
        # Step A: Guard against NaN or infinite numerical poisoning
        raw_array = np.array(payload.features, dtype=np.float32)
        if not np.isfinite(raw_array).all():
            raise HTTPException(
                status_code=422, 
                detail="Input features contain NaN or infinite values."
            )

        # Step B: Retrieve the loaded model from application state
        model = request.app.state.model

        # Step C: Dynamically map incoming numbers to the EXACT columns the model was trained on
        input_data = pd.DataFrame([payload.features], columns=model.feature_names_in_)

        # Step D: Execute model prediction
        prediction_class = int(model.predict(input_data)[0])
        
        # Step E: Retrieve probability confidence score if supported by estimator
        probability = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            probability = float(probs[prediction_class])

        return {
            "prediction": "Rain" if prediction_class == 1 else "No Rain",
            "probability": round(probability, 4),
            "raw_class": prediction_class
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Inference failure during /predict execution")
        # Return exact Python exception text to client for transparent debugging
        raise HTTPException(status_code=500, detail=f"Inference crash: {str(e)}")