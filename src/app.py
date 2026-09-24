import os
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import pandas as pd
import logging

# =====================================================================
# ZERO-DAY OOM DEFENSE: Clamp Threading Arenas BEFORE importing BLAS/ML
# Prevents Linux OpenMP/MKL from allocating 300MB+ RAM during startup
# =====================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

logger = logging.getLogger("uvicorn.error")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "weather_predictor.pkl")

# 1. Lifespan Manager for ML Artifacts
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = joblib.load(MODEL_PATH, mmap_mode="r")
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
        gc.collect()
        yield
    except FileNotFoundError:
        logger.critical(f"❌ CRITICAL: weather_predictor.pkl not found at {MODEL_PATH}.")
        raise RuntimeError("Model artifact missing; shutting down.")
    finally:
        app.state.model = None
        gc.collect()

app = FastAPI(title="Gunupur WRI Weather API", version="2.0.0", lifespan=lifespan)

# 2. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Pydantic Schemas (Updated for Frontend Compatibility)
class WeatherInput(BaseModel):
    RH2M: float
    T2MDEW: float
    QV2M: float
    T2MWET: float
    PS: float
    PSC: float
    TS: float
    T2M_MAX: float
    T2M_MIN: float
    ALLSKY_SFC_UV_INDEX: float
    WS50M: float
    WD50M: float
    WSC: float
    LATITUDE: float
    LONGITUDE: float

class ModelMetadataResponse(BaseModel):
    features: List[str]
    feature_count: int
    model_type: str

# 4. The Research Component: Workability Risk Index (WRI)
def calculate_workability_index(rain_prob: float, t2m_max: float, wind_speed: float) -> dict:
    base_score = 10.0
    
    # Mathematical Penalties
    rain_penalty = rain_prob * 6.0
    heat_penalty = max(0.0, (t2m_max - 32.0) * 0.3)
    wind_penalty = max(0.0, (wind_speed - 12.0) * 0.2)
    
    # Final Score (clamped between 0 and 10)
    wri_score = round(max(0.0, base_score - rain_penalty - heat_penalty - wind_penalty), 1)
    
    if wri_score >= 7.5:
        return {
            "score": wri_score,
            "status": "GREEN - Favorable",
            "farmer_advice": "Ideal conditions for sowing and pesticide spraying.",
            "labor_advice": "Safe for all outdoor construction and manual labor."
        }
    elif wri_score >= 4.5:
        return {
            "score": wri_score,
            "status": "YELLOW - Marginal Risk",
            "farmer_advice": "Caution: Avoid spraying chemicals. Rain or wind may cause runoff.",
            "labor_advice": "Plan indoor tasks if possible. Ensure hydration if heat penalty is high."
        }
    else:
        return {
            "score": wri_score,
            "status": "RED - Critical Disruption",
            "farmer_advice": "Halt field operations. High risk of crop damage or chemical runoff.",
            "labor_advice": "High risk of wage loss. Halt outdoor pouring and scaffolding work."
        }

# 5. Metadata Endpoint
@app.get("/api/model-info", response_model=ModelMetadataResponse)
def get_model_metadata(request: Request):
    model = request.app.state.model
    return {
        "features": list(model.feature_names_in_),
        "feature_count": len(model.feature_names_in_),
        "model_type": type(model).__name__
    }

# 6. Prescriptive Inference Endpoint
@app.post("/api/predict")
def predict_weather(payload: WeatherInput, request: Request):
    try:
        model = request.app.state.model
        
        # Dynamically map the Pydantic JSON to the exact column order the model expects
        input_dict = payload.model_dump()
        feature_values = [input_dict[f] for f in model.feature_names_in_]
        
        # Guard against NaN/Infinity
        if not np.isfinite(feature_values).all():
            raise HTTPException(status_code=422, detail="Input features contain NaN or infinite values.")
            
        input_data = pd.DataFrame([feature_values], columns=model.feature_names_in_)

        # Extract Class and Probability
        prediction_class = int(model.predict(input_data)[0])
        probs = model.predict_proba(input_data)[0]
        rain_prob = float(probs[1]) # Probability of class 1 (Rain)
        
        # Calculate Research WRI
        wri_data = calculate_workability_index(
            rain_prob=rain_prob, 
            t2m_max=payload.T2M_MAX, 
            wind_speed=payload.WS50M
        )

        return {
            "prediction_class": prediction_class,
            "rain_probability": round(rain_prob * 100, 2),
            "workability_index": wri_data
        }

    except Exception as e:
        logger.exception("Inference failure")
        raise HTTPException(status_code=500, detail=str(e))

# 7. Serve Frontend Dashboard (Must be at the bottom)
INDEX_DIR = os.path.join(BASE_DIR, "..", "Index")
app.mount("/", StaticFiles(directory=INDEX_DIR, html=True), name="static")