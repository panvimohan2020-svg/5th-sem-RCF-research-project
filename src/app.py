from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np

app = FastAPI()

# 1. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load your trained model (Fail Fast implementation)
# If this file is missing, the server will intentionally crash on startup.
try:
    model = joblib.load("../models/weather_predictor.pkl")
except FileNotFoundError:
    raise RuntimeError("CRITICAL STARTUP ERROR: weather_predictor.pkl not found in ../models/")

# 3. Define a strict schema requiring exactly 14 numerical features
class WeatherPayload(BaseModel):
    features: List[float] = Field(..., min_items=14, max_items=14)

# 4. The Prediction Endpoint
@app.post("/predict")
def predict_weather(payload: WeatherPayload):
    try:
        # Reshape data for the XGBoost model
        input_data = np.array(payload.features).reshape(1, -1)
        prediction = model.predict(input_data)
        
        # Convert NumPy int64/float32 to standard Python int for JSON serialization
        result = int(prediction[0])
        
        return {"prediction": "Rain" if result == 1 else "No Rain"}
    
    except Exception as e:
        # Catch unexpected mathematical or structural failures cleanly
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")