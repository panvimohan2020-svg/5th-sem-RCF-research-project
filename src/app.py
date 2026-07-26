from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np

app = FastAPI()

# 1. CORS Configuration: Crucial for allowing your HTML frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load your trained model
# Ensure this relative path correctly points to your .pkl file from the /src folder
try:
    model = joblib.load("../models/weather_predictor.pkl")
except FileNotFoundError:
    print("CRITICAL ERROR: Model file not found. Check your relative paths.")

# 3. Define a strict schema requiring exactly 14 numerical features (Precipitation removed)
class WeatherPayload(BaseModel):
    features: List[float] = Field(..., min_items=14, max_items=14)

# 4. The Prediction Endpoint
@app.post("/predict")
def predict_weather(payload: WeatherPayload):
    try:
        # Reshape data for the Random Forest model
        input_data = np.array(payload.features).reshape(1, -1)
        prediction = model.predict(input_data)
        
        # Convert NumPy int64 to standard Python int for JSON serialization
        result = int(prediction[0])
        
        return {"prediction": "Rain" if result == 1 else "No Rain"}
    
    except Exception as e:
        # Catch unexpected mathematical or structural failures cleanly
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")