from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("../models/shipment_delay_model.h5")
scaler = joblib.load("../models/scaler.pkl")

# Initialize API
app = FastAPI()

# Define Input Data Model
class ShipmentData(BaseModel):
    distance_km: float
    traffic_level: int
    weather_conditions: int

@app.post("/predict")
def predict_shipment_delay(data: ShipmentData):
    """
    Predict if a shipment will be delayed.
    """
    input_data = np.array([[data.distance_km, data.traffic_level, data.weather_conditions, 0]])
    input_data = scaler.transform(input_data)
    input_data = np.reshape(input_data, (1, 10, input_data.shape[1]))

    prediction = model.predict(input_data)
    return {"delay_prediction": int(prediction[0] > 0.5)}

# Run with: uvicorn src.api:app --reload
