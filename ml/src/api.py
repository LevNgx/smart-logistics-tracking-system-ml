from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from predict import predict_delay

app = FastAPI()

class ShipmentInput(BaseModel):
    shipments: List[List[float]]

@app.post("/predict")
def predict(input_data: ShipmentInput):
    if not input_data.shipments:
        return {"error": "No input provided"}
    
    prediction = predict_delay(input_data.shipments)
    return prediction

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)