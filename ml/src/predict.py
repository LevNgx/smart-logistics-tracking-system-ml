import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("../models/shipment_delay_model.h5")
scaler = joblib.load("../models/scaler.pkl")

def predict_delay(input_data):
    """
    Predicts shipment delay based on input data.
    """
    # Normalize input
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)

    # Reshape for LSTM
    input_data = np.reshape(input_data, (1, 10, input_data.shape[1]))

    # Predict
    prediction = model.predict(input_data)
    
    return {"delay_prediction": int(prediction[0] > 0.5)}

# Example Test
example_input = [[500, 1, 2, 0]] * 10  # Repeated for 10 timesteps
result = predict_delay(example_input)
print("🚀 Prediction Result:", result)
