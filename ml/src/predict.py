import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

ML_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDARD_SCALER_PATH = os.path.join(ML_PATH, 'models', 'standard_scaler.pkl')
LSTM_DELAY_MODEL_PATH = os.path.join(ML_PATH, 'models','lstm_delay_model.h5')
TRAINING_DATA_PATH = os.path.join(ML_PATH, 'data','shipment_data.csv')
XGB_DELAY_PREDICT_PATH = os.path.join(ML_PATH, 'models','xgb_delay_model.pkl')

def predict_delay(input_data):
    features = ["distance_km", "traffic_level", "weather_conditions", "hour_of_day", "weekend"]
    input_df = pd.DataFrame(input_data, columns=features)
    
    # Load scaler and transform input
    scaler = joblib.load(STANDARD_SCALER_PATH)
    input_scaled = scaler.transform(input_df)
    
    # Load XGBoost model and predict
    xgb_model = joblib.load(XGB_DELAY_PREDICT_PATH)
    xgb_prediction = xgb_model.predict_proba(input_scaled)[:, 1]
    
    # Load LSTM model and predict
    lstm_model = tf.keras.models.load_model(LSTM_DELAY_MODEL_PATH)
    input_lstm = np.reshape(input_scaled, (1, 5, input_scaled.shape[1]))
    lstm_prediction = lstm_model.predict(input_lstm)[0][0]
    
    # Combine predictions
    final_prediction = (0.8 * xgb_prediction[0]) + (0.2 * lstm_prediction)
    print("final prediction", final_prediction, int(final_prediction < 0.4), final_prediction<0.4)
    if final_prediction < 0.39:
        return {"delay_prediction": 1}
    else:
        return {"delay_prediction": 0}


# test_input_delayed =[
#     [3200, 0, 0, 7, 0],   # Past - Low traffic, clear weather, early morning, weekday
#     [3300, 0, 0, 8, 0],   # Past - Low traffic, clear weather, morning, weekday
#     [3400, 1, 0, 9, 0],   # Past - Medium traffic, clear weather, morning, weekday
#     [3500, 1, 0, 10, 0],  # Past - Medium traffic, clear weather, morning, weekday
#     [3600, 1, 0, 11, 0],  # 🚀 Prediction - No Delay
# ]

# print(predict_delay(test_input_delayed))