import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

ML_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Feature Engineering
    # df["hour_of_day"] = pd.to_datetime(df["date"]).dt.hour
    df["weekend"] = pd.to_datetime(df["date"]).dt.weekday >= 5
    # df["shipment_type"] = np.random.choice([0, 1], len(df))  # 0 = Standard, 1 = Express

    # Encode categorical variables
    traffic_mapping = {"Low": 0, "Medium": 1, "High": 2}
    weather_mapping = {"Clear": 0, "Rain": 1, "Snow": 2, "Fog": 3}
    df.replace({"traffic_level": traffic_mapping, "weather_conditions": weather_mapping}, inplace=True)

    # Select features
    features = ["distance_km", "traffic_level", "weather_conditions", "hour_of_day", "weekend"]
    X = df[features]
    y = df["delay_flag"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    STANDARD_SCALER_PATH = os.path.join(ML_PATH, 'models', 'standard_scaler.pkl')
    # Save the scaler
    joblib.dump(scaler, STANDARD_SCALER_PATH)
    
    return X_scaled, y