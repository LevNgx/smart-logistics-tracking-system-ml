import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    """
    Load shipment data, preprocess it for LSTM model training.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    print(df)
    df = df.sort_values(by='date')  # Ensure chronological order

    # Convert categorical columns to numerical values
    traffic_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    weather_mapping = {'Clear': 0, 'Rain': 1, 'Snow': 2, 'Fog': 3}
    
    df['traffic_level'] = df['traffic_level'].map(traffic_mapping)
    df['weather_conditions'] = df['weather_conditions'].map(weather_mapping)

    # Select relevant features
    features = df[['distance_km', 'traffic_level', 'weather_conditions', 'delay_flag']]

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    return df, scaled_features, scaler
