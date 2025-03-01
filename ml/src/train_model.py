import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib
import os
from data_processing import load_and_preprocess_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "shipment_data.csv")
# Load and preprocess data
df, scaled_features, scaler = load_and_preprocess_data(DATA_PATH)

# Prepare dataset for LSTM
X, y = [], []
time_steps = 3  # Number of past timesteps used for prediction
print(scaled_features)
for i in range(len(scaled_features) - time_steps):
    X.append(scaled_features[i:i+time_steps])
    y.append(scaled_features[i + time_steps, -1])  # Target: 'delay_flag'

X, y = np.array(X), np.array(y)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "shipment_delay_model.h5")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
# Save trained model
model.save(MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)

print("✅ Model training complete. Model saved!")
