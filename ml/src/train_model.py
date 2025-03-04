from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import numpy as np
from data_processing import load_and_preprocess_data
import os

ML_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DATA_PATH = os.path.join(ML_PATH, 'data','shipment_data.csv')
# Load preprocessed data
X_scaled, y = load_and_preprocess_data(TRAINING_DATA_PATH)

# Train XGBoost model
xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05)
xgb_model.fit(X_scaled, y)
XGB_DELAY_PREDICT_PATH = os.path.join(ML_PATH, 'models','xgb_delay_model.pkl')
joblib.dump(xgb_model, XGB_DELAY_PREDICT_PATH)

# Train LSTM model

time_steps = 5
X_lstm = np.array([X_scaled[i - time_steps:i] for i in range(time_steps, len(X_scaled))])
y_lstm = np.array(y[time_steps:])

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, X_scaled.shape[1])),
    Dropout(0.3),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

LSTM_DELAY_MODEL_PATH = os.path.join(ML_PATH, 'models','lstm_delay_model.h5')
lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, validation_split=0.2)

lstm_model.save(LSTM_DELAY_MODEL_PATH)
print("Training complete. Models saved!")